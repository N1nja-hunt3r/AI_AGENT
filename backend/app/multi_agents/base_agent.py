from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class AgentStatus(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class AgentPriority(IntEnum):
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AgentError(Exception):
    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")


class AgentNotInitializedError(AgentError):
    pass


class AgentTimeoutError(AgentError):
    pass


class AgentApprovalDeniedError(AgentError):
    pass


class AgentExecutionError(AgentError):
    pass


@dataclass(frozen=True)
class AgentMetadata:
    name: str
    version: str
    description: str = ""
    priority: AgentPriority = AgentPriority.NORMAL
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    timeout_seconds: float = 60.0
    approval_required: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    max_concurrent_tasks: int = 1
    author: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthCheckResult:
    agent_name: str
    status: HealthStatus
    latency_seconds: Optional[float]
    checked_at_epoch: float
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    requested_at_epoch: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    agent_name: str
    action: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    completed_at_epoch: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalCallback:
    async def request_approval(self, agent_name: str, task: AgentTask) -> bool:
        raise NotImplementedError


class AutoApproveCallback(ApprovalCallback):
    async def request_approval(self, agent_name: str, task: AgentTask) -> bool:
        return True


class Agent(ABC):
    def __init__(
        self,
        metadata: AgentMetadata,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        self._metadata = metadata
        self._status = AgentStatus.UNINITIALIZED
        self._approval_callback = approval_callback or AutoApproveCallback()
        self._last_health_check: Optional[HealthCheckResult] = None
        self._initialized_at_epoch: Optional[float] = None
        self._active_task_count = 0

    @property
    def metadata(self) -> AgentMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def priority(self) -> AgentPriority:
        return self._metadata.priority

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._metadata.capabilities

    @property
    def timeout_seconds(self) -> float:
        return self._metadata.timeout_seconds

    @property
    def approval_required(self) -> bool:
        return self._metadata.approval_required

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status in (AgentStatus.IDLE, AgentStatus.BUSY, AgentStatus.DEGRADED)

    @property
    def is_available(self) -> bool:
        return (
            self._status in (AgentStatus.IDLE, AgentStatus.DEGRADED)
            and self._active_task_count < self._metadata.max_concurrent_tasks
        )

    def supports(self, capability: str) -> bool:
        return capability in self._metadata.capabilities

    async def initialize(self) -> None:
        if self._status in (AgentStatus.IDLE, AgentStatus.BUSY, AgentStatus.DEGRADED):
            return
        self._status = AgentStatus.INITIALIZING
        try:
            await self._on_initialize()
            self._status = AgentStatus.IDLE
            self._initialized_at_epoch = time.time()
        except Exception as exc:
            self._status = AgentStatus.ERROR
            raise AgentError(self.name, f"initialization failed: {exc}") from exc

    async def shutdown(self) -> None:
        if self._status in (AgentStatus.SHUTDOWN, AgentStatus.UNINITIALIZED):
            self._status = AgentStatus.SHUTDOWN
            return
        self._status = AgentStatus.SHUTTING_DOWN
        try:
            await self._on_shutdown()
        finally:
            self._status = AgentStatus.SHUTDOWN

    async def health_check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            if not self.is_ready and self._status != AgentStatus.UNINITIALIZED:
                result = HealthCheckResult(
                    agent_name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_seconds=None,
                    checked_at_epoch=time.time(),
                    detail=f"agent status is {self._status.value}",
                )
                self._last_health_check = result
                return result

            health = await self._on_health_check()
            latency = time.monotonic() - start
            result = HealthCheckResult(
                agent_name=self.name,
                status=health.status,
                latency_seconds=latency,
                checked_at_epoch=time.time(),
                detail=health.detail,
                metadata=health.metadata,
            )
            self._last_health_check = result
            if health.status == HealthStatus.UNHEALTHY:
                self._status = AgentStatus.UNHEALTHY
            elif health.status == HealthStatus.DEGRADED and self._status == AgentStatus.IDLE:
                self._status = AgentStatus.DEGRADED
            return result
        except Exception as exc:
            result = HealthCheckResult(
                agent_name=self.name,
                status=HealthStatus.UNKNOWN,
                latency_seconds=time.monotonic() - start,
                checked_at_epoch=time.time(),
                detail=str(exc),
            )
            self._last_health_check = result
            return result

    async def execute(self, task: AgentTask) -> AgentResult:
        if not self.is_ready:
            raise AgentNotInitializedError(
                self.name, f"agent is not ready (status={self._status.value})"
            )
        if self._active_task_count >= self._metadata.max_concurrent_tasks:
            raise AgentExecutionError(
                self.name, "agent has reached max concurrent task capacity"
            )

        if self._metadata.approval_required:
            approved = await self._approval_callback.request_approval(self.name, task)
            if not approved:
                return AgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    action=task.action,
                    success=False,
                    error="approval denied",
                    duration_seconds=0.0,
                )

        timeout = task.timeout_seconds or self._metadata.timeout_seconds
        self._active_task_count += 1
        if self._status == AgentStatus.IDLE:
            self._status = AgentStatus.BUSY

        start = time.monotonic()
        try:
            import asyncio

            result_value = await asyncio.wait_for(self._on_execute(task), timeout=timeout)
            duration = time.monotonic() - start
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                action=task.action,
                success=True,
                result=result_value,
                duration_seconds=duration,
            )
        except TimeoutError:
            duration = time.monotonic() - start
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                action=task.action,
                success=False,
                error=f"execution timed out after {timeout}s",
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                action=task.action,
                success=False,
                error=str(exc),
                duration_seconds=duration,
            )
        finally:
            self._active_task_count = max(0, self._active_task_count - 1)
            if self._status == AgentStatus.BUSY and self._active_task_count == 0:
                self._status = AgentStatus.IDLE

    @abstractmethod
    async def _on_initialize(self) -> None: ...

    @abstractmethod
    async def _on_shutdown(self) -> None: ...

    @abstractmethod
    async def _on_health_check(self) -> HealthCheckResult: ...

    @abstractmethod
    async def _on_execute(self, task: AgentTask) -> Any: ...
