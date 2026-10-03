from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class CapabilityStatus(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class CapabilityPriority(IntEnum):
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


class CapabilityError(Exception):
    def __init__(self, capability_name: str, message: str) -> None:
        self.capability_name = capability_name
        super().__init__(f"[{capability_name}] {message}")


class CapabilityNotInitializedError(CapabilityError):
    pass


class CapabilityTimeoutError(CapabilityError):
    pass


class CapabilityApprovalDeniedError(CapabilityError):
    pass


class CapabilityExecutionError(CapabilityError):
    pass


@dataclass(frozen=True)
class CapabilityMetadata:
    name: str
    version: str
    description: str = ""
    priority: CapabilityPriority = CapabilityPriority.NORMAL
    timeout_seconds: float = 30.0
    approval_required: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    author: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthCheckResult:
    capability_name: str
    status: HealthStatus
    latency_seconds: Optional[float]
    checked_at_epoch: float
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    timeout_seconds: Optional[float] = None
    requested_at_epoch: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ExecutionResult:
    request_id: str
    capability_name: str
    action: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    completed_at_epoch: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalCallback:
    async def request_approval(
        self, capability_name: str, request: ExecutionRequest
    ) -> bool:
        raise NotImplementedError


class AutoApproveCallback(ApprovalCallback):
    async def request_approval(
        self, capability_name: str, request: ExecutionRequest
    ) -> bool:
        return True


class ComputerCapability(ABC):
    def __init__(
        self,
        metadata: CapabilityMetadata,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        self._metadata = metadata
        self._status = CapabilityStatus.UNINITIALIZED
        self._approval_callback = approval_callback or AutoApproveCallback()
        self._last_health_check: Optional[HealthCheckResult] = None
        self._initialized_at_epoch: Optional[float] = None

    @property
    def metadata(self) -> CapabilityMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def priority(self) -> CapabilityPriority:
        return self._metadata.priority

    @property
    def timeout_seconds(self) -> float:
        return self._metadata.timeout_seconds

    @property
    def approval_required(self) -> bool:
        return self._metadata.approval_required

    @property
    def status(self) -> CapabilityStatus:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status in (CapabilityStatus.READY, CapabilityStatus.DEGRADED)

    async def initialize(self) -> None:
        if self._status in (CapabilityStatus.READY, CapabilityStatus.DEGRADED):
            return
        self._status = CapabilityStatus.INITIALIZING
        try:
            await self._on_initialize()
            self._status = CapabilityStatus.READY
            self._initialized_at_epoch = time.time()
        except Exception as exc:
            self._status = CapabilityStatus.ERROR
            raise CapabilityError(self.name, f"initialization failed: {exc}") from exc

    async def shutdown(self) -> None:
        if self._status in (CapabilityStatus.SHUTDOWN, CapabilityStatus.UNINITIALIZED):
            self._status = CapabilityStatus.SHUTDOWN
            return
        self._status = CapabilityStatus.SHUTTING_DOWN
        try:
            await self._on_shutdown()
        finally:
            self._status = CapabilityStatus.SHUTDOWN

    async def health_check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            if not self.is_ready and self._status != CapabilityStatus.UNINITIALIZED:
                result = HealthCheckResult(
                    capability_name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_seconds=None,
                    checked_at_epoch=time.time(),
                    detail=f"capability status is {self._status.value}",
                )
                self._last_health_check = result
                return result

            health = await self._on_health_check()
            latency = time.monotonic() - start
            result = HealthCheckResult(
                capability_name=self.name,
                status=health.status,
                latency_seconds=latency,
                checked_at_epoch=time.time(),
                detail=health.detail,
                metadata=health.metadata,
            )
            self._last_health_check = result
            if health.status == HealthStatus.UNHEALTHY:
                self._status = CapabilityStatus.UNHEALTHY
            elif health.status == HealthStatus.DEGRADED and self._status == CapabilityStatus.READY:
                self._status = CapabilityStatus.DEGRADED
            return result
        except Exception as exc:
            result = HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.UNKNOWN,
                latency_seconds=time.monotonic() - start,
                checked_at_epoch=time.time(),
                detail=str(exc),
            )
            self._last_health_check = result
            return result

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if not self.is_ready:
            raise CapabilityNotInitializedError(
                self.name, f"capability is not ready (status={self._status.value})"
            )

        if self._metadata.approval_required:
            approved = await self._approval_callback.request_approval(self.name, request)
            if not approved:
                return ExecutionResult(
                    request_id=request.request_id,
                    capability_name=self.name,
                    action=request.action,
                    success=False,
                    error="approval denied",
                    duration_seconds=0.0,
                )

        timeout = request.timeout_seconds or self._metadata.timeout_seconds
        start = time.monotonic()
        try:
            import asyncio

            result_value = await asyncio.wait_for(self._on_execute(request), timeout=timeout)
            duration = time.monotonic() - start
            return ExecutionResult(
                request_id=request.request_id,
                capability_name=self.name,
                action=request.action,
                success=True,
                result=result_value,
                duration_seconds=duration,
            )
        except TimeoutError:
            duration = time.monotonic() - start
            return ExecutionResult(
                request_id=request.request_id,
                capability_name=self.name,
                action=request.action,
                success=False,
                error=f"execution timed out after {timeout}s",
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return ExecutionResult(
                request_id=request.request_id,
                capability_name=self.name,
                action=request.action,
                success=False,
                error=str(exc),
                duration_seconds=duration,
            )

    @abstractmethod
    async def _on_initialize(self) -> None: ...

    @abstractmethod
    async def _on_shutdown(self) -> None: ...

    @abstractmethod
    async def _on_health_check(self) -> HealthCheckResult: ...

    @abstractmethod
    async def _on_execute(self, request: ExecutionRequest) -> Any: ...
