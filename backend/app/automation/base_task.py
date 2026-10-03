from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"
    SHUTDOWN = "shutdown"


class TaskError(Exception):
    pass


class InvalidTransitionError(TaskError):
    pass


class TaskTimeoutError(TaskError):
    pass


_VALID_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.INITIALIZING, TaskStatus.CANCELLED}),
    TaskStatus.INITIALIZING: frozenset({TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.AWAITING_APPROVAL: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.PAUSED, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SHUTDOWN}),
    TaskStatus.PAUSED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.SHUTDOWN}),
    TaskStatus.COMPLETED: frozenset({TaskStatus.SHUTDOWN}),
    TaskStatus.FAILED: frozenset({TaskStatus.SHUTDOWN, TaskStatus.PENDING}),
    TaskStatus.CANCELLED: frozenset({TaskStatus.SHUTDOWN}),
    TaskStatus.SHUTDOWN: frozenset(),
}


@dataclass
class TaskMetadata:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_task"
    version: str = "1.0.0"
    priority: int = 100
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    dependencies: frozenset[str] = field(default_factory=frozenset)
    approval_required: bool = False
    artifact_support: bool = False
    created_at: float = field(default_factory=time.time)
    tags: frozenset[str] = field(default_factory=frozenset)
    owner: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "version": self.version,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "dependencies": sorted(self.dependencies),
            "approval_required": self.approval_required,
            "artifact_support": self.artifact_support,
            "created_at": self.created_at,
            "tags": sorted(self.tags),
            "owner": self.owner,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    artifacts: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 6)


class BaseTask(ABC):
    """Abstract base class for all automation tasks with a managed lifecycle."""

    def __init__(self, metadata: Optional[TaskMetadata] = None) -> None:
        self.metadata: TaskMetadata = metadata or TaskMetadata()
        self.status: TaskStatus = TaskStatus.PENDING
        self._started_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._lock = asyncio.Lock()
        self._last_error: Optional[str] = None
        self._artifacts: list[str] = []
        self._created_at = time.time()
        self._status_history: list[tuple[TaskStatus, float]] = [(TaskStatus.PENDING, self._created_at)]

    def _transition(self, new_status: TaskStatus) -> None:
        allowed = _VALID_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"task '{self.metadata.task_id}' cannot transition from {self.status.value} to {new_status.value}"
            )
        self.status = new_status
        self._status_history.append((new_status, time.time()))

    @property
    def status_history(self) -> list[tuple[TaskStatus, float]]:
        return list(self._status_history)

    @abstractmethod
    async def _on_initialize(self) -> None:
        """Subclasses implement setup logic here."""
        raise NotImplementedError

    @abstractmethod
    async def _on_execute(self) -> Any:
        """Subclasses implement core task logic here. Must respect cancellation/pause checkpoints."""
        raise NotImplementedError

    async def _on_cancel(self) -> None:
        """Subclasses may override to release resources on cancellation."""
        return None

    async def _on_pause(self) -> None:
        """Subclasses may override for pause-time bookkeeping."""
        return None

    async def _on_resume(self) -> None:
        """Subclasses may override for resume-time bookkeeping."""
        return None

    async def _on_shutdown(self) -> None:
        """Subclasses may override for teardown logic."""
        return None

    async def checkpoint(self) -> None:
        """Call periodically inside _on_execute to honor pause/cancel signals."""
        if self._cancel_event.is_set():
            raise asyncio.CancelledError(f"task '{self.metadata.task_id}' was cancelled")
        await self._pause_event.wait()

    async def initialize(self) -> None:
        async with self._lock:
            self._transition(TaskStatus.INITIALIZING)
            try:
                await self._on_initialize()
            except Exception as exc:
                self._last_error = str(exc)
                self._transition(TaskStatus.FAILED)
                raise TaskError(f"initialization failed for task '{self.metadata.task_id}': {exc}") from exc

            if self.metadata.approval_required:
                self._transition(TaskStatus.AWAITING_APPROVAL)
            else:
                self._transition(TaskStatus.RUNNING)

    async def execute(self) -> TaskResult:
        if self.status not in (TaskStatus.RUNNING,):
            if self.status == TaskStatus.AWAITING_APPROVAL:
                raise TaskError(f"task '{self.metadata.task_id}' is awaiting approval and cannot execute yet")
            if self.status == TaskStatus.PENDING:
                await self.initialize()
            else:
                raise InvalidTransitionError(f"cannot execute task in status {self.status.value}")

        self._started_at = time.time()
        try:
            output = await asyncio.wait_for(self._run_with_checkpoints(), timeout=self.metadata.timeout_seconds)
            self._ended_at = time.time()
            self._transition(TaskStatus.COMPLETED)
            return TaskResult(
                task_id=self.metadata.task_id, status=self.status, output=output,
                started_at=self._started_at, ended_at=self._ended_at, artifacts=list(self._artifacts),
            )
        except asyncio.TimeoutError as exc:
            self._ended_at = time.time()
            self._last_error = f"task exceeded timeout of {self.metadata.timeout_seconds}s"
            self._transition(TaskStatus.FAILED)
            raise TaskTimeoutError(self._last_error) from exc
        except asyncio.CancelledError:
            self._ended_at = time.time()
            self._transition(TaskStatus.CANCELLED)
            return TaskResult(
                task_id=self.metadata.task_id, status=self.status, error="cancelled",
                started_at=self._started_at, ended_at=self._ended_at, artifacts=list(self._artifacts),
            )
        except Exception as exc:
            self._ended_at = time.time()
            self._last_error = str(exc)
            self._transition(TaskStatus.FAILED)
            return TaskResult(
                task_id=self.metadata.task_id, status=self.status, error=str(exc),
                started_at=self._started_at, ended_at=self._ended_at, artifacts=list(self._artifacts),
            )

    async def _run_with_checkpoints(self) -> Any:
        await self.checkpoint()
        return await self._on_execute()

    async def cancel(self) -> None:
        async with self._lock:
            if self.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.SHUTDOWN):
                return
            self._cancel_event.set()
            self._pause_event.set()
            await self._on_cancel()
            if self.status in _VALID_TRANSITIONS and TaskStatus.CANCELLED in _VALID_TRANSITIONS[self.status]:
                self._transition(TaskStatus.CANCELLED)
            self._ended_at = time.time()

    async def pause(self) -> None:
        async with self._lock:
            self._transition(TaskStatus.PAUSED)
            self._pause_event.clear()
            await self._on_pause()

    async def resume(self) -> None:
        async with self._lock:
            self._transition(TaskStatus.RUNNING)
            self._pause_event.set()
            await self._on_resume()

    async def shutdown(self) -> None:
        async with self._lock:
            if self.status == TaskStatus.SHUTDOWN:
                return
            self._cancel_event.set()
            self._pause_event.set()
            await self._on_shutdown()
            if TaskStatus.SHUTDOWN in _VALID_TRANSITIONS.get(self.status, frozenset()):
                self._transition(TaskStatus.SHUTDOWN)
            else:
                self.status = TaskStatus.SHUTDOWN
                self._status_history.append((TaskStatus.SHUTDOWN, time.time()))

    def add_artifact(self, artifact_ref: str) -> None:
        if not self.metadata.artifact_support:
            raise TaskError(f"task '{self.metadata.task_id}' does not support artifacts")
        self._artifacts.append(artifact_ref)

    def can_retry(self) -> bool:
        return self.metadata.retry_count < self.metadata.max_retries

    def increment_retry(self) -> int:
        self.metadata.retry_count += 1
        return self.metadata.retry_count

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "task_id": self.metadata.task_id,
                "name": self.metadata.name,
                "version": self.metadata.version,
                "status": self.status.value,
                "priority": self.metadata.priority,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "timeout_seconds": self.metadata.timeout_seconds,
                "dependencies": sorted(self.metadata.dependencies),
                "approval_required": self.metadata.approval_required,
                "artifact_support": self.metadata.artifact_support,
                "artifact_count": len(self._artifacts),
                "last_error": self._last_error,
                "uptime_seconds": round(time.time() - self._created_at, 2),
            }
            healthy = self.status not in (TaskStatus.FAILED,)
            return HealthStatus(healthy=healthy, component=f"task:{self.metadata.name}", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component=f"task:{self.metadata.name}", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "BaseTask",
    "TaskMetadata",
    "TaskStatus",
    "TaskResult",
    "TaskError",
    "InvalidTransitionError",
    "TaskTimeoutError",
    "HealthStatus",
]
