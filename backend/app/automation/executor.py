from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Protocol, Union, runtime_checkable


class ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class ExecutorError(Exception):
    pass


class ExecutionNotFoundError(ExecutorError):
    pass


class ApprovalDeniedError(ExecutorError):
    pass


ActionFn = Callable[..., Union[Any, Awaitable[Any]]]
RollbackFn = Callable[[Any], Union[Any, Awaitable[Any]]]


@runtime_checkable
class ApprovalProvider(Protocol):
    async def request(self, action: str, requested_by: str, **kwargs: Any) -> Any: ...
    async def wait_for_resolution(self, request_id: str, *, timeout: Optional[float] = None) -> Any: ...


@runtime_checkable
class ArtifactProvider(Protocol):
    async def save(self, name: str, data: Any, **kwargs: Any) -> str: ...


@runtime_checkable
class LoggerProvider(Protocol):
    def info(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...


@runtime_checkable
class MetricsProvider(Protocol):
    def increment(self, name: str, value: float = 1.0, **kwargs: Any) -> None: ...
    def observe(self, name: str, value: float, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionRecord:
    execution_id: str
    name: str
    state: ExecutionState = ExecutionState.PENDING
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    attempt: int = 0
    max_retries: int = 0
    timeout_seconds: Optional[float] = None
    artifacts: list[str] = field(default_factory=list)
    approval_request_id: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id, "name": self.name, "state": self.state.value,
            "started_at": self.started_at, "ended_at": self.ended_at, "duration_seconds": self.duration_seconds,
            "error": self.error, "attempt": self.attempt, "max_retries": self.max_retries,
            "artifacts": self.artifacts, "approval_request_id": self.approval_request_id,
        }


class Executor:
    """Executes actions with timeout/retry/rollback handling and optional approval/artifact/logging/metrics integration."""

    def __init__(
        self,
        *,
        approval_provider: Optional[ApprovalProvider] = None,
        artifact_provider: Optional[ArtifactProvider] = None,
        logger: Optional[LoggerProvider] = None,
        metrics: Optional[MetricsProvider] = None,
        default_timeout_seconds: float = 60.0,
        default_max_retries: int = 0,
        default_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.approval_provider = approval_provider
        self.artifact_provider = artifact_provider
        self.logger = logger
        self.metrics = metrics
        self.default_timeout_seconds = default_timeout_seconds
        self.default_max_retries = default_max_retries
        self.default_retry_delay_seconds = default_retry_delay_seconds

        self._records: dict[str, ExecutionRecord] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._rollback_fns: dict[str, RollbackFn] = {}
        self._rollback_args: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._execute_count = 0
        self._retry_count_total = 0
        self._failure_count = 0

    def _log(self, level: str, message: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(message, **fields)
        except Exception:
            pass

    def _metric(self, kind: str, name: str, value: float = 1.0, **kwargs: Any) -> None:
        if self.metrics is None:
            return
        try:
            getattr(self.metrics, kind)(name, value, **kwargs)
        except Exception:
            pass

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def execute(
        self,
        action: ActionFn,
        *,
        name: Optional[str] = None,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        requires_approval: bool = False,
        approval_requested_by: str = "system",
        approval_timeout_seconds: Optional[float] = None,
        rollback: Optional[RollbackFn] = None,
        save_result_as_artifact: bool = False,
        artifact_name: Optional[str] = None,
    ) -> ExecutionRecord:
        self._execute_count += 1
        execution_id = str(uuid.uuid4())
        label = name or getattr(action, "__name__", "action")
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        effective_max_retries = max_retries if max_retries is not None else self.default_max_retries
        effective_retry_delay = retry_delay_seconds if retry_delay_seconds is not None else self.default_retry_delay_seconds

        record = ExecutionRecord(
            execution_id=execution_id, name=label, max_retries=effective_max_retries,
            timeout_seconds=effective_timeout,
        )
        self._records[execution_id] = record
        self._cancel_events[execution_id] = asyncio.Event()

        if requires_approval:
            if self.approval_provider is None:
                record.state = ExecutionState.FAILED
                record.error = "approval required but no approval_provider configured"
                self._log("error", record.error, execution_id=execution_id)
                return record

            record.state = ExecutionState.AWAITING_APPROVAL
            self._log("info", f"requesting approval for execution '{label}'", execution_id=execution_id)
            approval_request = await self.approval_provider.request(
                label, approval_requested_by, payload={"execution_id": execution_id}
            )
            request_id = getattr(approval_request, "request_id", None) or str(approval_request)
            record.approval_request_id = request_id

            resolution = await self.approval_provider.wait_for_resolution(request_id, timeout=approval_timeout_seconds)
            status_value = getattr(resolution, "status", None)
            status_str = getattr(status_value, "value", str(status_value))
            if status_str != "approved":
                record.state = ExecutionState.FAILED
                record.error = f"approval not granted (status={status_str})"
                self._log("error", record.error, execution_id=execution_id)
                raise ApprovalDeniedError(record.error)

        record.state = ExecutionState.RUNNING
        record.started_at = time.time()

        attempt = 0
        while True:
            attempt += 1
            record.attempt = attempt
            try:
                if self._cancel_events[execution_id].is_set():
                    record.state = ExecutionState.CANCELLED
                    record.ended_at = time.time()
                    return record

                result = await asyncio.wait_for(
                    self._call(action, *args, **(kwargs or {})), timeout=effective_timeout
                )
                record.result = result
                record.state = ExecutionState.COMPLETED
                record.ended_at = time.time()

                if rollback is not None:
                    self._rollback_fns[execution_id] = rollback
                    self._rollback_args[execution_id] = result

                if save_result_as_artifact and self.artifact_provider is not None:
                    ref = await self.artifact_provider.save(artifact_name or f"{label}_result", result)
                    record.artifacts.append(ref)

                self._metric("observe", "executor.duration_seconds", record.duration_seconds or 0.0, labels={"name": label})
                self._metric("increment", "executor.success_count", labels={"name": label})
                self._log("info", f"execution '{label}' completed", execution_id=execution_id, attempt=attempt)
                return record

            except asyncio.TimeoutError:
                record.error = f"execution exceeded timeout of {effective_timeout}s"
                self._log("error", record.error, execution_id=execution_id, attempt=attempt)
                if attempt > effective_max_retries:
                    record.state = ExecutionState.TIMED_OUT
                    record.ended_at = time.time()
                    self._failure_count += 1
                    self._metric("increment", "executor.timeout_count", labels={"name": label})
                    return record
                self._retry_count_total += 1
                await asyncio.sleep(effective_retry_delay)

            except asyncio.CancelledError:
                record.state = ExecutionState.CANCELLED
                record.ended_at = time.time()
                return record

            except Exception as exc:
                record.error = str(exc)
                self._log("error", f"execution '{label}' failed: {exc}", execution_id=execution_id, attempt=attempt)
                if attempt > effective_max_retries:
                    record.state = ExecutionState.FAILED
                    record.ended_at = time.time()
                    self._failure_count += 1
                    self._metric("increment", "executor.failure_count", labels={"name": label})
                    return record
                self._retry_count_total += 1
                self._metric("increment", "executor.retry_count", labels={"name": label})
                await asyncio.sleep(effective_retry_delay)

    async def cancel(self, execution_id: str) -> bool:
        if execution_id not in self._records:
            raise ExecutionNotFoundError(f"no execution with id {execution_id}")
        event = self._cancel_events.get(execution_id)
        if event is not None:
            event.set()
        record = self._records[execution_id]
        if record.state in (ExecutionState.PENDING, ExecutionState.RUNNING, ExecutionState.AWAITING_APPROVAL):
            record.state = ExecutionState.CANCELLED
            record.ended_at = time.time()
        return True

    async def retry(self, execution_id: str, action: ActionFn, **execute_kwargs: Any) -> ExecutionRecord:
        original = self._records.get(execution_id)
        if original is None:
            raise ExecutionNotFoundError(f"no execution with id {execution_id}")
        self._retry_count_total += 1
        return await self.execute(action, name=f"{original.name}.retry", **execute_kwargs)

    async def rollback(self, execution_id: str) -> ExecutionRecord:
        record = self._records.get(execution_id)
        if record is None:
            raise ExecutionNotFoundError(f"no execution with id {execution_id}")
        rollback_fn = self._rollback_fns.get(execution_id)
        if rollback_fn is None:
            raise ExecutorError(f"no rollback function registered for execution '{execution_id}'")

        try:
            await self._call(rollback_fn, self._rollback_args.get(execution_id))
            record.state = ExecutionState.ROLLED_BACK
            self._log("info", f"execution '{record.name}' rolled back", execution_id=execution_id)
        except Exception as exc:
            record.error = f"rollback failed: {exc}"
            self._log("error", record.error, execution_id=execution_id)
            raise ExecutorError(record.error) from exc
        return record

    def get(self, execution_id: str) -> Optional[ExecutionRecord]:
        return self._records.get(execution_id)

    def list_executions(self, *, state: Optional[ExecutionState] = None, limit: int = 200) -> list[ExecutionRecord]:
        items = list(self._records.values())
        if state is not None:
            items = [r for r in items if r.state == state]
        return items[-limit:]

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "execution_count": self._execute_count,
                "tracked_record_count": len(self._records),
                "retry_count_total": self._retry_count_total,
                "failure_count": self._failure_count,
                "approval_provider_configured": self.approval_provider is not None,
                "artifact_provider_configured": self.artifact_provider is not None,
                "logger_configured": self.logger is not None,
                "metrics_configured": self.metrics is not None,
            }
            return HealthStatus(healthy=True, component="executor", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="executor", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Executor",
    "ExecutionRecord",
    "ExecutionState",
    "ApprovalProvider",
    "ArtifactProvider",
    "LoggerProvider",
    "MetricsProvider",
    "ExecutorError",
    "ExecutionNotFoundError",
    "ApprovalDeniedError",
    "HealthStatus",
]
