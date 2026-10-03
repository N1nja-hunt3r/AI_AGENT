from __future__ import annotations

import asyncio
import heapq
import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class ScheduleKind(str, Enum):
    ONE_TIME = "one_time"
    DELAYED = "delayed"
    PERIODIC = "periodic"
    CRON = "cron"


class ScheduledTaskState(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SchedulerError(Exception):
    pass


class ScheduledTaskNotFoundError(SchedulerError):
    pass


CronNextFn = Callable[[str, float, Optional[str]], float]
CallableFn = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScheduledTask:
    schedule_id: str
    name: str
    kind: ScheduleKind
    func: CallableFn
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    next_run_at: float = field(default_factory=time.time)
    interval_seconds: Optional[float] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None
    max_retries: int = 0
    retry_delay_seconds: float = 5.0
    retry_count: int = 0
    state: ScheduledTaskState = ScheduledTaskState.SCHEDULED
    run_count: int = 0
    last_run_at: Optional[float] = None
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    max_runs: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "kind": self.kind.value,
            "priority": self.priority,
            "next_run_at": self.next_run_at,
            "interval_seconds": self.interval_seconds,
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "state": self.state.value,
            "run_count": self.run_count,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "max_runs": self.max_runs,
        }


@dataclass(order=True)
class _HeapEntry:
    next_run_at: float
    priority: int
    seq: int
    schedule_id: str = field(compare=False)


class Scheduler:
    """Async task scheduler supporting one-time, delayed, periodic, recurring, and cron schedules."""

    def __init__(self, *, cron_next_fn: Optional[CronNextFn] = None, tick_interval_seconds: float = 0.5) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._heap: list[_HeapEntry] = []
        self._seq_counter = itertools.count()
        self._cron_next_fn = cron_next_fn
        self.tick_interval_seconds = tick_interval_seconds
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._schedule_count = 0
        self._execution_count = 0
        self._failure_count = 0

    def _push_heap(self, task: ScheduledTask) -> None:
        heapq.heappush(self._heap, _HeapEntry(task.next_run_at, task.priority, next(self._seq_counter), task.schedule_id))

    def schedule(
        self,
        func: CallableFn,
        *,
        name: Optional[str] = None,
        kind: ScheduleKind = ScheduleKind.ONE_TIME,
        run_at: Optional[float] = None,
        delay_seconds: Optional[float] = None,
        interval_seconds: Optional[float] = None,
        cron_expression: Optional[str] = None,
        timezone: Optional[str] = None,
        priority: int = 100,
        max_retries: int = 0,
        retry_delay_seconds: float = 5.0,
        max_runs: Optional[int] = None,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> ScheduledTask:
        self._schedule_count += 1
        schedule_id = str(uuid.uuid4())
        now = time.time()

        if kind == ScheduleKind.ONE_TIME:
            next_run = run_at if run_at is not None else now
        elif kind == ScheduleKind.DELAYED:
            if delay_seconds is None:
                raise SchedulerError("delay_seconds is required for DELAYED schedules")
            next_run = now + delay_seconds
        elif kind == ScheduleKind.PERIODIC:
            if interval_seconds is None or interval_seconds <= 0:
                raise SchedulerError("interval_seconds (>0) is required for PERIODIC schedules")
            next_run = run_at if run_at is not None else now + interval_seconds
        elif kind == ScheduleKind.CRON:
            if cron_expression is None:
                raise SchedulerError("cron_expression is required for CRON schedules")
            if self._cron_next_fn is None:
                raise SchedulerError("no cron_next_fn configured; pass one to Scheduler() to enable CRON schedules")
            next_run = self._cron_next_fn(cron_expression, now, timezone)
        else:
            raise SchedulerError(f"unsupported schedule kind: {kind}")

        task = ScheduledTask(
            schedule_id=schedule_id, name=name or func.__class__.__name__, kind=kind, func=func,
            args=args, kwargs=kwargs or {}, priority=priority, next_run_at=next_run,
            interval_seconds=interval_seconds, cron_expression=cron_expression, timezone=timezone,
            max_retries=max_retries, retry_delay_seconds=retry_delay_seconds, max_runs=max_runs,
        )
        self._tasks[schedule_id] = task
        self._push_heap(task)
        return task

    def cancel(self, schedule_id: str) -> bool:
        task = self._tasks.get(schedule_id)
        if task is None:
            raise ScheduledTaskNotFoundError(f"no scheduled task with id {schedule_id}")
        task.state = ScheduledTaskState.CANCELLED
        return True

    def pause(self, schedule_id: str) -> bool:
        task = self._tasks.get(schedule_id)
        if task is None:
            raise ScheduledTaskNotFoundError(f"no scheduled task with id {schedule_id}")
        if task.state == ScheduledTaskState.SCHEDULED:
            task.state = ScheduledTaskState.PAUSED
        return True

    def resume(self, schedule_id: str) -> bool:
        task = self._tasks.get(schedule_id)
        if task is None:
            raise ScheduledTaskNotFoundError(f"no scheduled task with id {schedule_id}")
        if task.state == ScheduledTaskState.PAUSED:
            task.state = ScheduledTaskState.SCHEDULED
            self._push_heap(task)
        return True

    def reschedule(
        self,
        schedule_id: str,
        *,
        run_at: Optional[float] = None,
        delay_seconds: Optional[float] = None,
        interval_seconds: Optional[float] = None,
        cron_expression: Optional[str] = None,
    ) -> ScheduledTask:
        task = self._tasks.get(schedule_id)
        if task is None:
            raise ScheduledTaskNotFoundError(f"no scheduled task with id {schedule_id}")

        if run_at is not None:
            task.next_run_at = run_at
        elif delay_seconds is not None:
            task.next_run_at = time.time() + delay_seconds
        elif interval_seconds is not None:
            task.interval_seconds = interval_seconds
            task.next_run_at = time.time() + interval_seconds
        elif cron_expression is not None:
            if self._cron_next_fn is None:
                raise SchedulerError("no cron_next_fn configured; pass one to Scheduler() to enable CRON schedules")
            task.cron_expression = cron_expression
            task.next_run_at = self._cron_next_fn(cron_expression, time.time(), task.timezone)
        else:
            raise SchedulerError("must provide run_at, delay_seconds, interval_seconds, or cron_expression")

        if task.state in (ScheduledTaskState.COMPLETED, ScheduledTaskState.CANCELLED, ScheduledTaskState.FAILED):
            task.state = ScheduledTaskState.SCHEDULED
        self._push_heap(task)
        return task

    def get(self, schedule_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(schedule_id)

    def list_tasks(self, *, state: Optional[ScheduledTaskState] = None, kind: Optional[ScheduleKind] = None) -> list[ScheduledTask]:
        items = list(self._tasks.values())
        if state is not None:
            items = [t for t in items if t.state == state]
        if kind is not None:
            items = [t for t in items if t.kind == kind]
        return sorted(items, key=lambda t: (t.priority, t.next_run_at))

    async def _execute_task(self, task: ScheduledTask) -> None:
        if task.state in (ScheduledTaskState.CANCELLED, ScheduledTaskState.PAUSED):
            return

        task.state = ScheduledTaskState.RUNNING
        self._execution_count += 1
        try:
            result = task.func(*task.args, **task.kwargs)
            if asyncio.iscoroutine(result):
                await result
            task.run_count += 1
            task.last_run_at = time.time()
            task.retry_count = 0
            task.last_error = None

            if task.max_runs is not None and task.run_count >= task.max_runs:
                task.state = ScheduledTaskState.COMPLETED
                return

            if task.kind == ScheduleKind.ONE_TIME or task.kind == ScheduleKind.DELAYED:
                task.state = ScheduledTaskState.COMPLETED
            elif task.kind == ScheduleKind.PERIODIC:
                task.next_run_at = time.time() + (task.interval_seconds or 0)
                task.state = ScheduledTaskState.SCHEDULED
                self._push_heap(task)
            elif task.kind == ScheduleKind.CRON:
                if self._cron_next_fn is not None and task.cron_expression is not None:
                    task.next_run_at = self._cron_next_fn(task.cron_expression, time.time(), task.timezone)
                task.state = ScheduledTaskState.SCHEDULED
                self._push_heap(task)
        except Exception as exc:
            self._failure_count += 1
            task.last_error = str(exc)
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                task.next_run_at = time.time() + task.retry_delay_seconds
                task.state = ScheduledTaskState.SCHEDULED
                self._push_heap(task)
            else:
                task.state = ScheduledTaskState.FAILED

    async def run_due_tasks(self) -> int:
        """Run all currently due tasks once; returns count of tasks executed."""
        now = time.time()
        executed = 0
        pending_reinsert: list[_HeapEntry] = []

        while self._heap and self._heap[0].next_run_at <= now:
            entry = heapq.heappop(self._heap)
            task = self._tasks.get(entry.schedule_id)
            if task is None:
                continue
            if task.state in (ScheduledTaskState.CANCELLED, ScheduledTaskState.PAUSED):
                continue
            if task.next_run_at != entry.next_run_at:
                continue
            await self._execute_task(task)
            executed += 1

        for entry in pending_reinsert:
            heapq.heappush(self._heap, entry)
        return executed

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while self._running:
            async with self._lock:
                await self.run_due_tasks()
            await asyncio.sleep(self.tick_interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "task_count": len(self._tasks),
                "heap_size": len(self._heap),
                "running": self._running,
                "schedule_count": self._schedule_count,
                "execution_count": self._execution_count,
                "failure_count": self._failure_count,
                "scheduled_state_count": sum(1 for t in self._tasks.values() if t.state == ScheduledTaskState.SCHEDULED),
                "failed_state_count": sum(1 for t in self._tasks.values() if t.state == ScheduledTaskState.FAILED),
            }
            return HealthStatus(healthy=True, component="scheduler", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="scheduler", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "Scheduler",
    "ScheduledTask",
    "ScheduleKind",
    "ScheduledTaskState",
    "SchedulerError",
    "ScheduledTaskNotFoundError",
    "HealthStatus",
]
