from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class StepKind(str, Enum):
    ACTION = "action"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"


class WorkflowError(Exception):
    pass


class StepExecutionError(WorkflowError):
    pass


class CheckpointNotFoundError(WorkflowError):
    pass


ActionFn = Callable[["WorkflowContext"], Union[Any, Awaitable[Any]]]
ConditionFn = Callable[["WorkflowContext"], Union[bool, Awaitable[bool]]]
RollbackFn = Callable[["WorkflowContext"], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowContext:
    workflow_id: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@dataclass
class StepRecord:
    step_id: str
    name: str
    kind: StepKind
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id, "name": self.name, "kind": self.kind.value,
            "status": self.status.value, "started_at": self.started_at, "ended_at": self.ended_at,
            "error": self.error,
        }


@dataclass
class WorkflowStep:
    name: str
    kind: StepKind = StepKind.ACTION
    action: Optional[ActionFn] = None
    condition: Optional[ConditionFn] = None
    on_true: Optional["WorkflowStep"] = None
    on_false: Optional["WorkflowStep"] = None
    loop_condition: Optional[ConditionFn] = None
    loop_body: Optional["WorkflowStep"] = None
    max_iterations: int = 1000
    parallel_steps: list["WorkflowStep"] = field(default_factory=list)
    rollback: Optional[RollbackFn] = None
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Checkpoint:
    checkpoint_id: str
    workflow_id: str
    created_at: float
    context_snapshot: dict[str, Any]
    completed_step_ids: list[str]


@dataclass
class WorkflowRun:
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    steps: list[StepRecord] = field(default_factory=list)
    context: WorkflowContext = field(default_factory=lambda: WorkflowContext(workflow_id=""))
    error: Optional[str] = None
    executed_step_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id, "status": self.status.value,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "steps": [s.to_dict() for s in self.steps], "error": self.error,
        }


class WorkflowEngine:
    """Executes workflows with branches, loops, parallel steps, checkpoints, and rollback support."""

    def __init__(self, *, max_checkpoints_per_workflow: int = 50) -> None:
        self.max_checkpoints_per_workflow = max_checkpoints_per_workflow
        self._runs: dict[str, WorkflowRun] = {}
        self._step_lookup: dict[str, dict[str, WorkflowStep]] = {}
        self._rollback_stack: dict[str, list[WorkflowStep]] = {}
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._pause_events: dict[str, asyncio.Event] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._run_count = 0

    def _index_steps(self, workflow_id: str, step: WorkflowStep) -> None:
        self._step_lookup.setdefault(workflow_id, {})[step.step_id] = step
        if step.on_true is not None:
            self._index_steps(workflow_id, step.on_true)
        if step.on_false is not None:
            self._index_steps(workflow_id, step.on_false)
        if step.loop_body is not None:
            self._index_steps(workflow_id, step.loop_body)
        for sub_step in step.parallel_steps:
            self._index_steps(workflow_id, sub_step)

    async def _call(self, fn: Optional[Callable[..., Any]], context: WorkflowContext) -> Any:
        if fn is None:
            return None
        result = fn(context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _execute_step(self, workflow_id: str, step: WorkflowStep, run: WorkflowRun) -> Any:
        pause_event = self._pause_events[workflow_id]
        cancel_event = self._cancel_events[workflow_id]
        await pause_event.wait()
        if cancel_event.is_set():
            raise asyncio.CancelledError(f"workflow {workflow_id} cancelled")

        record = StepRecord(step_id=step.step_id, name=step.name, kind=step.kind, status=StepStatus.RUNNING, started_at=time.time())
        run.steps.append(record)

        try:
            if step.kind == StepKind.ACTION:
                result = await self._call(step.action, run.context)
                if step.rollback is not None:
                    self._rollback_stack.setdefault(workflow_id, []).append(step)

            elif step.kind == StepKind.CONDITION:
                cond_result = bool(await self._call(step.condition, run.context))
                branch = step.on_true if cond_result else step.on_false
                if branch is not None:
                    result = await self._execute_step(workflow_id, branch, run)
                else:
                    result = None

            elif step.kind == StepKind.LOOP:
                results = []
                iterations = 0
                while iterations < step.max_iterations:
                    if step.loop_condition is not None:
                        should_continue = bool(await self._call(step.loop_condition, run.context))
                        if not should_continue:
                            break
                    if step.loop_body is not None:
                        body_result = await self._execute_step(workflow_id, step.loop_body, run)
                        results.append(body_result)
                    iterations += 1
                    if step.loop_condition is None:
                        break
                result = results

            elif step.kind == StepKind.PARALLEL:
                tasks = [self._execute_step(workflow_id, sub, run) for sub in step.parallel_steps]
                result = await asyncio.gather(*tasks)

            else:
                raise StepExecutionError(f"unsupported step kind: {step.kind}")

            record.status = StepStatus.COMPLETED
            record.result = result
            record.ended_at = time.time()
            run.executed_step_order.append(step.step_id)
            return result

        except asyncio.CancelledError:
            record.status = StepStatus.SKIPPED
            record.ended_at = time.time()
            raise
        except Exception as exc:
            record.status = StepStatus.FAILED
            record.error = str(exc)
            record.ended_at = time.time()
            raise StepExecutionError(f"step '{step.name}' failed: {exc}") from exc

    async def run(
        self,
        root_step: WorkflowStep,
        *,
        workflow_id: Optional[str] = None,
        initial_data: Optional[dict[str, Any]] = None,
    ) -> WorkflowRun:
        self._run_count += 1
        wf_id = workflow_id or str(uuid.uuid4())
        context = WorkflowContext(workflow_id=wf_id, data=dict(initial_data or {}))
        run = WorkflowRun(workflow_id=wf_id, status=WorkflowStatus.RUNNING, started_at=time.time(), context=context)
        self._runs[wf_id] = run
        self._index_steps(wf_id, root_step)
        self._rollback_stack[wf_id] = []
        self._pause_events[wf_id] = asyncio.Event()
        self._pause_events[wf_id].set()
        self._cancel_events[wf_id] = asyncio.Event()
        self._checkpoints[wf_id] = []

        try:
            await self._execute_step(wf_id, root_step, run)
            run.status = WorkflowStatus.COMPLETED
        except asyncio.CancelledError:
            run.status = WorkflowStatus.CANCELLED
        except StepExecutionError as exc:
            run.status = WorkflowStatus.FAILED
            run.error = str(exc)
        finally:
            run.ended_at = time.time()

        return run

    async def pause(self, workflow_id: str) -> None:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        run.status = WorkflowStatus.PAUSED
        self._pause_events[workflow_id].clear()

    async def resume(self, workflow_id: str) -> None:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        run.status = WorkflowStatus.RUNNING
        self._pause_events[workflow_id].set()

    async def cancel(self, workflow_id: str) -> None:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        self._cancel_events[workflow_id].set()
        self._pause_events[workflow_id].set()

    async def checkpoint(self, workflow_id: str) -> Checkpoint:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4()), workflow_id=workflow_id, created_at=time.time(),
            context_snapshot=dict(run.context.data), completed_step_ids=list(run.executed_step_order),
        )
        bucket = self._checkpoints.setdefault(workflow_id, [])
        bucket.append(cp)
        if len(bucket) > self.max_checkpoints_per_workflow:
            self._checkpoints[workflow_id] = bucket[-self.max_checkpoints_per_workflow :]
        return cp

    def list_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        return list(self._checkpoints.get(workflow_id, []))

    async def restore_checkpoint(self, workflow_id: str, checkpoint_id: str) -> WorkflowContext:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        cp = next((c for c in self._checkpoints.get(workflow_id, []) if c.checkpoint_id == checkpoint_id), None)
        if cp is None:
            raise CheckpointNotFoundError(f"no checkpoint '{checkpoint_id}' for workflow '{workflow_id}'")
        run.context.data = dict(cp.context_snapshot)
        run.executed_step_order = list(cp.completed_step_ids)
        return run.context

    async def rollback(self, workflow_id: str) -> list[str]:
        run = self._runs.get(workflow_id)
        if run is None:
            raise WorkflowError(f"no workflow run with id {workflow_id}")
        rolled_back: list[str] = []
        stack = self._rollback_stack.get(workflow_id, [])
        while stack:
            step = stack.pop()
            try:
                await self._call(step.rollback, run.context)
                rolled_back.append(step.step_id)
                for record in run.steps:
                    if record.step_id == step.step_id:
                        record.status = StepStatus.ROLLED_BACK
            except Exception:
                continue
        run.status = WorkflowStatus.ROLLED_BACK
        return rolled_back

    def get_run(self, workflow_id: str) -> Optional[WorkflowRun]:
        return self._runs.get(workflow_id)

    def list_runs(self, *, status: Optional[WorkflowStatus] = None, limit: int = 200) -> list[WorkflowRun]:
        items = list(self._runs.values())
        if status is not None:
            items = [r for r in items if r.status == status]
        return items[-limit:]

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "run_count": self._run_count,
                "tracked_run_count": len(self._runs),
                "running_count": sum(1 for r in self._runs.values() if r.status == WorkflowStatus.RUNNING),
                "completed_count": sum(1 for r in self._runs.values() if r.status == WorkflowStatus.COMPLETED),
                "failed_count": sum(1 for r in self._runs.values() if r.status == WorkflowStatus.FAILED),
            }
            return HealthStatus(healthy=True, component="workflow_engine", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="workflow_engine", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "WorkflowEngine",
    "WorkflowStep",
    "WorkflowRun",
    "WorkflowContext",
    "WorkflowStatus",
    "StepStatus",
    "StepKind",
    "StepRecord",
    "Checkpoint",
    "WorkflowError",
    "StepExecutionError",
    "CheckpointNotFoundError",
    "HealthStatus",
]
