"""
executor_connector.py - Production-grade executor connector with graph, checkpoints, rollback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"
    AWAITING_APPROVAL = "awaiting_approval"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class CheckpointType(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"


class ApprovalResult(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStep:
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    index: int = 0
    name: str = ""
    description: str = ""
    capability: Optional[str] = None
    action: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    requires_approval: bool = False
    rollback_fn: Optional[str] = None
    retries: int = 3
    timeout: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0


@dataclass
class Checkpoint:
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = ""
    step_id: Optional[str] = None
    checkpoint_type: CheckpointType = CheckpointType.AUTO
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class Artifact:
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = ""
    step_id: str = ""
    name: str = ""
    content: Any = None
    artifact_type: str = "generic"
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionGraph:
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[ExecutionStep] = field(default_factory=list)
    edges: Dict[str, List[str]] = field(default_factory=dict)  # step_id -> [dep_step_ids]

    def add_step(self, step: ExecutionStep) -> None:
        self.steps.append(step)
        self.edges[step.step_id] = step.dependencies

    def get_ready_steps(self) -> List[ExecutionStep]:
        completed_ids: Set[str] = {
            s.step_id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.ROLLED_BACK)
            for s in self.steps
        )

    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


@dataclass
class ExecutionRecord:
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    graph: Optional[ExecutionGraph] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    total_tokens: int = 0
    cost_usd: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0
    checkpoints: List[Checkpoint] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorConnectorConfig:
    max_parallel_steps: int = 4
    default_step_timeout: float = 60.0
    default_step_retries: int = 3
    retry_delay: float = 1.0
    enable_checkpoints: bool = True
    checkpoint_dir: str = ".checkpoints"
    enable_artifacts: bool = True
    artifact_dir: str = ".artifacts"
    enable_approval: bool = False
    approval_timeout: float = 300.0
    enable_budget: bool = True
    enable_rollback: bool = True
    enable_event_bus: bool = True
    execution_timeout: float = 600.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExecutorConnectorError(Exception):
    pass

class StepExecutionError(ExecutorConnectorError):
    pass

class CheckpointError(ExecutorConnectorError):
    pass

class RollbackError(ExecutorConnectorError):
    pass

class ApprovalDeniedError(ExecutorConnectorError):
    pass

class ExecutionBudgetExceededError(ExecutorConnectorError):
    pass


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class _CheckpointManager:
    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._lock = asyncio.Lock()

    async def save(self, checkpoint: Checkpoint) -> str:
        async with self._lock:
            cps = self._checkpoints.setdefault(checkpoint.execution_id, [])
            cps.append(checkpoint)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{checkpoint.execution_id}_{checkpoint.checkpoint_id}.json"
            data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "execution_id": checkpoint.execution_id,
                "step_id": checkpoint.step_id,
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "state": checkpoint.state,
                "created_at": checkpoint.created_at,
                "description": checkpoint.description,
            }
            await asyncio.to_thread(path.write_text, json.dumps(data, default=str))
        except Exception as exc:
            logger.warning("Checkpoint write failed: %s", exc)
        return checkpoint.checkpoint_id

    async def load(self, execution_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        cps = self._checkpoints.get(execution_id, [])
        if checkpoint_id:
            return next((c for c in cps if c.checkpoint_id == checkpoint_id), None)
        return cps[-1] if cps else None

    async def list_checkpoints(self, execution_id: str) -> List[Checkpoint]:
        return list(self._checkpoints.get(execution_id, []))

    async def delete(self, execution_id: str) -> None:
        async with self._lock:
            self._checkpoints.pop(execution_id, None)


# ---------------------------------------------------------------------------
# Artifact Manager
# ---------------------------------------------------------------------------

class _ArtifactManager:
    def __init__(self, artifact_dir: str) -> None:
        self._dir = Path(artifact_dir)
        self._artifacts: Dict[str, List[Artifact]] = {}
        self._lock = asyncio.Lock()

    async def save(self, artifact: Artifact) -> str:
        async with self._lock:
            arts = self._artifacts.setdefault(artifact.execution_id, [])
            arts.append(artifact)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{artifact.execution_id}_{artifact.artifact_id}_{artifact.name}"
            content = artifact.content
            if isinstance(content, (dict, list)):
                content = json.dumps(content, default=str)
            if isinstance(content, str):
                artifact.size_bytes = len(content.encode())
                await asyncio.to_thread(path.write_text, content)
            elif isinstance(content, bytes):
                artifact.size_bytes = len(content)
                await asyncio.to_thread(path.write_bytes, content)
        except Exception as exc:
            logger.warning("Artifact write failed: %s", exc)
        return artifact.artifact_id

    async def get(self, artifact_id: str, execution_id: str) -> Optional[Artifact]:
        arts = self._artifacts.get(execution_id, [])
        return next((a for a in arts if a.artifact_id == artifact_id), None)

    async def list_artifacts(self, execution_id: str) -> List[Artifact]:
        return list(self._artifacts.get(execution_id, []))


# ---------------------------------------------------------------------------
# Approval Manager
# ---------------------------------------------------------------------------

class _ApprovalManager:
    def __init__(self, timeout: float = 300.0) -> None:
        self._timeout = timeout
        self._pending: Dict[str, asyncio.Future] = {}  # type: ignore[type-arg]
        self._lock = asyncio.Lock()

    async def request_approval(
        self,
        step: ExecutionStep,
        execution_id: str,
        approval_callback: Optional[Callable[..., Any]] = None,
    ) -> ApprovalResult:
        approval_id = f"{execution_id}:{step.step_id}"
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()  # type: ignore[type-arg]

        async with self._lock:
            self._pending[approval_id] = future

        logger.info(
            "Awaiting approval for step '%s' [%s] timeout=%.0fs",
            step.name, approval_id, self._timeout,
        )

        if approval_callback:
            try:
                if asyncio.iscoroutinefunction(approval_callback):
                    asyncio.create_task(approval_callback(step, approval_id))
                else:
                    asyncio.get_event_loop().run_in_executor(None, approval_callback, step, approval_id)
            except Exception as exc:
                logger.warning("Approval callback error: %s", exc)

        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
            return ApprovalResult(result)
        except asyncio.TimeoutError:
            return ApprovalResult.TIMEOUT
        finally:
            async with self._lock:
                self._pending.pop(approval_id, None)

    async def respond(self, approval_id: str, approved: bool) -> bool:
        async with self._lock:
            future = self._pending.get(approval_id)
        if future and not future.done():
            future.set_result(ApprovalResult.APPROVED.value if approved else ApprovalResult.REJECTED.value)
            return True
        return False


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class _EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[..., Any]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event: str, handler: Callable[..., Any]) -> None:
        async with self._lock:
            self._subscribers.setdefault(event, []).append(handler)

    async def publish(self, event: str, payload: Dict[str, Any]) -> None:
        handlers = self._subscribers.get(event, []) + self._subscribers.get("*", [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event, payload))
                else:
                    await asyncio.to_thread(handler, event, payload)
            except Exception as exc:
                logger.warning("EventBus handler error for '%s': %s", event, exc)


# ---------------------------------------------------------------------------
# Step executor
# ---------------------------------------------------------------------------

class _StepExecutor:
    def __init__(
        self,
        executor: Any,
        llm_connector: Any,
        capability_connector: Any,
        config: ExecutorConnectorConfig,
    ) -> None:
        self._executor = executor
        self._llm = llm_connector
        self._caps = capability_connector
        self._config = config
        self._rollback_registry: Dict[str, Callable[..., Any]] = {}

    def register_rollback(self, name: str, fn: Callable[..., Any]) -> None:
        self._rollback_registry[name] = fn

    async def execute_step(
        self,
        step: ExecutionStep,
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        last_exc: Optional[Exception] = None

        for attempt in range(step.retries):
            try:
                result = await asyncio.wait_for(
                    self._run_step(step, execution_context),
                    timeout=step.timeout,
                )
                step.status = StepStatus.COMPLETED
                step.completed_at = time.time()
                step.latency_ms = (step.completed_at - step.started_at) * 1000
                step.outputs = result if isinstance(result, dict) else {"result": result}
                return step.outputs
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning("Step '%s' timeout (attempt %d/%d)", step.name, attempt + 1, step.retries)
            except Exception as exc:
                last_exc = exc
                logger.warning("Step '%s' failed: %s (attempt %d/%d)", step.name, exc, attempt + 1, step.retries)
            if attempt < step.retries - 1:
                await asyncio.sleep(self._config.retry_delay * (2 ** attempt))

        step.status = StepStatus.FAILED
        step.error = str(last_exc)
        step.completed_at = time.time()
        if step.started_at:
            step.latency_ms = (step.completed_at - step.started_at) * 1000
        raise StepExecutionError(f"Step '{step.name}' failed after {step.retries} attempts: {last_exc}")

    async def _run_step(
        self,
        step: ExecutionStep,
        execution_context: Dict[str, Any],
    ) -> Any:
        # 1. Try registered executor
        if self._executor and hasattr(self._executor, "execute_step"):
            try:
                return await self._executor.execute_step(step, context=execution_context)
            except Exception as exc:
                logger.debug("Primary executor failed for step '%s': %s", step.name, exc)

        # 2. Try capability connector
        if step.capability and self._caps:
            try:
                result = await self._caps.execute(
                    step.capability,
                    step.action or "execute",
                    **{**step.inputs, **execution_context},
                )
                if result.success:
                    return result.result
            except Exception as exc:
                logger.debug("Capability execution failed for step '%s': %s", step.name, exc)

        # 3. LLM fallback
        if self._llm:
            messages = [
                {"role": "system", "content": "Execute the following task and return the result."},
                {"role": "user", "content": f"Task: {step.description}\nInputs: {json.dumps(step.inputs, default=str)}"},
            ]
            response = await self._llm.complete(messages=messages)
            return {
                "result": response.text,
                "tokens": response.total_tokens,
                "cost": response.cost_usd,
            }

        return {"result": f"Step '{step.name}' completed (no executor)", "tokens": 0, "cost": 0.0}

    async def rollback_step(self, step: ExecutionStep, context: Dict[str, Any]) -> bool:
        if step.rollback_fn and step.rollback_fn in self._rollback_registry:
            fn = self._rollback_registry[step.rollback_fn]
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(step, context)
                else:
                    await asyncio.to_thread(fn, step, context)
                step.status = StepStatus.ROLLED_BACK
                logger.info("Rolled back step '%s'", step.name)
                return True
            except Exception as exc:
                logger.error("Rollback failed for step '%s': %s", step.name, exc)
                raise RollbackError(f"Rollback failed for '{step.name}': {exc}") from exc

        if step.status in (StepStatus.COMPLETED, StepStatus.FAILED):
            step.status = StepStatus.ROLLED_BACK
            logger.info("Marked step '%s' as rolled back (no rollback_fn)", step.name)
            return True

        return False


# ---------------------------------------------------------------------------
# ExecutorConnector – Singleton
# ---------------------------------------------------------------------------

class ExecutorConnector:
    _instance: Optional["ExecutorConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = ExecutorConnectorConfig()
        self._checkpoint_mgr: Optional[_CheckpointManager] = None
        self._artifact_mgr: Optional[_ArtifactManager] = None
        self._approval_mgr: Optional[_ApprovalManager] = None
        self._event_bus: Optional[_EventBus] = None
        self._step_executor: Optional[_StepExecutor] = None
        self._initialized = False
        self._active_executions: Dict[str, ExecutionRecord] = {}
        self._execution_history: Dict[str, ExecutionRecord] = {}
        self._exec_lock = asyncio.Lock()

        # Injected
        self._executor: Any = None
        self._llm: Any = None
        self._capability_connector: Any = None
        self._budget_manager: Any = None
        self._approval_callback: Optional[Callable[..., Any]] = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ExecutorConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "ExecutorConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: Optional[ExecutorConnectorConfig] = None,
        executor: Any = None,
        llm_connector: Any = None,
        capability_connector: Any = None,
        budget_manager: Any = None,
        approval_callback: Optional[Callable[..., Any]] = None,
    ) -> "ExecutorConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        self._executor = executor
        self._llm = llm_connector
        self._capability_connector = capability_connector
        self._budget_manager = budget_manager
        self._approval_callback = approval_callback

        self._checkpoint_mgr = _CheckpointManager(self._config.checkpoint_dir)
        self._artifact_mgr = _ArtifactManager(self._config.artifact_dir)
        self._approval_mgr = _ApprovalManager(timeout=self._config.approval_timeout)
        self._event_bus = _EventBus()
        self._step_executor = _StepExecutor(
            executor=executor,
            llm_connector=llm_connector,
            capability_connector=capability_connector,
            config=self._config,
        )
        self._initialized = True
        logger.info("ExecutorConnector initialized: checkpoints=%s artifacts=%s approval=%s",
                    self._config.enable_checkpoints, self._config.enable_artifacts, self._config.enable_approval)
        return self

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        steps: List[Dict[str, Any]],
        execution_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        budget_usd: Optional[float] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionRecord:
        await self._ensure_initialized()

        # Build graph
        graph = ExecutionGraph()
        for idx, s in enumerate(steps):
            step = ExecutionStep(
                index=idx,
                name=s.get("name", f"step_{idx}"),
                description=s.get("description", str(s)),
                capability=s.get("capability"),
                action=s.get("action"),
                inputs=s.get("inputs", {}),
                dependencies=s.get("dependencies", []),
                requires_approval=s.get("requires_approval", False),
                rollback_fn=s.get("rollback_fn"),
                retries=s.get("retries", self._config.default_step_retries),
                timeout=s.get("timeout", self._config.default_step_timeout),
                metadata=s.get("metadata", {}),
            )
            graph.add_step(step)

        record = ExecutionRecord(
            execution_id=execution_id or str(uuid.uuid4()),
            graph=graph,
            status=ExecutionStatus.RUNNING,
            started_at=time.time(),
            metadata=metadata or {},
        )

        async with self._exec_lock:
            self._active_executions[record.execution_id] = record

        await self._publish("execution.started", {"execution_id": record.execution_id})

        exec_timeout = timeout or self._config.execution_timeout
        try:
            await asyncio.wait_for(
                self._run_execution(record, context or {}, budget_usd),
                timeout=exec_timeout,
            )
        except asyncio.TimeoutError:
            record.status = ExecutionStatus.FAILED
            record.error = f"Execution timed out after {exec_timeout}s"
            await self._publish("execution.timeout", {"execution_id": record.execution_id})
        except Exception as exc:
            record.status = ExecutionStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            logger.error("Execution [%s] failed:\n%s", record.execution_id, traceback.format_exc())
            await self._publish("execution.failed", {"execution_id": record.execution_id, "error": str(exc)})
        finally:
            record.completed_at = time.time()
            if record.started_at:
                record.latency_ms = (record.completed_at - record.started_at) * 1000
            async with self._exec_lock:
                self._active_executions.pop(record.execution_id, None)
                self._execution_history[record.execution_id] = record

        return record

    async def _run_execution(
        self,
        record: ExecutionRecord,
        context: Dict[str, Any],
        budget_usd: Optional[float],
    ) -> None:
        assert record.graph is not None
        graph = record.graph
        execution_context = dict(context)
        total_cost = 0.0

        while not graph.is_complete():
            ready = graph.get_ready_steps()
            if not ready:
                if graph.has_failures():
                    record.status = ExecutionStatus.FAILED
                    record.error = "Execution halted due to step failures"
                    return
                break  # deadlock guard

            # Checkpoint before batch
            if self._config.enable_checkpoints and self._checkpoint_mgr:
                cp = Checkpoint(
                    execution_id=record.execution_id,
                    checkpoint_type=CheckpointType.AUTO,
                    state={
                        "context": execution_context,
                        "step_statuses": {s.step_id: s.status.value for s in graph.steps},
                    },
                    description=f"Before executing {len(ready)} steps",
                )
                cp_id = await self._checkpoint_mgr.save(cp)
                record.checkpoints.append(cp)
                logger.debug("Checkpoint saved: %s", cp_id)

            # Execute ready steps in parallel (up to max_parallel)
            sem = asyncio.Semaphore(self._config.max_parallel_steps)
            tasks = [
                self._execute_single_step(step, record, execution_context, budget_usd, sem, total_cost)
                for step in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(ready, results):
                if isinstance(result, Exception):
                    step.status = StepStatus.FAILED
                    step.error = str(result)
                    record.steps_failed += 1
                    await self._publish("step.failed", {
                        "execution_id": record.execution_id,
                        "step_id": step.step_id,
                        "error": str(result),
                    })
                    if self._config.enable_rollback:
                        await self._rollback_completed_steps(record, execution_context)
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Step '{step.name}' failed: {result}"
                    return
                elif isinstance(result, dict):
                    step.outputs = result
                    execution_context[f"step_{step.index}_result"] = result.get("result")
                    step_cost = result.get("cost", 0.0)
                    total_cost += step_cost
                    record.cost_usd += step_cost
                    record.total_tokens += result.get("tokens", 0)
                    record.steps_completed += 1

                    # Save artifact
                    if self._config.enable_artifacts and self._artifact_mgr and result.get("result"):
                        artifact = Artifact(
                            execution_id=record.execution_id,
                            step_id=step.step_id,
                            name=f"step_{step.index}_output",
                            content=result.get("result"),
                            artifact_type="step_output",
                        )
                        await self._artifact_mgr.save(artifact)
                        record.artifacts.append(artifact)

                    # Budget check
                    if budget_usd and total_cost >= budget_usd:
                        raise ExecutionBudgetExceededError(
                            f"Budget ${budget_usd:.4f} exceeded at ${total_cost:.4f}"
                        )

        if graph.has_failures():
            record.status = ExecutionStatus.FAILED
        else:
            record.status = ExecutionStatus.COMPLETED
            all_results = [
                s.outputs.get("result", "") for s in graph.steps
                if s.status == StepStatus.COMPLETED
            ]
            record.result = "\n".join(str(r) for r in all_results if r)
            await self._publish("execution.completed", {"execution_id": record.execution_id})

    async def _execute_single_step(
        self,
        step: ExecutionStep,
        record: ExecutionRecord,
        context: Dict[str, Any],
        budget_usd: Optional[float],
        sem: asyncio.Semaphore,
        total_cost: float,
    ) -> Dict[str, Any]:
        async with sem:
            await self._publish("step.started", {
                "execution_id": record.execution_id,
                "step_id": step.step_id,
                "step_name": step.name,
            })

            # Approval gate
            if self._config.enable_approval and step.requires_approval and self._approval_mgr:
                step.status = StepStatus.AWAITING_APPROVAL
                approval = await self._approval_mgr.request_approval(
                    step, record.execution_id, self._approval_callback
                )
                if approval != ApprovalResult.APPROVED:
                    raise StepExecutionError(
                        f"Step '{step.name}' not approved: {approval.value}"
                    )

            assert self._step_executor is not None
            result = await self._step_executor.execute_step(step, context)
            await self._publish("step.completed", {
                "execution_id": record.execution_id,
                "step_id": step.step_id,
            })
            return result

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback(
        self,
        execution_id: str,
        target_checkpoint_id: Optional[str] = None,
    ) -> bool:
        await self._ensure_initialized()
        record = (
            self._active_executions.get(execution_id)
            or self._execution_history.get(execution_id)
        )
        if not record or not record.graph:
            logger.warning("Rollback: execution '%s' not found", execution_id)
            return False

        if self._config.enable_checkpoints and target_checkpoint_id and self._checkpoint_mgr:
            cp = await self._checkpoint_mgr.load(execution_id, target_checkpoint_id)
            if cp:
                logger.info("Rolling back to checkpoint %s", target_checkpoint_id)
                for step in record.graph.steps:
                    saved_status = cp.state.get("step_statuses", {}).get(step.step_id)
                    if saved_status:
                        step.status = StepStatus(saved_status)

        await self._rollback_completed_steps(record, {})
        record.status = ExecutionStatus.ROLLED_BACK
        await self._publish("execution.rolled_back", {"execution_id": execution_id})
        return True

    async def _rollback_completed_steps(
        self,
        record: ExecutionRecord,
        context: Dict[str, Any],
    ) -> None:
        if not record.graph or not self._step_executor:
            return
        completed = [
            s for s in reversed(record.graph.steps)
            if s.status == StepStatus.COMPLETED
        ]
        for step in completed:
            try:
                await self._step_executor.rollback_step(step, context)
            except RollbackError as exc:
                logger.error("Rollback error: %s", exc)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    async def checkpoint(
        self,
        execution_id: str,
        description: str = "",
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        await self._ensure_initialized()
        if not self._checkpoint_mgr:
            return None
        record = (
            self._active_executions.get(execution_id)
            or self._execution_history.get(execution_id)
        )
        graph_state: Dict[str, Any] = {}
        if record and record.graph:
            graph_state = {
                "step_statuses": {s.step_id: s.status.value for s in record.graph.steps},
                "steps_completed": record.steps_completed,
            }
        cp = Checkpoint(
            execution_id=execution_id,
            checkpoint_type=CheckpointType.MANUAL,
            state={**(state or {}), **graph_state},
            description=description,
        )
        cp_id = await self._checkpoint_mgr.save(cp)
        if record:
            record.checkpoints.append(cp)
        return cp_id

    async def restore_checkpoint(self, execution_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        if not self._checkpoint_mgr:
            return None
        cp = await self._checkpoint_mgr.load(execution_id, checkpoint_id)
        return cp.state if cp else None

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def save_artifact(
        self,
        execution_id: str,
        step_id: str,
        name: str,
        content: Any,
        artifact_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        await self._ensure_initialized()
        if not self._artifact_mgr:
            return None
        artifact = Artifact(
            execution_id=execution_id,
            step_id=step_id,
            name=name,
            content=content,
            artifact_type=artifact_type,
            metadata=metadata or {},
        )
        return await self._artifact_mgr.save(artifact)

    async def get_artifacts(self, execution_id: str) -> List[Artifact]:
        await self._ensure_initialized()
        if not self._artifact_mgr:
            return []
        return await self._artifact_mgr.list_artifacts(execution_id)

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    async def respond_to_approval(self, approval_id: str, approved: bool) -> bool:
        await self._ensure_initialized()
        if not self._approval_mgr:
            return False
        return await self._approval_mgr.respond(approval_id, approved)

    # ------------------------------------------------------------------
    # Event Bus
    # ------------------------------------------------------------------

    async def subscribe(self, event: str, handler: Callable[..., Any]) -> None:
        await self._ensure_initialized()
        if self._event_bus:
            await self._event_bus.subscribe(event, handler)

    async def _publish(self, event: str, payload: Dict[str, Any]) -> None:
        if self._event_bus and self._config.enable_event_bus:
            await self._event_bus.publish(event, payload)

    # ------------------------------------------------------------------
    # Rollback registry
    # ------------------------------------------------------------------

    def register_rollback(self, name: str, fn: Callable[..., Any]) -> None:
        if self._step_executor:
            self._step_executor.register_rollback(name, fn)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        return {
            "healthy": True,
            "initialized": self._initialized,
            "active_executions": len(self._active_executions),
            "total_executions": len(self._execution_history),
            "components": {
                "checkpoint_manager": self._checkpoint_mgr is not None,
                "artifact_manager": self._artifact_mgr is not None,
                "approval_manager": self._approval_mgr is not None,
                "event_bus": self._event_bus is not None,
                "step_executor": self._step_executor is not None,
                "llm_connector": self._llm is not None,
                "capability_connector": self._capability_connector is not None,
                "budget_manager": self._budget_manager is not None,
            },
            "config": {
                "max_parallel_steps": self._config.max_parallel_steps,
                "enable_checkpoints": self._config.enable_checkpoints,
                "enable_artifacts": self._config.enable_artifacts,
                "enable_approval": self._config.enable_approval,
                "enable_rollback": self._config.enable_rollback,
            },
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        async with self._exec_lock:
            active = list(self._active_executions.keys())
        for eid in active:
            logger.warning("Execution '%s' still active on shutdown", eid)
        self._initialized = False
        logger.info("ExecutorConnector shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        return (
            self._active_executions.get(execution_id)
            or self._execution_history.get(execution_id)
        )

    def list_active_executions(self) -> List[str]:
        return list(self._active_executions.keys())

    async def cancel_execution(self, execution_id: str) -> bool:
        async with self._exec_lock:
            record = self._active_executions.get(execution_id)
            if record:
                record.status = ExecutionStatus.CANCELLED
                self._execution_history[execution_id] = record
                self._active_executions.pop(execution_id)
                await self._publish("execution.cancelled", {"execution_id": execution_id})
                return True
        return False

    async def __aenter__(self) -> "ExecutorConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return (
            f"ExecutorConnector(active={len(self._active_executions)}, "
            f"history={len(self._execution_history)}, "
            f"initialized={self._initialized})"
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def get_executor_connector() -> ExecutorConnector:
    return ExecutorConnector.get_instance()


async def execute_steps(steps: List[Dict[str, Any]], **kwargs: Any) -> ExecutionRecord:
    return await get_executor_connector().execute(steps, **kwargs)
