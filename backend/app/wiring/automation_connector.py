"""
automation_connector.py - Production-grade automation connector.
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
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from croniter import croniter  # type: ignore[import]
    _CRONITER_AVAILABLE = True
except ImportError:
    _CRONITER_AVAILABLE = False
    croniter = None  # type: ignore

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    aioredis = None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"
    AWAITING_APPROVAL = "awaiting_approval"
    SCHEDULED = "scheduled"


class TriggerType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    EVENT = "event"
    MANUAL = "manual"
    WEBHOOK = "webhook"
    CONDITION = "condition"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NotificationType(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    SLACK = "slack"
    CUSTOM = "custom"


class QueuePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    handler: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    retries: int = 3
    retry_delay: float = 5.0
    timeout: float = 300.0
    priority: QueuePriority = QueuePriority.NORMAL
    tags: List[str] = field(default_factory=list)
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    attempt: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class TriggerDefinition:
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger_type: TriggerType = TriggerType.MANUAL
    cron_expression: Optional[str] = None
    interval_seconds: Optional[float] = None
    event_name: Optional[str] = None
    condition: Optional[str] = None
    task_definition: Optional[TaskDefinition] = None
    workflow_id: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_fired: Optional[float] = None
    next_fire: Optional[float] = None


@dataclass
class WorkflowStep:
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task: Optional[TaskDefinition] = None
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    on_failure: str = "stop"  # stop | continue | retry
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowDefinition:
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    triggers: List[TriggerDefinition] = field(default_factory=list)
    max_parallel: int = 4
    timeout: float = 3600.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkflowState:
    workflow_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    step_states: Dict[str, TaskState] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationRecord:
    notification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notification_type: NotificationType = NotificationType.LOG
    recipient: str = ""
    subject: str = ""
    body: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_at: Optional[float] = None
    success: bool = False


@dataclass
class ArtifactRecord:
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    name: str = ""
    content: Any = None
    artifact_type: str = "generic"
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    approval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    workflow_id: Optional[str] = None
    description: str = ""
    requester: str = "system"
    approvers: List[str] = field(default_factory=list)
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    approved: Optional[bool] = None
    approver: Optional[str] = None
    comments: str = ""


@dataclass
class AutomationConnectorConfig:
    max_workers: int = 10
    default_timeout: float = 300.0
    default_retries: int = 3
    default_retry_delay: float = 5.0
    queue_max_size: int = 1000
    enable_redis_queue: bool = False
    redis_url: str = "redis://localhost:6379"
    redis_queue_key: str = "automation:queue"
    cron_tick_interval: float = 60.0
    enable_notifications: bool = True
    enable_artifacts: bool = True
    enable_approval: bool = True
    approval_timeout: float = 300.0
    event_bus_connector: Any = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AutomationConnectorError(Exception):
    pass

class TaskNotFoundError(AutomationConnectorError):
    pass

class WorkflowNotFoundError(AutomationConnectorError):
    pass

class TriggerNotFoundError(AutomationConnectorError):
    pass

class ApprovalDeniedError(AutomationConnectorError):
    pass

class QueueFullError(AutomationConnectorError):
    pass


# ---------------------------------------------------------------------------
# Task State Manager
# ---------------------------------------------------------------------------

class _TaskStateManager:
    def __init__(self) -> None:
        self._states: Dict[str, TaskState] = {}
        self._lock = asyncio.Lock()

    async def create(self, task_id: str) -> TaskState:
        state = TaskState(task_id=task_id)
        async with self._lock:
            self._states[task_id] = state
        return state

    async def get(self, task_id: str) -> Optional[TaskState]:
        return self._states.get(task_id)

    async def update(self, task_id: str, **kwargs: Any) -> Optional[TaskState]:
        async with self._lock:
            state = self._states.get(task_id)
            if state:
                for k, v in kwargs.items():
                    if hasattr(state, k):
                        setattr(state, k, v)
        return state

    async def delete(self, task_id: str) -> bool:
        async with self._lock:
            return bool(self._states.pop(task_id, None))

    async def list_by_status(self, status: TaskStatus) -> List[TaskState]:
        return [s for s in self._states.values() if s.status == status]

    async def all_states(self) -> Dict[str, TaskState]:
        return dict(self._states)

    async def count(self) -> int:
        return len(self._states)


# ---------------------------------------------------------------------------
# Queue Manager
# ---------------------------------------------------------------------------

class _QueueManager:
    def __init__(self, config: AutomationConnectorConfig) -> None:
        self._config = config
        self._queues: Dict[QueuePriority, asyncio.Queue] = {
            QueuePriority.CRITICAL: asyncio.Queue(maxsize=config.queue_max_size),
            QueuePriority.HIGH: asyncio.Queue(maxsize=config.queue_max_size),
            QueuePriority.NORMAL: asyncio.Queue(maxsize=config.queue_max_size),
            QueuePriority.LOW: asyncio.Queue(maxsize=config.queue_max_size),
        }
        self._redis_client: Any = None

    async def _get_redis(self) -> Any:
        if not _REDIS_AVAILABLE:
            return None
        if self._redis_client is None:
            self._redis_client = aioredis.from_url(
                self._config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis_client

    async def enqueue(self, task: TaskDefinition) -> None:
        if self._config.enable_redis_queue:
            client = await self._get_redis()
            if client:
                await client.lpush(
                    f"{self._config.redis_queue_key}:{task.priority.value}",
                    json.dumps(task.__dict__, default=str),
                )
                return

        queue = self._queues[task.priority]
        if queue.full():
            raise QueueFullError(f"Queue {task.priority.value} is full")
        await queue.put(task)

    async def dequeue(self, timeout: float = 1.0) -> Optional[TaskDefinition]:
        # Priority order
        for priority in (QueuePriority.CRITICAL, QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW):
            if self._config.enable_redis_queue:
                client = await self._get_redis()
                if client:
                    raw = await client.rpop(f"{self._config.redis_queue_key}:{priority.value}")
                    if raw:
                        data = json.loads(raw)
                        return TaskDefinition(**{k: v for k, v in data.items() if k in TaskDefinition.__dataclass_fields__})
            else:
                queue = self._queues[priority]
                if not queue.empty():
                    return queue.get_nowait()
        return None

    async def size(self) -> int:
        if self._config.enable_redis_queue:
            client = await self._get_redis()
            if client:
                total = 0
                for p in QueuePriority:
                    total += await client.llen(f"{self._config.redis_queue_key}:{p.value}")
                return total
        return sum(q.qsize() for q in self._queues.values())

    async def health_check(self) -> bool:
        if self._config.enable_redis_queue:
            try:
                client = await self._get_redis()
                return await client.ping() if client else False
            except Exception:
                return False
        return True

    async def close(self) -> None:
        if self._redis_client:
            await self._redis_client.aclose()


# ---------------------------------------------------------------------------
# Cron Manager
# ---------------------------------------------------------------------------

class _CronManager:
    def __init__(self, tick_interval: float = 60.0) -> None:
        self._triggers: Dict[str, TriggerDefinition] = {}
        self._lock = asyncio.Lock()
        self._tick_interval = tick_interval
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._fire_callbacks: List[Callable[[TriggerDefinition], Any]] = []

    def add_fire_callback(self, cb: Callable[[TriggerDefinition], Any]) -> None:
        self._fire_callbacks.append(cb)

    async def register(self, trigger: TriggerDefinition) -> None:
        if trigger.trigger_type == TriggerType.CRON and trigger.cron_expression:
            if _CRONITER_AVAILABLE:
                cron = croniter(trigger.cron_expression, time.time())
                trigger.next_fire = cron.get_next(float)
            else:
                trigger.next_fire = time.time() + 3600.0
        elif trigger.trigger_type == TriggerType.INTERVAL and trigger.interval_seconds:
            trigger.next_fire = time.time() + trigger.interval_seconds
        async with self._lock:
            self._triggers[trigger.trigger_id] = trigger

    async def unregister(self, trigger_id: str) -> bool:
        async with self._lock:
            return bool(self._triggers.pop(trigger_id, None))

    async def start(self) -> None:
        self._task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _tick_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._tick_interval)
                await self._check_triggers()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("CronManager tick error: %s", exc)

    async def _check_triggers(self) -> None:
        now = time.time()
        async with self._lock:
            triggers = list(self._triggers.values())
        for trigger in triggers:
            if not trigger.enabled:
                continue
            if trigger.next_fire and now >= trigger.next_fire:
                trigger.last_fired = now
                if trigger.trigger_type == TriggerType.CRON and trigger.cron_expression and _CRONITER_AVAILABLE:
                    cron = croniter(trigger.cron_expression, now)
                    trigger.next_fire = cron.get_next(float)
                elif trigger.trigger_type == TriggerType.INTERVAL and trigger.interval_seconds:
                    trigger.next_fire = now + trigger.interval_seconds
                for cb in self._fire_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.create_task(cb(trigger))
                        else:
                            asyncio.get_event_loop().run_in_executor(None, cb, trigger)
                    except Exception as exc:
                        logger.warning("Trigger fire callback error: %s", exc)

    def get_trigger(self, trigger_id: str) -> Optional[TriggerDefinition]:
        return self._triggers.get(trigger_id)

    def list_triggers(self) -> List[TriggerDefinition]:
        return list(self._triggers.values())


# ---------------------------------------------------------------------------
# Trigger Manager
# ---------------------------------------------------------------------------

class _TriggerManager:
    def __init__(self, cron_manager: _CronManager) -> None:
        self._cron = cron_manager
        self._event_triggers: Dict[str, List[TriggerDefinition]] = {}
        self._webhook_triggers: Dict[str, TriggerDefinition] = {}
        self._lock = asyncio.Lock()
        self._fire_callbacks: List[Callable[[TriggerDefinition, Dict[str, Any]], Any]] = []

    def add_fire_callback(self, cb: Callable[[TriggerDefinition, Dict[str, Any]], Any]) -> None:
        self._fire_callbacks.append(cb)

    async def register(self, trigger: TriggerDefinition) -> None:
        if trigger.trigger_type in (TriggerType.CRON, TriggerType.INTERVAL):
            await self._cron.register(trigger)
        elif trigger.trigger_type == TriggerType.EVENT and trigger.event_name:
            async with self._lock:
                self._event_triggers.setdefault(trigger.event_name, []).append(trigger)
        elif trigger.trigger_type == TriggerType.WEBHOOK:
            async with self._lock:
                self._webhook_triggers[trigger.trigger_id] = trigger
        logger.debug("Registered trigger '%s' type=%s", trigger.name, trigger.trigger_type.value)

    async def unregister(self, trigger_id: str) -> bool:
        await self._cron.unregister(trigger_id)
        async with self._lock:
            for triggers in self._event_triggers.values():
                triggers[:] = [t for t in triggers if t.trigger_id != trigger_id]
            self._webhook_triggers.pop(trigger_id, None)
        return True

    async def fire_event(self, event_name: str, payload: Dict[str, Any]) -> int:
        async with self._lock:
            triggers = list(self._event_triggers.get(event_name, []))
        fired = 0
        for trigger in triggers:
            if not trigger.enabled:
                continue
            trigger.last_fired = time.time()
            fired += 1
            for cb in self._fire_callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.create_task(cb(trigger, payload))
                    else:
                        asyncio.get_event_loop().run_in_executor(None, cb, trigger, payload)
                except Exception as exc:
                    logger.warning("Event trigger fire error: %s", exc)
        return fired

    async def fire_webhook(self, trigger_id: str, payload: Dict[str, Any]) -> bool:
        trigger = self._webhook_triggers.get(trigger_id)
        if not trigger or not trigger.enabled:
            return False
        trigger.last_fired = time.time()
        for cb in self._fire_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(trigger, payload))
            except Exception as exc:
                logger.warning("Webhook trigger fire error: %s", exc)
        return True


# ---------------------------------------------------------------------------
# Notification Manager
# ---------------------------------------------------------------------------

class _NotificationManager:
    def __init__(self) -> None:
        self._handlers: Dict[NotificationType, List[Callable[..., Any]]] = {}
        self._history: List[NotificationRecord] = []
        self._lock = asyncio.Lock()

    def register_handler(self, ntype: NotificationType, handler: Callable[..., Any]) -> None:
        self._handlers.setdefault(ntype, []).append(handler)

    async def notify(
        self,
        notification_type: NotificationType,
        subject: str,
        body: str,
        recipient: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationRecord:
        record = NotificationRecord(
            notification_type=notification_type,
            recipient=recipient,
            subject=subject,
            body=body,
            metadata=metadata or {},
        )
        handlers = self._handlers.get(notification_type, [])
        if not handlers:
            logger.info("[Notification][%s] %s: %s", notification_type.value, subject, body[:100])
            record.success = True
            record.sent_at = time.time()
        else:
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(record)
                    else:
                        await asyncio.to_thread(handler, record)
                    record.success = True
                    record.sent_at = time.time()
                except Exception as exc:
                    logger.warning("Notification handler error: %s", exc)
        async with self._lock:
            self._history.append(record)
        return record

    async def get_history(self, limit: int = 100) -> List[NotificationRecord]:
        return self._history[-limit:]


# ---------------------------------------------------------------------------
# Artifact Handler
# ---------------------------------------------------------------------------

class _ArtifactHandler:
    def __init__(self) -> None:
        self._artifacts: Dict[str, List[ArtifactRecord]] = {}
        self._lock = asyncio.Lock()

    async def save(self, artifact: ArtifactRecord) -> str:
        content = artifact.content
        if isinstance(content, (dict, list)):
            artifact.size_bytes = len(json.dumps(content, default=str).encode())
        elif isinstance(content, str):
            artifact.size_bytes = len(content.encode())
        elif isinstance(content, bytes):
            artifact.size_bytes = len(content)
        async with self._lock:
            self._artifacts.setdefault(artifact.task_id, []).append(artifact)
        return artifact.artifact_id

    async def get(self, artifact_id: str) -> Optional[ArtifactRecord]:
        for arts in self._artifacts.values():
            for a in arts:
                if a.artifact_id == artifact_id:
                    return a
        return None

    async def list_by_task(self, task_id: str) -> List[ArtifactRecord]:
        return list(self._artifacts.get(task_id, []))

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock:
            for arts in self._artifacts.values():
                for i, a in enumerate(arts):
                    if a.artifact_id == artifact_id:
                        arts.pop(i)
                        return True
        return False


# ---------------------------------------------------------------------------
# Approval Handler
# ---------------------------------------------------------------------------

class _ApprovalHandler:
    def __init__(self, timeout: float = 300.0) -> None:
        self._timeout = timeout
        self._pending: Dict[str, asyncio.Future] = {}  # type: ignore[type-arg]
        self._history: Dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()

    async def request(
        self,
        approval_request: ApprovalRequest,
        notification_fn: Optional[Callable[..., Any]] = None,
    ) -> bool:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()  # type: ignore[type-arg]
        async with self._lock:
            self._pending[approval_request.approval_id] = future
            self._history[approval_request.approval_id] = approval_request

        if notification_fn:
            try:
                if asyncio.iscoroutinefunction(notification_fn):
                    asyncio.create_task(notification_fn(approval_request))
                else:
                    asyncio.get_event_loop().run_in_executor(None, notification_fn, approval_request)
            except Exception as exc:
                logger.warning("Approval notification error: %s", exc)

        logger.info("Awaiting approval [%s] timeout=%.0fs", approval_request.approval_id, self._timeout)
        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
            approval_request.resolved_at = time.time()
            approval_request.approved = result
            return result
        except asyncio.TimeoutError:
            approval_request.approved = False
            return False
        finally:
            async with self._lock:
                self._pending.pop(approval_request.approval_id, None)

    async def respond(self, approval_id: str, approved: bool, approver: str = "", comments: str = "") -> bool:
        async with self._lock:
            future = self._pending.get(approval_id)
            record = self._history.get(approval_id)
        if record:
            record.approver = approver
            record.comments = comments
        if future and not future.done():
            future.set_result(approved)
            return True
        return False

    async def get_pending(self) -> List[ApprovalRequest]:
        async with self._lock:
            return [self._history[aid] for aid in self._pending]


# ---------------------------------------------------------------------------
# Workflow Engine
# ---------------------------------------------------------------------------

class _WorkflowEngine:
    def __init__(
        self,
        task_runner: Callable[..., Any],
        state_manager: _TaskStateManager,
        config: AutomationConnectorConfig,
    ) -> None:
        self._run_task = task_runner
        self._state_mgr = state_manager
        self._config = config
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._runs: Dict[str, WorkflowState] = {}
        self._lock = asyncio.Lock()

    async def register(self, workflow: WorkflowDefinition) -> None:
        async with self._lock:
            self._workflows[workflow.workflow_id] = workflow
        logger.debug("Registered workflow '%s'", workflow.name)

    async def run(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")

        state = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=time.time(),
            metadata=inputs or {},
        )
        async with self._lock:
            self._runs[state.run_id] = state

        try:
            await asyncio.wait_for(
                self._execute_workflow(workflow, state, inputs or {}),
                timeout=workflow.timeout,
            )
        except asyncio.TimeoutError:
            state.status = WorkflowStatus.FAILED
            state.error = f"Workflow timed out after {workflow.timeout}s"
        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            logger.error("Workflow [%s] error: %s", workflow_id, traceback.format_exc())
        finally:
            state.completed_at = time.time()
            if state.started_at:
                state.latency_ms = (state.completed_at - state.started_at) * 1000

        return state

    async def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        context: Dict[str, Any],
    ) -> None:
        completed: Set[str] = set()

        while True:
            ready = [
                s for s in workflow.steps
                if s.status == TaskStatus.PENDING
                and all(dep in completed for dep in s.dependencies)
            ]
            if not ready:
                break

            sem = asyncio.Semaphore(workflow.max_parallel)

            async def _run_step(wstep: WorkflowStep) -> None:
                async with sem:
                    if wstep.task is None:
                        wstep.status = TaskStatus.COMPLETED
                        completed.add(wstep.step_id)
                        return
                    wstep.task.inputs.update(context)
                    result = await self._run_task(wstep.task)
                    wstep.result = result
                    if isinstance(result, dict) and result.get("error"):
                        wstep.status = TaskStatus.FAILED
                        wstep.error = result.get("error")
                        if wstep.on_failure == "stop":
                            raise StepFailedError(f"Step '{wstep.name}' failed")
                    else:
                        wstep.status = TaskStatus.COMPLETED
                        completed.add(wstep.step_id)
                        context[f"step_{wstep.name}_result"] = result

            tasks = [_run_step(s) for s in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    state.status = WorkflowStatus.FAILED
                    state.error = str(r)
                    return

        all_done = all(
            s.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for s in workflow.steps  # type: ignore[attr-defined]
        )
        state.status = WorkflowStatus.COMPLETED if all_done else WorkflowStatus.FAILED
        state.result = context

    def get_run(self, run_id: str) -> Optional[WorkflowState]:
        return self._runs.get(run_id)


class StepFailedError(Exception):
    pass


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class _Scheduler:
    def __init__(self) -> None:
        self._scheduled: Dict[str, asyncio.TimerHandle] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def schedule_once(
        self,
        task_id: str,
        delay: float,
        callback: Callable[[], Any],
    ) -> str:
        loop = asyncio.get_event_loop()
        handle = loop.call_later(delay, lambda: asyncio.ensure_future(callback() if asyncio.iscoroutinefunction(callback) else asyncio.to_thread(callback)))
        self._scheduled[task_id] = handle
        return task_id

    async def cancel_scheduled(self, task_id: str) -> bool:
        handle = self._scheduled.pop(task_id, None)
        if handle:
            handle.cancel()
            return True
        return False

    def count(self) -> int:
        return len(self._scheduled)


# ---------------------------------------------------------------------------
# AutomationConnector – Singleton
# ---------------------------------------------------------------------------

class AutomationConnector:
    _instance: Optional["AutomationConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = AutomationConnectorConfig()
        self._state_mgr: Optional[_TaskStateManager] = None
        self._queue_mgr: Optional[_QueueManager] = None
        self._cron_mgr: Optional[_CronManager] = None
        self._trigger_mgr: Optional[_TriggerManager] = None
        self._notification_mgr: Optional[_NotificationManager] = None
        self._artifact_handler: Optional[_ArtifactHandler] = None
        self._approval_handler: Optional[_ApprovalHandler] = None
        self._workflow_engine: Optional[_WorkflowEngine] = None
        self._scheduler: Optional[_Scheduler] = None
        self._initialized = False
        self._worker_tasks: List[asyncio.Task] = []  # type: ignore[type-arg]
        self._registered_handlers: Dict[str, Callable[..., Any]] = {}
        self._executor_connector: Any = None
        self._event_bus: Any = None
        self._stats: Dict[str, int] = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "workflows_run": 0,
            "triggers_fired": 0,
            "notifications_sent": 0,
        }

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "AutomationConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "AutomationConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: Optional[AutomationConnectorConfig] = None,
        executor_connector: Any = None,
        event_bus: Any = None,
    ) -> "AutomationConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        self._executor_connector = executor_connector
        self._event_bus = event_bus

        self._state_mgr = _TaskStateManager()
        self._queue_mgr = _QueueManager(self._config)
        self._cron_mgr = _CronManager(self._config.cron_tick_interval)
        self._trigger_mgr = _TriggerManager(self._cron_mgr)
        self._notification_mgr = _NotificationManager()
        self._artifact_handler = _ArtifactHandler()
        self._approval_handler = _ApprovalHandler(self._config.approval_timeout)
        self._scheduler = _Scheduler()

        self._workflow_engine = _WorkflowEngine(
            task_runner=self._run_task_definition,
            state_manager=self._state_mgr,
            config=self._config,
        )

        # Wire cron -> trigger -> task
        self._cron_mgr.add_fire_callback(self._on_cron_trigger)
        self._trigger_mgr.add_fire_callback(self._on_trigger_fire)

        # Start cron
        await self._cron_mgr.start()

        # Start workers
        for i in range(self._config.max_workers):
            t = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(t)

        self._initialized = True
        logger.info(
            "AutomationConnector initialized: workers=%d queue_type=%s",
            self._config.max_workers,
            "redis" if self._config.enable_redis_queue else "memory",
        )
        return self

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, name: str, fn: Callable[..., Any]) -> None:
        self._registered_handlers[name] = fn
        logger.debug("Registered task handler '%s'", name)

    def register_notification_handler(
        self, ntype: NotificationType, fn: Callable[..., Any]
    ) -> None:
        if self._notification_mgr:
            self._notification_mgr.register_handler(ntype, fn)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    async def schedule(
        self,
        task: TaskDefinition,
        delay: float = 0.0,
        trigger: Optional[TriggerDefinition] = None,
    ) -> str:
        await self._ensure_initialized()
        assert self._state_mgr and self._queue_mgr and self._scheduler

        await self._state_mgr.create(task.task_id)
        self._stats["tasks_scheduled"] += 1

        if trigger:
            trigger.task_definition = task
            await self._trigger_mgr.register(trigger)  # type: ignore[union-attr]
            await self._state_mgr.update(task.task_id, status=TaskStatus.SCHEDULED)
            logger.info("Task '%s' scheduled via trigger '%s'", task.name, trigger.name)
            return task.task_id

        if delay > 0:
            async def _delayed_enqueue() -> None:
                await self._queue_mgr.enqueue(task)  # type: ignore[union-attr]
                await self._state_mgr.update(task.task_id, status=TaskStatus.QUEUED)  # type: ignore[union-attr]

            await self._scheduler.schedule_once(task.task_id, delay, _delayed_enqueue)
            await self._state_mgr.update(task.task_id, status=TaskStatus.SCHEDULED)
        else:
            if task.requires_approval and self._config.enable_approval:
                await self._state_mgr.update(task.task_id, status=TaskStatus.AWAITING_APPROVAL)
                approval_req = ApprovalRequest(
                    task_id=task.task_id,
                    description=f"Approve task: {task.name}",
                    timeout=self._config.approval_timeout,
                )
                approved = await self._approval_handler.request(approval_req)  # type: ignore[union-attr]
                if not approved:
                    await self._state_mgr.update(task.task_id, status=TaskStatus.CANCELLED)
                    raise ApprovalDeniedError(f"Task '{task.name}' approval denied")

            await self._queue_mgr.enqueue(task)
            await self._state_mgr.update(task.task_id, status=TaskStatus.QUEUED)

        logger.info("Scheduled task '%s' [%s] priority=%s", task.name, task.task_id, task.priority.value)
        return task.task_id

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: TaskDefinition,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> TaskState:
        await self._ensure_initialized()
        await self.schedule(task, delay=0.0)

        if not wait:
            state = await self._state_mgr.get(task.task_id)  # type: ignore[union-attr]
            return state or TaskState(task_id=task.task_id)

        exec_timeout = timeout or task.timeout
        deadline = time.time() + exec_timeout
        while time.time() < deadline:
            state = await self._state_mgr.get(task.task_id)  # type: ignore[union-attr]
            if state and state.status in (
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ):
                return state
            await asyncio.sleep(0.5)

        state = await self._state_mgr.get(task.task_id)  # type: ignore[union-attr]
        if state:
            state.status = TaskStatus.FAILED
            state.error = f"Execute timed out after {exec_timeout}s"
        return state or TaskState(task_id=task.task_id, status=TaskStatus.FAILED)

    async def execute_workflow(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        await self._ensure_initialized()
        self._stats["workflows_run"] += 1
        state = await self._workflow_engine.run(workflow_id, inputs)  # type: ignore[union-attr]
        if self._config.enable_notifications and self._notification_mgr:
            await self._notification_mgr.notify(
                NotificationType.LOG,
                subject=f"Workflow {workflow_id} {state.status.value}",
                body=f"Run ID: {state.run_id} | Error: {state.error}",
            )
        return state

    # ------------------------------------------------------------------
    # Cancel / Resume
    # ------------------------------------------------------------------

    async def cancel(self, task_id: str) -> bool:
        await self._ensure_initialized()
        state = await self._state_mgr.get(task_id)  # type: ignore[union-attr]
        if not state:
            return False
        if state.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        await self._state_mgr.update(
            task_id, status=TaskStatus.CANCELLED, error="Cancelled by user"
        )
        await self._scheduler.cancel_scheduled(task_id)  # type: ignore[union-attr]
        await self._publish_event("task.cancelled", {"task_id": task_id})
        logger.info("Cancelled task '%s'", task_id)
        return True

    async def resume(self, task_id: str) -> bool:
        await self._ensure_initialized()
        state = await self._state_mgr.get(task_id)  # type: ignore[union-attr]
        if not state or state.status != TaskStatus.PAUSED:
            return False
        await self._state_mgr.update(task_id, status=TaskStatus.QUEUED)
        logger.info("Resumed task '%s'", task_id)
        return True

    async def pause(self, task_id: str) -> bool:
        await self._ensure_initialized()
        state = await self._state_mgr.get(task_id)  # type: ignore[union-attr]
        if not state or state.status != TaskStatus.RUNNING:
            return False
        await self._state_mgr.update(task_id, status=TaskStatus.PAUSED)
        return True

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    async def register_workflow(self, workflow: WorkflowDefinition) -> None:
        await self._ensure_initialized()
        await self._workflow_engine.register(workflow)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    async def register_trigger(self, trigger: TriggerDefinition) -> str:
        await self._ensure_initialized()
        await self._trigger_mgr.register(trigger)  # type: ignore[union-attr]
        return trigger.trigger_id

    async def unregister_trigger(self, trigger_id: str) -> bool:
        await self._ensure_initialized()
        return await self._trigger_mgr.unregister(trigger_id)  # type: ignore[union-attr]

    async def fire_event_trigger(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> int:
        await self._ensure_initialized()
        count = await self._trigger_mgr.fire_event(event_name, payload or {})  # type: ignore[union-attr]
        self._stats["triggers_fired"] += count
        return count

    async def fire_webhook_trigger(self, trigger_id: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        await self._ensure_initialized()
        fired = await self._trigger_mgr.fire_webhook(trigger_id, payload or {})  # type: ignore[union-attr]
        if fired:
            self._stats["triggers_fired"] += 1
        return fired

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def notify(
        self,
        notification_type: NotificationType,
        subject: str,
        body: str,
        recipient: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationRecord:
        await self._ensure_initialized()
        record = await self._notification_mgr.notify(  # type: ignore[union-attr]
            notification_type, subject, body, recipient, metadata
        )
        if record.success:
            self._stats["notifications_sent"] += 1
        return record

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def save_artifact(
        self,
        task_id: str,
        name: str,
        content: Any,
        artifact_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        await self._ensure_initialized()
        artifact = ArtifactRecord(
            task_id=task_id,
            name=name,
            content=content,
            artifact_type=artifact_type,
            metadata=metadata or {},
        )
        return await self._artifact_handler.save(artifact)  # type: ignore[union-attr]

    async def get_artifacts(self, task_id: str) -> List[ArtifactRecord]:
        await self._ensure_initialized()
        return await self._artifact_handler.list_by_task(task_id)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------

    async def respond_to_approval(
        self,
        approval_id: str,
        approved: bool,
        approver: str = "",
        comments: str = "",
    ) -> bool:
        await self._ensure_initialized()
        return await self._approval_handler.respond(approval_id, approved, approver, comments)  # type: ignore[union-attr]

    async def get_pending_approvals(self) -> List[ApprovalRequest]:
        await self._ensure_initialized()
        return await self._approval_handler.get_pending()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Task state
    # ------------------------------------------------------------------

    async def get_task_state(self, task_id: str) -> Optional[TaskState]:
        await self._ensure_initialized()
        return await self._state_mgr.get(task_id)  # type: ignore[union-attr]

    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TaskState]:
        await self._ensure_initialized()
        if status:
            return await self._state_mgr.list_by_status(status)  # type: ignore[union-attr]
        states = await self._state_mgr.all_states()  # type: ignore[union-attr]
        return list(states.values())

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        logger.debug("Automation worker %d started", worker_id)
        while True:
            try:
                task = await self._queue_mgr.dequeue(timeout=1.0)  # type: ignore[union-attr]
                if task is None:
                    await asyncio.sleep(0.5)
                    continue
                await self._run_task_definition(task)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Worker %d error: %s", worker_id, exc)

    async def _run_task_definition(self, task: TaskDefinition) -> Dict[str, Any]:
        assert self._state_mgr
        await self._state_mgr.update(
            task.task_id,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
        )
        await self._publish_event("task.started", {"task_id": task.task_id, "name": task.name})

        state = await self._state_mgr.get(task.task_id)
        attempt = 0
        last_result: Dict[str, Any] = {}

        while attempt < task.retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                if state:
                    state.attempt = attempt
                result = await asyncio.wait_for(
                    self._dispatch_task(task),
                    timeout=task.timeout,
                )
                latency = (time.perf_counter() - t0) * 1000
                await self._state_mgr.update(
                    task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    completed_at=time.time(),
                    latency_ms=latency,
                )
                self._stats["tasks_completed"] += 1
                await self._publish_event("task.completed", {"task_id": task.task_id})
                if self._config.enable_notifications and self._notification_mgr:
                    asyncio.create_task(self._notification_mgr.notify(
                        NotificationType.LOG,
                        subject=f"Task '{task.name}' completed",
                        body=f"Task ID: {task.task_id}",
                    ))
                return {"result": result, "tokens": 0, "cost": 0.0}
            except asyncio.TimeoutError:
                logger.warning("Task '%s' timed out (attempt %d/%d)", task.name, attempt, task.retries)
                last_result = {"error": f"Timeout after {task.timeout}s"}
                if attempt < task.retries:
                    await self._state_mgr.update(task.task_id, status=TaskStatus.RETRYING)
                    await asyncio.sleep(task.retry_delay * (2 ** (attempt - 1)))
            except Exception as exc:
                logger.warning("Task '%s' error: %s (attempt %d/%d)", task.name, exc, attempt, task.retries)
                last_result = {"error": str(exc)}
                if attempt < task.retries:
                    await self._state_mgr.update(task.task_id, status=TaskStatus.RETRYING)
                    await asyncio.sleep(task.retry_delay * (2 ** (attempt - 1)))

        await self._state_mgr.update(
            task.task_id,
            status=TaskStatus.FAILED,
            error=last_result.get("error", "Unknown error"),
            completed_at=time.time(),
        )
        self._stats["tasks_failed"] += 1
        await self._publish_event("task.failed", {"task_id": task.task_id, "error": last_result.get("error")})
        return {**last_result, "tokens": 0, "cost": 0.0}

    async def _dispatch_task(self, task: TaskDefinition) -> Any:
        # 1. Registered handler
        if task.handler and task.handler in self._registered_handlers:
            fn = self._registered_handlers[task.handler]
            if asyncio.iscoroutinefunction(fn):
                return await fn(task)
            return await asyncio.to_thread(fn, task)

        # 2. Executor connector
        if self._executor_connector and hasattr(self._executor_connector, "execute"):
            steps = [{"description": task.description or task.name, **task.inputs}]
            record = await self._executor_connector.execute(
                steps=steps, execution_id=task.task_id, context=task.inputs
            )
            return record.result

        # Default
        logger.warning("No handler for task '%s'; returning inputs", task.name)
        return task.inputs

    # ------------------------------------------------------------------
    # Trigger callbacks
    # ------------------------------------------------------------------

    async def _on_cron_trigger(self, trigger: TriggerDefinition) -> None:
        if trigger.task_definition:
            new_task = TaskDefinition(
                task_id=str(uuid.uuid4()),
                name=trigger.task_definition.name,
                description=trigger.task_definition.description,
                handler=trigger.task_definition.handler,
                inputs=dict(trigger.task_definition.inputs),
                retries=trigger.task_definition.retries,
                retry_delay=trigger.task_definition.retry_delay,
                timeout=trigger.task_definition.timeout,
                priority=trigger.task_definition.priority,
            )
            await self._queue_mgr.enqueue(new_task)  # type: ignore[union-attr]
            await self._state_mgr.create(new_task.task_id)  # type: ignore[union-attr]
            await self._state_mgr.update(new_task.task_id, status=TaskStatus.QUEUED)  # type: ignore[union-attr]
            self._stats["triggers_fired"] += 1

    async def _on_trigger_fire(self, trigger: TriggerDefinition, payload: Dict[str, Any]) -> None:
        if trigger.task_definition:
            new_task = TaskDefinition(
                task_id=str(uuid.uuid4()),
                name=trigger.task_definition.name,
                handler=trigger.task_definition.handler,
                inputs={**trigger.task_definition.inputs, **payload},
                retries=trigger.task_definition.retries,
                timeout=trigger.task_definition.timeout,
                priority=trigger.task_definition.priority,
            )
            await self._queue_mgr.enqueue(new_task)  # type: ignore[union-attr]
            await self._state_mgr.create(new_task.task_id)  # type: ignore[union-attr]
            await self._state_mgr.update(new_task.task_id, status=TaskStatus.QUEUED)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Event bus
    # ------------------------------------------------------------------

    async def _publish_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self._event_bus and hasattr(self._event_bus, "publish"):
            try:
                await self._event_bus.publish(event, payload)
            except Exception as exc:
                logger.warning("Event publish error '%s': %s", event, exc)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        queue_size = await self._queue_mgr.size()  # type: ignore[union-attr]
        queue_healthy = await self._queue_mgr.health_check()  # type: ignore[union-attr]
        task_count = await self._state_mgr.count()  # type: ignore[union-attr]
        pending_approvals = await self._approval_handler.get_pending()  # type: ignore[union-attr]
        return {
            "healthy": queue_healthy,
            "initialized": self._initialized,
            "workers": len(self._worker_tasks),
            "queue_size": queue_size,
            "queue_healthy": queue_healthy,
            "task_count": task_count,
            "triggers": len(self._trigger_mgr.list_triggers()) if self._trigger_mgr else 0,  # type: ignore[union-attr, attr-defined]
            "pending_approvals": len(pending_approvals),
            "stats": dict(self._stats),
            "components": {
                "scheduler": self._scheduler is not None,
                "queue_manager": self._queue_mgr is not None,
                "cron_manager": self._cron_mgr is not None,
                "trigger_manager": self._trigger_mgr is not None,
                "workflow_engine": self._workflow_engine is not None,
                "notification_manager": self._notification_mgr is not None,
                "artifact_handler": self._artifact_handler is not None,
                "approval_handler": self._approval_handler is not None,
                "executor_connector": self._executor_connector is not None,
                "event_bus": self._event_bus is not None,
            },
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        logger.info("AutomationConnector shutting down...")
        for t in self._worker_tasks:
            t.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        if self._cron_mgr:
            await self._cron_mgr.stop()
        if self._queue_mgr:
            await self._queue_mgr.close()
        self._initialized = False
        logger.info("AutomationConnector shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def __aenter__(self) -> "AutomationConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return (
            f"AutomationConnector(workers={len(self._worker_tasks)}, "
            f"initialized={self._initialized}, "
            f"stats={self._stats})"
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def get_automation_connector() -> AutomationConnector:
    return AutomationConnector.get_instance()


async def schedule_task(task: TaskDefinition, **kwargs: Any) -> str:
    return await get_automation_connector().schedule(task, **kwargs)


async def execute_task(task: TaskDefinition, **kwargs: Any) -> TaskState:
    return await get_automation_connector().execute(task, **kwargs)
