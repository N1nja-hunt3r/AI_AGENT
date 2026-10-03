from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union


class TriggerKind(str, Enum):
    TIME = "time"
    EVENT = "event"
    WEBHOOK = "webhook"
    CONDITION = "condition"
    MANUAL = "manual"
    FILE = "file"


class TriggerState(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    FIRED = "fired"


class TriggerManagerError(Exception):
    pass


class TriggerNotFoundError(TriggerManagerError):
    pass


CallbackFn = Callable[["TriggerFireContext"], Union[Any, Awaitable[Any]]]
ConditionFn = Callable[[], Union[bool, Awaitable[bool]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriggerFireContext:
    trigger_id: str
    kind: TriggerKind
    fired_at: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trigger:
    trigger_id: str
    name: str
    kind: TriggerKind
    callback: CallbackFn
    state: TriggerState = TriggerState.ACTIVE
    created_at: float = field(default_factory=time.time)
    fire_count: int = 0
    last_fired_at: Optional[float] = None

    run_at: Optional[float] = None
    event_name: Optional[str] = None
    webhook_token: Optional[str] = None
    condition_fn: Optional[ConditionFn] = None
    poll_interval_seconds: float = 5.0
    watch_path: Optional[str] = None
    last_file_hash: Optional[str] = None
    last_file_mtime: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id, "name": self.name, "kind": self.kind.value,
            "state": self.state.value, "created_at": self.created_at, "fire_count": self.fire_count,
            "last_fired_at": self.last_fired_at, "run_at": self.run_at, "event_name": self.event_name,
            "watch_path": self.watch_path,
        }


def _file_digest(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class TriggerManager:
    """Manages time, event, webhook, condition, manual, and file-based triggers."""

    def __init__(self) -> None:
        self._triggers: dict[str, Trigger] = {}
        self._event_index: dict[str, set[str]] = {}
        self._webhook_index: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._register_count = 0
        self._fire_count = 0

    async def _invoke(self, trigger: Trigger, payload: Optional[dict[str, Any]] = None) -> Any:
        self._fire_count += 1
        trigger.fire_count += 1
        trigger.last_fired_at = time.time()
        context = TriggerFireContext(trigger_id=trigger.trigger_id, kind=trigger.kind, fired_at=trigger.last_fired_at, payload=payload or {})
        result = trigger.callback(context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def register(
        self,
        name: str,
        callback: CallbackFn,
        *,
        kind: TriggerKind = TriggerKind.MANUAL,
        run_at: Optional[float] = None,
        event_name: Optional[str] = None,
        webhook_token: Optional[str] = None,
        condition_fn: Optional[ConditionFn] = None,
        poll_interval_seconds: float = 5.0,
        watch_path: Optional[str] = None,
    ) -> Trigger:
        self._register_count += 1
        trigger_id = str(uuid.uuid4())

        if kind == TriggerKind.TIME and run_at is None:
            raise TriggerManagerError("run_at is required for TIME triggers")
        if kind == TriggerKind.EVENT and event_name is None:
            raise TriggerManagerError("event_name is required for EVENT triggers")
        if kind == TriggerKind.WEBHOOK and webhook_token is None:
            webhook_token = uuid.uuid4().hex
        if kind == TriggerKind.CONDITION and condition_fn is None:
            raise TriggerManagerError("condition_fn is required for CONDITION triggers")
        if kind == TriggerKind.FILE and watch_path is None:
            raise TriggerManagerError("watch_path is required for FILE triggers")

        trigger = Trigger(
            trigger_id=trigger_id, name=name, kind=kind, callback=callback,
            run_at=run_at, event_name=event_name, webhook_token=webhook_token,
            condition_fn=condition_fn, poll_interval_seconds=poll_interval_seconds,
            watch_path=watch_path,
        )

        if kind == TriggerKind.FILE and watch_path is not None:
            p = Path(watch_path)
            trigger.last_file_hash = _file_digest(p)
            try:
                trigger.last_file_mtime = p.stat().st_mtime
            except OSError:
                trigger.last_file_mtime = None

        self._triggers[trigger_id] = trigger

        if kind == TriggerKind.EVENT and event_name is not None:
            self._event_index.setdefault(event_name, set()).add(trigger_id)
        if kind == TriggerKind.WEBHOOK and webhook_token is not None:
            self._webhook_index[webhook_token] = trigger_id

        return trigger

    def unregister(self, trigger_id: str) -> bool:
        trigger = self._triggers.pop(trigger_id, None)
        if trigger is None:
            raise TriggerNotFoundError(f"no trigger with id {trigger_id}")
        if trigger.kind == TriggerKind.EVENT and trigger.event_name is not None:
            self._event_index.get(trigger.event_name, set()).discard(trigger_id)
        if trigger.kind == TriggerKind.WEBHOOK and trigger.webhook_token is not None:
            self._webhook_index.pop(trigger.webhook_token, None)
        return True

    def enable(self, trigger_id: str) -> None:
        trigger = self._triggers.get(trigger_id)
        if trigger is None:
            raise TriggerNotFoundError(f"no trigger with id {trigger_id}")
        trigger.state = TriggerState.ACTIVE

    def disable(self, trigger_id: str) -> None:
        trigger = self._triggers.get(trigger_id)
        if trigger is None:
            raise TriggerNotFoundError(f"no trigger with id {trigger_id}")
        trigger.state = TriggerState.DISABLED

    async def trigger(self, trigger_id: str, *, payload: Optional[dict[str, Any]] = None) -> Any:
        """Manually fire a trigger regardless of kind (used for MANUAL triggers or forced firing)."""
        t = self._triggers.get(trigger_id)
        if t is None:
            raise TriggerNotFoundError(f"no trigger with id {trigger_id}")
        if t.state != TriggerState.ACTIVE:
            raise TriggerManagerError(f"trigger '{trigger_id}' is not active (state={t.state.value})")
        return await self._invoke(t, payload)

    async def fire_event(self, event_name: str, *, payload: Optional[dict[str, Any]] = None) -> list[Any]:
        trigger_ids = self._event_index.get(event_name, set())
        results = []
        for tid in list(trigger_ids):
            t = self._triggers.get(tid)
            if t is not None and t.state == TriggerState.ACTIVE:
                results.append(await self._invoke(t, payload))
        return results

    async def fire_webhook(self, webhook_token: str, *, payload: Optional[dict[str, Any]] = None) -> Any:
        trigger_id = self._webhook_index.get(webhook_token)
        if trigger_id is None:
            raise TriggerNotFoundError("no trigger registered for webhook token")
        t = self._triggers.get(trigger_id)
        if t is None or t.state != TriggerState.ACTIVE:
            raise TriggerManagerError("webhook trigger is not active")
        return await self._invoke(t, payload)

    async def check(self) -> list[Any]:
        """Evaluate TIME, CONDITION, and FILE triggers; fires any that are due. Returns results of fired triggers."""
        results: list[Any] = []
        now = time.time()

        for t in list(self._triggers.values()):
            if t.state != TriggerState.ACTIVE:
                continue

            if t.kind == TriggerKind.TIME and t.run_at is not None and t.run_at <= now:
                results.append(await self._invoke(t))
                t.state = TriggerState.FIRED

            elif t.kind == TriggerKind.CONDITION and t.condition_fn is not None:
                cond_result = t.condition_fn()
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
                if cond_result:
                    results.append(await self._invoke(t))

            elif t.kind == TriggerKind.FILE and t.watch_path is not None:
                p = Path(t.watch_path)
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if t.last_file_mtime is None or mtime != t.last_file_mtime:
                    new_hash = _file_digest(p)
                    if new_hash != t.last_file_hash:
                        results.append(await self._invoke(t, {"path": str(p)}))
                    t.last_file_mtime = mtime
                    t.last_file_hash = new_hash

        return results

    def get(self, trigger_id: str) -> Optional[Trigger]:
        return self._triggers.get(trigger_id)

    def list_triggers(self, *, kind: Optional[TriggerKind] = None, state: Optional[TriggerState] = None) -> list[Trigger]:
        items = list(self._triggers.values())
        if kind is not None:
            items = [t for t in items if t.kind == kind]
        if state is not None:
            items = [t for t in items if t.state == state]
        return items

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "trigger_count": len(self._triggers),
                "active_count": sum(1 for t in self._triggers.values() if t.state == TriggerState.ACTIVE),
                "disabled_count": sum(1 for t in self._triggers.values() if t.state == TriggerState.DISABLED),
                "register_count": self._register_count,
                "fire_count": self._fire_count,
                "event_index_size": len(self._event_index),
                "webhook_index_size": len(self._webhook_index),
            }
            return HealthStatus(healthy=True, component="trigger_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="trigger_manager", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "TriggerManager",
    "Trigger",
    "TriggerKind",
    "TriggerState",
    "TriggerFireContext",
    "TriggerManagerError",
    "TriggerNotFoundError",
    "HealthStatus",
]
