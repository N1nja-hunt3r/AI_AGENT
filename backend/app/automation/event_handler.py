from __future__ import annotations

import asyncio
import fnmatch
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union


class EventHandlerError(Exception):
    pass


class SubscriptionNotFoundError(EventHandlerError):
    pass


CallbackFn = Callable[["Event"], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Event:
    event_id: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    published_at: float = field(default_factory=time.time)
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "name": self.name, "payload": self.payload,
            "published_at": self.published_at, "source": self.source,
        }


@dataclass
class Subscription:
    subscription_id: str
    pattern: str
    callback: CallbackFn
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    error_count: int = 0
    active: bool = True


@dataclass
class DispatchResult:
    event: Event
    subscription_id: str
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


class EventHandler:
    """Publish/subscribe event dispatcher with glob-pattern matching, history, and metrics."""

    def __init__(self, *, max_history: int = 10_000, max_concurrent_callbacks: int = 100) -> None:
        self.max_history = max_history
        self._subscriptions: dict[str, Subscription] = {}
        self._history: list[Event] = []
        self._dispatch_log: list[DispatchResult] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_callbacks)
        self._created_at = time.time()
        self._publish_count = 0
        self._dispatch_count = 0
        self._dispatch_failures = 0

    def subscribe(self, pattern: str, callback: CallbackFn) -> Subscription:
        subscription = Subscription(subscription_id=str(uuid.uuid4()), pattern=pattern, callback=callback)
        self._subscriptions[subscription.subscription_id] = subscription
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        sub = self._subscriptions.pop(subscription_id, None)
        if sub is None:
            raise SubscriptionNotFoundError(f"no subscription with id {subscription_id}")
        return True

    def pause_subscription(self, subscription_id: str) -> None:
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            raise SubscriptionNotFoundError(f"no subscription with id {subscription_id}")
        sub.active = False

    def resume_subscription(self, subscription_id: str) -> None:
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            raise SubscriptionNotFoundError(f"no subscription with id {subscription_id}")
        sub.active = True

    def _matching_subscriptions(self, event_name: str) -> list[Subscription]:
        return [
            sub for sub in self._subscriptions.values()
            if sub.active and fnmatch.fnmatchcase(event_name, sub.pattern)
        ]

    async def _dispatch_to(self, sub: Subscription, event: Event) -> DispatchResult:
        async with self._semaphore:
            start = time.time()
            try:
                result = sub.callback(event)
                if asyncio.iscoroutine(result):
                    await result
                sub.call_count += 1
                dispatch_result = DispatchResult(
                    event=event, subscription_id=sub.subscription_id, success=True,
                    duration_seconds=time.time() - start,
                )
            except Exception as exc:
                sub.error_count += 1
                self._dispatch_failures += 1
                dispatch_result = DispatchResult(
                    event=event, subscription_id=sub.subscription_id, success=False,
                    error=str(exc), duration_seconds=time.time() - start,
                )
            self._dispatch_count += 1
            self._dispatch_log.append(dispatch_result)
            if len(self._dispatch_log) > self.max_history:
                self._dispatch_log = self._dispatch_log[-self.max_history :]
            return dispatch_result

    async def publish(self, name: str, *, payload: Optional[dict[str, Any]] = None, source: Optional[str] = None) -> list[DispatchResult]:
        self._publish_count += 1
        event = Event(event_id=str(uuid.uuid4()), name=name, payload=payload or {}, source=source)
        self._history.append(event)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        matching = self._matching_subscriptions(name)
        if not matching:
            return []

        results = await asyncio.gather(*(self._dispatch_to(sub, event) for sub in matching))
        return list(results)

    def publish_sync(self, name: str, *, payload: Optional[dict[str, Any]] = None, source: Optional[str] = None) -> Event:
        """Records the event without invoking async callbacks; intended for non-async contexts."""
        self._publish_count += 1
        event = Event(event_id=str(uuid.uuid4()), name=name, payload=payload or {}, source=source)
        self._history.append(event)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]
        return event

    def list_subscriptions(self, *, pattern: Optional[str] = None, active_only: bool = False) -> list[Subscription]:
        items = list(self._subscriptions.values())
        if pattern is not None:
            items = [s for s in items if s.pattern == pattern]
        if active_only:
            items = [s for s in items if s.active]
        return items

    def history(self, limit: int = 200, *, name_pattern: Optional[str] = None) -> list[Event]:
        items = self._history
        if name_pattern is not None:
            items = [e for e in items if fnmatch.fnmatchcase(e.name, name_pattern)]
        return items[-limit:]

    def dispatch_history(self, limit: int = 200, *, success: Optional[bool] = None) -> list[DispatchResult]:
        items = self._dispatch_log
        if success is not None:
            items = [d for d in items if d.success == success]
        return items[-limit:]

    def metrics(self) -> dict[str, Any]:
        return {
            "subscription_count": len(self._subscriptions),
            "active_subscription_count": sum(1 for s in self._subscriptions.values() if s.active),
            "publish_count": self._publish_count,
            "dispatch_count": self._dispatch_count,
            "dispatch_failures": self._dispatch_failures,
            "history_size": len(self._history),
        }

    async def publish_async(self, name: str, **kwargs: Any) -> list[DispatchResult]:
        return await self.publish(name, **kwargs)

    async def subscribe_async(self, pattern: str, callback: CallbackFn) -> Subscription:
        async with self._lock:
            return self.subscribe(pattern, callback)

    async def unsubscribe_async(self, subscription_id: str) -> bool:
        async with self._lock:
            return self.unsubscribe(subscription_id)

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                **self.metrics(),
            }
            return HealthStatus(healthy=True, component="event_handler", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="event_handler", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "EventHandler",
    "Event",
    "Subscription",
    "DispatchResult",
    "EventHandlerError",
    "SubscriptionNotFoundError",
    "HealthStatus",
]
