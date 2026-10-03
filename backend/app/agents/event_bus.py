"""
Event Bus - Asynchronous event system for the AI Agent Platform.

This module provides comprehensive event management including:
- Publish/subscribe pattern
- Async event handlers with priorities
- Middleware pipeline for event processing
- Event filtering and routing
- Multi-agent event isolation and broadcasting
- Event history and replay
- Dead letter queue for failed events
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
EventDataT = TypeVar("EventDataT")


# =============================================================================
# ENUMS
# =============================================================================


class EventPriority(Enum):
    """Priority levels for event handlers."""

    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100
    CRITICAL = 150
    SYSTEM = 200


class EventStatus(Enum):
    """Status of an event."""

    PENDING = auto()
    PROCESSING = auto()
    DELIVERED = auto()
    PARTIALLY_DELIVERED = auto()
    FAILED = auto()
    CANCELLED = auto()
    EXPIRED = auto()


class PropagationMode(Enum):
    """How events propagate to handlers."""

    ALL = auto()  # Deliver to all handlers
    FIRST_MATCH = auto()  # Stop after first handler
    UNTIL_HANDLED = auto()  # Stop when handler returns True
    BROADCAST = auto()  # Broadcast to all agents


class EventScope(Enum):
    """Scope of event visibility."""

    LOCAL = auto()  # Current agent only
    AGENT = auto()  # Specific agent
    SESSION = auto()  # Current session
    GLOBAL = auto()  # All agents


# =============================================================================
# EVENT TYPES
# =============================================================================


@dataclass(frozen=True)
class EventType:
    """
    Event type identifier with hierarchical naming.

    Supports wildcard matching for subscriptions.
    Example: "agent.task.completed" matches "agent.*" and "agent.task.*"
    """

    name: str
    namespace: str = ""

    @property
    def full_name(self) -> str:
        """Get full event type name."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name

    @property
    def parts(self) -> tuple[str, ...]:
        """Get name parts for hierarchical matching."""
        return tuple(self.full_name.split("."))

    def matches(self, pattern: str) -> bool:
        """
        Check if event type matches a pattern.

        Supports wildcards:
        - "*" matches any single part
        - "**" matches any number of parts
        """
        pattern_parts = pattern.split(".")
        event_parts = list(self.parts)

        i = j = 0
        while i < len(pattern_parts) and j < len(event_parts):
            if pattern_parts[i] == "**":
                # Match rest of event
                if i == len(pattern_parts) - 1:
                    return True
                # Try matching remaining pattern
                for k in range(j, len(event_parts) + 1):
                    remaining_event = ".".join(event_parts[k:])
                    remaining_pattern = ".".join(pattern_parts[i + 1:])
                    if EventType(remaining_event).matches(remaining_pattern):
                        return True
                return False
            elif pattern_parts[i] == "*":
                i += 1
                j += 1
            elif pattern_parts[i] == event_parts[j]:
                i += 1
                j += 1
            else:
                return False

        return i == len(pattern_parts) and j == len(event_parts)

    def __str__(self) -> str:
        return self.full_name


# Common event types
class Events:
    """Common event type constants."""

    # Agent events
    AGENT_STARTED = EventType("started", "agent")
    AGENT_STOPPED = EventType("stopped", "agent")
    AGENT_ERROR = EventType("error", "agent")

    # Task events
    TASK_CREATED = EventType("created", "task")
    TASK_STARTED = EventType("started", "task")
    TASK_COMPLETED = EventType("completed", "task")
    TASK_FAILED = EventType("failed", "task")
    TASK_CANCELLED = EventType("cancelled", "task")

    # Plan events
    PLAN_CREATED = EventType("created", "plan")
    PLAN_STEP_STARTED = EventType("step.started", "plan")
    PLAN_STEP_COMPLETED = EventType("step.completed", "plan")
    PLAN_STEP_FAILED = EventType("step.failed", "plan")
    PLAN_COMPLETED = EventType("completed", "plan")

    # Tool events
    TOOL_INVOKED = EventType("invoked", "tool")
    TOOL_COMPLETED = EventType("completed", "tool")
    TOOL_FAILED = EventType("failed", "tool")

    # Memory events
    MEMORY_STORED = EventType("stored", "memory")
    MEMORY_RETRIEVED = EventType("retrieved", "memory")
    MEMORY_CLEARED = EventType("cleared", "memory")

    # LLM events
    LLM_REQUEST = EventType("request", "llm")
    LLM_RESPONSE = EventType("response", "llm")
    LLM_ERROR = EventType("error", "llm")
    LLM_STREAM_CHUNK = EventType("stream.chunk", "llm")

    # Approval events
    APPROVAL_REQUESTED = EventType("requested", "approval")
    APPROVAL_GRANTED = EventType("granted", "approval")
    APPROVAL_DENIED = EventType("denied", "approval")
    APPROVAL_TIMEOUT = EventType("timeout", "approval")

    # System events
    SYSTEM_STARTUP = EventType("startup", "system")
    SYSTEM_SHUTDOWN = EventType("shutdown", "system")
    SYSTEM_ERROR = EventType("error", "system")

    # Budget events
    BUDGET_WARNING = EventType("warning", "budget")
    BUDGET_EXCEEDED = EventType("exceeded", "budget")


# =============================================================================
# EVENT
# =============================================================================


@dataclass
class Event(Generic[EventDataT]):
    """
    Event container with metadata.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        data: Event payload
        timestamp: When event was created
        source: Event source identifier
        agent_id: Originating agent
        session_id: Associated session
        correlation_id: For tracking related events
        causation_id: ID of event that caused this one
        scope: Event visibility scope
        priority: Event priority
        ttl_seconds: Time-to-live (None = forever)
        metadata: Additional metadata
        status: Current status
        propagation: How event propagates
    """

    event_id: str
    event_type: EventType
    data: EventDataT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    agent_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    scope: EventScope = EventScope.LOCAL
    priority: EventPriority = EventPriority.NORMAL
    ttl_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: EventStatus = EventStatus.PENDING
    propagation: PropagationMode = PropagationMode.ALL

    @property
    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get event age in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def with_data(self, data: Any) -> Event[Any]:
        """Create new event with different data."""
        return Event(
            event_id=self.event_id,
            event_type=self.event_type,
            data=data,
            timestamp=self.timestamp,
            source=self.source,
            agent_id=self.agent_id,
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            scope=self.scope,
            priority=self.priority,
            ttl_seconds=self.ttl_seconds,
            metadata=dict(self.metadata),
            status=self.status,
            propagation=self.propagation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.full_name,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "scope": self.scope.name,
            "priority": self.priority.name,
            "status": self.status.name,
            "metadata": self.metadata,
        }


def create_event(
    event_type: EventType | str,
    data: Any = None,
    source: str = "",
    agent_id: str | None = None,
    session_id: str | None = None,
    correlation_id: str | None = None,
    priority: EventPriority = EventPriority.NORMAL,
    scope: EventScope = EventScope.LOCAL,
    ttl_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event[Any]:
    """
    Factory function to create an event.

    Args:
        event_type: Event type or string name
        data: Event payload
        source: Event source
        agent_id: Originating agent
        session_id: Associated session
        correlation_id: Correlation ID
        priority: Event priority
        scope: Event scope
        ttl_seconds: Time-to-live
        metadata: Additional metadata

    Returns:
        New Event instance
    """
    if isinstance(event_type, str):
        parts = event_type.rsplit(".", 1)
        if len(parts) == 2:
            event_type = EventType(name=parts[1], namespace=parts[0])
        else:
            event_type = EventType(name=event_type)

    return Event(
        event_id=str(uuid4()),
        event_type=event_type,
        data=data,
        source=source,
        agent_id=agent_id,
        session_id=session_id,
        correlation_id=correlation_id,
        priority=priority,
        scope=scope,
        ttl_seconds=ttl_seconds,
        metadata=metadata or {},
    )


# =============================================================================
# HANDLER TYPES
# =============================================================================

# Handler function types
SyncHandler = Callable[[Event[Any]], Any]
AsyncHandler = Callable[[Event[Any]], Awaitable[Any]]
EventHandler = SyncHandler | AsyncHandler


@dataclass
class HandlerRegistration:
    """
    Registration info for an event handler.

    Attributes:
        handler_id: Unique handler identifier
        handler: The handler function
        pattern: Event pattern to match
        priority: Handler priority
        agent_id: Owning agent (None = global)
        is_async: Whether handler is async
        once: Remove after first invocation
        filter_fn: Optional filter function
        timeout_seconds: Handler timeout
        enabled: Whether handler is active
        invocation_count: Number of times invoked
        last_invoked: Last invocation time
        error_count: Number of errors
    """

    handler_id: str
    handler: EventHandler
    pattern: str
    priority: EventPriority = EventPriority.NORMAL
    agent_id: str | None = None
    is_async: bool = False
    once: bool = False
    filter_fn: Callable[[Event[Any]], bool] | None = None
    timeout_seconds: float | None = None
    enabled: bool = True
    invocation_count: int = 0
    last_invoked: datetime | None = None
    error_count: int = 0

    def matches(self, event: Event[Any]) -> bool:
        """Check if handler matches event."""
        if not self.enabled:
            return False

        # Check pattern match
        if not event.event_type.matches(self.pattern):
            return False

        # Check agent scope
        if self.agent_id is not None:
            if event.scope == EventScope.LOCAL and event.agent_id != self.agent_id:
                return False
            if event.scope == EventScope.AGENT and event.agent_id != self.agent_id:
                return False

        # Check custom filter
        if self.filter_fn is not None:
            try:
                if not self.filter_fn(event):
                    return False
            except Exception:
                return False

        return True


# =============================================================================
# MIDDLEWARE
# =============================================================================


class EventMiddleware(ABC):
    """Abstract base for event middleware."""

    @abstractmethod
    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        """
        Process event through middleware.

        Args:
            event: Event to process
            next_middleware: Next middleware in chain

        Returns:
            Processed event or None to stop propagation
        """
        ...


class LoggingMiddleware(EventMiddleware):
    """Middleware that logs all events."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.DEBUG,
        log_data: bool = False,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_level = log_level
        self._log_data = log_data

    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        start_time = time.perf_counter()

        self._logger.log(
            self._log_level,
            f"Event: {event.event_type} (id={event.event_id[:8]})"
            + (f" data={event.data}" if self._log_data else ""),
        )

        result = await next_middleware(event)

        elapsed = (time.perf_counter() - start_time) * 1000
        self._logger.log(
            self._log_level,
            f"Event processed: {event.event_type} in {elapsed:.2f}ms",
        )

        return result


class FilterMiddleware(EventMiddleware):
    """Middleware that filters events based on criteria."""

    def __init__(
        self,
        allow_types: set[str] | None = None,
        block_types: set[str] | None = None,
        allow_scopes: set[EventScope] | None = None,
        min_priority: EventPriority = EventPriority.LOWEST,
    ) -> None:
        self._allow_types = allow_types
        self._block_types = block_types or set()
        self._allow_scopes = allow_scopes
        self._min_priority = min_priority

    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        # Check blocked types
        if event.event_type.full_name in self._block_types:
            return None

        # Check allowed types
        if self._allow_types and event.event_type.full_name not in self._allow_types:
            return None

        # Check scope
        if self._allow_scopes and event.scope not in self._allow_scopes:
            return None

        # Check priority
        if event.priority.value < self._min_priority.value:
            return None

        return await next_middleware(event)


class TransformMiddleware(EventMiddleware):
    """Middleware that transforms event data."""

    def __init__(
        self,
        transform_fn: Callable[[Event[Any]], Event[Any]],
    ) -> None:
        self._transform = transform_fn

    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        transformed = self._transform(event)
        return await next_middleware(transformed)


class RetryMiddleware(EventMiddleware):
    """Middleware that retries failed event processing."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        exponential_backoff: bool = True,
    ) -> None:
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._exponential = exponential_backoff

    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await next_middleware(event)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_delay
                    if self._exponential:
                        delay *= 2 ** attempt
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error
        return None


class ThrottleMiddleware(EventMiddleware):
    """Middleware that throttles event processing rate."""

    def __init__(
        self,
        max_events_per_second: float = 100.0,
        per_event_type: bool = False,
    ) -> None:
        self._max_rate = max_events_per_second
        self._per_type = per_event_type
        self._last_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def process(
        self,
        event: Event[Any],
        next_middleware: Callable[[Event[Any]], Awaitable[Event[Any] | None]],
    ) -> Event[Any] | None:
        key = event.event_type.full_name if self._per_type else "__global__"
        min_interval = 1.0 / self._max_rate

        async with self._lock:
            now = time.perf_counter()
            last_time = self._last_times.get(key, 0)
            elapsed = now - last_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_times[key] = time.perf_counter()

        return await next_middleware(event)


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""

    event: Event[Any]
    error: str
    handler_id: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0


class DeadLetterQueue:
    """Queue for failed events."""

    def __init__(self, max_size: int = 1000) -> None:
        self._queue: list[DeadLetterEntry] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def add(
        self,
        event: Event[Any],
        error: str,
        handler_id: str | None = None,
    ) -> None:
        """Add failed event to queue."""
        async with self._lock:
            entry = DeadLetterEntry(
                event=event,
                error=error,
                handler_id=handler_id,
            )
            self._queue.append(entry)

            # Trim if over max size
            if len(self._queue) > self._max_size:
                self._queue = self._queue[-self._max_size:]

    async def get_all(self) -> list[DeadLetterEntry]:
        """Get all entries."""
        return list(self._queue)

    async def get_by_type(self, event_type: str) -> list[DeadLetterEntry]:
        """Get entries by event type."""
        return [
            e for e in self._queue
            if e.event.event_type.matches(event_type)
        ]

    async def retry(
        self,
        event_bus: EventBus,
        max_retries: int = 3,
    ) -> int:
        """Retry failed events."""
        retried = 0

        async with self._lock:
            remaining: list[DeadLetterEntry] = []

            for entry in self._queue:
                if entry.retry_count < max_retries:
                    entry.retry_count += 1
                    try:
                        await event_bus.publish(entry.event)
                        retried += 1
                    except Exception:
                        remaining.append(entry)
                else:
                    remaining.append(entry)

            self._queue = remaining

        return retried

    async def clear(self) -> int:
        """Clear the queue."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)


# =============================================================================
# EVENT HISTORY
# =============================================================================


class EventHistory:
    """Maintains history of published events."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float | None = 3600,
    ) -> None:
        self._events: list[Event[Any]] = []
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def add(self, event: Event[Any]) -> None:
        """Add event to history."""
        async with self._lock:
            self._events.append(event)

            # Trim old events
            if len(self._events) > self._max_size:
                self._events = self._events[-self._max_size:]

    async def get_recent(
        self,
        count: int = 100,
        event_type: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Event[Any]]:
        """Get recent events matching criteria."""
        events = list(reversed(self._events))

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type.matches(event_type)]

        # Filter by agent
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]

        # Filter by session
        if session_id:
            events = [e for e in events if e.session_id == session_id]

        # Filter expired
        if self._ttl:
            events = [e for e in events if e.age_seconds <= self._ttl]

        return events[:count]

    async def get_by_correlation(self, correlation_id: str) -> list[Event[Any]]:
        """Get events by correlation ID."""
        return [e for e in self._events if e.correlation_id == correlation_id]

    async def replay(
        self,
        event_bus: EventBus,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Replay events through event bus."""
        events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type.matches(event_type)]

        if since:
            events = [e for e in events if e.timestamp >= since]

        for event in events:
            await event_bus.publish(event)

        return len(events)

    async def clear(self) -> int:
        """Clear history."""
        async with self._lock:
            count = len(self._events)
            self._events.clear()
            return count

    @property
    def size(self) -> int:
        """Get history size."""
        return len(self._events)


# =============================================================================
# EVENT BUS
# =============================================================================


@dataclass
class EventBusConfig:
    """
    Configuration for EventBus.

    Attributes:
        max_handlers_per_event: Maximum handlers per event type
        default_timeout_seconds: Default handler timeout
        enable_history: Enable event history
        history_max_size: Maximum history size
        enable_dead_letter: Enable dead letter queue
        dead_letter_max_size: Maximum dead letter queue size
        enable_metrics: Enable metrics collection
    """

    max_handlers_per_event: int = 100
    default_timeout_seconds: float = 30.0
    enable_history: bool = True
    history_max_size: int = 10000
    enable_dead_letter: bool = True
    dead_letter_max_size: int = 1000
    enable_metrics: bool = True


@dataclass
class EventMetrics:
    """Metrics for event bus."""

    events_published: int = 0
    events_delivered: int = 0
    events_failed: int = 0
    handlers_invoked: int = 0
    handlers_failed: int = 0
    total_processing_time_ms: float = 0.0
    events_by_type: dict[str, int] = field(default_factory=dict)


class EventBus:
    """
    Central event bus for publish/subscribe messaging.

    Supports async handlers, priorities, middleware,
    and multi-agent event isolation.
    """

    def __init__(
        self,
        config: EventBusConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize event bus.

        Args:
            config: Bus configuration
            logger: Optional logger
        """
        self._config = config or EventBusConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Handler storage
        self._handlers: dict[str, list[HandlerRegistration]] = defaultdict(list)
        self._handler_index: dict[str, HandlerRegistration] = {}

        # Middleware chain
        self._middleware: list[EventMiddleware] = []

        # Components
        self._history: EventHistory | None = None
        if self._config.enable_history:
            self._history = EventHistory(max_size=self._config.history_max_size)

        self._dead_letter: DeadLetterQueue | None = None
        if self._config.enable_dead_letter:
            self._dead_letter = DeadLetterQueue(max_size=self._config.dead_letter_max_size)

        # Metrics
        self._metrics = EventMetrics() if self._config.enable_metrics else None

        # State
        self._lock = asyncio.Lock()
        self._running = True

        # Agent views (for multi-agent support)
        self._agent_views: dict[str, AgentEventView] = {}

    # -------------------------------------------------------------------------
    # Subscribe
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        pattern: str | EventType,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        agent_id: str | None = None,
        once: bool = False,
        filter_fn: Callable[[Event[Any]], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """
        Subscribe a handler to events matching pattern.

        Args:
            pattern: Event pattern (supports wildcards)
            handler: Handler function
            priority: Handler priority
            agent_id: Owning agent
            once: Remove after first invocation
            filter_fn: Optional filter function
            timeout_seconds: Handler timeout

        Returns:
            Handler ID for unsubscribing
        """
        if isinstance(pattern, EventType):
            pattern = pattern.full_name

        handler_id = str(uuid4())
        is_async = asyncio.iscoroutinefunction(handler)

        registration = HandlerRegistration(
            handler_id=handler_id,
            handler=handler,
            pattern=pattern,
            priority=priority,
            agent_id=agent_id,
            is_async=is_async,
            once=once,
            filter_fn=filter_fn,
            timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
        )

        # Add to handlers list
        self._handlers[pattern].append(registration)
        self._handler_index[handler_id] = registration

        # Sort by priority (highest first)
        self._handlers[pattern].sort(key=lambda h: h.priority.value, reverse=True)

        self._logger.debug(f"Subscribed handler {handler_id[:8]} to {pattern}")

        return handler_id

    def on(
        self,
        pattern: str | EventType,
        priority: EventPriority = EventPriority.NORMAL,
        agent_id: str | None = None,
        once: bool = False,
        filter_fn: Callable[[Event[Any]], bool] | None = None,
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator for subscribing handlers.

        Usage:
            @bus.on("task.completed")
            async def handle_task(event):
                ...
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(
                pattern=pattern,
                handler=handler,
                priority=priority,
                agent_id=agent_id,
                once=once,
                filter_fn=filter_fn,
            )
            return handler

        return decorator

    def once(
        self,
        pattern: str | EventType,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Callable[[EventHandler], EventHandler]:
        """Decorator for one-time handlers."""
        return self.on(pattern, priority=priority, once=True)

    # -------------------------------------------------------------------------
    # Unsubscribe
    # -------------------------------------------------------------------------

    def unsubscribe(self, handler_id: str) -> bool:
        """
        Unsubscribe a handler.

        Args:
            handler_id: Handler ID from subscribe()

        Returns:
            True if handler was found and removed
        """
        registration = self._handler_index.pop(handler_id, None)
        if registration is None:
            return False

        pattern = registration.pattern
        if pattern in self._handlers:
            self._handlers[pattern] = [
                h for h in self._handlers[pattern]
                if h.handler_id != handler_id
            ]

            if not self._handlers[pattern]:
                del self._handlers[pattern]

        self._logger.debug(f"Unsubscribed handler {handler_id[:8]}")
        return True

    def unsubscribe_all(self, pattern: str | None = None, agent_id: str | None = None) -> int:
        """
        Unsubscribe multiple handlers.

        Args:
            pattern: Pattern to match (None = all)
            agent_id: Agent ID to match (None = all)

        Returns:
            Number of handlers removed
        """
        removed = 0

        handlers_to_remove: list[str] = []

        for handler_id, registration in self._handler_index.items():
            if pattern and registration.pattern != pattern:
                continue
            if agent_id and registration.agent_id != agent_id:
                continue
            handlers_to_remove.append(handler_id)

        for handler_id in handlers_to_remove:
            if self.unsubscribe(handler_id):
                removed += 1

        return removed

    # -------------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------------

    async def publish(
        self,
        event: Event[Any] | EventType | str,
        data: Any = None,
        **kwargs: Any,
    ) -> Event[Any]:
        """
        Publish an event to all matching handlers.

        Args:
            event: Event instance, EventType, or string
            data: Event data (if event is type/string)
            **kwargs: Additional event parameters

        Returns:
            Published event
        """
        # Create event if needed
        if isinstance(event, (EventType, str)):
            event = create_event(event, data=data, **kwargs)
        elif data is not None:
            event = event.with_data(data)

        # Check if expired
        if event.is_expired:
            event.status = EventStatus.EXPIRED
            return event

        # Process through middleware
        processed_event = await self._process_middleware(event)
        if processed_event is None:
            event.status = EventStatus.CANCELLED
            return event

        event = processed_event
        event.status = EventStatus.PROCESSING

        # Record in history
        if self._history:
            await self._history.add(event)

        # Update metrics
        if self._metrics:
            self._metrics.events_published += 1
            type_name = event.event_type.full_name
            self._metrics.events_by_type[type_name] = (
                self._metrics.events_by_type.get(type_name, 0) + 1
            )

        # Find matching handlers
        handlers = self._get_matching_handlers(event)

        if not handlers:
            event.status = EventStatus.DELIVERED
            return event

        # Invoke handlers
        start_time = time.perf_counter()
        delivered = 0
        failed = 0
        handlers_to_remove: list[str] = []

        for registration in handlers:
            try:
                result = await self._invoke_handler(registration, event)

                registration.invocation_count += 1
                registration.last_invoked = datetime.now(timezone.utc)
                delivered += 1

                if self._metrics:
                    self._metrics.handlers_invoked += 1

                # Check for once handlers
                if registration.once:
                    handlers_to_remove.append(registration.handler_id)

                # Check propagation mode
                if event.propagation == PropagationMode.FIRST_MATCH:
                    break
                if event.propagation == PropagationMode.UNTIL_HANDLED and result:
                    break

            except Exception as e:
                registration.error_count += 1
                failed += 1

                if self._metrics:
                    self._metrics.handlers_failed += 1

                self._logger.error(
                    f"Handler {registration.handler_id[:8]} failed: {e}"
                )

                # Add to dead letter queue
                if self._dead_letter:
                    await self._dead_letter.add(
                        event=event,
                        error=str(e),
                        handler_id=registration.handler_id,
                    )

        # Remove once handlers
        for handler_id in handlers_to_remove:
            self.unsubscribe(handler_id)

        # Update metrics
        if self._metrics:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._metrics.total_processing_time_ms += elapsed
            self._metrics.events_delivered += 1 if delivered > 0 else 0
            self._metrics.events_failed += 1 if failed > 0 and delivered == 0 else 0

        # Set final status
        if delivered > 0 and failed == 0:
            event.status = EventStatus.DELIVERED
        elif delivered > 0 and failed > 0:
            event.status = EventStatus.PARTIALLY_DELIVERED
        else:
            event.status = EventStatus.FAILED

        return event

    async def emit(
        self,
        event_type: EventType | str,
        data: Any = None,
        **kwargs: Any,
    ) -> Event[Any]:
        """Alias for publish with event type."""
        return await self.publish(event_type, data=data, **kwargs)

    async def broadcast(
        self,
        event_type: EventType | str,
        data: Any = None,
        **kwargs: Any,
    ) -> Event[Any]:
        """Broadcast event to all agents."""
        return await self.publish(
            event_type,
            data=data,
            scope=EventScope.GLOBAL,
            propagation=PropagationMode.BROADCAST,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Middleware
    # -------------------------------------------------------------------------

    def use(self, middleware: EventMiddleware) -> None:
        """Add middleware to the processing chain."""
        self._middleware.append(middleware)

    def remove_middleware(self, middleware: EventMiddleware) -> bool:
        """Remove middleware from the chain."""
        if middleware in self._middleware:
            self._middleware.remove(middleware)
            return True
        return False

    async def _process_middleware(self, event: Event[Any]) -> Event[Any] | None:
        """Process event through middleware chain."""
        if not self._middleware:
            return event

        async def terminal(e: Event[Any]) -> Event[Any] | None:
            return e

        # Build chain from end to start
        chain = terminal
        for middleware in reversed(self._middleware):
            current_middleware = middleware
            next_chain = chain

            async def make_chain(
                e: Event[Any],
                mw: EventMiddleware = current_middleware,
                nxt: Callable[[Event[Any]], Awaitable[Event[Any] | None]] = next_chain,
            ) -> Event[Any] | None:
                return await mw.process(e, nxt)

            chain = make_chain

        return await chain(event)

    # -------------------------------------------------------------------------
    # Handler Invocation
    # -------------------------------------------------------------------------

    def _get_matching_handlers(self, event: Event[Any]) -> list[HandlerRegistration]:
        """Get handlers matching an event."""
        matching: list[HandlerRegistration] = []

        for pattern, handlers in self._handlers.items():
            if event.event_type.matches(pattern):
                for handler in handlers:
                    if handler.matches(event):
                        matching.append(handler)

        # Sort by priority
        matching.sort(key=lambda h: h.priority.value, reverse=True)

        return matching

    async def _invoke_handler(
        self,
        registration: HandlerRegistration,
        event: Event[Any],
    ) -> Any:
        """Invoke a single handler."""
        handler = registration.handler
        timeout = registration.timeout_seconds

        if registration.is_async:
            if timeout:
                return await asyncio.wait_for(
                    handler(event),  # type: ignore
                    timeout=timeout,
                )
            else:
                return await handler(event)  # type: ignore
        else:
            # Run sync handler in executor
            loop = asyncio.get_event_loop()
            if timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, handler, event),
                    timeout=timeout,
                )
            else:
                return await loop.run_in_executor(None, handler, event)

    # -------------------------------------------------------------------------
    # Multi-Agent Support
    # -------------------------------------------------------------------------

    def get_agent_view(self, agent_id: str) -> AgentEventView:
        """
        Get an agent-scoped view of the event bus.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentEventView for the agent
        """
        if agent_id not in self._agent_views:
            self._agent_views[agent_id] = AgentEventView(self, agent_id)

        return self._agent_views[agent_id]

    def remove_agent_view(self, agent_id: str) -> bool:
        """Remove an agent view and its handlers."""
        if agent_id in self._agent_views:
            # Unsubscribe all agent handlers
            self.unsubscribe_all(agent_id=agent_id)
            del self._agent_views[agent_id]
            return True
        return False

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def has_handlers(self, pattern: str) -> bool:
        """Check if pattern has any handlers."""
        return pattern in self._handlers and len(self._handlers[pattern]) > 0

    def handler_count(self, pattern: str | None = None) -> int:
        """Get number of handlers."""
        if pattern:
            return len(self._handlers.get(pattern, []))
        return len(self._handler_index)

    def list_patterns(self) -> list[str]:
        """List all subscribed patterns."""
        return list(self._handlers.keys())

    def get_handler_info(self, handler_id: str) -> dict[str, Any] | None:
        """Get information about a handler."""
        registration = self._handler_index.get(handler_id)
        if registration is None:
            return None

        return {
            "handler_id": registration.handler_id,
            "pattern": registration.pattern,
            "priority": registration.priority.name,
            "agent_id": registration.agent_id,
            "is_async": registration.is_async,
            "once": registration.once,
            "enabled": registration.enabled,
            "invocation_count": registration.invocation_count,
            "error_count": registration.error_count,
            "last_invoked": registration.last_invoked.isoformat() if registration.last_invoked else None,
        }

    def enable_handler(self, handler_id: str) -> bool:
        """Enable a handler."""
        registration = self._handler_index.get(handler_id)
        if registration:
            registration.enabled = True
            return True
        return False

    def disable_handler(self, handler_id: str) -> bool:
        """Disable a handler."""
        registration = self._handler_index.get(handler_id)
        if registration:
            registration.enabled = False
            return True
        return False

    # -------------------------------------------------------------------------
    # History & Dead Letter
    # -------------------------------------------------------------------------

    @property
    def history(self) -> EventHistory | None:
        """Get event history."""
        return self._history

    @property
    def dead_letter(self) -> DeadLetterQueue | None:
        """Get dead letter queue."""
        return self._dead_letter

    async def replay_history(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Replay events from history."""
        if self._history is None:
            return 0
        return await self._history.replay(self, event_type, since)

    async def retry_dead_letters(self, max_retries: int = 3) -> int:
        """Retry events in dead letter queue."""
        if self._dead_letter is None:
            return 0
        return await self._dead_letter.retry(self, max_retries)

    # -------------------------------------------------------------------------
    # Metrics & Stats
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Get event bus metrics."""
        if self._metrics is None:
            return {}

        avg_time = 0.0
        if self._metrics.events_delivered > 0:
            avg_time = self._metrics.total_processing_time_ms / self._metrics.events_delivered

        return {
            "events_published": self._metrics.events_published,
            "events_delivered": self._metrics.events_delivered,
            "events_failed": self._metrics.events_failed,
            "handlers_invoked": self._metrics.handlers_invoked,
            "handlers_failed": self._metrics.handlers_failed,
            "avg_processing_time_ms": avg_time,
            "events_by_type": dict(self._metrics.events_by_type),
            "handler_count": len(self._handler_index),
            "pattern_count": len(self._handlers),
            "history_size": self._history.size if self._history else 0,
            "dead_letter_size": self._dead_letter.size if self._dead_letter else 0,
        }

    def reset_metrics(self) -> None:
        """Reset metrics."""
        if self._metrics:
            self._metrics = EventMetrics()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False

        # Emit shutdown event
        await self.emit(Events.SYSTEM_SHUTDOWN, {"reason": "shutdown"})

        # Clear handlers
        self._handlers.clear()
        self._handler_index.clear()

        self._logger.info("Event bus shutdown complete")


# =============================================================================
# AGENT EVENT VIEW
# =============================================================================


class AgentEventView:
    """
    Agent-scoped view of the event bus.

    Provides isolated event handling for multi-agent systems.
    """

    def __init__(self, bus: EventBus, agent_id: str) -> None:
        """
        Initialize agent view.

        Args:
            bus: Parent event bus
            agent_id: Agent identifier
        """
        self._bus = bus
        self._agent_id = agent_id
        self._handler_ids: set[str] = set()

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._agent_id

    def subscribe(
        self,
        pattern: str | EventType,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
        filter_fn: Callable[[Event[Any]], bool] | None = None,
    ) -> str:
        """Subscribe handler scoped to this agent."""
        handler_id = self._bus.subscribe(
            pattern=pattern,
            handler=handler,
            priority=priority,
            agent_id=self._agent_id,
            once=once,
            filter_fn=filter_fn,
        )
        self._handler_ids.add(handler_id)
        return handler_id

    def on(
        self,
        pattern: str | EventType,
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
    ) -> Callable[[EventHandler], EventHandler]:
        """Decorator for subscribing handlers."""
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(pattern, handler, priority, once)
            return handler
        return decorator

    def unsubscribe(self, handler_id: str) -> bool:
        """Unsubscribe a handler."""
        if handler_id in self._handler_ids:
            self._handler_ids.discard(handler_id)
            return self._bus.unsubscribe(handler_id)
        return False

    def unsubscribe_all(self) -> int:
        """Unsubscribe all handlers for this agent."""
        count = 0
        for handler_id in list(self._handler_ids):
            if self._bus.unsubscribe(handler_id):
                count += 1
        self._handler_ids.clear()
        return count

    async def publish(
        self,
        event_type: EventType | str,
        data: Any = None,
        scope: EventScope = EventScope.AGENT,
        **kwargs: Any,
    ) -> Event[Any]:
        """Publish event from this agent."""
        return await self._bus.publish(
            event_type,
            data=data,
            agent_id=self._agent_id,
            scope=scope,
            **kwargs,
        )

    async def emit(
        self,
        event_type: EventType | str,
        data: Any = None,
        **kwargs: Any,
    ) -> Event[Any]:
        """Emit event from this agent."""
        return await self.publish(event_type, data=data, **kwargs)

    async def broadcast(
        self,
        event_type: EventType | str,
        data: Any = None,
        **kwargs: Any,
    ) -> Event[Any]:
        """Broadcast event to all agents."""
        return await self._bus.broadcast(
            event_type,
            data=data,
            agent_id=self._agent_id,
            **kwargs,
        )

    def handler_count(self) -> int:
        """Get number of handlers for this agent."""
        return len(self._handler_ids)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_event_bus(
    enable_history: bool = True,
    enable_dead_letter: bool = True,
    enable_logging: bool = True,
    log_level: int = logging.DEBUG,
    logger: logging.Logger | None = None,
) -> EventBus:
    """
    Factory function to create configured EventBus.

    Args:
        enable_history: Enable event history
        enable_dead_letter: Enable dead letter queue
        enable_logging: Add logging middleware
        log_level: Logging level
        logger: Optional logger

    Returns:
        Configured EventBus
    """
    config = EventBusConfig(
        enable_history=enable_history,
        enable_dead_letter=enable_dead_letter,
    )

    bus = EventBus(config=config, logger=logger)

    if enable_logging:
        bus.use(LoggingMiddleware(logger=logger, log_level=log_level))

    return bus


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EventPriority",
    "EventStatus",
    "PropagationMode",
    "EventScope",
    # Event Types
    "EventType",
    "Events",
    # Event
    "Event",
    "create_event",
    # Handlers
    "EventHandler",
    "SyncHandler",
    "AsyncHandler",
    "HandlerRegistration",
    # Middleware
    "EventMiddleware",
    "LoggingMiddleware",
    "FilterMiddleware",
    "TransformMiddleware",
    "RetryMiddleware",
    "ThrottleMiddleware",
    # Dead Letter
    "DeadLetterEntry",
    "DeadLetterQueue",
    # History
    "EventHistory",
    # Event Bus
    "EventBusConfig",
    "EventMetrics",
    "EventBus",
    # Agent View
    "AgentEventView",
    # Factory
    "create_event_bus",
]
