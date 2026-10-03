"""
Middleware - Extensible middleware system for the AI Agent Platform.

This module provides a comprehensive middleware framework including:
- Before/after execution hooks
- Error handling hooks
- Logging middleware
- Metrics collection middleware
- Approval workflow middleware
- Async-first design
- Middleware chain with priority ordering
- Context propagation
- Conditional middleware execution
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")


# =============================================================================
# ENUMS
# =============================================================================


class MiddlewarePhase(Enum):
    """Phases of middleware execution."""

    BEFORE = "before"
    AFTER = "after"
    ERROR = "error"
    FINALLY = "finally"


class MiddlewarePriority(Enum):
    """Priority levels for middleware ordering."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


class MiddlewareAction(Enum):
    """Actions middleware can take."""

    CONTINUE = auto()  # Continue to next middleware
    SKIP = auto()  # Skip remaining middleware in phase
    ABORT = auto()  # Abort entire chain
    RETRY = auto()  # Retry the operation


# =============================================================================
# CONTEXT
# =============================================================================


@dataclass
class MiddlewareContext:
    """
    Context passed through middleware chain.

    Attributes:
        request_id: Unique request identifier
        session_id: Session identifier
        user_id: User identifier
        agent_id: Agent identifier
        operation: Operation being performed
        timestamp: Request timestamp
        metadata: Additional context data
        state: Mutable state for middleware communication
        errors: Collected errors
        metrics: Collected metrics
        skip_remaining: Flag to skip remaining middleware
        abort: Flag to abort chain
    """

    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    operation: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    skip_remaining: bool = False
    abort: bool = False

    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)

    def add_error(self, error: Exception) -> None:
        """Add an error to the context."""
        self.errors.append(error)

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the context."""
        self.metrics[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "state": {k: str(v) for k, v in self.state.items()},
            "error_count": len(self.errors),
            "metrics": self.metrics,
        }


# =============================================================================
# MIDDLEWARE RESULT
# =============================================================================


@dataclass
class MiddlewareResult:
    """
    Result from middleware execution.

    Attributes:
        action: Action to take
        modified_request: Modified request (if any)
        modified_response: Modified response (if any)
        error: Error if action is ABORT
        metadata: Additional result data
    """

    action: MiddlewareAction = MiddlewareAction.CONTINUE
    modified_request: Any = None
    modified_response: Any = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def continue_(cls, **kwargs: Any) -> MiddlewareResult:
        """Create a continue result."""
        return cls(action=MiddlewareAction.CONTINUE, **kwargs)

    @classmethod
    def skip(cls, **kwargs: Any) -> MiddlewareResult:
        """Create a skip result."""
        return cls(action=MiddlewareAction.SKIP, **kwargs)

    @classmethod
    def abort(cls, error: Exception | None = None, **kwargs: Any) -> MiddlewareResult:
        """Create an abort result."""
        return cls(action=MiddlewareAction.ABORT, error=error, **kwargs)

    @classmethod
    def retry(cls, **kwargs: Any) -> MiddlewareResult:
        """Create a retry result."""
        return cls(action=MiddlewareAction.RETRY, **kwargs)


# =============================================================================
# MIDDLEWARE PROTOCOL
# =============================================================================


@runtime_checkable
class Middleware(Protocol):
    """Protocol for middleware implementations."""

    @property
    def name(self) -> str:
        """Middleware name."""
        ...

    @property
    def priority(self) -> int:
        """Middleware priority (lower = higher priority)."""
        ...

    async def before(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute before the main operation."""
        ...

    async def after(
        self,
        request: Any,
        response: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute after the main operation."""
        ...

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute when an error occurs."""
        ...


# =============================================================================
# BASE MIDDLEWARE
# =============================================================================


class BaseMiddleware(ABC):
    """
    Abstract base class for middleware implementations.

    Provides default implementations for all hooks.
    """

    def __init__(
        self,
        name: str | None = None,
        priority: int = MiddlewarePriority.NORMAL.value,
        enabled: bool = True,
        conditions: list[Callable[[MiddlewareContext], bool]] | None = None,
    ) -> None:
        """
        Initialize middleware.

        Args:
            name: Middleware name
            priority: Execution priority
            enabled: Whether middleware is enabled
            conditions: Conditions for execution
        """
        self._name = name or self.__class__.__name__
        self._priority = priority
        self._enabled = enabled
        self._conditions = conditions or []

    @property
    def name(self) -> str:
        """Get middleware name."""
        return self._name

    @property
    def priority(self) -> int:
        """Get middleware priority."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set middleware enabled state."""
        self._enabled = value

    def should_execute(self, context: MiddlewareContext) -> bool:
        """Check if middleware should execute."""
        if not self._enabled:
            return False

        for condition in self._conditions:
            if not condition(context):
                return False

        return True

    def add_condition(self, condition: Callable[[MiddlewareContext], bool]) -> None:
        """Add an execution condition."""
        self._conditions.append(condition)

    async def before(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute before hook. Override in subclass."""
        return MiddlewareResult.continue_()

    async def after(
        self,
        request: Any,
        response: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute after hook. Override in subclass."""
        return MiddlewareResult.continue_()

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Execute error hook. Override in subclass."""
        return MiddlewareResult.continue_()

    async def on_finally(
        self,
        request: Any,
        response: Any | None,
        error: Exception | None,
        context: MiddlewareContext,
    ) -> None:
        """Execute finally hook. Override in subclass."""
        pass


# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================


class LoggingMiddleware(BaseMiddleware):
    """
    Middleware for request/response logging.

    Logs request details, response details, errors, and timing.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_request: bool = True,
        log_response: bool = True,
        log_errors: bool = True,
        log_timing: bool = True,
        sensitive_fields: set[str] | None = None,
        max_body_length: int = 1000,
        **kwargs: Any,
    ) -> None:
        """
        Initialize logging middleware.

        Args:
            logger: Logger instance
            log_request: Log request details
            log_response: Log response details
            log_errors: Log errors
            log_timing: Log timing information
            sensitive_fields: Fields to redact
            max_body_length: Maximum body length to log
            **kwargs: Base middleware arguments
        """
        super().__init__(name="LoggingMiddleware", **kwargs)
        self._logger = logger or logging.getLogger(__name__)
        self._log_request = log_request
        self._log_response = log_response
        self._log_errors = log_errors
        self._log_timing = log_timing
        self._sensitive_fields = sensitive_fields or {"password", "token", "api_key", "secret"}
        self._max_body_length = max_body_length

    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive fields from data."""
        if isinstance(data, dict):
            return {
                k: "[REDACTED]" if k.lower() in self._sensitive_fields else self._redact_sensitive(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]
        return data

    def _truncate(self, text: str) -> str:
        """Truncate text to max length."""
        if len(text) > self._max_body_length:
            return text[:self._max_body_length] + "...[truncated]"
        return text

    async def before(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Log request details."""
        context.set("_logging_start_time", time.perf_counter())

        if self._log_request:
            request_data = self._redact_sensitive(
                request if isinstance(request, dict) else str(request)
            )
            self._logger.info(
                f"[{context.request_id}] Request started",
                extra={
                    "request_id": context.request_id,
                    "operation": context.operation,
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "request": self._truncate(str(request_data)),
                },
            )

        return MiddlewareResult.continue_()

    async def after(
        self,
        request: Any,
        response: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Log response details."""
        if self._log_response:
            response_data = self._redact_sensitive(
                response if isinstance(response, dict) else str(response)
            )
            self._logger.info(
                f"[{context.request_id}] Request completed",
                extra={
                    "request_id": context.request_id,
                    "operation": context.operation,
                    "response": self._truncate(str(response_data)),
                },
            )

        return MiddlewareResult.continue_()

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Log error details."""
        if self._log_errors:
            self._logger.error(
                f"[{context.request_id}] Request failed: {error}",
                extra={
                    "request_id": context.request_id,
                    "operation": context.operation,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                exc_info=True,
            )

        return MiddlewareResult.continue_()

    async def on_finally(
        self,
        request: Any,
        response: Any | None,
        error: Exception | None,
        context: MiddlewareContext,
    ) -> None:
        """Log timing information."""
        if self._log_timing:
            start_time = context.get("_logging_start_time")
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.add_metric("duration_ms", duration_ms)

                self._logger.info(
                    f"[{context.request_id}] Request duration: {duration_ms:.2f}ms",
                    extra={
                        "request_id": context.request_id,
                        "duration_ms": duration_ms,
                        "success": error is None,
                    },
                )


# =============================================================================
# METRICS MIDDLEWARE
# =============================================================================


@dataclass
class MetricsCollector:
    """Collector for metrics data."""

    counters: dict[str, int] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=dict)
    timers: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self.gauges[key] = value

    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def timer(self, name: str, duration_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timer value."""
        key = self._make_key(name, tags)
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(duration_ms)

    def decrement(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Decrement a counter."""
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) - value

    def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
        """Create a metric key with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all metrics."""
        stats: dict[str, Any] = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {},
            "timers": {},
        }

        for name, values in self.histograms.items():
            if values:
                stats["histograms"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }

        for name, values in self.timers.items():
            if values:
                sorted_values = sorted(values)
                stats["timers"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted_values[len(sorted_values) // 2],
                    "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) >= 20 else max(values),
                    "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) >= 100 else max(values),
                }

        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()


class MetricsMiddleware(BaseMiddleware):
    """
    Middleware for collecting metrics.

    Tracks request counts, latencies, error rates, and custom metrics.
    """

    def __init__(
        self,
        collector: MetricsCollector | None = None,
        prefix: str = "agent",
        track_latency: bool = True,
        track_errors: bool = True,
        track_requests: bool = True,
        custom_tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize metrics middleware.

        Args:
            collector: Metrics collector
            prefix: Metric name prefix
            track_latency: Track request latency
            track_errors: Track error counts
            track_requests: Track request counts
            custom_tags: Custom tags for all metrics
            **kwargs: Base middleware arguments
        """
        super().__init__(name="MetricsMiddleware", priority=MiddlewarePriority.HIGHEST.value, **kwargs)
        self._collector = collector or MetricsCollector()
        self._prefix = prefix
        self._track_latency = track_latency
        self._track_errors = track_errors
        self._track_requests = track_requests
        self._custom_tags = custom_tags or {}

    @property
    def collector(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._collector

    def _get_tags(self, context: MiddlewareContext) -> dict[str, str]:
        """Get tags for metrics."""
        tags = dict(self._custom_tags)
        if context.operation:
            tags["operation"] = context.operation
        if context.agent_id:
            tags["agent_id"] = context.agent_id
        return tags

    async def before(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Record request start."""
        context.set("_metrics_start_time", time.perf_counter())

        if self._track_requests:
            tags = self._get_tags(context)
            self._collector.increment(f"{self._prefix}_requests_total", tags=tags)
            self._collector.increment(f"{self._prefix}_requests_active", tags=tags)

        return MiddlewareResult.continue_()

    async def after(
        self,
        request: Any,
        response: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Record successful request."""
        tags = self._get_tags(context)
        tags["status"] = "success"

        if self._track_requests:
            self._collector.increment(f"{self._prefix}_requests_completed", tags=tags)

        return MiddlewareResult.continue_()

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Record error metrics."""
        if self._track_errors:
            tags = self._get_tags(context)
            tags["error_type"] = type(error).__name__
            self._collector.increment(f"{self._prefix}_errors_total", tags=tags)

        return MiddlewareResult.continue_()

    async def on_finally(
        self,
        request: Any,
        response: Any | None,
        error: Exception | None,
        context: MiddlewareContext,
    ) -> None:
        """Record latency and decrement active requests."""
        if self._track_latency:
            start_time = context.get("_metrics_start_time")
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                tags = self._get_tags(context)
                self._collector.timer(f"{self._prefix}_latency_ms", duration_ms, tags=tags)
                context.add_metric("latency_ms", duration_ms)

        if self._track_requests:
            tags = self._get_tags(context)
            self._collector.decrement(f"{self._prefix}_requests_active", tags=tags)


# =============================================================================
# RETRY MIDDLEWARE
# =============================================================================


class RetryMiddleware(BaseMiddleware):
    """
    Middleware that retries failed operations.

    Supports configurable retry count, backoff strategy, and error filtering.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize retry middleware.

        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff multiplier
            retryable_exceptions: Exceptions that trigger retry
            **kwargs: Base middleware arguments
        """
        super().__init__(name="RetryMiddleware", **kwargs)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._backoff_factor = backoff_factor
        self._retryable_exceptions = retryable_exceptions or (Exception,)

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Handle retry logic on error."""
        if not isinstance(error, self._retryable_exceptions):
            return MiddlewareResult.continue_()

        retry_count = context.get("_retry_count", 0) + 1

        if retry_count > self._max_retries:
            return MiddlewareResult.continue_()

        context.set("_retry_count", retry_count)

        delay = min(self._base_delay * (self._backoff_factor ** (retry_count - 1)), self._max_delay)
        await asyncio.sleep(delay)

        return MiddlewareResult.retry(metadata={"retry_count": retry_count, "delay": delay})


# =============================================================================
# TIMING MIDDLEWARE
# =============================================================================


class TimingMiddleware(BaseMiddleware):
    """
    Middleware for detailed timing and performance tracking.

    Adds timing information to the context metrics.
    """

    def __init__(
        self,
        warn_threshold_ms: float = 5000.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize timing middleware.

        Args:
            warn_threshold_ms: Threshold in ms to log warnings
            **kwargs: Base middleware arguments
        """
        super().__init__(name="TimingMiddleware", **kwargs)
        self._warn_threshold_ms = warn_threshold_ms
        self._logger = logging.getLogger(__name__)

    async def before(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Record start time."""
        context.set("_timing_start", time.perf_counter_ns())
        return MiddlewareResult.continue_()

    async def on_finally(
        self,
        request: Any,
        response: Any | None,
        error: Exception | None,
        context: MiddlewareContext,
    ) -> None:
        """Record timing breakdown."""
        start_ns = context.get("_timing_start")
        if start_ns is None:
            return

        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms = elapsed_ns / 1_000_000

        context.add_metric("total_duration_ms", elapsed_ms)
        context.add_metric("total_duration_ns", elapsed_ns)

        if elapsed_ms > self._warn_threshold_ms:
            self._logger.warning(
                f"[{context.request_id}] Slow operation: {elapsed_ms:.2f}ms",
                extra={
                    "request_id": context.request_id,
                    "operation": context.operation,
                    "duration_ms": elapsed_ms,
                    "threshold_ms": self._warn_threshold_ms,
                },
            )


# =============================================================================
# MIDDLEWARE CHAIN
# =============================================================================


class MiddlewareChain:
    """
    Orchestrates middleware execution in priority order.

    Runs middleware through before/after/error/finally phases
    with proper priority ordering and context propagation.
    """

    def __init__(
        self,
        middlewares: list[Middleware] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize middleware chain.

        Args:
            middlewares: Initial list of middleware
            logger: Optional logger
        """
        self._middlewares: list[Middleware] = []
        self._logger = logger or logging.getLogger(__name__)
        self._handler: Callable[[Any], Awaitable[Any]] | None = None

        if middlewares:
            for m in middlewares:
                self.add(m)

    def add(self, middleware: Middleware) -> None:
        """Add middleware to the chain."""
        self._middlewares.append(middleware)
        self._middlewares.sort(key=lambda m: m.priority)

    def remove(self, name: str) -> bool:
        """Remove middleware by name."""
        for i, m in enumerate(self._middlewares):
            if m.name == name:
                self._middlewares.pop(i)
                return True
        return False

    def get(self, name: str) -> Middleware | None:
        """Get middleware by name."""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    @property
    def middlewares(self) -> list[Middleware]:
        """Get sorted middleware list."""
        return list(self._middlewares)

    def disable(self, name: str) -> bool:
        """Disable middleware by name."""
        mw = self.get(name)
        if mw and hasattr(mw, "enabled"):
            mw.enabled = False  # type: ignore[assignment]
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable middleware by name."""
        mw = self.get(name)
        if mw and hasattr(mw, "enabled"):
            mw.enabled = True  # type: ignore[assignment]
            return True
        return False

    async def execute(
        self,
        request: Any,
        operation: str | None = None,
        context: MiddlewareContext | None = None,
    ) -> tuple[Any, MiddlewareContext]:
        """
        Execute the full middleware chain.

        Args:
            request: The request to process
            operation: Operation name
            context: Optional existing context

        Returns:
            Tuple of (response, context)
        """
        ctx = context or MiddlewareContext(operation=operation)
        response: Any = None
        error: Exception | None = None
        max_retries = 5

        for attempt in range(max_retries):
            # --- Before phase ---
            skip_after = False
            for mw in self._middlewares:
                if not self._should_execute(mw, ctx):
                    continue

                try:
                    result = await mw.before(request, ctx)
                except Exception as e:
                    ctx.add_error(e)
                    self._logger.error(f"Middleware '{mw.name}' before error: {e}")
                    result = MiddlewareResult.abort(e)

                action = result.action
                if result.modified_request is not None:
                    request = result.modified_request

                if action == MiddlewareAction.ABORT:
                    error = result.error or Exception("Aborted by middleware")
                    ctx.abort = True
                    skip_after = True
                    break
                elif action == MiddlewareAction.SKIP:
                    skip_after = True
                    break
                elif action == MiddlewareAction.RETRY:
                    error = None
                    break

            if ctx.abort:
                break

            if action == MiddlewareAction.RETRY:
                continue

            # --- Main operation ---
            if not skip_after:
                try:
                    response = await self._execute_operation(request)
                except Exception as e:
                    error = e
                    ctx.add_error(e)

            # --- Error phase ---
            if error is not None:
                for mw in self._middlewares:
                    if not self._should_execute(mw, ctx):
                        continue

                    try:
                        result = await mw.on_error(request, error, ctx)
                    except Exception as e:
                        ctx.add_error(e)
                        result = MiddlewareResult.abort(e)

                    action = result.action
                    if result.modified_response is not None:
                        response = result.modified_response

                    if action == MiddlewareAction.ABORT:
                        ctx.abort = True
                        break
                    elif action == MiddlewareAction.RETRY:
                        error = None
                        ctx.set("_retry_error", str(error) if error else "unknown")
                        break

                if action == MiddlewareAction.RETRY:
                    continue

            # --- After phase ---
            if not skip_after and error is None:
                for mw in self._middlewares:
                    if not self._should_execute(mw, ctx):
                        continue

                    try:
                        result = await mw.after(request, response, ctx)
                    except Exception as e:
                        ctx.add_error(e)
                        result = MiddlewareResult.abort(e)

                    action = result.action
                    if result.modified_response is not None:
                        response = result.modified_response

                    if action == MiddlewareAction.ABORT:
                        error = result.error or Exception("Aborted in after phase")
                        ctx.abort = True
                        break

            # --- Finally phase ---
            for mw in self._middlewares:
                if not self._should_execute(mw, ctx):
                    continue

                try:
                    await mw.on_finally(request, response, error, ctx)  # type: ignore[attr-defined]
                except Exception as e:
                    self._logger.warning(f"Middleware '{mw.name}' finally error: {e}")
                    ctx.add_error(e)

            break

        return response, ctx

    async def _execute_operation(self, request: Any) -> Any:
        """
        Execute the main operation.

        Override in subclass or set via handler.
        """
        if self._handler is not None:
            return await self._handler(request)
        return request

    def set_handler(self, handler: Callable[[Any], Awaitable[Any]]) -> None:
        """Set the main operation handler."""
        self._handler = handler

    def _should_execute(self, middleware: Middleware, context: MiddlewareContext) -> bool:
        """Check if middleware should execute."""
        if hasattr(middleware, "should_execute"):
            return middleware.should_execute(context)  # type: ignore[union-attr]
        return True

    def reset(self) -> None:
        """Reset chain state."""
        self._handler = None
        for mw in self._middlewares:
            if hasattr(mw, "enabled"):
                mw.enabled = True  # type: ignore[assignment]


# =============================================================================
# CHAIN BUILDER
# =============================================================================


def create_middleware_chain(
    log_requests: bool = True,
    collect_metrics: bool = True,
    max_retries: int = 0,
    **kwargs: Any,
) -> MiddlewareChain:
    """
    Factory function to create a configured MiddlewareChain.

    Args:
        log_requests: Enable logging middleware
        collect_metrics: Enable metrics middleware
        max_retries: Max retries (0 = no retry middleware)
        **kwargs: Additional arguments

    Returns:
        Configured MiddlewareChain
    """
    chain = MiddlewareChain()

    if max_retries > 0:
        chain.add(RetryMiddleware(max_retries=max_retries))

    if collect_metrics:
        chain.add(MetricsMiddleware(**kwargs))

    if log_requests:
        chain.add(LoggingMiddleware(**kwargs))

    chain.add(TimingMiddleware(**kwargs))

    return chain


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MiddlewarePhase",
    "MiddlewarePriority",
    "MiddlewareAction",
    # Context
    "MiddlewareContext",
    # Result
    "MiddlewareResult",
    # Protocol
    "Middleware",
    # Base
    "BaseMiddleware",
    # Middleware
    "LoggingMiddleware",
    "MetricsCollector",
    "MetricsMiddleware",
    "RetryMiddleware",
    "TimingMiddleware",
    # Chain
    "MiddlewareChain",
    # Factory
    "create_middleware_chain",
]
