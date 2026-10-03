from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import SpanKind, Status, StatusCode
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class TelemetryError(Exception):
    pass


class TraceNotFoundError(TelemetryError):
    pass


class TraceStatus(str, Enum):
    ACTIVE = "active"
    OK = "ok"
    ERROR = "error"


@dataclass
class TraceEvent:
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    trace_id: str
    span_name: str
    started_at: float
    parent_trace_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    status: TraceStatus = TraceStatus.ACTIVE
    ended_at: Optional[float] = None
    error_message: Optional[str] = None
    _otel_span: Optional[Any] = field(default=None, repr=False)
    _otel_cm: Optional[Any] = field(default=None, repr=False)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_name": self.span_name,
            "parent_trace_id": self.parent_trace_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "attributes": self.attributes,
            "error_message": self.error_message,
            "events": [{"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes} for e in self.events],
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Telemetry:
    """OpenTelemetry-backed distributed tracing and event collection, with in-memory fallback."""

    def __init__(
        self,
        *,
        service_name: str = "service",
        use_console_exporter: bool = False,
        max_completed_traces: int = 5000,
    ) -> None:
        self.service_name = service_name
        self.max_completed_traces = max_completed_traces
        self._active_traces: dict[str, TraceContext] = {}
        self._completed_traces: list[TraceContext] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._trace_count = 0
        self._event_count = 0
        self._otel_enabled = False
        self._tracer: Optional[Any] = None

        if _HAS_OTEL:
            try:
                resource = Resource.create({"service.name": service_name})
                provider = TracerProvider(resource=resource)
                if use_console_exporter:
                    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                otel_trace.set_tracer_provider(provider)
                self._tracer = otel_trace.get_tracer(service_name)
                self._otel_enabled = True
            except Exception:
                self._otel_enabled = False
                self._tracer = None

    def start_trace(
        self,
        span_name: str,
        *,
        parent_trace_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> TraceContext:
        self._trace_count += 1
        trace_id = str(uuid.uuid4())
        context = TraceContext(
            trace_id=trace_id,
            span_name=span_name,
            started_at=time.time(),
            parent_trace_id=parent_trace_id,
            attributes=dict(attributes or {}),
        )

        if self._otel_enabled and self._tracer is not None:
            try:
                cm = self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL)
                span = cm.__enter__()
                for key, value in (attributes or {}).items():
                    span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
                context._otel_span = span
                context._otel_cm = cm
            except Exception:
                pass

        self._active_traces[trace_id] = context
        return context

    def add_event(self, trace_id: str, event_name: str, *, attributes: Optional[dict[str, Any]] = None) -> TraceEvent:
        self._event_count += 1
        context = self._active_traces.get(trace_id)
        if context is None:
            raise TraceNotFoundError(f"no active trace with id {trace_id}")

        event = TraceEvent(name=event_name, attributes=dict(attributes or {}))
        context.events.append(event)

        if context._otel_span is not None:
            try:
                context._otel_span.add_event(event_name, attributes={
                    k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in event.attributes.items()
                })
            except Exception:
                pass

        return event

    def set_attribute(self, trace_id: str, key: str, value: Any) -> None:
        context = self._active_traces.get(trace_id)
        if context is None:
            raise TraceNotFoundError(f"no active trace with id {trace_id}")
        context.attributes[key] = value
        if context._otel_span is not None:
            try:
                context._otel_span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
            except Exception:
                pass

    def end_trace(
        self,
        trace_id: str,
        *,
        status: TraceStatus = TraceStatus.OK,
        error_message: Optional[str] = None,
    ) -> TraceContext:
        context = self._active_traces.pop(trace_id, None)
        if context is None:
            raise TraceNotFoundError(f"no active trace with id {trace_id}")

        context.ended_at = time.time()
        context.status = status
        context.error_message = error_message

        if context._otel_span is not None:
            try:
                if status == TraceStatus.ERROR:
                    context._otel_span.set_status(Status(StatusCode.ERROR, error_message or ""))
                else:
                    context._otel_span.set_status(Status(StatusCode.OK))
            except Exception:
                pass
        if context._otel_cm is not None:
            try:
                context._otel_cm.__exit__(None, None, None)
            except Exception:
                pass

        context._otel_span = None
        context._otel_cm = None

        self._completed_traces.append(context)
        if len(self._completed_traces) > self.max_completed_traces:
            self._completed_traces = self._completed_traces[-self.max_completed_traces :]

        return context

    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]
        return next((t for t in self._completed_traces if t.trace_id == trace_id), None)

    def list_active_traces(self) -> list[TraceContext]:
        return list(self._active_traces.values())

    def list_completed_traces(self, limit: int = 100) -> list[TraceContext]:
        return self._completed_traces[-limit:]

    async def start_trace_async(self, span_name: str, **kwargs: Any) -> TraceContext:
        async with self._lock:
            return self.start_trace(span_name, **kwargs)

    async def end_trace_async(self, trace_id: str, **kwargs: Any) -> TraceContext:
        async with self._lock:
            return self.end_trace(trace_id, **kwargs)

    async def add_event_async(self, trace_id: str, event_name: str, **kwargs: Any) -> TraceEvent:
        async with self._lock:
            return self.add_event(trace_id, event_name, **kwargs)

    def health_check(self) -> HealthStatus:
        try:
            probe = self.start_trace("__health_check__")
            self.add_event(probe.trace_id, "probe_event")
            self.end_trace(probe.trace_id, status=TraceStatus.OK)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "service_name": self.service_name,
                "otel_enabled": self._otel_enabled,
                "active_trace_count": len(self._active_traces),
                "completed_trace_count": len(self._completed_traces),
                "trace_count": self._trace_count,
                "event_count": self._event_count,
            }
            return HealthStatus(healthy=True, component="telemetry", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="telemetry", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Telemetry",
    "TraceContext",
    "TraceEvent",
    "TraceStatus",
    "TelemetryError",
    "TraceNotFoundError",
    "HealthStatus",
]
