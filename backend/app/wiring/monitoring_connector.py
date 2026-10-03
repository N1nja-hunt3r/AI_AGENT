"""
monitoring_connector.py
-----------------------
Connects the monitoring subsystem: Logger, Metrics, Telemetry, Profiler,
Alerts, Dashboard, HealthMonitor, Traces, OpenTelemetry.
Compatible with monitoring/ and middleware_loader.py.
"""

from __future__ import annotations

import asyncio
import cProfile
import io
import logging
import pstats
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

# ---------------------------------------------------------------------------
# OpenTelemetry – optional hard dependency; graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Types / Enums
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


MetricValue = Union[int, float]
Labels = Dict[str, str]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricPoint:
    name: str
    value: MetricValue
    labels: Labels = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass
class TraceSpan:
    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    parent_id: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    labels: Labels = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class HealthReport:
    status: HealthStatus
    component: str
    details: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DashboardSnapshot:
    timestamp: float
    metrics: Dict[str, List[MetricPoint]]
    active_alerts: List[Alert]
    health_reports: List[HealthReport]
    recent_spans: List[TraceSpan]


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------


class _Logger:
    """Structured logger wrapping stdlib logging."""

    def __init__(self, name: str = "nova.monitoring") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def log(
        self,
        level: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        lvl = getattr(logging, level.upper(), logging.INFO)
        extra_msg = f" | {context}" if context else ""
        self._logger.log(lvl, f"{message}{extra_msg}")

    def debug(self, msg: str, **ctx: Any) -> None:
        self.log("DEBUG", msg, context=ctx or None)

    def info(self, msg: str, **ctx: Any) -> None:
        self.log("INFO", msg, context=ctx or None)

    def warning(self, msg: str, **ctx: Any) -> None:
        self.log("WARNING", msg, context=ctx or None)

    def error(self, msg: str, **ctx: Any) -> None:
        self.log("ERROR", msg, context=ctx or None)

    def critical(self, msg: str, **ctx: Any) -> None:
        self.log("CRITICAL", msg, context=ctx or None)


class _MetricsStore:
    """In-memory metrics storage with aggregation support."""

    def __init__(self) -> None:
        self._points: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record(self, point: MetricPoint) -> None:
        async with self._lock:
            self._points[point.name].append(point)

    def aggregate(
        self,
        name: str,
        method: str = "last",
        window_seconds: float = 60.0,
    ) -> Optional[MetricValue]:
        series = self._points.get(name, [])
        now = time.time()
        windowed = [p for p in series if now - p.timestamp <= window_seconds]
        if not windowed:
            return None
        values = [p.value for p in windowed]
        if method == "last":
            return windowed[-1].value
        if method == "sum":
            return sum(values)
        if method == "avg":
            return sum(values) / len(values)
        if method == "max":
            return max(values)
        if method == "min":
            return min(values)
        if method == "count":
            return len(values)
        return windowed[-1].value

    def snapshot(self) -> Dict[str, List[MetricPoint]]:
        return dict(self._points)

    def clear(self) -> None:
        self._points.clear()


class _Telemetry:
    """Lightweight telemetry event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[..., Any]) -> None:
        self._subscribers[event].append(handler)

    async def emit(self, event: str, payload: Dict[str, Any]) -> None:
        for handler in self._subscribers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, payload)
                else:
                    handler(event, payload)
            except Exception:
                pass  # telemetry must never crash caller


class _Profiler:
    """cProfile wrapper with async/sync context managers."""

    def __init__(self) -> None:
        self._profiles: Dict[str, str] = {}

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        pr = cProfile.Profile()
        pr.enable()
        try:
            yield
        finally:
            pr.disable()
            stream = io.StringIO()
            ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
            ps.print_stats(20)
            self._profiles[name] = stream.getvalue()

    @asynccontextmanager
    async def async_profile(self, name: str) -> AsyncIterator[None]:
        pr = cProfile.Profile()
        pr.enable()
        try:
            yield
        finally:
            pr.disable()
            stream = io.StringIO()
            ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
            ps.print_stats(20)
            self._profiles[name] = stream.getvalue()

    def get_report(self, name: str) -> Optional[str]:
        return self._profiles.get(name)

    def all_reports(self) -> Dict[str, str]:
        return dict(self._profiles)


class _AlertManager:
    """Alert lifecycle management."""

    def __init__(self, logger: _Logger) -> None:
        self._logger = logger
        self._alerts: Dict[str, Alert] = {}
        self._handlers: List[Callable[[Alert], Any]] = []

    def add_handler(self, handler: Callable[[Alert], Any]) -> None:
        self._handlers.append(handler)

    async def fire(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        labels: Optional[Labels] = None,
    ) -> Alert:
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            message=message,
            labels=labels or {},
        )
        self._alerts[alert.alert_id] = alert
        self._logger.log(severity.value, f"[ALERT] {title}: {message}")
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception:
                pass
        return alert

    def resolve(self, alert_id: str) -> bool:
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            return True
        return False

    def active(self) -> List[Alert]:
        return [a for a in self._alerts.values() if not a.resolved]

    def all_alerts(self) -> List[Alert]:
        return list(self._alerts.values())


class _Dashboard:
    """Read-only dashboard aggregating all subsystem data."""

    def __init__(
        self,
        metrics: _MetricsStore,
        alerts: _AlertManager,
        health: "_HealthMonitor",
        traces: "_TraceManager",
    ) -> None:
        self._metrics = metrics
        self._alerts = alerts
        self._health = health
        self._traces = traces

    def snapshot(self) -> DashboardSnapshot:
        return DashboardSnapshot(
            timestamp=time.time(),
            metrics=self._metrics.snapshot(),
            active_alerts=self._alerts.active(),
            health_reports=self._health.all_reports(),
            recent_spans=self._traces.recent(limit=50),
        )

    def summary(self) -> Dict[str, Any]:
        snap = self.snapshot()
        return {
            "timestamp": snap.timestamp,
            "metric_series": len(snap.metrics),
            "active_alerts": len(snap.active_alerts),
            "health_checks": len(snap.health_reports),
            "recent_spans": len(snap.recent_spans),
        }


class _HealthMonitor:
    """Component health tracking and reporting."""

    def __init__(self) -> None:
        self._reports: Dict[str, HealthReport] = {}
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        self._checks[name] = check_fn

    async def run_check(self, component: str) -> HealthReport:
        results: Dict[str, bool] = {}
        for name, fn in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(fn):
                    results[name] = await fn()
                else:
                    results[name] = fn()
            except Exception:
                results[name] = False

        failed = [k for k, v in results.items() if not v]
        if not results:
            status = HealthStatus.UNKNOWN
        elif not failed:
            status = HealthStatus.HEALTHY
        elif len(failed) < len(results):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        report = HealthReport(
            status=status,
            component=component,
            checks=results,
            details={"failed_checks": failed},
        )
        self._reports[component] = report
        return report

    def all_reports(self) -> List[HealthReport]:
        return list(self._reports.values())

    def overall_status(self) -> HealthStatus:
        reports = self.all_reports()
        if not reports:
            return HealthStatus.UNKNOWN
        statuses = {r.status for r in reports}
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(r.status == HealthStatus.HEALTHY for r in reports):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN


class _TraceManager:
    """Lightweight span-based tracing with OTel integration."""

    def __init__(self, use_otel: bool = False) -> None:
        self._spans: List[TraceSpan] = []
        self._active: Dict[str, TraceSpan] = {}
        self._otel_tracer: Optional[Any] = None
        self._otel_spans: Dict[str, Any] = {}
        if use_otel and _OTEL_AVAILABLE:
            self._otel_tracer = otel_trace.get_tracer("nova.monitoring")

    def start_span(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
            parent_id=parent_id,
        )
        self._active[span.span_id] = span
        if self._otel_tracer:
            otel_span = self._otel_tracer.start_span(name)
            self._otel_spans[span.span_id] = otel_span
        return span

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        span = self._active.pop(span_id, None)
        if span is None:
            return None
        span.end_time = time.time()
        span.status = status
        if attributes:
            span.attributes.update(attributes)
        self._spans.append(span)
        if span_id in self._otel_spans:
            otel_span = self._otel_spans.pop(span_id)
            if status != "OK" and _OTEL_AVAILABLE:
                otel_span.set_status(StatusCode.ERROR)
            otel_span.end()
        return span

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[TraceSpan]:
        s = self.start_span(name, attributes=attributes)
        try:
            yield s
        except Exception as exc:
            self.end_span(s.span_id, status=f"ERROR: {exc}")
            raise
        else:
            self.end_span(s.span_id)

    @asynccontextmanager
    async def async_span(self, name: str, **attributes: Any) -> AsyncIterator[TraceSpan]:
        s = self.start_span(name, attributes=attributes)
        try:
            yield s
        except Exception as exc:
            self.end_span(s.span_id, status=f"ERROR: {exc}")
            raise
        else:
            self.end_span(s.span_id)

    def recent(self, limit: int = 50) -> List[TraceSpan]:
        return self._spans[-limit:]


class _OTelBridge:
    """OpenTelemetry SDK bootstrap – traces + metrics."""

    def __init__(self, service_name: str = "nova") -> None:
        self._service_name = service_name
        self._tracer_provider: Optional[Any] = None
        self._meter_provider: Optional[Any] = None
        self._meter: Optional[Any] = None

    def initialize(self) -> None:
        if not _OTEL_AVAILABLE:
            return
        resource = Resource.create({"service.name": self._service_name})

        # Tracer
        tp = TracerProvider(resource=resource)
        tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        otel_trace.set_tracer_provider(tp)
        self._tracer_provider = tp

        # Meter
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        mp = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(mp)
        self._meter_provider = mp
        self._meter = otel_metrics.get_meter(self._service_name)

    def record_counter(self, name: str, value: int = 1, labels: Optional[Labels] = None) -> None:
        if self._meter is None:
            return
        counter = self._meter.create_counter(name)
        counter.add(value, labels or {})

    def record_gauge(self, name: str, value: float, labels: Optional[Labels] = None) -> None:
        if self._meter is None:
            return
        self._meter.create_observable_gauge(
            name,
            callbacks=[lambda _: [(value, labels or {})]],
        )

    def shutdown(self) -> None:
        if self._tracer_provider and _OTEL_AVAILABLE:
            self._tracer_provider.shutdown()
        if self._meter_provider and _OTEL_AVAILABLE:
            self._meter_provider.shutdown()


# ---------------------------------------------------------------------------
# MonitoringConnector – singleton façade
# ---------------------------------------------------------------------------


class MonitoringConnector:
    """
    Singleton façade connecting all monitoring subsystem components.

    Usage
    -----
    connector = MonitoringConnector.instance()
    await connector.initialize()
    await connector.record_metric("cpu_usage", 42.5, labels={"host": "node-1"})
    connector.log("INFO", "System started")
    async with connector.trace("request_handler") as span:
        ...
    report = await connector.health_check()
    await connector.shutdown()
    """

    _instance: Optional["MonitoringConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, service_name: str = "nova") -> None:
        self._service_name = service_name
        self._initialized: bool = False

        # Sub-components
        self.logger = _Logger(name=f"{service_name}.monitoring")
        self.metrics = _MetricsStore()
        self.telemetry = _Telemetry()
        self.profiler = _Profiler()
        self.traces = _TraceManager(use_otel=_OTEL_AVAILABLE)
        self.alerts = _AlertManager(self.logger)
        self.health = _HealthMonitor()
        self.otel = _OTelBridge(service_name=service_name)
        self.dashboard = _Dashboard(self.metrics, self.alerts, self.health, self.traces)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    async def instance(cls, service_name: str = "nova") -> "MonitoringConnector":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(service_name=service_name)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["MonitoringConnector"]:
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(
        self,
        *,
        alert_handlers: Optional[List[Callable[[Alert], Any]]] = None,
        health_checks: Optional[Dict[str, Callable[[], bool]]] = None,
        telemetry_subscribers: Optional[Dict[str, Callable[..., Any]]] = None,
        otel_enabled: bool = True,
    ) -> None:
        if self._initialized:
            self.logger.warning("MonitoringConnector already initialized; skipping.")
            return

        # OTel
        if otel_enabled:
            self.otel.initialize()

        # Alert handlers
        for handler in alert_handlers or []:
            self.alerts.add_handler(handler)

        # Health checks
        for name, fn in (health_checks or {}).items():
            self.health.register_check(name, fn)

        # Telemetry subscribers
        for event, handler in (telemetry_subscribers or {}).items():
            self.telemetry.subscribe(event, handler)

        # Default health check: self-check
        self.health.register_check(
            "monitoring_connector",
            lambda: self._initialized or True,
        )

        self._initialized = True
        await self.telemetry.emit("monitoring.initialized", {"service": self._service_name})
        self.logger.info("MonitoringConnector initialized", service=self._service_name)

    async def shutdown(self) -> None:
        if not self._initialized:
            return
        await self.telemetry.emit("monitoring.shutdown", {"service": self._service_name})
        self.otel.shutdown()
        self._initialized = False
        self.logger.info("MonitoringConnector shut down", service=self._service_name)
        MonitoringConnector._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def record_metric(
        self,
        name: str,
        value: MetricValue,
        *,
        labels: Optional[Labels] = None,
        unit: str = "",
    ) -> None:
        point = MetricPoint(name=name, value=value, labels=labels or {}, unit=unit)
        await self.metrics.record(point)
        await self.telemetry.emit("metric.recorded", {"name": name, "value": value})

    def log(
        self,
        level: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.log(level, message, context=context)

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        **attributes: Any,
    ) -> AsyncIterator[TraceSpan]:
        async with self.traces.async_span(name, **attributes) as span:
            yield span
            await self.telemetry.emit(
                "trace.completed",
                {"name": name, "duration_ms": span.duration_ms},
            )

    async def health_check(
        self,
        component: str = "nova",
    ) -> HealthReport:
        report = await self.health.run_check(component)
        await self.telemetry.emit(
            "health.checked",
            {"component": component, "status": report.status.value},
        )
        if report.status == HealthStatus.UNHEALTHY:
            await self.alerts.fire(
                title=f"Health check failed: {component}",
                message=f"Failed checks: {report.details.get('failed_checks', [])}",
                severity=AlertSeverity.CRITICAL,
                labels={"component": component},
            )
        return report

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def fire_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        labels: Optional[Labels] = None,
    ) -> Alert:
        return await self.alerts.fire(title, message, severity=severity, labels=labels)

    def aggregate_metric(
        self,
        name: str,
        method: str = "last",
        window_seconds: float = 60.0,
    ) -> Optional[MetricValue]:
        return self.metrics.aggregate(name, method=method, window_seconds=window_seconds)

    def get_dashboard(self) -> DashboardSnapshot:
        return self.dashboard.snapshot()

    def get_summary(self) -> Dict[str, Any]:
        return self.dashboard.summary()

    def profile_context(self, name: str) -> Iterator[None]:
        return self.profiler.profile(name)  # type: ignore[return-value]

    def async_profile_context(self, name: str) -> AsyncIterator[None]:
        return self.profiler.async_profile(name)  # type: ignore[return-value]

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def overall_health(self) -> HealthStatus:
        return self.health.overall_status()

    # ------------------------------------------------------------------
    # middleware_loader.py compatibility
    # ------------------------------------------------------------------

    @classmethod
    def from_middleware_loader(
        cls,
        config: Optional[Dict[str, Any]] = None,
    ) -> "MonitoringConnector":
        """
        Factory used by middleware_loader.py.

        Expected config keys (all optional)
        ------------------------------------
        service_name : str
        otel_enabled : bool
        alert_handlers : list[callable]
        health_checks : dict[str, callable]
        """
        cfg = config or {}
        connector = cls(service_name=cfg.get("service_name", "nova"))
        return connector

    def get_component(self, name: str) -> Optional[Any]:
        """
        Return a named sub-component.
        Allows middleware_loader.py to access internals by name.
        """
        mapping: Dict[str, Any] = {
            "logger": self.logger,
            "metrics": self.metrics,
            "telemetry": self.telemetry,
            "profiler": self.profiler,
            "traces": self.traces,
            "alerts": self.alerts,
            "health": self.health,
            "otel": self.otel,
            "dashboard": self.dashboard,
        }
        return mapping.get(name)

    def __repr__(self) -> str:
        return (
            f"MonitoringConnector("
            f"service={self._service_name!r}, "
            f"initialized={self._initialized}, "
            f"health={self.overall_health.value})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def get_connector(service_name: str = "nova") -> MonitoringConnector:
    """Retrieve or create the singleton connector."""
    return await MonitoringConnector.instance(service_name=service_name)


async def quick_setup(
    service_name: str = "nova",
    otel_enabled: bool = True,
    **init_kwargs: Any,
) -> MonitoringConnector:
    """One-call convenience for middleware_loader.py or tests."""
    connector = await get_connector(service_name=service_name)
    if not connector.is_initialized:
        await connector.initialize(otel_enabled=otel_enabled, **init_kwargs)
    return connector
