"""
monitoring_loader.py

Initializes and registers all monitoring subsystem components: logger,
metrics, telemetry, traces, profiler, health_monitor, alerts and
dashboard.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus, Registry, get_registry

logger = logging.getLogger("monitoring_loader")

_COMPONENT_NAMES: List[str] = [
    "logger",
    "metrics",
    "telemetry",
    "traces",
    "profiler",
    "health_monitor",
    "alerts",
    "dashboard",
]

_CLASS_MAP: Dict[str, str] = {
    "logger": "StructuredLogger",
    "metrics": "MetricsCollector",
    "telemetry": "TelemetryService",
    "traces": "TraceManager",
    "profiler": "Profiler",
    "health_monitor": "HealthMonitor",
    "alerts": "AlertManager",
    "dashboard": "DashboardService",
}


class _StubMonitoringComponent:
    """Fallback no-op implementation used when a real module is unavailable."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._initialized = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return f"<StubMonitoringComponent name={self.name!r}>"


def _import_component(module_name: str, class_name: str) -> Any:
    try:
        module = __import__(f"app.monitoring.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        logger.warning(
            "monitoring.%s.%s not found; using stub fallback.", module_name, class_name
        )
        return None


@dataclass
class MonitoringLoaderConfig:
    enabled: bool = True
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    component_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fail_fast: bool = False


class MonitoringLoader:
    """Loads, wires and supervises all monitoring components."""

    def __init__(
        self,
        container: Optional[DependencyContainer] = None,
        config: Optional[MonitoringLoaderConfig] = None,
    ) -> None:
        self._container = container or get_container()
        self._registry: Registry = get_registry("monitoring")
        self._config = config or MonitoringLoaderConfig()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._metrics: Dict[str, int] = {"init_success": 0, "init_failure": 0}

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            if not self._config.enabled:
                logger.info("Monitoring subsystem disabled via configuration.")
                return

            await self.register_components()

            for name in self._registry.list():
                if self._config.feature_flags.get(name, True) is False:
                    logger.info("Skipping disabled monitoring component '%s'.", name)
                    continue
                component = self._registry.get(name)
                try:
                    cfg = self._config.component_config.get(name, {})
                    if hasattr(component, "initialize"):
                        await component.initialize(cfg)
                    self._metrics["init_success"] += 1
                    logger.info("Initialized monitoring component '%s'.", name)
                except Exception:
                    self._metrics["init_failure"] += 1
                    logger.exception("Failed to initialize monitoring component '%s'.", name)
                    if self._config.fail_fast:
                        raise

            self._initialized = True

    async def register_components(self) -> None:
        for name in _COMPONENT_NAMES:
            if self._registry.has(name):
                continue
            cls = _import_component(name, _CLASS_MAP[name])
            instance = cls() if cls is not None else _StubMonitoringComponent(name)
            self._registry.register(
                name,
                instance,
                version="1.0.0",
                metadata={"kind": "monitoring"},
                health_check=getattr(instance, "health_check", None),
            )
            self._container.register_singleton(f"monitoring:{name}", instance)

    def get_component(self, name: str) -> Any:
        return self._registry.get(name)

    def list_components(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    async def shutdown(self) -> None:
        async with self._lock:
            for name in self._registry.list():
                component = self._registry.get(name)
                try:
                    if hasattr(component, "shutdown"):
                        await component.shutdown()
                except Exception:
                    logger.exception("Error shutting down monitoring component '%s'.", name)
            self._initialized = False

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            metrics_component = self._registry.get("metrics")
            if hasattr(metrics_component, "record"):
                metrics_component.record(name, value, tags or {})
        except Exception:
            logger.debug("Metrics component unavailable; dropping metric '%s'.", name)

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_loader_lock = asyncio.Lock()
_monitoring_loader: Optional[MonitoringLoader] = None


async def get_monitoring_loader() -> MonitoringLoader:
    global _monitoring_loader
    async with _loader_lock:
        if _monitoring_loader is None:
            _monitoring_loader = MonitoringLoader()
        return _monitoring_loader
