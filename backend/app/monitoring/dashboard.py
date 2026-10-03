from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class DashboardError(Exception):
    pass


class SourceNotFoundError(DashboardError):
    pass


class OverallStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


MetricsSourceFn = Callable[[], Union[dict[str, Any], Awaitable[dict[str, Any]]]]
HealthSourceFn = Callable[[], Union[Any, Awaitable[Any]]]
CapabilitySourceFn = Callable[[], Union[dict[str, Any], Awaitable[dict[str, Any]]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CapabilityOverview:
    name: str
    status: str
    invocation_count: int = 0
    error_count: int = 0
    avg_latency_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "invocation_count": self.invocation_count,
            "error_count": self.error_count,
            "avg_latency_seconds": self.avg_latency_seconds,
            "metadata": self.metadata,
        }


@dataclass
class DashboardSnapshot:
    generated_at: float
    overall_status: OverallStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    capabilities: list[CapabilityOverview] = field(default_factory=list)
    system: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status.value,
            "metrics": self.metrics,
            "health": self.health,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "system": self.system,
            "errors": self.errors,
        }


def _normalize_health(raw: Any) -> tuple[bool, dict[str, Any]]:
    if raw is None:
        return False, {"error": "health source returned None"}
    if hasattr(raw, "healthy"):
        details = dict(getattr(raw, "details", {}) or {})
        return bool(raw.healthy), details
    if isinstance(raw, dict):
        healthy = bool(raw.get("healthy", False))
        return healthy, {k: v for k, v in raw.items() if k != "healthy"}
    if isinstance(raw, bool):
        return raw, {}
    return False, {"error": f"unrecognized health result type: {type(raw).__name__}"}


class Dashboard:
    """Aggregates metrics, health, and capability data into unified status reports with export support."""

    def __init__(self, *, name: str = "dashboard", max_snapshot_history: int = 500) -> None:
        self.name = name
        self.max_snapshot_history = max_snapshot_history
        self._metrics_sources: dict[str, MetricsSourceFn] = {}
        self._health_sources: dict[str, HealthSourceFn] = {}
        self._capability_sources: dict[str, CapabilitySourceFn] = {}
        self._system_provider: Optional[Callable[[], Union[dict[str, Any], Awaitable[dict[str, Any]]]]] = None
        self._snapshot_history: list[DashboardSnapshot] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._generate_count = 0

    def register_metrics_source(self, name: str, source_fn: MetricsSourceFn) -> None:
        self._metrics_sources[name] = source_fn

    def register_health_source(self, name: str, source_fn: HealthSourceFn) -> None:
        self._health_sources[name] = source_fn

    def register_capability_source(self, name: str, source_fn: CapabilitySourceFn) -> None:
        self._capability_sources[name] = source_fn

    def set_system_provider(self, provider: Callable[[], Union[dict[str, Any], Awaitable[dict[str, Any]]]]) -> None:
        self._system_provider = provider

    def unregister_metrics_source(self, name: str) -> bool:
        return self._metrics_sources.pop(name, None) is not None

    def unregister_health_source(self, name: str) -> bool:
        return self._health_sources.pop(name, None) is not None

    def unregister_capability_source(self, name: str) -> bool:
        return self._capability_sources.pop(name, None) is not None

    def _resolve_sync(self, fn: Callable[[], Any]) -> Any:
        result = fn()
        if asyncio.iscoroutine(result):
            raise DashboardError("async source function used with synchronous generate(); use generate_async() instead")
        return result

    def generate(self) -> DashboardSnapshot:
        self._generate_count += 1
        errors: dict[str, str] = {}

        metrics: dict[str, Any] = {}
        for name, source_fn in self._metrics_sources.items():
            try:
                metrics[name] = self._resolve_sync(source_fn)
            except Exception as exc:
                errors[f"metrics:{name}"] = str(exc)

        health: dict[str, Any] = {}
        healthy_flags: list[bool] = []
        for name, source_fn in self._health_sources.items():
            try:
                raw = self._resolve_sync(source_fn)
                ok, details = _normalize_health(raw)
                health[name] = {"healthy": ok, "details": details}
                healthy_flags.append(ok)
            except Exception as exc:
                errors[f"health:{name}"] = str(exc)
                health[name] = {"healthy": False, "details": {"error": str(exc)}}
                healthy_flags.append(False)

        capabilities: list[CapabilityOverview] = []
        for name, source_fn in self._capability_sources.items():
            try:
                raw = self._resolve_sync(source_fn)
                capabilities.append(CapabilityOverview(
                    name=name,
                    status=str(raw.get("status", "unknown")),
                    invocation_count=int(raw.get("invocation_count", 0)),
                    error_count=int(raw.get("error_count", 0)),
                    avg_latency_seconds=raw.get("avg_latency_seconds"),
                    metadata={k: v for k, v in raw.items() if k not in {"status", "invocation_count", "error_count", "avg_latency_seconds"}},
                ))
            except Exception as exc:
                errors[f"capability:{name}"] = str(exc)
                capabilities.append(CapabilityOverview(name=name, status="error", metadata={"error": str(exc)}))

        system: dict[str, Any] = {}
        if self._system_provider is not None:
            try:
                system = self._resolve_sync(self._system_provider)
            except Exception as exc:
                errors["system"] = str(exc)

        overall = self._compute_overall(healthy_flags, errors)
        snapshot = DashboardSnapshot(
            generated_at=time.time(), overall_status=overall, metrics=metrics,
            health=health, capabilities=capabilities, system=system, errors=errors,
        )
        self._archive(snapshot)
        return snapshot

    async def generate_async(self) -> DashboardSnapshot:
        self._generate_count += 1
        errors: dict[str, str] = {}

        async def _resolve(fn: Callable[[], Any]) -> Any:
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result

        metrics: dict[str, Any] = {}
        for name, source_fn in self._metrics_sources.items():
            try:
                metrics[name] = await _resolve(source_fn)
            except Exception as exc:
                errors[f"metrics:{name}"] = str(exc)

        health: dict[str, Any] = {}
        healthy_flags: list[bool] = []
        for name, source_fn in self._health_sources.items():
            try:
                raw = await _resolve(source_fn)
                ok, details = _normalize_health(raw)
                health[name] = {"healthy": ok, "details": details}
                healthy_flags.append(ok)
            except Exception as exc:
                errors[f"health:{name}"] = str(exc)
                health[name] = {"healthy": False, "details": {"error": str(exc)}}
                healthy_flags.append(False)

        capabilities: list[CapabilityOverview] = []
        for name, source_fn in self._capability_sources.items():
            try:
                raw = await _resolve(source_fn)
                capabilities.append(CapabilityOverview(
                    name=name,
                    status=str(raw.get("status", "unknown")),
                    invocation_count=int(raw.get("invocation_count", 0)),
                    error_count=int(raw.get("error_count", 0)),
                    avg_latency_seconds=raw.get("avg_latency_seconds"),
                    metadata={k: v for k, v in raw.items() if k not in {"status", "invocation_count", "error_count", "avg_latency_seconds"}},
                ))
            except Exception as exc:
                errors[f"capability:{name}"] = str(exc)
                capabilities.append(CapabilityOverview(name=name, status="error", metadata={"error": str(exc)}))

        system: dict[str, Any] = {}
        if self._system_provider is not None:
            try:
                system = await _resolve(self._system_provider)
            except Exception as exc:
                errors["system"] = str(exc)

        overall = self._compute_overall(healthy_flags, errors)
        snapshot = DashboardSnapshot(
            generated_at=time.time(), overall_status=overall, metrics=metrics,
            health=health, capabilities=capabilities, system=system, errors=errors,
        )
        async with self._lock:
            self._archive(snapshot)
        return snapshot

    @staticmethod
    def _compute_overall(healthy_flags: list[bool], errors: dict[str, str]) -> OverallStatus:
        if not healthy_flags and not errors:
            return OverallStatus.UNKNOWN
        if errors and not healthy_flags:
            return OverallStatus.UNHEALTHY
        if all(healthy_flags) and not errors:
            return OverallStatus.HEALTHY
        if not any(healthy_flags) and healthy_flags:
            return OverallStatus.UNHEALTHY
        return OverallStatus.DEGRADED

    def _archive(self, snapshot: DashboardSnapshot) -> None:
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self.max_snapshot_history:
            self._snapshot_history = self._snapshot_history[-self.max_snapshot_history :]

    def report(self, *, snapshot: Optional[DashboardSnapshot] = None) -> dict[str, Any]:
        snap = snapshot or (self._snapshot_history[-1] if self._snapshot_history else self.generate())
        return snap.to_dict()

    def history(self, limit: int = 50) -> list[DashboardSnapshot]:
        return self._snapshot_history[-limit:]

    def export(self, *, fmt: str = "json", snapshot: Optional[DashboardSnapshot] = None) -> str:
        snap = snapshot or (self._snapshot_history[-1] if self._snapshot_history else self.generate())
        data = snap.to_dict()
        if fmt == "json":
            return json.dumps(data, default=str, sort_keys=True)
        if fmt == "summary":
            lines = [
                f"Dashboard: {self.name}",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snap.generated_at))}",
                f"Overall status: {snap.overall_status.value}",
                f"Health sources: {len(snap.health)} (healthy={sum(1 for h in snap.health.values() if h.get('healthy'))})",
                f"Capabilities: {len(snap.capabilities)}",
                f"Metrics sources: {len(snap.metrics)}",
                f"Errors: {len(snap.errors)}",
            ]
            return "\n".join(lines)
        raise DashboardError(f"unsupported export format: {fmt}")

    async def export_async(self, *, fmt: str = "json", snapshot: Optional[DashboardSnapshot] = None) -> str:
        snap = snapshot or (self._snapshot_history[-1] if self._snapshot_history else await self.generate_async())
        return self.export(fmt=fmt, snapshot=snap)

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "name": self.name,
                "metrics_source_count": len(self._metrics_sources),
                "health_source_count": len(self._health_sources),
                "capability_source_count": len(self._capability_sources),
                "system_provider_registered": self._system_provider is not None,
                "snapshot_history_count": len(self._snapshot_history),
                "generate_count": self._generate_count,
            }
            return HealthStatus(healthy=True, component="dashboard", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="dashboard", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Dashboard",
    "DashboardSnapshot",
    "CapabilityOverview",
    "OverallStatus",
    "DashboardError",
    "SourceNotFoundError",
    "HealthStatus",
]
