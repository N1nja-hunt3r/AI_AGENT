from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class RegistryError(Exception):
    pass


class MonitorNotFoundError(RegistryError):
    pass


class DuplicateMonitorError(RegistryError):
    pass


@runtime_checkable
class HealthCheckable(Protocol):
    def health_check(self) -> Any: ...


class MonitorStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass
class MonitorVersion:
    major: int
    minor: int
    patch: int

    @staticmethod
    def parse(version_str: str) -> "MonitorVersion":
        try:
            parts = version_str.strip().lstrip("v").split(".")
            major, minor, patch = (int(p) for p in (parts + ["0", "0", "0"])[:3])
            return MonitorVersion(major, minor, patch)
        except (ValueError, IndexError) as exc:
            raise RegistryError(f"invalid version string '{version_str}'") from exc

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "MonitorVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MonitorVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __le__(self, other: "MonitorVersion") -> bool:
        return self < other or self == other


@dataclass
class MonitorRecord:
    name: str
    instance: Any
    version: MonitorVersion
    priority: int = 100
    status: MonitorStatus = MonitorStatus.REGISTERED
    registered_at: float = field(default_factory=time.time)
    tags: frozenset[str] = field(default_factory=frozenset)
    last_health_check: Optional[Any] = None
    last_health_check_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MonitorRegistry:
    """Central registry for observability/monitoring modules: register, discover, prioritize, version, health-check."""

    def __init__(self) -> None:
        self._monitors: dict[str, MonitorRecord] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._discovery_count = 0

    def register(
        self,
        name: str,
        instance: Any,
        *,
        version: str = "1.0.0",
        priority: int = 100,
        tags: Optional[frozenset[str]] = None,
        replace: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MonitorRecord:
        if name in self._monitors and not replace:
            existing = self._monitors[name]
            new_version = MonitorVersion.parse(version)
            if new_version <= existing.version:
                raise DuplicateMonitorError(
                    f"monitor '{name}' already registered at version {existing.version}; "
                    f"use replace=True or a higher version than {new_version}"
                )

        record = MonitorRecord(
            name=name,
            instance=instance,
            version=MonitorVersion.parse(version),
            priority=priority,
            status=MonitorStatus.ACTIVE,
            tags=tags or frozenset(),
            metadata=metadata or {},
        )
        self._monitors[name] = record
        return record

    def unregister(self, name: str) -> bool:
        return self._monitors.pop(name, None) is not None

    def get(self, name: str) -> Any:
        record = self._monitors.get(name)
        if record is None:
            raise MonitorNotFoundError(f"no monitor registered under name '{name}'")
        if record.status == MonitorStatus.DISABLED:
            raise RegistryError(f"monitor '{name}' is disabled")
        return record.instance

    def get_record(self, name: str) -> Optional[MonitorRecord]:
        return self._monitors.get(name)

    def enable(self, name: str) -> None:
        record = self._monitors.get(name)
        if record is None:
            raise MonitorNotFoundError(f"no monitor registered under name '{name}'")
        record.status = MonitorStatus.ACTIVE

    def disable(self, name: str) -> None:
        record = self._monitors.get(name)
        if record is None:
            raise MonitorNotFoundError(f"no monitor registered under name '{name}'")
        record.status = MonitorStatus.DISABLED

    def list_monitors(
        self,
        *,
        tag: Optional[str] = None,
        status: Optional[MonitorStatus] = None,
        sort_by_priority: bool = True,
    ) -> list[MonitorRecord]:
        records = list(self._monitors.values())
        if tag is not None:
            records = [r for r in records if tag in r.tags]
        if status is not None:
            records = [r for r in records if r.status == status]
        if sort_by_priority:
            records.sort(key=lambda r: r.priority)
        return records

    def discover(
        self,
        monitor_specs: list[tuple[str, str, str]],
        *,
        priority_start: int = 100,
        priority_step: int = 10,
    ) -> list[MonitorRecord]:
        """
        Discover and register monitors dynamically.
        monitor_specs: list of (registry_name, python_module_path, class_name) tuples.
        """
        records: list[MonitorRecord] = []
        priority = priority_start
        for registry_name, module_path, class_name in monitor_specs:
            self._discovery_count += 1
            try:
                py_module = importlib.import_module(module_path)
                cls = getattr(py_module, class_name)
                instance = cls() if inspect.isclass(cls) else cls
                version = getattr(py_module, "__version__", "1.0.0")
                record = self.register(
                    registry_name, instance, version=version, priority=priority, replace=True,
                    tags=frozenset({"auto-discovered"}),
                )
                records.append(record)
            except Exception as exc:
                failed_record = MonitorRecord(
                    name=registry_name, instance=None, version=MonitorVersion(0, 0, 0),
                    priority=priority, status=MonitorStatus.FAILED, metadata={"error": str(exc)},
                )
                self._monitors[registry_name] = failed_record
                records.append(failed_record)
            priority += priority_step
        return records

    def health_status(self, name: str) -> HealthStatus:
        record = self._monitors.get(name)
        if record is None:
            return HealthStatus(healthy=False, component=name, details={"error": "monitor not registered"})
        if record.instance is None:
            return HealthStatus(healthy=False, component=name, details={"error": "monitor failed to load", **record.metadata})

        try:
            if hasattr(record.instance, "health_check") and callable(record.instance.health_check):
                result = record.instance.health_check()
                record.last_health_check = result
                record.last_health_check_at = time.time()
                healthy = getattr(result, "healthy", bool(result))
                details = getattr(result, "details", {}) if not isinstance(result, dict) else result
                return HealthStatus(healthy=healthy, component=name, details=dict(details) if details else {})
            return HealthStatus(healthy=True, component=name, details={"note": "monitor has no health_check method"})
        except Exception as exc:
            return HealthStatus(healthy=False, component=name, details={"error": str(exc)})

    def health_check(self) -> HealthStatus:
        try:
            monitor_results = {name: self.health_status(name) for name in self._monitors}
            all_healthy = all(r.healthy for r in monitor_results.values())
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_monitor_count": len(self._monitors),
                "active_monitor_count": sum(1 for r in self._monitors.values() if r.status == MonitorStatus.ACTIVE),
                "disabled_monitor_count": sum(1 for r in self._monitors.values() if r.status == MonitorStatus.DISABLED),
                "failed_monitor_count": sum(1 for r in self._monitors.values() if r.status == MonitorStatus.FAILED),
                "discovery_attempts": self._discovery_count,
                "monitors": {
                    name: {"healthy": res.healthy, "details": res.details}
                    for name, res in monitor_results.items()
                },
            }
            return HealthStatus(healthy=all_healthy, component="registry", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="registry", details={"error": str(exc)})

    async def register_async(self, name: str, instance: Any, **kwargs: Any) -> MonitorRecord:
        async with self._lock:
            return self.register(name, instance, **kwargs)

    async def unregister_async(self, name: str) -> bool:
        async with self._lock:
            return self.unregister(name)

    async def discover_async(self, monitor_specs: list[tuple[str, str, str]], **kwargs: Any) -> list[MonitorRecord]:
        async with self._lock:
            return await asyncio.to_thread(self.discover, monitor_specs, **kwargs)

    async def health_status_async(self, name: str) -> HealthStatus:
        record = self._monitors.get(name)
        if record is None:
            return HealthStatus(healthy=False, component=name, details={"error": "monitor not registered"})
        if record.instance is None:
            return HealthStatus(healthy=False, component=name, details={"error": "monitor failed to load", **record.metadata})

        try:
            if hasattr(record.instance, "health_check_async") and callable(record.instance.health_check_async):
                result = await record.instance.health_check_async()
            elif hasattr(record.instance, "health_check") and callable(record.instance.health_check):
                result = await asyncio.to_thread(record.instance.health_check)
            else:
                return HealthStatus(healthy=True, component=name, details={"note": "monitor has no health_check method"})

            record.last_health_check = result
            record.last_health_check_at = time.time()
            healthy = getattr(result, "healthy", bool(result))
            details = getattr(result, "details", {}) if not isinstance(result, dict) else result
            return HealthStatus(healthy=healthy, component=name, details=dict(details) if details else {})
        except Exception as exc:
            return HealthStatus(healthy=False, component=name, details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            names = list(self._monitors.keys())
            results = await asyncio.gather(*(self.health_status_async(name) for name in names))
            monitor_results = dict(zip(names, results))
            all_healthy = all(r.healthy for r in monitor_results.values())
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_monitor_count": len(self._monitors),
                "monitors": {name: {"healthy": res.healthy, "details": res.details} for name, res in monitor_results.items()},
            }
            return HealthStatus(healthy=all_healthy, component="registry", details=details)


__all__ = [
    "MonitorRegistry",
    "MonitorRecord",
    "MonitorStatus",
    "MonitorVersion",
    "HealthCheckable",
    "RegistryError",
    "MonitorNotFoundError",
    "DuplicateMonitorError",
    "HealthStatus",
]
