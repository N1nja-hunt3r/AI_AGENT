from __future__ import annotations

import asyncio
import importlib
import inspect
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class RegistryError(Exception):
    pass


class ModuleNotFoundError_(RegistryError):
    pass


class DuplicateModuleError(RegistryError):
    pass


class CircularDependencyError(RegistryError):
    pass


class MissingDependencyError(RegistryError):
    pass


@runtime_checkable
class HealthCheckable(Protocol):
    def health_check(self) -> Any: ...


class ModuleStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass
class ModuleVersion:
    major: int
    minor: int
    patch: int

    @staticmethod
    def parse(version_str: str) -> "ModuleVersion":
        try:
            parts = version_str.strip().lstrip("v").split(".")
            major, minor, patch = (int(p) for p in (parts + ["0", "0", "0"])[:3])
            return ModuleVersion(major, minor, patch)
        except (ValueError, IndexError) as exc:
            raise RegistryError(f"invalid version string '{version_str}'") from exc

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "ModuleVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuleVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __le__(self, other: "ModuleVersion") -> bool:
        return self < other or self == other


@dataclass
class ModuleRecord:
    name: str
    instance: Any
    version: ModuleVersion
    priority: int = 100
    status: ModuleStatus = ModuleStatus.REGISTERED
    dependencies: frozenset[str] = field(default_factory=frozenset)
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


class AutomationModuleRegistry:
    """Thread-safe central registry for automation modules: register, discover, prioritize, version, resolve dependencies, health-check."""

    def __init__(self) -> None:
        self._modules: dict[str, ModuleRecord] = {}
        self._thread_lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._created_at = time.time()
        self._discovery_count = 0

    def register(
        self,
        name: str,
        instance: Any,
        *,
        version: str = "1.0.0",
        priority: int = 100,
        dependencies: Optional[frozenset[str]] = None,
        tags: Optional[frozenset[str]] = None,
        replace: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ModuleRecord:
        with self._thread_lock:
            deps = dependencies or frozenset()

            if name in self._modules and not replace:
                existing = self._modules[name]
                new_version = ModuleVersion.parse(version)
                if new_version <= existing.version:
                    raise DuplicateModuleError(
                        f"module '{name}' already registered at version {existing.version}; "
                        f"use replace=True or a higher version than {new_version}"
                    )

            for dep in deps:
                if dep not in self._modules:
                    raise MissingDependencyError(f"module '{name}' depends on unregistered module '{dep}'")

            record = ModuleRecord(
                name=name, instance=instance, version=ModuleVersion.parse(version),
                priority=priority, status=ModuleStatus.ACTIVE, dependencies=deps,
                tags=tags or frozenset(), metadata=metadata or {},
            )
            self._modules[name] = record

            if self._has_cycle():
                del self._modules[name]
                raise CircularDependencyError(f"registering module '{name}' would create a circular dependency")

            return record

    def _has_cycle(self) -> bool:
        visited: set[str] = set()
        stack: set[str] = set()

        def visit(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            record = self._modules.get(node)
            if record is not None:
                for dep in record.dependencies:
                    if dep in self._modules and visit(dep):
                        return True
            stack.discard(node)
            return False

        return any(visit(name) for name in list(self._modules.keys()))

    def unregister(self, name: str) -> bool:
        with self._thread_lock:
            dependents = [
                rec.name for rec in self._modules.values()
                if name in rec.dependencies and rec.name != name
            ]
            if dependents:
                raise RegistryError(f"cannot unregister '{name}': required by {dependents}")
            return self._modules.pop(name, None) is not None

    def get(self, name: str) -> Any:
        with self._thread_lock:
            record = self._modules.get(name)
            if record is None:
                raise ModuleNotFoundError_(f"no module registered under name '{name}'")
            if record.status == ModuleStatus.DISABLED:
                raise RegistryError(f"module '{name}' is disabled")
            return record.instance

    def get_record(self, name: str) -> Optional[ModuleRecord]:
        with self._thread_lock:
            return self._modules.get(name)

    def enable(self, name: str) -> None:
        with self._thread_lock:
            record = self._modules.get(name)
            if record is None:
                raise ModuleNotFoundError_(f"no module registered under name '{name}'")
            record.status = ModuleStatus.ACTIVE

    def disable(self, name: str) -> None:
        with self._thread_lock:
            record = self._modules.get(name)
            if record is None:
                raise ModuleNotFoundError_(f"no module registered under name '{name}'")
            record.status = ModuleStatus.DISABLED

    def resolve_load_order(self) -> list[str]:
        """Topologically sorts modules by dependency, tie-broken by priority (lower first)."""
        with self._thread_lock:
            in_degree: dict[str, int] = {name: 0 for name in self._modules}
            dependents_map: dict[str, list[str]] = {name: [] for name in self._modules}

            for name, record in self._modules.items():
                for dep in record.dependencies:
                    if dep in self._modules:
                        in_degree[name] += 1
                        dependents_map[dep].append(name)

            ready = sorted(
                [n for n, deg in in_degree.items() if deg == 0],
                key=lambda n: self._modules[n].priority,
            )
            order: list[str] = []

            while ready:
                ready.sort(key=lambda n: self._modules[n].priority)
                current = ready.pop(0)
                order.append(current)
                for dependent in dependents_map[current]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        ready.append(dependent)

            if len(order) != len(self._modules):
                raise CircularDependencyError("dependency graph contains a cycle; cannot resolve load order")

            return order

    def list_modules(
        self,
        *,
        tag: Optional[str] = None,
        status: Optional[ModuleStatus] = None,
        sort_by_priority: bool = True,
    ) -> list[ModuleRecord]:
        with self._thread_lock:
            records = list(self._modules.values())
            if tag is not None:
                records = [r for r in records if tag in r.tags]
            if status is not None:
                records = [r for r in records if r.status == status]
            if sort_by_priority:
                records.sort(key=lambda r: r.priority)
            return records

    def discover(
        self,
        module_specs: list[tuple[str, str, str]],
        *,
        priority_start: int = 100,
        priority_step: int = 10,
    ) -> list[ModuleRecord]:
        """
        Discover and register modules dynamically.
        module_specs: list of (registry_name, python_module_path, class_name) tuples.
        """
        records: list[ModuleRecord] = []
        priority = priority_start
        with self._thread_lock:
            for registry_name, module_path, class_name in module_specs:
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
                    failed_record = ModuleRecord(
                        name=registry_name, instance=None, version=ModuleVersion(0, 0, 0),
                        priority=priority, status=ModuleStatus.FAILED, metadata={"error": str(exc)},
                    )
                    self._modules[registry_name] = failed_record
                    records.append(failed_record)
                priority += priority_step
        return records

    def health_status(self, name: str) -> HealthStatus:
        with self._thread_lock:
            record = self._modules.get(name)
        if record is None:
            return HealthStatus(healthy=False, component=name, details={"error": "module not registered"})
        if record.instance is None:
            return HealthStatus(healthy=False, component=name, details={"error": "module failed to load", **record.metadata})

        try:
            if hasattr(record.instance, "health_check") and callable(record.instance.health_check):
                result = record.instance.health_check()
                record.last_health_check = result
                record.last_health_check_at = time.time()
                healthy = getattr(result, "healthy", bool(result))
                details = getattr(result, "details", {}) if not isinstance(result, dict) else result
                return HealthStatus(healthy=healthy, component=name, details=dict(details) if details else {})
            return HealthStatus(healthy=True, component=name, details={"note": "module has no health_check method"})
        except Exception as exc:
            return HealthStatus(healthy=False, component=name, details={"error": str(exc)})

    def health_check(self) -> HealthStatus:
        with self._thread_lock:
            names = list(self._modules.keys())
            module_count = len(self._modules)
            active_count = sum(1 for r in self._modules.values() if r.status == ModuleStatus.ACTIVE)
            disabled_count = sum(1 for r in self._modules.values() if r.status == ModuleStatus.DISABLED)
            failed_count = sum(1 for r in self._modules.values() if r.status == ModuleStatus.FAILED)

        try:
            module_results = {name: self.health_status(name) for name in names}
            all_healthy = all(r.healthy for r in module_results.values())
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_module_count": module_count,
                "active_module_count": active_count,
                "disabled_module_count": disabled_count,
                "failed_module_count": failed_count,
                "discovery_attempts": self._discovery_count,
                "modules": {name: {"healthy": res.healthy, "details": res.details} for name, res in module_results.items()},
            }
            return HealthStatus(healthy=all_healthy, component="registry", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="registry", details={"error": str(exc)})

    async def register_async(self, name: str, instance: Any, **kwargs: Any) -> ModuleRecord:
        async with self._async_lock:
            return await asyncio.to_thread(lambda: self.register(name, instance, **kwargs))

    async def unregister_async(self, name: str) -> bool:
        async with self._async_lock:
            return await asyncio.to_thread(self.unregister, name)

    async def discover_async(self, module_specs: list[tuple[str, str, str]], **kwargs: Any) -> list[ModuleRecord]:
        async with self._async_lock:
            return await asyncio.to_thread(lambda: self.discover(module_specs, **kwargs))

    async def health_status_async(self, name: str) -> HealthStatus:
        with self._thread_lock:
            record = self._modules.get(name)
        if record is None:
            return HealthStatus(healthy=False, component=name, details={"error": "module not registered"})
        if record.instance is None:
            return HealthStatus(healthy=False, component=name, details={"error": "module failed to load", **record.metadata})

        try:
            if hasattr(record.instance, "health_check_async") and callable(record.instance.health_check_async):
                result = await record.instance.health_check_async()
            elif hasattr(record.instance, "health_check") and callable(record.instance.health_check):
                result = await asyncio.to_thread(record.instance.health_check)
            else:
                return HealthStatus(healthy=True, component=name, details={"note": "module has no health_check method"})

            record.last_health_check = result
            record.last_health_check_at = time.time()
            healthy = getattr(result, "healthy", bool(result))
            details = getattr(result, "details", {}) if not isinstance(result, dict) else result
            return HealthStatus(healthy=healthy, component=name, details=dict(details) if details else {})
        except Exception as exc:
            return HealthStatus(healthy=False, component=name, details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._async_lock:
            with self._thread_lock:
                names = list(self._modules.keys())
                module_count = len(self._modules)

            results = await asyncio.gather(*(self.health_status_async(name) for name in names))
            module_results = dict(zip(names, results))
            all_healthy = all(r.healthy for r in module_results.values())
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_module_count": module_count,
                "modules": {name: {"healthy": res.healthy, "details": res.details} for name, res in module_results.items()},
            }
            return HealthStatus(healthy=all_healthy, component="registry", details=details)


__all__ = [
    "AutomationModuleRegistry",
    "ModuleRecord",
    "ModuleStatus",
    "ModuleVersion",
    "HealthCheckable",
    "RegistryError",
    "ModuleNotFoundError_",
    "DuplicateModuleError",
    "CircularDependencyError",
    "MissingDependencyError",
    "HealthStatus",
]
