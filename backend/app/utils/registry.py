"""
registry.py

Thread-safe registry for discovering, registering, versioning, and
health-checking pluggable utility components (e.g. tools, providers,
agents) by name and priority.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

HealthCheckFn = Callable[[], bool]


class RegistryError(Exception):
    """Base error for registry operations."""


class DuplicateRegistrationError(RegistryError):
    """Raised when registering a name/version pair that already exists."""


class EntryNotFoundError(RegistryError):
    """Raised when a requested registry entry does not exist."""


@dataclass
class RegistryEntry:
    name: str
    version: str
    factory: Callable[..., Any]
    priority: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)
    health_check: Optional[HealthCheckFn] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


class Registry:
    """Thread-safe registry supporting versioned, prioritized entries."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, Dict[str, RegistryEntry]] = {}
        self._default_versions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        version: str = "1.0.0",
        priority: int = 0,
        tags: Optional[tuple[str, ...]] = None,
        health_check: Optional[HealthCheckFn] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        make_default: bool = True,
    ) -> RegistryEntry:
        """Register a factory under a name/version, optionally as the default version."""
        with self._lock:
            versions = self._entries.setdefault(name, {})
            if version in versions and not overwrite:
                raise DuplicateRegistrationError(
                    f"'{name}@{version}' is already registered"
                )
            entry = RegistryEntry(
                name=name,
                version=version,
                factory=factory,
                priority=priority,
                tags=tags or (),
                health_check=health_check,
                metadata=metadata or {},
            )
            versions[version] = entry
            if make_default or name not in self._default_versions:
                self._default_versions[name] = version
            logger.debug("Registered %s (priority=%d)", entry.key, priority)
            return entry

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """Unregister a specific version, or all versions if version is None."""
        with self._lock:
            if name not in self._entries:
                return False
            if version is None:
                del self._entries[name]
                self._default_versions.pop(name, None)
                return True
            removed = self._entries[name].pop(version, None) is not None
            if removed and self._default_versions.get(name) == version:
                remaining = list(self._entries[name].keys())
                if remaining:
                    self._default_versions[name] = remaining[-1]
                else:
                    self._default_versions.pop(name, None)
                    del self._entries[name]
            return removed

    def get_entry(self, name: str, version: Optional[str] = None) -> RegistryEntry:
        """Retrieve a registry entry by name, defaulting to its current default version."""
        with self._lock:
            versions = self._entries.get(name)
            if not versions:
                raise EntryNotFoundError(f"No entries registered under '{name}'")
            target_version = version or self._default_versions.get(name)
            if target_version is None or target_version not in versions:
                raise EntryNotFoundError(f"No version '{version}' registered for '{name}'")
            return versions[target_version]

    def resolve(self, name: str, *args: Any, version: Optional[str] = None, **kwargs: Any) -> Any:
        """Instantiate/construct the registered factory for `name`."""
        entry = self.get_entry(name, version)
        return entry.factory(*args, **kwargs)

    def set_default_version(self, name: str, version: str) -> None:
        with self._lock:
            versions = self._entries.get(name)
            if not versions or version not in versions:
                raise EntryNotFoundError(f"Cannot set default; '{name}@{version}' not registered")
            self._default_versions[name] = version

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries.keys())

    def list_versions(self, name: str) -> List[str]:
        with self._lock:
            versions = self._entries.get(name, {})
            return sorted(versions.keys())

    def list_entries(self, *, tag: Optional[str] = None) -> List[RegistryEntry]:
        """List all entries (across all names/versions), optionally filtered by tag."""
        with self._lock:
            all_entries = [
                entry for versions in self._entries.values() for entry in versions.values()
            ]
        if tag:
            all_entries = [e for e in all_entries if tag in e.tags]
        return sorted(all_entries, key=lambda e: (-e.priority, e.name, e.version))

    def best_for_tag(self, tag: str) -> Optional[RegistryEntry]:
        """Return the highest-priority entry matching a given tag."""
        candidates = self.list_entries(tag=tag)
        return candidates[0] if candidates else None

    def contains(self, name: str, version: Optional[str] = None) -> bool:
        with self._lock:
            versions = self._entries.get(name)
            if not versions:
                return False
            return version is None or version in versions

    def health_check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Run health checks for one entry or all registered entries."""
        with self._lock:
            entries = (
                [self.get_entry(name)] if name else
                [e for versions in self._entries.values() for e in versions.values()]
            )

        results: Dict[str, Any] = {}
        all_healthy = True
        for entry in entries:
            if entry.health_check is None:
                results[entry.key] = {"status": "unknown", "reason": "no health_check defined"}
                continue
            try:
                healthy = bool(entry.health_check())
            except Exception as exc:  # noqa: BLE001
                healthy = False
                results[entry.key] = {"status": "unhealthy", "error": str(exc)}
                all_healthy = False
                continue
            results[entry.key] = {"status": "healthy" if healthy else "unhealthy"}
            if not healthy:
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checked": len(entries),
            "details": results,
        }

    def discover_package(
        self,
        package_name: str,
        *,
        predicate: Optional[Callable[[Type[Any]], bool]] = None,
        register_with: Optional[Callable[[Type[Any]], None]] = None,
    ) -> List[Type[Any]]:
        """
        Import all modules within a package and collect classes matching
        an optional predicate. If `register_with` is provided, invoke it
        for each discovered class to perform custom registration.
        """
        discovered: List[Type[Any]] = []
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            raise RegistryError(f"Cannot import package '{package_name}': {exc}") from exc

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            return discovered

        for module_info in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
            try:
                module = importlib.import_module(module_info.name)
            except ImportError:
                logger.exception("Failed to import module during discovery: %s", module_info.name)
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if predicate and not predicate(obj):
                    continue
                discovered.append(obj)
                if register_with:
                    register_with(obj)

        return discovered

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._default_versions.clear()


_global_registry: Optional[Registry] = None
_global_registry_lock = threading.Lock()


def get_registry() -> Registry:
    """Return the process-wide singleton Registry, creating it if needed."""
    global _global_registry
    with _global_registry_lock:
        if _global_registry is None:
            _global_registry = Registry()
        return _global_registry


def reset_registry() -> None:
    """Reset the singleton Registry (primarily for testing)."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None


def register(
    name: str,
    *,
    version: str = "1.0.0",
    priority: int = 0,
    tags: Optional[tuple[str, ...]] = None,
    health_check: Optional[HealthCheckFn] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    make_default: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Class/function decorator that registers the target into the global Registry."""

    def decorator(factory: Callable[..., T]) -> Callable[..., T]:
        get_registry().register(
            name,
            factory,
            version=version,
            priority=priority,
            tags=tags,
            health_check=health_check,
            metadata=metadata,
            overwrite=overwrite,
            make_default=make_default,
        )
        return factory

    return decorator
