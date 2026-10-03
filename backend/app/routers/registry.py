"""
registry.py

Thread-safe registry for FastAPI routers: registration, package-based
discovery, versioning, and aggregate health checks across all mounted
routers.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import APIRouter, FastAPI

logger = logging.getLogger(__name__)

HealthCheckFn = Callable[[], Union[bool, Awaitable[bool]]]


class RouterRegistryError(Exception):
    """Base error for router registry operations."""


class DuplicateRouterError(RouterRegistryError):
    """Raised when registering a name/version pair that already exists."""


class RouterNotFoundError(RouterRegistryError):
    """Raised when a requested router entry does not exist."""


@dataclass
class RouterEntry:
    name: str
    router: APIRouter
    version: str = "1.0.0"
    priority: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)
    health_check: Optional[HealthCheckFn] = None
    enabled: bool = True

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


class RouterRegistry:
    """Thread-safe registry for discovering and mounting FastAPI routers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, Dict[str, RouterEntry]] = {}
        self._default_versions: Dict[str, str] = {}
        self._mounted: set[str] = set()

    def register(
        self,
        name: str,
        router: APIRouter,
        *,
        version: str = "1.0.0",
        priority: int = 0,
        tags: Optional[tuple[str, ...]] = None,
        health_check: Optional[HealthCheckFn] = None,
        enabled: bool = True,
        overwrite: bool = False,
        make_default: bool = True,
    ) -> RouterEntry:
        """Register a router instance under a name/version."""
        with self._lock:
            versions = self._entries.setdefault(name, {})
            if version in versions and not overwrite:
                raise DuplicateRouterError(f"Router '{name}@{version}' already registered")
            entry = RouterEntry(
                name=name,
                router=router,
                version=version,
                priority=priority,
                tags=tags or (),
                health_check=health_check,
                enabled=enabled,
            )
            versions[version] = entry
            if make_default or name not in self._default_versions:
                self._default_versions[name] = version
            logger.debug("Registered router %s (priority=%d)", entry.key, priority)
            return entry

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
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

    def get_entry(self, name: str, version: Optional[str] = None) -> RouterEntry:
        with self._lock:
            versions = self._entries.get(name)
            if not versions:
                raise RouterNotFoundError(f"No router registered under '{name}'")
            target_version = version or self._default_versions.get(name)
            if target_version is None or target_version not in versions:
                raise RouterNotFoundError(f"No version '{version}' registered for '{name}'")
            return versions[target_version]

    def set_default_version(self, name: str, version: str) -> None:
        with self._lock:
            versions = self._entries.get(name)
            if not versions or version not in versions:
                raise RouterNotFoundError(f"Cannot set default; '{name}@{version}' not registered")
            self._default_versions[name] = version

    def list_entries(self, *, tag: Optional[str] = None, enabled_only: bool = False) -> List[RouterEntry]:
        with self._lock:
            all_entries = [
                entry for versions in self._entries.values() for entry in versions.values()
            ]
        if tag:
            all_entries = [e for e in all_entries if tag in e.tags]
        if enabled_only:
            all_entries = [e for e in all_entries if e.enabled]
        return sorted(all_entries, key=lambda e: (-e.priority, e.name, e.version))

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries.keys())

    def enable(self, name: str, version: Optional[str] = None) -> None:
        self.get_entry(name, version).enabled = True

    def disable(self, name: str, version: Optional[str] = None) -> None:
        self.get_entry(name, version).enabled = False

    def mount_all(
        self,
        app: FastAPI,
        *,
        prefix: str = "",
        only_enabled: bool = True,
        dependencies: Optional[List[Any]] = None,
    ) -> List[str]:
        """Mount all registered (default-version) routers onto a FastAPI app."""
        mounted: List[str] = []
        with self._lock:
            for name, default_version in self._default_versions.items():
                entry = self._entries[name][default_version]
                if only_enabled and not entry.enabled:
                    continue
                if entry.key in self._mounted:
                    continue
                include_kwargs: Dict[str, Any] = {}
                if prefix:
                    include_kwargs["prefix"] = prefix
                if dependencies:
                    include_kwargs["dependencies"] = dependencies
                app.include_router(entry.router, **include_kwargs)
                self._mounted.add(entry.key)
                mounted.append(entry.key)
                logger.info("Mounted router: %s", entry.key)
        return mounted

    def mount(self, app: FastAPI, name: str, *, version: Optional[str] = None, prefix: str = "") -> str:
        """Mount a single registered router by name/version."""
        entry = self.get_entry(name, version)
        include_kwargs: Dict[str, Any] = {}
        if prefix:
            include_kwargs["prefix"] = prefix
        app.include_router(entry.router, **include_kwargs)
        with self._lock:
            self._mounted.add(entry.key)
        logger.info("Mounted router: %s", entry.key)
        return entry.key

    def discover_package(
        self,
        package_name: str,
        *,
        router_attr: str = "router",
        name_prefix: str = "",
        priority_attr: str = "ROUTER_PRIORITY",
        version_attr: str = "ROUTER_VERSION",
        tags_attr: str = "ROUTER_TAGS",
        auto_register: bool = True,
    ) -> List[str]:
        """
        Walk a package's modules, find any module exposing an APIRouter
        instance under `router_attr`, and optionally auto-register it.
        Returns the list of discovered router names.
        """
        discovered: List[str] = []
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            raise RouterRegistryError(f"Cannot import package '{package_name}': {exc}") from exc

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            return discovered

        for module_info in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
            try:
                module = importlib.import_module(module_info.name)
            except ImportError:
                logger.exception("Failed to import module during router discovery: %s", module_info.name)
                continue

            candidate = getattr(module, router_attr, None)
            if candidate is None or not isinstance(candidate, APIRouter):
                continue

            module_short_name = module_info.name.rsplit(".", 1)[-1]
            router_name = f"{name_prefix}{module_short_name}"
            priority = getattr(module, priority_attr, 0)
            version = getattr(module, version_attr, "1.0.0")
            tags = tuple(getattr(module, tags_attr, ()))

            discovered.append(router_name)
            if auto_register:
                try:
                    self.register(
                        router_name,
                        candidate,
                        version=version,
                        priority=priority,
                        tags=tags,
                        overwrite=True,
                    )
                except DuplicateRouterError:
                    logger.debug("Router '%s' already registered; skipping", router_name)

        return discovered

    async def health_check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Run health checks for one router entry, or all registered entries."""
        import inspect

        with self._lock:
            entries = (
                [self.get_entry(name)] if name else
                [e for versions in self._entries.values() for e in versions.values()]
            )

        results: Dict[str, Any] = {}
        all_healthy = True
        for entry in entries:
            if entry.health_check is None:
                results[entry.key] = {
                    "status": "unknown",
                    "enabled": entry.enabled,
                    "reason": "no health_check defined",
                }
                continue
            try:
                outcome = entry.health_check()
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                healthy = bool(outcome)
            except Exception as exc:  # noqa: BLE001
                healthy = False
                results[entry.key] = {"status": "unhealthy", "enabled": entry.enabled, "error": str(exc)}
                all_healthy = False
                continue
            results[entry.key] = {"status": "healthy" if healthy else "unhealthy", "enabled": entry.enabled}
            if not healthy:
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checked": len(entries),
            "details": results,
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._default_versions.clear()
            self._mounted.clear()


_global_router_registry: Optional[RouterRegistry] = None
_global_router_registry_lock = threading.Lock()


def get_router_registry() -> RouterRegistry:
    """Return the process-wide singleton RouterRegistry, creating it if needed."""
    global _global_router_registry
    with _global_router_registry_lock:
        if _global_router_registry is None:
            _global_router_registry = RouterRegistry()
        return _global_router_registry


def reset_router_registry() -> None:
    """Reset the singleton RouterRegistry (primarily for testing)."""
    global _global_router_registry
    with _global_router_registry_lock:
        _global_router_registry = None


def register_router(
    name: str,
    *,
    version: str = "1.0.0",
    priority: int = 0,
    tags: Optional[tuple[str, ...]] = None,
    health_check: Optional[HealthCheckFn] = None,
    enabled: bool = True,
    overwrite: bool = False,
    make_default: bool = True,
) -> Callable[[APIRouter], APIRouter]:
    """Decorator-style helper to register a module-level `router` object."""

    def decorator(router: APIRouter) -> APIRouter:
        get_router_registry().register(
            name,
            router,
            version=version,
            priority=priority,
            tags=tags,
            health_check=health_check,
            enabled=enabled,
            overwrite=overwrite,
            make_default=make_default,
        )
        return router

    return decorator
