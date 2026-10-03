"""
router_loader.py

Discovers and registers FastAPI routers, including API versioning and
health-check reporting for the routing layer.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI

from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus, Registry, get_registry

logger = logging.getLogger("router_loader")


@dataclass(frozen=True)
class RouterSpec:
    name: str
    router: APIRouter
    prefix: str
    version: str = "v1"
    tags: Optional[List[str]] = None


class RouterLoader:
    """Loads and mounts versioned API routers onto a FastAPI application."""

    def __init__(
        self,
        package: str = "app.routers",
        container: Optional[DependencyContainer] = None,
        default_version: str = "v1",
    ) -> None:
        self._package = package
        self._container = container or get_container()
        self._registry: Registry = get_registry("routers")
        self._default_version = default_version
        self._mounted = False

    def discover(self) -> List[RouterSpec]:
        specs: List[RouterSpec] = []
        try:
            package = importlib.import_module(self._package)
        except ImportError:
            logger.warning("Router package '%s' not found; no routers discovered.", self._package)
            return specs

        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{self._package}."):
            try:
                module = importlib.import_module(module_info.name)
                router = getattr(module, "router", None)
                if not isinstance(router, APIRouter):
                    continue
                short_name = module_info.name.rsplit(".", 1)[-1]
                version = getattr(module, "API_VERSION", self._default_version)
                prefix = getattr(module, "PREFIX", f"/api/{version}/{short_name}")
                tags = getattr(module, "TAGS", [short_name])
                specs.append(
                    RouterSpec(name=short_name, router=router, prefix=prefix, version=version, tags=tags)
                )
            except Exception:
                logger.exception("Failed to inspect router module '%s'.", module_info.name)
        return specs

    def register_routers(self, app: FastAPI, extra_specs: Optional[List[RouterSpec]] = None) -> List[str]:
        registered: List[str] = []
        specs = self.discover() + (extra_specs or [])
        for spec in specs:
            try:
                app.include_router(spec.router, prefix=spec.prefix, tags=spec.tags)
                self._registry.register(
                    spec.name,
                    spec.router,
                    version=spec.version,
                    metadata={"prefix": spec.prefix, "tags": spec.tags},
                    overwrite=True,
                )
                self._container.register_router(spec.name, spec.router)
                registered.append(spec.name)
                logger.info(
                    "Mounted router '%s' at '%s' (version=%s).", spec.name, spec.prefix, spec.version
                )
            except Exception:
                logger.exception("Failed to mount router '%s'.", spec.name)
        self._mounted = True
        return registered

    def list_routers(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    @property
    def is_mounted(self) -> bool:
        return self._mounted


_router_loader: Optional[RouterLoader] = None


def get_router_loader() -> RouterLoader:
    global _router_loader
    if _router_loader is None:
        _router_loader = RouterLoader()
    return _router_loader
