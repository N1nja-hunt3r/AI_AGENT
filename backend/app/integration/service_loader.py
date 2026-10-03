"""
service_loader.py

Discovers, loads and registers system services, integrating with the
global component registry and dependency container, with async
loading and health-check support.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
from typing import Any, Dict, List, Optional

from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus, Registry, get_registry

logger = logging.getLogger("service_loader")


class ServiceLoader:
    """Loads service modules and registers them for injection."""

    def __init__(
        self,
        package: str = "app.services",
        container: Optional[DependencyContainer] = None,
    ) -> None:
        self._package = package
        self._container = container or get_container()
        self._registry: Registry = get_registry("services")
        self._lock = asyncio.Lock()
        self._loaded = False

    async def discover(self) -> List[str]:
        try:
            package = importlib.import_module(self._package)
        except ImportError:
            logger.warning("Service package '%s' not found.", self._package)
            return []

        return [
            module_info.name
            for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{self._package}.")
        ]

    async def load_all(self) -> Dict[str, Any]:
        async with self._lock:
            loaded: Dict[str, Any] = {}
            for module_name in await self.discover():
                try:
                    module = importlib.import_module(module_name)
                    factory = getattr(module, "create_service", None) or getattr(module, "Service", None)
                    if factory is None:
                        continue
                    instance = factory() if callable(factory) else factory
                    if hasattr(instance, "initialize"):
                        await instance.initialize()
                    short_name = module_name.rsplit(".", 1)[-1]
                    await self.register_service(short_name, instance)
                    loaded[short_name] = instance
                except Exception:
                    logger.exception("Failed to load service module '%s'.", module_name)
            self._loaded = True
            return loaded

    async def register_service(self, name: str, instance: Any) -> None:
        self._registry.register(
            name,
            instance,
            version=getattr(instance, "version", "1.0.0"),
            metadata={"kind": "service"},
            health_check=getattr(instance, "health_check", None),
            overwrite=True,
        )
        self._container.register_service(name, instance)

    def get_service(self, name: str) -> Any:
        return self._registry.get(name)

    def list_services(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


_service_loader_lock = asyncio.Lock()
_service_loader: Optional[ServiceLoader] = None


async def get_service_loader() -> ServiceLoader:
    global _service_loader
    async with _service_loader_lock:
        if _service_loader is None:
            _service_loader = ServiceLoader()
        return _service_loader
