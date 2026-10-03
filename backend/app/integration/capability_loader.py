"""
capability_loader.py

Discovers, loads and registers system capabilities, integrating with
the global component registry and dependency container, with async
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

logger = logging.getLogger("capability_loader")


class CapabilityLoadError(RuntimeError):
    pass


class CapabilityLoader:
    """Loads capability modules and registers them for injection."""

    def __init__(
        self,
        package: str = "app.capabilities",
        container: Optional[DependencyContainer] = None,
    ) -> None:
        self._package = package
        self._container = container or get_container()
        self._registry: Registry = get_registry("capabilities")
        self._lock = asyncio.Lock()
        self._loaded = False

    async def discover(self) -> List[str]:
        """Discover capability module names within the configured package."""
        try:
            package = importlib.import_module(self._package)
        except ImportError:
            logger.warning("Capability package '%s' not found.", self._package)
            return []

        names: List[str] = []
        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{self._package}."):
            names.append(module_info.name)
        return names

    async def load_all(self) -> Dict[str, Any]:
        async with self._lock:
            loaded: Dict[str, Any] = {}
            module_names = await self.discover()
            for module_name in module_names:
                try:
                    module = importlib.import_module(module_name)
                    factory = getattr(module, "create_capability", None)
                    if factory is None:
                        for attr_name in dir(module):
                            if attr_name.endswith("Capability"):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type) and attr.__name__ != "Capability":
                                    factory = attr
                                    break
                    if factory is None:
                        continue
                    instance = factory()
                    if hasattr(instance, "initialize"):
                        await instance.initialize()
                    short_name = module_name.rsplit(".", 1)[-1]
                    await self.register_capability(short_name, instance)
                    loaded[short_name] = instance
                except Exception:
                    logger.exception("Failed to load capability module '%s'.", module_name)
            self._loaded = True
            return loaded

    async def register_capability(self, name: str, instance: Any) -> None:
        self._registry.register(
            name,
            instance,
            version=getattr(instance, "version", "1.0.0"),
            metadata={"kind": "capability"},
            health_check=getattr(instance, "health_check", None),
            overwrite=True,
        )
        self._container.register_capability(name, instance)

    def get_capability(self, name: str) -> Any:
        return self._registry.get(name)

    def list_capabilities(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


_capability_loader_lock = asyncio.Lock()
_capability_loader: Optional[CapabilityLoader] = None


async def get_capability_loader() -> CapabilityLoader:
    global _capability_loader
    async with _capability_loader_lock:
        if _capability_loader is None:
            _capability_loader = CapabilityLoader()
        return _capability_loader
