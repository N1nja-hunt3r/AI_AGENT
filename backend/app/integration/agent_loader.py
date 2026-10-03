"""
agent_loader.py

Discovers, loads and registers system agents, integrating with the
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

logger = logging.getLogger("agent_loader")


class AgentLoader:
    """Loads agent modules and registers them for injection."""

    def __init__(
        self,
        package: str = "app.agents",
        container: Optional[DependencyContainer] = None,
    ) -> None:
        self._package = package
        self._container = container or get_container()
        self._registry: Registry = get_registry("agents")
        self._lock = asyncio.Lock()
        self._loaded = False

    async def discover(self) -> List[str]:
        try:
            package = importlib.import_module(self._package)
        except ImportError:
            logger.warning("Agent package '%s' not found.", self._package)
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
                    factory = getattr(module, "create_agent", None)
                    if factory is None:
                        for attr_name in dir(module):
                            if attr_name.endswith("Agent") or attr_name.endswith("Manager") or attr_name.endswith("Controller"):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type) and attr_name != "Agent":
                                    factory = attr
                                    break
                    if factory is None:
                        continue
                    instance = factory()
                    if hasattr(instance, "initialize"):
                        await instance.initialize()
                    short_name = module_name.rsplit(".", 1)[-1]
                    await self.register_agent(short_name, instance)
                    loaded[short_name] = instance
                except Exception:
                    logger.exception("Failed to load agent module '%s'.", module_name)
            self._loaded = True
            return loaded

    async def register_agent(self, name: str, instance: Any) -> None:
        self._registry.register(
            name,
            instance,
            version=getattr(instance, "version", "1.0.0"),
            metadata={"kind": "agent"},
            health_check=getattr(instance, "health_check", None),
            overwrite=True,
        )
        self._container.register_agent(name, instance)

    def get_agent(self, name: str) -> Any:
        return self._registry.get(name)

    def list_agents(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


_agent_loader_lock = asyncio.Lock()
_agent_loader: Optional[AgentLoader] = None


async def get_agent_loader() -> AgentLoader:
    global _agent_loader
    async with _agent_loader_lock:
        if _agent_loader is None:
            _agent_loader = AgentLoader()
        return _agent_loader
