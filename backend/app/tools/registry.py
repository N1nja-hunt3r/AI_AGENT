"""Thread-safe registry for tool registration, discovery, and execution."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.tools.base_tool import (
    BaseTool,
    ToolHealth,
    ToolPriority,
    ToolResult,
    ToolStateError,
    ToolStatus,
)

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Base error for registry operations."""


class ToolNotFoundError(RegistryError):
    """Raised when a requested tool/version cannot be found."""


class ToolAlreadyRegisteredError(RegistryError):
    """Raised when registering a tool name/version that already exists."""


class DependencyError(RegistryError):
    """Raised when tool dependencies cannot be satisfied (missing or cyclic)."""


def _parse_version(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for segment in version.split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


@dataclass(frozen=True)
class RegistryEntry:
    tool: BaseTool[Any, Any]
    dependencies: Tuple[str, ...]
    registered_at: float


@dataclass(frozen=True)
class DiscoveryResult:
    order: List[str]
    initialized: List[str]
    failed: Dict[str, str]


class ToolRegistry:
    """Thread-safe registry managing tool registration, versions, dependencies,
    priority-aware discovery/initialization, health checks, and execution.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tools: Dict[str, Dict[str, RegistryEntry]] = {}
        self._active_version: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tool: BaseTool[Any, Any],
        dependencies: Optional[Sequence[str]] = None,
        replace: bool = False,
        set_active: bool = True,
    ) -> None:
        """Register a tool instance under its name and version."""
        name = tool.name
        version = tool.version
        deps = tuple(dependencies or ())

        with self._lock:
            versions = self._tools.setdefault(name, {})
            if version in versions and not replace:
                raise ToolAlreadyRegisteredError(
                    f"Tool '{name}' version '{version}' is already registered."
                )
            versions[version] = RegistryEntry(
                tool=tool, dependencies=deps, registered_at=time.time()
            )
            if set_active or name not in self._active_version:
                self._active_version[name] = version
            logger.info("Registered tool '%s' version '%s'.", name, version)

    def unregister(self, name: str, version: Optional[str] = None) -> None:
        """Unregister a tool. If version is None, removes all versions of it."""
        with self._lock:
            if name not in self._tools:
                raise ToolNotFoundError(f"Tool '{name}' is not registered.")

            if version is None:
                removed = list(self._tools[name].keys())
                del self._tools[name]
                self._active_version.pop(name, None)
                logger.info("Unregistered all versions of tool '%s': %s", name, removed)
                return

            versions = self._tools[name]
            if version not in versions:
                raise ToolNotFoundError(f"Tool '{name}' version '{version}' is not registered.")
            del versions[version]

            if not versions:
                del self._tools[name]
                self._active_version.pop(name, None)
            elif self._active_version.get(name) == version:
                self._active_version[name] = max(versions.keys(), key=_parse_version)

            logger.info("Unregistered tool '%s' version '%s'.", name, version)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, name: str, version: Optional[str] = None) -> BaseTool[Any, Any]:
        with self._lock:
            return self._get_entry(name, version).tool

    def _get_entry(self, name: str, version: Optional[str] = None) -> RegistryEntry:
        versions = self._tools.get(name)
        if not versions:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        resolved_version = version or self._active_version.get(name)
        if resolved_version is None or resolved_version not in versions:
            raise ToolNotFoundError(f"Tool '{name}' version '{version}' is not registered.")
        return versions[resolved_version]

    def list_tools(self) -> List[Tuple[str, str]]:
        with self._lock:
            return [
                (name, version)
                for name, versions in self._tools.items()
                for version in versions
            ]

    def set_active_version(self, name: str, version: str) -> None:
        with self._lock:
            versions = self._tools.get(name)
            if not versions or version not in versions:
                raise ToolNotFoundError(f"Tool '{name}' version '{version}' is not registered.")
            self._active_version[name] = version

    def get_priority(self, name: str, version: Optional[str] = None) -> ToolPriority:
        with self._lock:
            return self._get_entry(name, version).tool.priority

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _resolve_order(self, names: Optional[Sequence[str]] = None) -> List[str]:
        """Topologically sort tools (including transitive dependencies) by
        dependency order, breaking ties by descending priority."""
        with self._lock:
            target_names = list(names) if names is not None else list(self._tools.keys())
            all_nodes: Set[str] = set()

            def collect(node: str, stack: Set[str]) -> None:
                if node in all_nodes:
                    return
                if node not in self._tools:
                    raise DependencyError(f"Dependency '{node}' is not registered.")
                if node in stack:
                    raise DependencyError(f"Circular dependency detected involving '{node}'.")
                stack.add(node)
                for dep in self._get_entry(node).dependencies:
                    collect(dep, stack)
                stack.discard(node)
                all_nodes.add(node)

            for name in target_names:
                collect(name, set())

            deps_map: Dict[str, Set[str]] = {
                node: set(self._get_entry(node).dependencies) for node in all_nodes
            }
            dependents: Dict[str, List[str]] = {node: [] for node in all_nodes}
            remaining_in_degree: Dict[str, int] = {}
            for node, deps in deps_map.items():
                remaining_in_degree[node] = len(deps)
                for dep in deps:
                    dependents[dep].append(node)

            def priority_of(node: str) -> int:
                return int(self._get_entry(node).tool.priority)

            ready = [node for node, degree in remaining_in_degree.items() if degree == 0]
            order: List[str] = []

            while ready:
                ready.sort(key=priority_of, reverse=True)
                node = ready.pop(0)
                order.append(node)
                for dependent in dependents[node]:
                    remaining_in_degree[dependent] -= 1
                    if remaining_in_degree[dependent] == 0:
                        ready.append(dependent)

            if len(order) != len(all_nodes):
                raise DependencyError("Circular dependency detected among registered tools.")

            return order

    # ------------------------------------------------------------------
    # Discovery / bulk initialization
    # ------------------------------------------------------------------

    async def discover(self, names: Optional[Sequence[str]] = None) -> DiscoveryResult:
        """Resolve dependency/priority order and initialize tools accordingly."""
        order = self._resolve_order(names)
        initialized: List[str] = []
        failed: Dict[str, str] = {}

        for name in order:
            with self._lock:
                entry = self._get_entry(name)
            tool = entry.tool

            unmet = [dep for dep in entry.dependencies if dep in failed]
            if unmet:
                failed[name] = f"Unmet dependencies: {', '.join(unmet)}"
                continue

            if tool.status is ToolStatus.READY:
                initialized.append(name)
                continue

            try:
                await tool.initialize()
                initialized.append(name)
            except Exception as exc:  # noqa: BLE001 - capture per-tool failure
                logger.error("Failed to initialize tool '%s': %s", name, exc)
                failed[name] = str(exc)

        return DiscoveryResult(order=order, initialized=initialized, failed=failed)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self, names: Optional[Sequence[str]] = None) -> Dict[str, ToolHealth]:
        """Run health checks concurrently for the given (or all) registered tools."""
        with self._lock:
            target = list(names) if names is not None else [n for n, _ in self.list_tools()]
            entries = {name: self._get_entry(name) for name in target}

        async def _check(name: str, entry: RegistryEntry) -> Tuple[str, ToolHealth]:
            health = await entry.tool.health_check()
            return name, health

        results = await asyncio.gather(*(_check(name, entry) for name, entry in entries.items()))
        return dict(results)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        name: str,
        input_data: Any,
        version: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult[Any]:
        """Execute a registered tool by name (and optional version)."""
        with self._lock:
            entry = self._get_entry(name, version)
        tool = entry.tool

        if tool.status is not ToolStatus.READY:
            raise ToolStateError(
                f"Tool '{name}' is not ready for execution (status={tool.status.value})."
            )

        return await tool.execute(input_data, timeout=timeout)
