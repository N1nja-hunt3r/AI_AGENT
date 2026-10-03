from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from app.computer_use.base_computer import (
    ComputerCapability,
    ExecutionRequest,
    ExecutionResult,
    HealthCheckResult,
    HealthStatus,
)


class RegistryError(Exception):
    pass


class CapabilityAlreadyRegisteredError(RegistryError):
    pass


class CapabilityNotRegisteredError(RegistryError):
    pass


class CircularDependencyError(RegistryError):
    pass


class MissingDependencyError(RegistryError):
    pass


@dataclass(frozen=True)
class RegistryEntry:
    capability: ComputerCapability
    registered_at_epoch: float


@dataclass(frozen=True)
class DiscoveryFilter:
    tag: Optional[str] = None
    name_prefix: Optional[str] = None
    min_priority: Optional[int] = None
    ready_only: bool = False


class CapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, capability: ComputerCapability, replace: bool = False
    ) -> None:
        async with self._lock:
            name = capability.name
            if name in self._entries and not replace:
                raise CapabilityAlreadyRegisteredError(
                    f"capability '{name}' is already registered"
                )
            self._validate_dependencies_present(capability)
            self._entries[name] = RegistryEntry(
                capability=capability, registered_at_epoch=time.time()
            )

    async def unregister(self, name: str, shutdown: bool = True) -> None:
        async with self._lock:
            entry = self._entries.pop(name, None)
            if entry is None:
                raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")
            dependents = self._find_dependents_locked(name)
            if dependents:
                self._entries[name] = entry
                raise RegistryError(
                    f"cannot unregister '{name}': required by {sorted(dependents)}"
                )
            if shutdown:
                await entry.capability.shutdown()

    def _validate_dependencies_present(self, capability: ComputerCapability) -> None:
        for dependency in capability.metadata.dependencies:
            if dependency not in self._entries and dependency != capability.name:
                pass

    def _find_dependents_locked(self, name: str) -> set[str]:
        dependents: set[str] = set()
        for other_name, entry in self._entries.items():
            if name in entry.capability.metadata.dependencies:
                dependents.add(other_name)
        return dependents

    def get(self, name: str) -> ComputerCapability:
        entry = self._entries.get(name)
        if entry is None:
            raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")
        return entry.capability

    def discover(self, filter_: Optional[DiscoveryFilter] = None) -> list[ComputerCapability]:
        capabilities = [entry.capability for entry in self._entries.values()]

        if filter_ is None:
            return self._sort_by_priority(capabilities)

        filtered = capabilities
        if filter_.tag is not None:
            filtered = [c for c in filtered if filter_.tag in c.metadata.tags]
        if filter_.name_prefix is not None:
            filtered = [c for c in filtered if c.name.startswith(filter_.name_prefix)]
        if filter_.min_priority is not None:
            filtered = [c for c in filtered if int(c.priority) >= filter_.min_priority]
        if filter_.ready_only:
            filtered = [c for c in filtered if c.is_ready]

        return self._sort_by_priority(filtered)

    @staticmethod
    def _sort_by_priority(capabilities: list[ComputerCapability]) -> list[ComputerCapability]:
        return sorted(capabilities, key=lambda c: int(c.priority), reverse=True)

    def resolve_dependency_order(self, names: Optional[list[str]] = None) -> list[str]:
        target_names = names if names is not None else list(self._entries.keys())
        for name in target_names:
            if name not in self._entries:
                raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")

        visited: set[str] = set()
        in_progress: set[str] = set()
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in in_progress:
                raise CircularDependencyError(
                    f"circular dependency detected involving '{name}'"
                )
            in_progress.add(name)
            entry = self._entries.get(name)
            if entry is None:
                raise MissingDependencyError(f"dependency '{name}' is not registered")
            for dependency in entry.capability.metadata.dependencies:
                if dependency not in self._entries:
                    raise MissingDependencyError(
                        f"'{name}' depends on unregistered capability '{dependency}'"
                    )
                visit(dependency)
            in_progress.discard(name)
            visited.add(name)
            ordered.append(name)

        for name in target_names:
            visit(name)

        return ordered

    async def initialize_all(self, names: Optional[list[str]] = None) -> dict[str, Optional[str]]:
        order = self.resolve_dependency_order(names)
        results: dict[str, Optional[str]] = {}
        for name in order:
            capability = self.get(name)
            try:
                await capability.initialize()
                results[name] = None
            except Exception as exc:
                results[name] = str(exc)
        return results

    async def shutdown_all(self, names: Optional[list[str]] = None) -> dict[str, Optional[str]]:
        order = self.resolve_dependency_order(names)
        results: dict[str, Optional[str]] = {}
        for name in reversed(order):
            capability = self.get(name)
            try:
                await capability.shutdown()
                results[name] = None
            except Exception as exc:
                results[name] = str(exc)
        return results

    async def execute(self, name: str, request: ExecutionRequest) -> ExecutionResult:
        capability = self.get(name)
        return await capability.execute(request)

    async def health_check(self, name: str) -> HealthCheckResult:
        capability = self.get(name)
        return await capability.health_check()

    async def health_check_all(
        self, names: Optional[list[str]] = None
    ) -> dict[str, HealthCheckResult]:
        target_names = names if names is not None else list(self._entries.keys())
        capabilities = [(name, self.get(name)) for name in target_names]

        results = await asyncio.gather(
            *(capability.health_check() for _, capability in capabilities),
            return_exceptions=True,
        )

        health_map: dict[str, HealthCheckResult] = {}
        for (name, _), result in zip(capabilities, results):
            if isinstance(result, HealthCheckResult):
                health_map[name] = result
            else:
                health_map[name] = HealthCheckResult(
                    capability_name=name,
                    status=HealthStatus.UNKNOWN,
                    latency_seconds=None,
                    checked_at_epoch=time.time(),
                    detail=str(result),
                )
        return health_map

    def list_names(self) -> list[str]:
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
