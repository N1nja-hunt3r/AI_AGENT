"""
registry.py

Generic, thread-safe component registry providing registration,
discovery, versioning and health-check aggregation for capabilities,
services, agents, routers, security components and monitoring
components across the system.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentRecord:
    name: str
    component: Any
    version: str = "1.0.0"
    namespace: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    health_check: Optional[Callable[[], Any]] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_checked_at: Optional[float] = None


class ComponentAlreadyRegisteredError(RuntimeError):
    pass


class ComponentNotFoundError(KeyError):
    pass


class Registry:
    """Thread-safe registry for arbitrary system components."""

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace
        self._lock = threading.RLock()
        self._records: Dict[str, ComponentRecord] = {}

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(
        self,
        name: str,
        component: Any,
        *,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        health_check: Optional[Callable[[], Any]] = None,
        overwrite: bool = False,
    ) -> ComponentRecord:
        with self._lock:
            if name in self._records and not overwrite:
                raise ComponentAlreadyRegisteredError(
                    f"Component '{name}' already registered in namespace "
                    f"'{self._namespace}'."
                )
            record = ComponentRecord(
                name=name,
                component=component,
                version=version,
                namespace=self._namespace,
                metadata=metadata or {},
                health_check=health_check,
            )
            self._records[name] = record
            return record

    def unregister(self, name: str) -> None:
        with self._lock:
            self._records.pop(name, None)

    def get(self, name: str) -> Any:
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise ComponentNotFoundError(
                    f"Component '{name}' not found in namespace "
                    f"'{self._namespace}'."
                )
            return record.component

    def get_record(self, name: str) -> ComponentRecord:
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise ComponentNotFoundError(
                    f"Component '{name}' not found in namespace "
                    f"'{self._namespace}'."
                )
            return record

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._records

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._records.keys())

    def list_records(self) -> List[ComponentRecord]:
        with self._lock:
            return list(self._records.values())

    def discover(self, predicate: Callable[[ComponentRecord], bool]) -> List[ComponentRecord]:
        with self._lock:
            return [r for r in self._records.values() if predicate(r)]

    def get_version(self, name: str) -> str:
        return self.get_record(name).version

    async def health_check(self, name: str) -> HealthStatus:
        record = self.get_record(name)
        status = HealthStatus.UNKNOWN
        if record.health_check is not None:
            try:
                result = record.health_check()
                if hasattr(result, "__await__"):
                    result = await result  # type: ignore[assignment]
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            except Exception:
                status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.HEALTHY
        with self._lock:
            record.last_status = status
            record.last_checked_at = time.time()
        return status

    async def health_check_all(self) -> Dict[str, HealthStatus]:
        results: Dict[str, HealthStatus] = {}
        for name in self.list():
            results[name] = await self.health_check(name)
        return results

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


_global_registries: Dict[str, Registry] = {}
_global_lock = threading.RLock()


def get_registry(namespace: str = "default") -> Registry:
    """Return (creating if necessary) the process-wide registry for a namespace."""
    with _global_lock:
        if namespace not in _global_registries:
            _global_registries[namespace] = Registry(namespace=namespace)
        return _global_registries[namespace]
