"""
registry.py

Central registry for managing named VectorStore instances with lifecycle
management, priority-based selection, and health monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.vector_db.base_vector_store import (
    HealthStatus,
    VectorStore,
)

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Base exception for registry failures."""

class StoreNotFoundError(RegistryError):
    """Raised when a requested store is not registered."""

class StoreAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a store with a duplicate name."""


@dataclass(frozen=True)
class RegistryHealthStatus:
    healthy: bool
    store_count: int
    healthy_store_count: int
    stores: Dict[str, bool]
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


class Registry:
    """Manages named VectorStore instances with lifecycle and priority
    support."""

    def __init__(self) -> None:
        self._stores: Dict[str, VectorStore] = {}
        self._lock = asyncio.Lock()

    async def register(self, name: str, store: VectorStore, initialize: bool = True) -> None:
        async with self._lock:
            if name in self._stores:
                raise StoreAlreadyRegisteredError(f"Store '{name}' is already registered.")
            self._stores[name] = store
        if initialize:
            await store.initialize()
        logger.info("Registered vector store '%s' (backend=%s, priority=%s).", name, store.metadata.backend, store.priority.name)

    async def unregister(self, name: str, shutdown: bool = True) -> None:
        async with self._lock:
            store = self._stores.pop(name, None)
        if store is None:
            raise StoreNotFoundError(f"Store '{name}' is not registered.")
        if shutdown:
            await store.shutdown()
        logger.info("Unregistered vector store '%s'.", name)

    async def get(self, name: str) -> VectorStore:
        async with self._lock:
            store = self._stores.get(name)
        if store is None:
            raise StoreNotFoundError(f"Store '{name}' is not registered.")
        return store

    async def list_stores(self) -> Dict[str, VectorStore]:
        async with self._lock:
            return dict(self._stores)

    async def names(self) -> List[str]:
        async with self._lock:
            return list(self._stores.keys())

    async def select(self, *, min_priority: Optional[int] = None, backend: Optional[str] = None) -> VectorStore:
        """Select the highest-priority registered store, optionally filtered
        by minimum priority or backend type."""
        async with self._lock:
            candidates = list(self._stores.items())
        if backend is not None:
            candidates = [(n, s) for n, s in candidates if s.metadata.backend == backend]
        if min_priority is not None:
            candidates = [(n, s) for n, s in candidates if s.priority.value >= min_priority]
        if not candidates:
            raise StoreNotFoundError("No store matches the given criteria.")
        candidates.sort(key=lambda pair: pair[1].priority.value, reverse=True)
        return candidates[0][1]

    async def health_check(self) -> RegistryHealthStatus:
        start = time.monotonic()
        async with self._lock:
            stores = dict(self._stores)

        results: Dict[str, bool] = {}
        for name, store in stores.items():
            try:
                result = await store.health_check()
                results[name] = result.status == HealthStatus.HEALTHY
            except Exception as exc:
                results[name] = False
                logger.warning("Health check failed for store '%s': %s", name, exc)

        latency_ms = (time.monotonic() - start) * 1000
        healthy_count = sum(1 for v in results.values() if v)
        all_healthy = healthy_count == len(results) and len(results) > 0
        return RegistryHealthStatus(
            healthy=all_healthy,
            store_count=len(results),
            healthy_store_count=healthy_count,
            stores=results,
            latency_ms=latency_ms,
            message="ok" if all_healthy else f"{healthy_count}/{len(results)} stores healthy",
        )

    async def shutdown_all(self) -> None:
        async with self._lock:
            stores = list(self._stores.items())
            self._stores.clear()
        for name, store in stores:
            try:
                await store.shutdown()
                logger.info("Shut down store '%s'.", name)
            except Exception as exc:
                logger.error("Error shutting down store '%s': %s", name, exc)
