"""
index_manager.py

Production-grade vector index lifecycle manager: create, delete, optimize,
rebuild, and version indexes against a pluggable backend, with async support
and health checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class IndexManagerError(Exception):
    """Base exception for index management failures."""


class IndexAlreadyExistsError(IndexManagerError):
    """Raised when attempting to create an index that already exists."""


class IndexNotFoundError(IndexManagerError):
    """Raised when an operation targets a non-existent index."""


class IndexOperationInProgressError(IndexManagerError):
    """Raised when a conflicting operation is already running on an index."""


# --------------------------------------------------------------------------
# Enums / data classes
# --------------------------------------------------------------------------

class IndexStatus(str, Enum):
    CREATING = "creating"
    READY = "ready"
    OPTIMIZING = "optimizing"
    REBUILDING = "rebuilding"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class IndexConfig:
    dimension: int
    metric: DistanceMetric = DistanceMetric.COSINE
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexMetadata:
    name: str
    version: int
    physical_name: str
    config: IndexConfig
    status: IndexStatus
    vector_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_optimized_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    backend_healthy: bool
    index_count: int
    indexes_in_error: List[str]
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------

class IndexBackend(Protocol):
    """Abstraction over the underlying vector index engine
    (e.g. FAISS, HNSW, Pinecone, Qdrant, pgvector)."""

    async def create(self, physical_name: str, config: IndexConfig) -> None: ...
    async def delete(self, physical_name: str) -> None: ...
    async def optimize(self, physical_name: str) -> None: ...
    async def count(self, physical_name: str) -> int: ...
    async def exists(self, physical_name: str) -> bool: ...
    async def health_check(self) -> bool: ...


RebuildPopulateFn = Callable[[str], Awaitable[None]]
"""A callback that populates a freshly created physical index, given its
physical name. Supplied by the caller of rebuild_index()."""


# --------------------------------------------------------------------------
# IndexManager
# --------------------------------------------------------------------------

class IndexManager:
    """Manages the lifecycle of named, versioned vector indexes on top of a
    pluggable backend. Supports zero-downtime rebuilds via physical
    name versioning (e.g. ``my_index__v2``) with atomic alias promotion."""

    def __init__(self, backend: IndexBackend) -> None:
        self._backend = backend
        self._indexes: Dict[str, IndexMetadata] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    def _lock_for(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    @staticmethod
    def _physical_name(name: str, version: int) -> str:
        return f"{name}__v{version}"

    async def create_index(self, name: str, config: IndexConfig) -> IndexMetadata:
        """Create a brand-new index at version 1."""
        async with self._registry_lock:
            if name in self._indexes:
                raise IndexAlreadyExistsError(f"Index '{name}' already exists.")
            placeholder = IndexMetadata(
                name=name,
                version=1,
                physical_name=self._physical_name(name, 1),
                config=config,
                status=IndexStatus.CREATING,
            )
            self._indexes[name] = placeholder

        lock = self._lock_for(name)
        async with lock:
            try:
                await self._backend.create(placeholder.physical_name, config)
                placeholder.status = IndexStatus.READY
                placeholder.updated_at = time.time()
                logger.info("Created index '%s' (physical='%s').", name, placeholder.physical_name)
            except Exception as exc:  # noqa: BLE001
                placeholder.status = IndexStatus.ERROR
                placeholder.error_message = str(exc)
                logger.error("Failed to create index '%s': %s", name, exc)
                raise IndexManagerError(f"Failed to create index '{name}': {exc}") from exc

        return placeholder

    async def delete_index(self, name: str) -> None:
        """Delete an index and its underlying physical storage."""
        metadata = await self._require_index(name)
        lock = self._lock_for(name)
        async with lock:
            metadata.status = IndexStatus.DELETING
            try:
                await self._backend.delete(metadata.physical_name)
                metadata.status = IndexStatus.DELETED
                metadata.updated_at = time.time()
                logger.info("Deleted index '%s'.", name)
            except Exception as exc:  # noqa: BLE001
                metadata.status = IndexStatus.ERROR
                metadata.error_message = str(exc)
                raise IndexManagerError(f"Failed to delete index '{name}': {exc}") from exc

        async with self._registry_lock:
            self._indexes.pop(name, None)
            self._locks.pop(name, None)

    async def optimize_index(self, name: str) -> IndexMetadata:
        """Trigger backend-specific optimization (e.g. compaction, re-clustering)."""
        metadata = await self._require_index(name)
        lock = self._lock_for(name)
        async with lock:
            if metadata.status != IndexStatus.READY:
                raise IndexOperationInProgressError(
                    f"Cannot optimize index '{name}' while status is '{metadata.status.value}'."
                )
            metadata.status = IndexStatus.OPTIMIZING
            try:
                await self._backend.optimize(metadata.physical_name)
                metadata.status = IndexStatus.READY
                metadata.last_optimized_at = time.time()
                metadata.updated_at = time.time()
                logger.info("Optimized index '%s'.", name)
            except Exception as exc:  # noqa: BLE001
                metadata.status = IndexStatus.ERROR
                metadata.error_message = str(exc)
                raise IndexManagerError(f"Failed to optimize index '{name}': {exc}") from exc
        return metadata

    async def rebuild_index(
        self,
        name: str,
        populate_fn: RebuildPopulateFn,
        config: Optional[IndexConfig] = None,
    ) -> IndexMetadata:
        """Rebuild an index into a new version with zero downtime.

        Creates a new physical index, invokes ``populate_fn`` to fill it with
        vectors, then atomically promotes it as the active version and
        deletes the previous physical index.
        """
        metadata = await self._require_index(name)
        lock = self._lock_for(name)
        async with lock:
            if metadata.status != IndexStatus.READY:
                raise IndexOperationInProgressError(
                    f"Cannot rebuild index '{name}' while status is '{metadata.status.value}'."
                )

            old_physical_name = metadata.physical_name
            new_version = metadata.version + 1
            new_physical_name = self._physical_name(name, new_version)
            new_config = config or metadata.config

            metadata.status = IndexStatus.REBUILDING
            try:
                await self._backend.create(new_physical_name, new_config)
                await populate_fn(new_physical_name)
                new_count = await self._backend.count(new_physical_name)

                # Atomic promotion
                metadata.physical_name = new_physical_name
                metadata.version = new_version
                metadata.config = new_config
                metadata.vector_count = new_count
                metadata.status = IndexStatus.READY
                metadata.updated_at = time.time()

                await self._backend.delete(old_physical_name)
                logger.info(
                    "Rebuilt index '%s' to version %s (%s vectors).",
                    name, new_version, new_count,
                )
            except Exception as exc:  # noqa: BLE001
                metadata.status = IndexStatus.ERROR
                metadata.error_message = str(exc)
                raise IndexManagerError(f"Failed to rebuild index '{name}': {exc}") from exc

        return metadata

    async def get_index(self, name: str) -> IndexMetadata:
        return await self._require_index(name)

    async def list_indexes(self) -> List[IndexMetadata]:
        async with self._registry_lock:
            return list(self._indexes.values())

    async def _require_index(self, name: str) -> IndexMetadata:
        async with self._registry_lock:
            metadata = self._indexes.get(name)
        if metadata is None:
            raise IndexNotFoundError(f"Index '{name}' does not exist.")
        return metadata

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            backend_healthy = await self._backend.health_check()
        except Exception as exc:  # noqa: BLE001
            backend_healthy = False
            logger.error("Backend health check failed: %s", exc)

        async with self._registry_lock:
            indexes_in_error = [
                meta.name for meta in self._indexes.values() if meta.status == IndexStatus.ERROR
            ]
            index_count = len(self._indexes)

        latency_ms = (time.monotonic() - start) * 1000
        healthy = backend_healthy and not indexes_in_error
        message = "ok" if healthy else f"backend_healthy={backend_healthy}, errors={indexes_in_error}"
        return HealthStatus(
            healthy=healthy,
            backend_healthy=backend_healthy,
            index_count=index_count,
            indexes_in_error=indexes_in_error,
            latency_ms=latency_ms,
            message=message,
        )
