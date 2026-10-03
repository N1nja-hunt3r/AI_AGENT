"""
metadata_store.py

Production-grade metadata store with namespace isolation, CRUD operations,
operator-based filtering, lightweight text search, and async health checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

MetadataDict = Dict[str, Any]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class MetadataStoreError(Exception):
    """Base exception for metadata store failures."""


class MetadataNotFoundError(MetadataStoreError):
    """Raised when a requested metadata record does not exist."""


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class MetadataRecord:
    id: str
    namespace: str
    data: MetadataDict
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    backend: str
    namespace_count: int
    record_count: int
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------

class MetadataBackend(Protocol):
    async def upsert(self, namespace: str, record_id: str, data: MetadataDict) -> MetadataRecord: ...
    async def get(self, namespace: str, record_id: str) -> Optional[MetadataRecord]: ...
    async def delete(self, namespace: str, record_id: str) -> bool: ...
    async def delete_batch(self, namespace: str, record_ids: Sequence[str]) -> int: ...
    async def list_all(self, namespace: str) -> List[MetadataRecord]: ...
    async def list_namespaces(self) -> List[str]: ...
    async def count(self, namespace: Optional[str] = None) -> int: ...
    async def health_check(self) -> bool: ...


# --------------------------------------------------------------------------
# In-memory backend (default)
# --------------------------------------------------------------------------

class InMemoryMetadataBackend:
    """Default namespace-partitioned in-memory metadata backend with TTL
    expiration."""

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self._data: Dict[str, Dict[str, MetadataRecord]] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl

    def _purge_expired(self, bucket: Dict[str, MetadataRecord]) -> None:
        now = time.time()
        expired_keys = [k for k, v in bucket.items() if v.expires_at is not None and v.expires_at <= now]
        for k in expired_keys:
            del bucket[k]

    async def upsert(self, namespace: str, record_id: str, data: MetadataDict) -> MetadataRecord:
        async with self._lock:
            bucket = self._data.setdefault(namespace, {})
            self._purge_expired(bucket)
            existing = bucket.get(record_id)
            now = time.time()
            expires_at = None
            ttl = data.pop("_ttl", self._default_ttl)
            if ttl is not None:
                expires_at = now + ttl
            if existing is not None:
                existing.data = {**existing.data, **data}
                existing.updated_at = now
                existing.expires_at = expires_at
                record = existing
            else:
                record = MetadataRecord(
                    id=record_id, namespace=namespace, data=dict(data),
                    created_at=now, updated_at=now, expires_at=expires_at,
                )
                bucket[record_id] = record
            return record

    async def get(self, namespace: str, record_id: str) -> Optional[MetadataRecord]:
        async with self._lock:
            bucket = self._data.get(namespace, {})
            self._purge_expired(bucket)
            record = bucket.get(record_id)
            if record is None:
                return None
            return record

    async def delete(self, namespace: str, record_id: str) -> bool:
        async with self._lock:
            bucket = self._data.get(namespace)
            if bucket is None or record_id not in bucket:
                return False
            del bucket[record_id]
            return True

    async def delete_batch(self, namespace: str, record_ids: Sequence[str]) -> int:
        async with self._lock:
            bucket = self._data.get(namespace)
            if bucket is None:
                return 0
            count = 0
            for rid in record_ids:
                if rid in bucket:
                    del bucket[rid]
                    count += 1
            return count

    async def list_all(self, namespace: str) -> List[MetadataRecord]:
        async with self._lock:
            bucket = self._data.get(namespace, {})
            self._purge_expired(bucket)
            return list(bucket.values())

    async def list_namespaces(self) -> List[str]:
        async with self._lock:
            return list(self._data.keys())

    async def count(self, namespace: Optional[str] = None) -> int:
        async with self._lock:
            if namespace is not None:
                bucket = self._data.get(namespace, {})
                self._purge_expired(bucket)
                return len(bucket)
            total = 0
            for bucket in self._data.values():
                self._purge_expired(bucket)
                total += len(bucket)
            return total

    async def health_check(self) -> bool:
        return True


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda value, target: value == target,
    "$ne": lambda value, target: value != target,
    "$gt": lambda value, target: value is not None and value > target,
    "$gte": lambda value, target: value is not None and value >= target,
    "$lt": lambda value, target: value is not None and value < target,
    "$lte": lambda value, target: value is not None and value <= target,
    "$in": lambda value, target: value in target,
    "$nin": lambda value, target: value not in target,
    "$contains": lambda value, target: isinstance(value, str) and target in value,
    "$exists": lambda value, target: (value is not None) == bool(target),
}


def matches_filter(data: MetadataDict, filters: Optional[MetadataDict]) -> bool:
    if not filters:
        return True
    for key, condition in filters.items():
        value = data.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                fn = _OPERATORS.get(op)
                if fn is None:
                    raise MetadataStoreError(f"Unsupported filter operator: {op}")
                if not fn(value, target):
                    return False
        else:
            if value != condition:
                return False
    return True


# --------------------------------------------------------------------------
# MetadataStore
# --------------------------------------------------------------------------

class MetadataStore:
    """Namespace-aware metadata store with CRUD, filtering, and search."""

    def __init__(self, backend: Optional[MetadataBackend] = None) -> None:
        self._backend: MetadataBackend = backend or InMemoryMetadataBackend()

    async def store_metadata(
        self, record_id: str, data: MetadataDict, namespace: str = "default"
    ) -> MetadataRecord:
        """Create or fully upsert a metadata record."""
        return await self._backend.upsert(namespace, record_id, data)

    async def update_metadata(
        self, record_id: str, data: MetadataDict, namespace: str = "default"
    ) -> MetadataRecord:
        """Partially update an existing metadata record (merge semantics)."""
        existing = await self._backend.get(namespace, record_id)
        if existing is None:
            raise MetadataNotFoundError(
                f"No metadata record '{record_id}' in namespace '{namespace}'."
            )
        return await self._backend.upsert(namespace, record_id, data)

    async def delete_metadata(self, record_id: str, namespace: str = "default") -> None:
        deleted = await self._backend.delete(namespace, record_id)
        if not deleted:
            raise MetadataNotFoundError(
                f"No metadata record '{record_id}' in namespace '{namespace}'."
            )

    async def delete_metadata_batch(self, record_ids: Sequence[str], namespace: str = "default") -> int:
        deleted = await self._backend.delete_batch(namespace, record_ids)
        return deleted

    async def get_metadata(self, record_id: str, namespace: str = "default") -> MetadataRecord:
        record = await self._backend.get(namespace, record_id)
        if record is None:
            raise MetadataNotFoundError(
                f"No metadata record '{record_id}' in namespace '{namespace}'."
            )
        return record

    async def filter_metadata(
        self,
        filters: MetadataDict,
        namespace: str = "default",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[MetadataRecord]:
        """Return records in a namespace matching operator-based filters with
        pagination."""
        records = await self._backend.list_all(namespace)
        matched = [record for record in records if matches_filter(record.data, filters)]
        if offset:
            matched = matched[offset:]
        if limit is not None:
            matched = matched[:limit]
        return matched

    async def search_metadata(
        self,
        query: str,
        namespace: str = "default",
        fields: Optional[Sequence[str]] = None,
        case_sensitive: bool = False,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[MetadataRecord]:
        """Lightweight substring search across string-valued metadata fields
        with pagination."""
        records = await self._backend.list_all(namespace)
        needle = query if case_sensitive else query.lower()
        results: List[MetadataRecord] = []

        for record in records:
            candidate_fields = fields if fields is not None else record.data.keys()
            for field_name in candidate_fields:
                value = record.data.get(field_name)
                if not isinstance(value, str):
                    continue
                haystack = value if case_sensitive else value.lower()
                if needle in haystack:
                    results.append(record)
                    break

        if offset:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    async def list_namespaces(self) -> List[str]:
        return await self._backend.list_namespaces()

    async def list_metadata(self, namespace: str = "default", offset: int = 0, limit: Optional[int] = None) -> List[MetadataRecord]:
        records = await self._backend.list_all(namespace)
        if offset:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            backend_healthy = await self._backend.health_check()
            namespaces = await self._backend.list_namespaces()
            record_count = await self._backend.count()
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=backend_healthy,
                backend=type(self._backend).__name__,
                namespace_count=len(namespaces),
                record_count=record_count,
                latency_ms=latency_ms,
                message="ok" if backend_healthy else "backend unhealthy",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=False,
                backend=type(self._backend).__name__,
                namespace_count=0,
                record_count=0,
                latency_ms=latency_ms,
                message=str(exc),
            )
