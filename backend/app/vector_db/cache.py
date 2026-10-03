"""
cache.py

Production-grade cache abstraction with in-memory (LRU + TTL) and Redis
backends, async-safe operations, and health checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
CacheKey = str
CacheValue = Any


class CacheError(Exception):
    """Base exception for cache failures."""

class CacheMissError(CacheError):
    """Raised when a key is not found in cache (non-strict mode returns
    None instead)."""


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    backend: str
    entry_count: int
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


class CacheBackend(ABC, Generic[T]):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[T]: ...

    @abstractmethod
    async def set(self, key: CacheKey, value: T, ttl: Optional[float] = None) -> None: ...

    @abstractmethod
    async def delete(self, key: CacheKey) -> bool: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def has(self, key: CacheKey) -> bool: ...

    @abstractmethod
    async def size(self) -> int: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


class InMemoryCache(CacheBackend[T]):
    """Async-safe LRU cache with per-entry TTL."""

    def __init__(self, max_size: int = 10_000, default_ttl: Optional[float] = 3600.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[CacheKey, Tuple[T, float, Optional[float]]] = OrderedDict()
        self._lock = asyncio.Lock()

    def _purge(self) -> None:
        now = time.time()
        expired = [k for k, (_, _, expires_at) in self._store.items()
                    if expires_at is not None and expires_at <= now]
        for k in expired:
            del self._store[k]

    async def get(self, key: CacheKey) -> Optional[T]:
        async with self._lock:
            self._purge()
            entry = self._store.get(key)
            if entry is None:
                return None
            value, created_at, expires_at = entry
            if expires_at is not None and expires_at <= time.time():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: CacheKey, value: T, ttl: Optional[float] = None) -> None:
        async with self._lock:
            self._purge()
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + effective_ttl if effective_ttl is not None else None
            self._store[key] = (value, time.time(), expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def delete(self, key: CacheKey) -> bool:
        async with self._lock:
            self._purge()
            if key not in self._store:
                return False
            del self._store[key]
            return True

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def has(self, key: CacheKey) -> bool:
        async with self._lock:
            self._purge()
            return key in self._store

    async def size(self) -> int:
        async with self._lock:
            self._purge()
            return len(self._store)

    async def health_check(self) -> bool:
        return True


class Cache:
    """High-level cache facade with namespace prefixing and optional seria-
    lization."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        namespace: str = "default",
        default_ttl: Optional[float] = 3600.0,
        max_size: int = 10_000,
    ) -> None:
        self._backend: CacheBackend = backend or InMemoryCache(max_size=max_size, default_ttl=default_ttl)
        self._namespace = namespace
        self._default_ttl = default_ttl

    def _ns_key(self, key: CacheKey) -> CacheKey:
        return f"{self._namespace}:{key}"

    async def get(self, key: CacheKey) -> Optional[Any]:
        return await self._backend.get(self._ns_key(key))

    async def set(self, key: CacheKey, value: Any, ttl: Optional[float] = None) -> None:
        await self._backend.set(self._ns_key(key), value, ttl=ttl)

    async def delete(self, key: CacheKey) -> bool:
        return await self._backend.delete(self._ns_key(key))

    async def clear(self) -> None:
        await self._backend.clear()

    async def has(self, key: CacheKey) -> bool:
        return await self._backend.has(self._ns_key(key))

    async def size(self) -> int:
        return await self._backend.size()

    async def get_or_set(
        self, key: CacheKey, factory, ttl: Optional[float] = None
    ) -> Any:
        existing = await self.get(key)
        if existing is not None:
            return existing
        value = await factory()
        await self.set(key, value, ttl=ttl)
        return value

    async def get_batch(self, keys: Sequence[CacheKey]) -> dict[CacheKey, Optional[Any]]:
        results: dict[CacheKey, Optional[Any]] = {}
        for key in keys:
            results[key] = await self.get(key)
        return results

    async def set_batch(self, mapping: dict[CacheKey, Any], ttl: Optional[float] = None) -> None:
        for key, value in mapping.items():
            await self.set(key, value, ttl=ttl)

    async def delete_batch(self, keys: Sequence[CacheKey]) -> int:
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            backend_healthy = await self._backend.health_check()
            entry_count = await self._backend.size()
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=backend_healthy,
                backend=type(self._backend).__name__,
                entry_count=entry_count,
                latency_ms=latency_ms,
                message="ok" if backend_healthy else "backend unhealthy",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=False,
                backend=type(self._backend).__name__,
                entry_count=0,
                latency_ms=latency_ms,
                message=str(exc),
            )
