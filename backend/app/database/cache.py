"""Caching layer with in-memory TTL/LRU backend and optional Redis backend."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def health_check(self) -> bool: ...


@dataclass
class CacheConfig:
    backend: str = os.getenv("CACHE_BACKEND", "memory")
    default_ttl_seconds: int = int(os.getenv("CACHE_DEFAULT_TTL", "300"))
    max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class _Entry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: Optional[float]) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() > self.expires_at


class InMemoryCache:
    """Async-safe in-memory cache with TTL expiry and LRU eviction."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._store: "OrderedDict[str, _Entry]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self.config.default_ttl_seconds
        expires_at = time.monotonic() + ttl if ttl > 0 else None
        async with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self.config.max_size:
                self._store.popitem(last=False)
            self._store[key] = _Entry(value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def evict_expired(self) -> int:
        """Remove all expired entries, returning the count removed."""
        async with self._lock:
            expired = [k for k, v in self._store.items() if v.is_expired()]
            for key in expired:
                del self._store[key]
            return len(expired)

    async def health_check(self) -> bool:
        try:
            probe_key = "__health_check__"
            await self.set(probe_key, "ok", ttl=5)
            value = await self.get(probe_key)
            await self.delete(probe_key)
            return value == "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("In-memory cache health check failed: %s", exc)
            return False


class RedisCache:
    """Async Redis-backed cache implementing the same interface as InMemoryCache."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis  # optional dependency, imported lazily

            self._client = redis.from_url(self.config.redis_url, decode_responses=True)
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        import json

        client = self._get_client()
        raw = await client.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        import json

        ttl = ttl if ttl is not None else self.config.default_ttl_seconds
        client = self._get_client()
        await client.set(key, json.dumps(value), ex=ttl if ttl > 0 else None)

    async def delete(self, key: str) -> None:
        client = self._get_client()
        await client.delete(key)

    async def clear(self) -> None:
        client = self._get_client()
        await client.flushdb()

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            return bool(await client.ping())
        except Exception as exc:  # noqa: BLE001
            logger.error("Redis cache health check failed: %s", exc)
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def get_cache(config: Optional[CacheConfig] = None) -> CacheBackend:
    """Factory returning the configured cache backend (memory or redis)."""
    cfg = config or CacheConfig()
    if cfg.backend == "redis":
        return RedisCache(cfg)
    return InMemoryCache(cfg)


_cache_instance: Optional[CacheBackend] = None


def get_cache_singleton() -> CacheBackend:
    """Module-level singleton accessor for the configured cache backend."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = get_cache()
    return _cache_instance
