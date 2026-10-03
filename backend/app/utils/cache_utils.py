"""
cache_utils.py

Caching utilities: an in-memory TTL+LRU cache, an optional Redis-backed
cache wrapper, and decorators for memoizing sync/async function results.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    value: Any
    expires_at: Optional[float]

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() > self.expires_at


class RedisLike(Protocol):
    """Minimal protocol describing the subset of Redis client methods used."""

    def get(self, key: str) -> Optional[bytes]: ...

    def set(self, key: str, value: bytes, ex: Optional[int] = None) -> Any: ...

    def delete(self, key: str) -> Any: ...

    def exists(self, key: str) -> Any: ...


class LRUTTLCache:
    """Thread-safe in-memory cache combining LRU eviction with per-key TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return default
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return default
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl if effective_ttl else None
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def contains(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._store[key]
                return False
            return True

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class RedisCache:
    """Thin wrapper exposing a uniform cache interface over a Redis client."""

    def __init__(self, client: RedisLike, prefix: str = "cache:", default_ttl: Optional[int] = None) -> None:
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            raw = self._client.get(self._full_key(key))
        except Exception:  # noqa: BLE001
            logger.exception("RedisCache.get failed for key=%s", key)
            return default
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            payload = json.dumps(value).encode("utf-8")
            self._client.set(self._full_key(key), payload, ex=effective_ttl)
        except Exception:  # noqa: BLE001
            logger.exception("RedisCache.set failed for key=%s", key)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._full_key(key))
        except Exception:  # noqa: BLE001
            logger.exception("RedisCache.delete failed for key=%s", key)

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(self._full_key(key)))
        except Exception:  # noqa: BLE001
            logger.exception("RedisCache.exists failed for key=%s", key)
            return False


def _make_cache_key(prefix: str, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    raw = json.dumps({"args": args, "kwargs": kwargs}, default=str, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def cached(
    cache: Optional[LRUTTLCache] = None,
    ttl: Optional[float] = None,
    key_prefix: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: memoize a synchronous function's results in an LRUTTLCache."""
    local_cache = cache or LRUTTLCache()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        prefix = key_prefix or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_key = _make_cache_key(prefix, args, kwargs)
            sentinel = object()
            cached_value = local_cache.get(cache_key, sentinel)
            if cached_value is not sentinel:
                return cached_value  # type: ignore[return-value]
            result = func(*args, **kwargs)
            local_cache.set(cache_key, result, ttl=ttl)
            return result

        wrapper.cache = local_cache  # type: ignore[attr-defined]
        wrapper.cache_clear = local_cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_cached(
    cache: Optional[LRUTTLCache] = None,
    ttl: Optional[float] = None,
    key_prefix: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: memoize an async function's results in an LRUTTLCache."""
    local_cache = cache or LRUTTLCache()
    locks: Dict[str, asyncio.Lock] = {}
    locks_guard = threading.Lock()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        prefix = key_prefix or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = _make_cache_key(prefix, args, kwargs)
            sentinel = object()
            cached_value = local_cache.get(cache_key, sentinel)
            if cached_value is not sentinel:
                return cached_value

            with locks_guard:
                lock = locks.setdefault(cache_key, asyncio.Lock())

            async with lock:
                cached_value = local_cache.get(cache_key, sentinel)
                if cached_value is not sentinel:
                    return cached_value
                result = await func(*args, **kwargs)
                local_cache.set(cache_key, result, ttl=ttl)
                return result

        wrapper.cache = local_cache  # type: ignore[attr-defined]
        wrapper.cache_clear = local_cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator
