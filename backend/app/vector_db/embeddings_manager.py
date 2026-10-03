"""
embeddings_manager.py

Production-grade embeddings manager supporting OpenAI, SentenceTransformers,
and Ollama embedding providers with caching, batching, retries, async
support, and health checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

EmbeddingVector = List[float]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class EmbeddingError(Exception):
    """Base exception for embedding-related failures."""


class ProviderNotAvailableError(EmbeddingError):
    """Raised when the requested embedding provider's dependencies are missing."""


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request exceeds the configured timeout."""


class EmbeddingRetryExhaustedError(EmbeddingError):
    """Raised when all retry attempts for an embedding request have failed."""


# --------------------------------------------------------------------------
# Enums / Data classes
# --------------------------------------------------------------------------

class EmbeddingProviderType(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    embedding: EmbeddingVector
    model: str
    provider: EmbeddingProviderType
    dimension: int
    cached: bool = False
    latency_ms: float = 0.0


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    provider: EmbeddingProviderType
    model: str
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    jitter: float = 0.25
    retryable_exceptions: Tuple[type, ...] = (EmbeddingError, ConnectionError, TimeoutError, OSError)


# --------------------------------------------------------------------------
# Cache protocol + default in-memory implementation
# --------------------------------------------------------------------------

class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[EmbeddingVector]: ...
    async def set(self, key: str, value: EmbeddingVector, ttl: Optional[int] = None) -> None: ...
    async def health_check(self) -> bool: ...


class InMemoryTTLCache:
    """Simple async-safe LRU cache with per-entry TTL, used as the default
    cache backend when no external cache (e.g. Redis) is supplied."""

    def __init__(self, max_size: int = 10_000, default_ttl: Optional[int] = 3600) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: "OrderedDict[str, Tuple[EmbeddingVector, Optional[float]]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[EmbeddingVector]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at < time.time():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: EmbeddingVector, ttl: Optional[int] = None) -> None:
        async with self._lock:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + effective_ttl if effective_ttl else None
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def health_check(self) -> bool:
        return True

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)


# --------------------------------------------------------------------------
# Provider base class
# --------------------------------------------------------------------------

class BaseEmbeddingProvider(ABC):
    provider_type: EmbeddingProviderType
    model: str

    @abstractmethod
    async def embed_one(self, text: str) -> EmbeddingVector: ...

    @abstractmethod
    async def embed_many(self, texts: Sequence[str]) -> List[EmbeddingVector]: ...

    @abstractmethod
    async def health_check(self) -> HealthStatus: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


# --------------------------------------------------------------------------
# OpenAI provider
# --------------------------------------------------------------------------

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    provider_type = EmbeddingProviderType.OPENAI

    def __init__(
        self,
        model: str = "nvidia/nv-embed-v1",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "openai package is not installed. Run `pip install openai`."
            ) from exc

        self.model = model
        self._dimension = dimension
        self._timeout = timeout
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_one(self, text: str) -> EmbeddingVector:
        result = await self.embed_many([text])
        return result[0]

    async def embed_many(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            await self.embed_one("health check probe")
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(True, self.provider_type, self.model, latency_ms, "ok")
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(False, self.provider_type, self.model, latency_ms, str(exc))


# --------------------------------------------------------------------------
# SentenceTransformers provider (local, CPU/GPU bound -> run in executor)
# --------------------------------------------------------------------------

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    provider_type = EmbeddingProviderType.SENTENCE_TRANSFORMERS

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "sentence-transformers package is not installed. "
                "Run `pip install sentence-transformers`."
            ) from exc

        self.model = model
        self._st_model = SentenceTransformer(model, device=device)
        self._dimension = int(self._st_model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_sync(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        embeddings = self._st_model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return [vector.tolist() for vector in embeddings]

    async def embed_one(self, text: str) -> EmbeddingVector:
        result = await self.embed_many([text])
        return result[0]

    async def embed_many(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            await self.embed_one("health check probe")
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(True, self.provider_type, self.model, latency_ms, "ok")
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(False, self.provider_type, self.model, latency_ms, str(exc))


# --------------------------------------------------------------------------
# Ollama provider
# --------------------------------------------------------------------------

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    provider_type = EmbeddingProviderType.OLLAMA

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "httpx package is not installed. Run `pip install httpx`."
            ) from exc

        self.model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._httpx = httpx
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingError("Dimension unknown until first embed call for Ollama provider.")
        return self._dimension

    async def embed_one(self, text: str) -> EmbeddingVector:
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._host}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            payload = response.json()
            embedding = payload.get("embedding")
            if embedding is None:
                raise EmbeddingError(f"Ollama returned no embedding for model '{self.model}'.")
            self._dimension = len(embedding)
            return embedding

    async def embed_many(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        # Ollama's embeddings endpoint is single-input; fan out concurrently.
        results = await asyncio.gather(*(self.embed_one(text) for text in texts))
        return list(results)

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            async with self._httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._host}/api/tags")
                response.raise_for_status()
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(True, self.provider_type, self.model, latency_ms, "ok")
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(False, self.provider_type, self.model, latency_ms, str(exc))


# --------------------------------------------------------------------------
# Retry helper
# --------------------------------------------------------------------------

async def _with_retry(
    func,
    retry_config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retry_config.retryable_exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == retry_config.max_retries:
                break
            delay = min(retry_config.base_delay * (2 ** attempt), retry_config.max_delay)
            delay += random.uniform(0, retry_config.jitter)
            logger.warning(
                "Embedding call failed (attempt %s/%s): %s. Retrying in %.2fs.",
                attempt + 1,
                retry_config.max_retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise EmbeddingRetryExhaustedError(
        f"Exhausted {retry_config.max_retries} retries: {last_exc}"
    ) from last_exc


# --------------------------------------------------------------------------
# EmbeddingsManager
# --------------------------------------------------------------------------

class EmbeddingsManager:
    """Coordinates embedding generation across providers with caching,
    batching, retries, and health monitoring."""

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        cache: Optional[CacheBackend] = None,
        retry_config: Optional[RetryConfig] = None,
        cache_ttl: Optional[int] = 3600,
        default_batch_size: int = 100,
    ) -> None:
        self._provider = provider
        self._cache: CacheBackend = cache or InMemoryTTLCache(default_ttl=cache_ttl)
        self._retry_config = retry_config or RetryConfig()
        self._cache_ttl = cache_ttl
        self._default_batch_size = default_batch_size

    @staticmethod
    def _cache_key(text: str, model: str, provider: EmbeddingProviderType) -> str:
        digest = hashlib.sha256(f"{provider.value}:{model}:{text}".encode("utf-8")).hexdigest()
        return f"emb:{digest}"

    async def embed(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """Embed a single text, optionally using the cache."""
        start = time.monotonic()
        key = self._cache_key(text, self._provider.model, self._provider.provider_type)

        if use_cache:
            cached_vector = await self._cache.get(key)
            if cached_vector is not None:
                latency_ms = (time.monotonic() - start) * 1000
                return EmbeddingResult(
                    text=text,
                    embedding=cached_vector,
                    model=self._provider.model,
                    provider=self._provider.provider_type,
                    dimension=len(cached_vector),
                    cached=True,
                    latency_ms=latency_ms,
                )

        vector = await _with_retry(self._provider.embed_one, self._retry_config, text)

        if use_cache:
            await self._cache.set(key, vector, ttl=self._cache_ttl)

        latency_ms = (time.monotonic() - start) * 1000
        return EmbeddingResult(
            text=text,
            embedding=vector,
            model=self._provider.model,
            provider=self._provider.provider_type,
            dimension=len(vector),
            cached=False,
            latency_ms=latency_ms,
        )

    async def embed_batch(
        self,
        texts: Sequence[str],
        use_cache: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """Embed multiple texts, batching uncached items for provider calls
        while preserving original order."""
        if not texts:
            return []

        effective_batch_size = batch_size or self._default_batch_size
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        to_fetch: List[Tuple[int, str]] = []

        if use_cache:
            for idx, text in enumerate(texts):
                key = self._cache_key(text, self._provider.model, self._provider.provider_type)
                cached_vector = await self._cache.get(key)
                if cached_vector is not None:
                    results[idx] = EmbeddingResult(
                        text=text,
                        embedding=cached_vector,
                        model=self._provider.model,
                        provider=self._provider.provider_type,
                        dimension=len(cached_vector),
                        cached=True,
                    )
                else:
                    to_fetch.append((idx, text))
        else:
            to_fetch = list(enumerate(texts))

        for start_idx in range(0, len(to_fetch), effective_batch_size):
            chunk = to_fetch[start_idx : start_idx + effective_batch_size]
            chunk_texts = [text for _, text in chunk]
            start = time.monotonic()
            vectors = await _with_retry(self._provider.embed_many, self._retry_config, chunk_texts)
            latency_ms = (time.monotonic() - start) * 1000

            for (idx, text), vector in zip(chunk, vectors):
                results[idx] = EmbeddingResult(
                    text=text,
                    embedding=vector,
                    model=self._provider.model,
                    provider=self._provider.provider_type,
                    dimension=len(vector),
                    cached=False,
                    latency_ms=latency_ms / max(len(chunk), 1),
                )
                if use_cache:
                    key = self._cache_key(text, self._provider.model, self._provider.provider_type)
                    await self._cache.set(key, vector, ttl=self._cache_ttl)

        return [result for result in results if result is not None]

    async def cache_embeddings(
        self,
        texts: Sequence[str],
        embeddings: Sequence[EmbeddingVector],
        ttl: Optional[int] = None,
    ) -> None:
        """Pre-populate the cache with externally computed embeddings."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must be the same length.")
        effective_ttl = ttl if ttl is not None else self._cache_ttl
        for text, vector in zip(texts, embeddings):
            key = self._cache_key(text, self._provider.model, self._provider.provider_type)
            await self._cache.set(key, list(vector), ttl=effective_ttl)

    async def health_check(self) -> HealthStatus:
        """Check provider and cache health."""
        provider_status = await self._provider.health_check()
        cache_ok = await self._cache.health_check()
        if not cache_ok:
            return HealthStatus(
                healthy=False,
                provider=provider_status.provider,
                model=provider_status.model,
                latency_ms=provider_status.latency_ms,
                message=f"{provider_status.message}; cache backend unhealthy",
            )
        return provider_status

    @property
    def dimension(self) -> int:
        return self._provider.dimension
