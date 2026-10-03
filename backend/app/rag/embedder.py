from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbedderBackend(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    MOCK = "mock"


@dataclass
class EmbeddingResult:
    text: str
    embedding: List[float]
    model: str
    dim: int
    cached: bool = False
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "model": self.model,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class EmbedderConfig:
    backend: EmbedderBackend = EmbedderBackend.OPENAI
    openai_api_key: Optional[str] = None
    openai_model: str = "nvidia/nv-embed-v1"
    openai_dimensions: Optional[int] = None
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_max_size: int = 10_000
    normalize: bool = True
    timeout: float = 30.0


class EmbeddingCache:
    def __init__(self, max_size: int) -> None:
        self._store: Dict[str, List[float]] = {}
        self._max = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        k = self._key(text, model)
        v = self._store.get(k)
        if v is not None:
            self._hits += 1
            return v
        self._misses += 1
        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        if len(self._store) >= self._max:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[self._key(text, model)] = embedding

    def set_batch(self, texts: List[str], model: str, embeddings: List[List[float]]) -> None:
        for text, emb in zip(texts, embeddings):
            self.set(text, model, emb)

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


class OpenAIEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._client: Any = None

    async def _client_instance(self) -> Any:
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self._config.openai_api_key,
                    timeout=self._config.timeout,
                )
            except ImportError as exc:
                raise ImportError("pip install openai") from exc
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        client = await self._client_instance()
        kwargs: Dict[str, Any] = {
            "model": self._config.openai_model,
            "input": texts,
        }
        if self._config.openai_dimensions:
            kwargs["dimensions"] = self._config.openai_dimensions
        resp = await client.embeddings.create(**kwargs)
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    @property
    def model_name(self) -> str:
        return self._config.openai_model

    async def health(self) -> bool:
        try:
            client = await self._client_instance()
            await client.embeddings.create(model=self._config.openai_model, input=["health"])
            return True
        except Exception:
            return False


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._config.sentence_transformer_model)
            except ImportError as exc:
                raise ImportError("pip install sentence-transformers") from exc
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = await asyncio.to_thread(self._load)
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    @property
    def model_name(self) -> str:
        return self._config.sentence_transformer_model

    async def health(self) -> bool:
        try:
            await self.embed(["health"])
            return True
        except Exception:
            return False


class MockEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import math
        results: List[List[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [(b - 128) / 128.0 for b in h]
            while len(vec) < self._dim:
                vec.extend(vec[:self._dim - len(vec)])
            vec = vec[:self._dim]
            mag = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / mag for x in vec])
        return results

    @property
    def model_name(self) -> str:
        return "mock"

    async def health(self) -> bool:
        return True


async def _with_retry(fn: Any, max_retries: int, delay: float) -> Any:
    last: Exception = Exception("No attempts")
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last = exc
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))
    raise last


def _normalize(vec: List[float]) -> List[float]:
    import math
    mag = math.sqrt(sum(x * x for x in vec)) or 1e-10
    return [x / mag for x in vec]


class Embedder:
    def __init__(self, config: Optional[EmbedderConfig] = None) -> None:
        self._config = config or EmbedderConfig()
        self._cache = EmbeddingCache(self._config.cache_max_size)
        self._backend = self._build_backend()
        self._total_embedded = 0

    def _build_backend(self) -> Any:
        b = self._config.backend
        if b == EmbedderBackend.OPENAI:
            return OpenAIEmbedder(self._config)
        if b == EmbedderBackend.SENTENCE_TRANSFORMER:
            return SentenceTransformerEmbedder(self._config)
        return MockEmbedder()

    async def embed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        results = await self.embed_batch([text], metadata=[metadata] if metadata else None)
        return results[0]

    async def embed_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[EmbeddingResult]:
        if not texts:
            return []
        model = self._backend.model_name
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            if self._config.cache_enabled:
                cached = self._cache.get(text, model)
                if cached is not None:
                    results[i] = EmbeddingResult(
                        text=text,
                        embedding=cached,
                        model=model,
                        dim=len(cached),
                        cached=True,
                        metadata=(metadata[i] if metadata else None) or {},
                    )
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            embeddings = await self._embed_in_batches(uncached_texts)
            for idx, (orig_idx, text, emb) in enumerate(
                zip(uncached_indices, uncached_texts, embeddings)
            ):
                if self._config.normalize:
                    emb = _normalize(emb)
                if self._config.cache_enabled:
                    self._cache.set(text, model, emb)
                results[orig_idx] = EmbeddingResult(
                    text=text,
                    embedding=emb,
                    model=model,
                    dim=len(emb),
                    cached=False,
                    metadata=(metadata[orig_idx] if metadata else None) or {},
                )
            self._total_embedded += len(uncached_texts)

        return [r for r in results if r is not None]

    async def _embed_in_batches(self, texts: List[str]) -> List[List[float]]:
        size = self._config.batch_size
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), size):
            batch = texts[i:i + size]
            embeddings = await _with_retry(
                lambda b=batch: self._backend.embed(b),
                self._config.max_retries,
                self._config.retry_delay,
            )
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def cache_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
    ) -> None:
        model = self._backend.model_name
        for text, emb in zip(texts, embeddings):
            self._cache.set(text, model, emb)

    async def health_check(self) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            ok = await self._backend.health()
            latency = (time.monotonic() - t0) * 1000
            return {
                "healthy": ok,
                "backend": self._config.backend.value,
                "model": self._backend.model_name,
                "latency_ms": round(latency, 2),
                "cache": self._cache.stats(),
                "total_embedded": self._total_embedded,
            }
        except Exception as exc:
            return {
                "healthy": False,
                "backend": self._config.backend.value,
                "error": str(exc),
                "latency_ms": (time.monotonic() - t0) * 1000,
            }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self._config.backend.value,
            "model": self._backend.model_name,
            "total_embedded": self._total_embedded,
            "cache": self._cache.stats(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
