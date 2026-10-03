"""
embeddings.py
=============
Embedding engine for the AI Operating System.

Responsibilities
----------------
- Generate vector embeddings via OpenAI or SentenceTransformers.
- In-memory LRU cache with optional TTL to avoid redundant API calls.
- Batch embedding with configurable concurrency and retry logic.
- Pluggable backend registry: swap or extend without touching callers.
- Future ChromaDB vector-store integration hook.
- Health checks per backend.

Compatible with
---------------
- retriever.py
- long_term.py
- memory_capability.py
- rag_capability.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    embedding: List[float]
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class EmbeddingCache:
    """
    Thread-safe async LRU cache with optional TTL.

    Parameters
    ----------
    max_size: Maximum number of entries (LRU eviction when exceeded).
    ttl:      Time-to-live in seconds; None = never expire.
    """

    def __init__(self, max_size: int = 4096, ttl: Optional[float] = 3600.0) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(text: str, model: str) -> str:
        payload = f"{model}::{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, text: str, model: str) -> Optional[List[float]]:
        k = self._key(text, model)
        async with self._lock:
            entry = self._store.get(k)
            if entry is None:
                self._misses += 1
                return None
            if self._ttl is not None and (time.monotonic() - entry.created_at) > self._ttl:
                del self._store[k]
                self._misses += 1
                return None
            # Move to end (most-recently used)
            self._store.move_to_end(k)
            entry.hit_count += 1
            self._hits += 1
            return entry.embedding

    async def set(self, text: str, model: str, embedding: List[float]) -> None:
        k = self._key(text, model)
        async with self._lock:
            if k in self._store:
                self._store.move_to_end(k)
                self._store[k].embedding = embedding
                return
            self._store[k] = _CacheEntry(embedding=embedding)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def invalidate(self, text: str, model: str) -> None:
        k = self._key(text, model)
        async with self._lock:
            self._store.pop(k, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------


class EmbeddingBackend(ABC):
    """Abstract base for all embedding providers."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    async def embed_one(self, text: str, model: Optional[str] = None) -> List[float]: ...

    @abstractmethod
    async def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]: ...

    @abstractmethod
    async def health(self) -> Dict[str, Any]: ...

    @property
    def dimension(self) -> Optional[int]:
        return None


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIBackend(EmbeddingBackend):
    """
    OpenAI Embeddings API backend (text-embedding-3-small / ada-002 / etc.).

    Parameters
    ----------
    api_key:        OpenAI API key; falls back to OPENAI_API_KEY env var.
    model:          Embedding model name.
    max_retries:    Retry attempts on transient errors.
    retry_delay:    Base delay (seconds) between retries (exponential back-off).
    batch_size:     Max texts per API call.
    timeout:        HTTP request timeout in seconds.
    """

    name = "openai"
    default_model = "nvidia/nv-embed-v1"

    _DIMENSIONS: Dict[str, int] = {
        "nvidia/nv-embed-v1": 1024,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/nv-embed-v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 256,
        timeout: float = 30.0,
    ) -> None:
        import os

        self._api_key = api_key or os.environ.get("NVIDIA_EMBED_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        self.default_model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
            return self._client
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

    async def _call_api(self, texts: List[str], model: str) -> List[List[float]]:
        client = self._ensure_client()
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(self._max_retries):
            try:
                response = await client.embeddings.create(input=texts, model=model)
                return [item.embedding for item in response.data]
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "[OpenAIBackend] Attempt %d failed: %s. Retrying in %.1fs.",
                        attempt + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise last_exc

    async def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self._call_api([text], model or self.default_model)
        return results[0]

    async def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        m = model or self.default_model
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i: i + self._batch_size]
            embeddings.extend(await self._call_api(chunk, m))
        return embeddings

    @property
    def dimension(self) -> Optional[int]:
        return self._DIMENSIONS.get(self.default_model)

    async def health(self) -> Dict[str, Any]:
        try:
            await self.embed_one("health check")
            return {"backend": self.name, "status": "healthy", "model": self.default_model}
        except Exception as exc:
            return {"backend": self.name, "status": "unhealthy", "error": str(exc)}


# ---------------------------------------------------------------------------
# SentenceTransformers backend
# ---------------------------------------------------------------------------


class SentenceTransformersBackend(EmbeddingBackend):
    """
    Local SentenceTransformers backend (no external API required).

    Parameters
    ----------
    model:       HuggingFace model name or local path.
    device:      'cpu' | 'cuda' | 'mps' | None (auto-detect).
    batch_size:  Sentences per encode call.
    normalize:   L2-normalise output vectors.
    """

    name = "sentence_transformers"
    default_model = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.default_model = model
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model: Any = None
        self._dim: Optional[int] = None
        self._current_model: Optional[str] = None
        self._model_lock = asyncio.Lock()

    async def _ensure_model(self, model_name: str) -> Any:
        async with self._model_lock:
            if self._model is not None and self._current_model == model_name:  # type: ignore[union-attr]
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                loop = asyncio.get_event_loop()
                m = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(model_name, device=self._device),
                )
                self._model = m
                self._current_model = model_name
                self._dim = m.get_sentence_embedding_dimension()
                logger.info(
                    "[STBackend] Loaded model=%s dim=%d device=%s",
                    model_name, self._dim, self._device,
                )
                return m
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def _encode(self, model: Any, texts: List[str]) -> List[List[float]]:
        vecs = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    async def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        m = await self._ensure_model(model or self.default_model)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._encode, m, [text])
        return results[0]

    async def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        m = await self._ensure_model(model or self.default_model)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode, m, texts)

    @property
    def dimension(self) -> Optional[int]:
        return self._dim

    async def health(self) -> Dict[str, Any]:
        try:
            await self.embed_one("health check")
            return {
                "backend": self.name,
                "status": "healthy",
                "model": self.default_model,
                "dimension": self._dim,
                "device": self._device,
            }
        except Exception as exc:
            return {"backend": self.name, "status": "unhealthy", "error": str(exc)}


# ---------------------------------------------------------------------------
# Naive fallback backend (no external deps)
# ---------------------------------------------------------------------------


class NaiveBackend(EmbeddingBackend):
    """
    Bag-of-words hash-trick embedder. No dependencies.
    Use only for testing or when no real backend is available.
    """

    name = "naive"
    default_model = "naive-512"
    _DIM = 512

    def _vectorise(self, text: str) -> List[float]:
        import re

        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._DIM
        for token in tokens:
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._vectorise, text)

    async def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: [self._vectorise(t) for t in texts]
        )

    @property
    def dimension(self) -> Optional[int]:
        return self._DIM

    async def health(self) -> Dict[str, Any]:
        return {"backend": self.name, "status": "healthy", "model": self.default_model}


# ---------------------------------------------------------------------------
# ChromaDB vector store hook (Future)
# ---------------------------------------------------------------------------


class ChromaDBStore:
    """
    Future ChromaDB integration for storing and searching embeddings natively.
    Wired into EmbeddingEngine.store_to_chroma() / search_in_chroma().
    """

    def __init__(
        self,
        collection_name: str = "embeddings",
        persist_dir: Optional[str] = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._collection: Any = None

    async def _ensure(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore

            loop = asyncio.get_event_loop()

            def _init() -> Any:
                client = (
                    chromadb.PersistentClient(path=self._persist_dir)
                    if self._persist_dir
                    else chromadb.EphemeralClient()
                )
                return client.get_or_create_collection(self._collection_name)

            self._collection = await loop.run_in_executor(None, _init)
            return self._collection
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        col = await self._ensure()
        loop = asyncio.get_event_loop()
        kwargs: Dict[str, Any] = dict(
            ids=ids, embeddings=embeddings, documents=documents
        )
        if metadatas:
            kwargs["metadatas"] = metadatas
        await loop.run_in_executor(None, lambda: col.upsert(**kwargs))

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        col = await self._ensure()
        loop = asyncio.get_event_loop()
        kwargs: Dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        raw = await loop.run_in_executor(None, lambda: col.query(**kwargs))
        results: List[Dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 1.0,
            })
        return results

    async def delete(self, ids: List[str]) -> None:
        col = await self._ensure()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: col.delete(ids=ids))

    async def health(self) -> bool:
        try:
            await self._ensure()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """
    Unified embedding engine with caching, batching, and multi-backend support.

    Parameters
    ----------
    backend:         Primary EmbeddingBackend instance.
    cache:           EmbeddingCache instance; pass None to disable caching.
    chroma_store:    Optional ChromaDBStore for vector persistence.
    max_concurrency: Max parallel embed_one coroutines during batch processing.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        cache: Optional[EmbeddingCache] = None,
        chroma_store: Optional[ChromaDBStore] = None,
        max_concurrency: int = 16,
    ) -> None:
        self._backend = backend
        self._cache = cache if cache is not None else EmbeddingCache()
        self._chroma = chroma_store
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._total_embedded = 0
        self._total_cached = 0

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Embed a single text string.

        Returns cached vector if available; otherwise calls the backend.
        """
        m = model or self._backend.default_model
        cached = await self._cache.get(text, m)
        if cached is not None:
            self._total_cached += 1
            return cached

        async with self._semaphore:
            # Double-check after acquiring semaphore
            cached = await self._cache.get(text, m)
            if cached is not None:
                self._total_cached += 1
                return cached

            embedding = await self._backend.embed_one(text, m)
            await self._cache.set(text, m, embedding)
            self._total_embedded += 1
            return embedding

    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        *,
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Embed a list of texts efficiently.

        Cache-hit texts are returned immediately; remaining texts are sent to
        the backend in a single batched call, then cached individually.

        Parameters
        ----------
        texts:     Input strings (preserves order in output).
        model:     Override default model.
        use_cache: Whether to consult / populate the cache.

        Returns
        -------
        List of embedding vectors in the same order as *texts*.
        """
        if not texts:
            return []

        m = model or self._backend.default_model
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if use_cache:
            for i, text in enumerate(texts):
                vec = await self._cache.get(text, m)
                if vec is not None:
                    results[i] = vec
                    self._total_cached += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        if uncached_texts:
            async with self._semaphore:
                batch_embeddings = await self._backend.embed_batch(uncached_texts, m)

            for idx, embedding in zip(uncached_indices, batch_embeddings):
                results[idx] = embedding
                if use_cache:
                    await self._cache.set(texts[idx], m, embedding)
                self._total_embedded += 1

        return [r for r in results if r is not None]

    async def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed a list of document dicts in-place.

        Each dict is updated with an 'embedding' key.
        Returns the mutated list.
        """
        texts = [str(doc.get(text_field, "")) for doc in documents]
        embeddings = await self.embed_batch(texts, model)
        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb
        return documents

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Return cosine similarity in [0, 1] between two unit vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return max(0.0, min(1.0, dot / (na * nb)))

    async def most_similar(
        self,
        query: str,
        candidates: List[str],
        model: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return the top_k most similar strings from *candidates* to *query*.

        Returns
        -------
        List of (text, similarity_score) sorted descending.
        """
        all_texts = [query] + candidates
        embeddings = await self.embed_batch(all_texts, model)
        query_vec = embeddings[0]
        candidate_vecs = embeddings[1:]

        scored = [
            (text, self.cosine_similarity(query_vec, vec))
            for text, vec in zip(candidates, candidate_vecs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # ChromaDB integration
    # ------------------------------------------------------------------

    async def store_to_chroma(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> None:
        """Embed *texts* and upsert into ChromaDB."""
        if self._chroma is None:
            raise RuntimeError("No ChromaDBStore configured.")
        embeddings = await self.embed_batch(texts, model)
        await self._chroma.upsert(ids, embeddings, texts, metadatas)

    async def search_in_chroma(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed *query* and search ChromaDB for nearest neighbours."""
        if self._chroma is None:
            raise RuntimeError("No ChromaDBStore configured.")
        query_vec = await self.embed(query, model)
        return await self._chroma.query(query_vec, top_k=top_k, where=where)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def clear_cache(self) -> None:
        await self._cache.clear()
        logger.info("[EmbeddingEngine] Cache cleared.")

    async def invalidate(self, text: str, model: Optional[str] = None) -> None:
        m = model or self._backend.default_model
        await self._cache.invalidate(text, m)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        backend_health = await self._backend.health()
        cache_stats = await self._cache.stats()
        chroma_ok: Optional[bool] = None
        if self._chroma is not None:
            chroma_ok = await self._chroma.health()

        return {
            "status": "healthy" if backend_health.get("status") == "healthy" else "degraded",
            "backend": backend_health,
            "cache": cache_stats,
            "chroma_connected": chroma_ok,
            "total_embedded": self._total_embedded,
            "total_cached_hits": self._total_cached,
            "dimension": self._backend.dimension,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKEND_REGISTRY: Dict[str, type] = {
    "openai": OpenAIBackend,
    "sentence_transformers": SentenceTransformersBackend,
    "naive": NaiveBackend,
}


def register_backend(name: str, cls: type) -> None:
    """Register a custom EmbeddingBackend class under *name*."""
    _BACKEND_REGISTRY[name] = cls
    logger.info("[EmbeddingEngine] Registered backend: %s", name)


def list_backends() -> List[str]:
    return list(_BACKEND_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedding_engine(
    backend: str = "naive",
    *,
    # OpenAI options
    openai_api_key: Optional[str] = None,
    openai_model: str = "nvidia/nv-embed-v1",
    openai_batch_size: int = 256,
    # SentenceTransformers options
    st_model: str = "all-MiniLM-L6-v2",
    st_device: Optional[str] = None,
    st_batch_size: int = 64,
    # Cache options
    cache_max_size: int = 4096,
    cache_ttl: Optional[float] = 3600.0,
    disable_cache: bool = False,
    # ChromaDB options
    chroma_collection: Optional[str] = None,
    chroma_persist_dir: Optional[str] = None,
    # Engine options
    max_concurrency: int = 16,
) -> EmbeddingEngine:
    """
    Build an EmbeddingEngine from named configuration.

    Parameters
    ----------
    backend:           'openai' | 'sentence_transformers' | 'naive' | custom.
    openai_api_key:    API key (or OPENAI_API_KEY env var).
    openai_model:      OpenAI model name.
    openai_batch_size: Texts per OpenAI API call.
    st_model:          SentenceTransformers model name or path.
    st_device:         Torch device ('cpu', 'cuda', 'mps').
    st_batch_size:     Texts per ST encode call.
    cache_max_size:    LRU cache capacity (entries).
    cache_ttl:         Cache TTL in seconds; None = no expiry.
    disable_cache:     Bypass cache entirely.
    chroma_collection: ChromaDB collection name (enables Chroma store).
    chroma_persist_dir: Directory for ChromaDB persistence.
    max_concurrency:   Semaphore width for concurrent embed calls.
    """
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_instance: EmbeddingBackend

    if backend == "openai":
        backend_instance = OpenAIBackend(
            api_key=openai_api_key,
            model=openai_model,
            batch_size=openai_batch_size,
        )
    elif backend == "sentence_transformers":
        backend_instance = SentenceTransformersBackend(
            model=st_model,
            device=st_device,
            batch_size=st_batch_size,
        )
    else:
        backend_instance = _BACKEND_REGISTRY[backend]()

    cache: Optional[EmbeddingCache] = (
        None if disable_cache else EmbeddingCache(max_size=cache_max_size, ttl=cache_ttl)
    )

    chroma: Optional[ChromaDBStore] = None
    if chroma_collection:
        chroma = ChromaDBStore(
            collection_name=chroma_collection,
            persist_dir=chroma_persist_dir,
        )

    logger.info(
        "[EmbeddingEngine] Created — backend=%s cache=%s chroma=%s",
        backend,
        "disabled" if disable_cache else f"lru(max={cache_max_size},ttl={cache_ttl}s)",
        chroma_collection or "none",
    )

    return EmbeddingEngine(
        backend=backend_instance,
        cache=cache,
        chroma_store=chroma,
        max_concurrency=max_concurrency,
    )
