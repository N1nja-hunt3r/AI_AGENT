"""
memory_connector.py - Production-grade memory connector with multi-backend support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

try:
    import chromadb

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    aioredis = None  # type: ignore

try:
    import pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False
    pinecone = None  # type: ignore

try:
    import weaviate
    _WEAVIATE_AVAILABLE = True
except ImportError:
    _WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    _OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    _OPENAI_EMBEDDINGS_AVAILABLE = False
    _AsyncOpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class MemoryBackend(str, Enum):
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class EmbeddingProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    SIMPLE_HASH = "simple_hash"


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    RECENCY = "recency"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"
    EXACT = "exact"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: MemoryTier = MemoryTier.SHORT_TERM
    embedding: Optional[List[float]] = field(default=None, repr=False)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # seconds; None = permanent
    tags: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        self.accessed_at = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "tier": self.tier.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "tags": self.tags,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        entry = cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            tier=MemoryTier(data.get("tier", MemoryTier.SHORT_TERM.value)),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", time.time()),
            accessed_at=data.get("accessed_at", time.time()),
            access_count=data.get("access_count", 0),
            ttl=data.get("ttl"),
            tags=data.get("tags", []),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
        )
        return entry


@dataclass
class RetrievalResult:
    entry: MemoryEntry
    score: float
    strategy: RetrievalStrategy


@dataclass
class MemoryStats:
    total_entries: int = 0
    short_term_count: int = 0
    long_term_count: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    working_count: int = 0
    total_retrievals: int = 0
    total_stores: int = 0
    total_deletes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    backend: str = ""
    embedding_provider: str = ""


@dataclass
class MemoryConfig:
    backend: MemoryBackend = MemoryBackend.IN_MEMORY
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SIMPLE_HASH
    short_term_ttl: float = 3600.0          # 1 hour
    long_term_ttl: Optional[float] = None   # permanent
    max_short_term_entries: int = 1000
    max_long_term_entries: int = 100_000
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "nvidia/nv-embed-v1"
    openai_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "memory:"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "agent_memory"
    chroma_in_memory: bool = True
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "agent-memory"
    pinecone_environment: str = "us-east-1"
    weaviate_url: str = "http://localhost:8080"
    weaviate_class: str = "AgentMemory"
    similarity_threshold: float = 0.5
    default_top_k: int = 5
    enable_importance_decay: bool = True
    importance_decay_rate: float = 0.01
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MemoryConnectorError(Exception):
    """Base memory connector error."""


class MemoryBackendError(MemoryConnectorError):
    """Backend-specific error."""


class EmbeddingError(MemoryConnectorError):
    """Embedding generation error."""


class MemoryNotFoundError(MemoryConnectorError):
    """Memory entry not found."""


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class _EmbeddingEngine:
    """Unified embedding engine supporting multiple providers."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._st_model: Any = None
        self._openai_client: Any = None
        self._embedding_dim: int = 384
        self._initialized = False

    async def initialize(self) -> None:
        provider = self.config.embedding_provider
        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning(
                    "sentence-transformers not available; falling back to simple_hash"
                )
                self.config = MemoryConfig(
                    **{**self.config.__dict__, "embedding_provider": EmbeddingProvider.SIMPLE_HASH}
                )
            else:
                self._st_model = await asyncio.to_thread(
                    SentenceTransformer, self.config.embedding_model
                )
                self._embedding_dim = self._st_model.get_sentence_embedding_dimension()
                logger.info("SentenceTransformer loaded: %s dim=%d", self.config.embedding_model, self._embedding_dim)

        elif provider == EmbeddingProvider.OPENAI:
            if not _OPENAI_EMBEDDINGS_AVAILABLE:
                logger.warning("openai not available; falling back to simple_hash")
                self.config = MemoryConfig(
                    **{**self.config.__dict__, "embedding_provider": EmbeddingProvider.SIMPLE_HASH}
                )
            else:
                api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
                self._openai_client = _AsyncOpenAI(api_key=api_key)
                self._embedding_dim = 1536
                logger.info("OpenAI embeddings configured: %s", self.config.openai_embedding_model)

        elif provider == EmbeddingProvider.SIMPLE_HASH:
            self._embedding_dim = 128
            logger.info("Using simple hash embeddings (dim=128)")

        self._initialized = True

    async def embed(self, text: str) -> List[float]:
        if not self._initialized:
            await self.initialize()

        provider = self.config.embedding_provider
        try:
            if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS and self._st_model:
                vec = await asyncio.to_thread(self._st_model.encode, text)
                return vec.tolist() if hasattr(vec, "tolist") else list(vec)

            elif provider == EmbeddingProvider.OPENAI and self._openai_client:
                response = await self._openai_client.embeddings.create(
                    model=self.config.openai_embedding_model,
                    input=text,
                )
                return response.data[0].embedding

            else:
                return self._hash_embed(text)

        except Exception as exc:
            logger.warning("Embedding failed: %s; falling back to hash", exc)
            return self._hash_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()

        provider = self.config.embedding_provider
        try:
            if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS and self._st_model:
                vecs = await asyncio.to_thread(self._st_model.encode, texts)
                return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

            elif provider == EmbeddingProvider.OPENAI and self._openai_client:
                response = await self._openai_client.embeddings.create(
                    model=self.config.openai_embedding_model,
                    input=texts,
                )
                return [d.embedding for d in response.data]

            else:
                return [self._hash_embed(t) for t in texts]

        except Exception as exc:
            logger.warning("Batch embedding failed: %s; falling back to hash", exc)
            return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str, dim: int = 128) -> List[float]:
        """Deterministic pseudo-embedding via SHA256 hashing."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        extended = (digest * ((dim * 4 // len(digest)) + 2))[: dim * 4]
        floats: List[float] = []
        for i in range(0, len(extended) - 3, 4):
            val = int.from_bytes(extended[i : i + 4], "little", signed=True)
            floats.append(val / 2_147_483_648.0)
        vec = floats[:dim]
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        if _NUMPY_AVAILABLE:
            va, vb = np.array(a), np.array(b)
            denom = (np.linalg.norm(va) * np.linalg.norm(vb))
            return float(np.dot(va, vb) / denom) if denom else 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if (na * nb) else 0.0

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

class _InMemoryBackend:
    """Thread-safe in-memory storage with LRU-style eviction."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._store: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

    async def store(self, entry: MemoryEntry) -> None:
        async with self._lock:
            tier = entry.tier
            tier_entries = [e for e in self._store.values() if e.tier == tier]
            limit = (
                self.config.max_short_term_entries
                if tier == MemoryTier.SHORT_TERM
                else self.config.max_long_term_entries
            )
            if len(tier_entries) >= limit:
                # Evict oldest accessed
                oldest = min(tier_entries, key=lambda e: e.accessed_at)
                self._store.pop(oldest.id, None)
            self._store[entry.id] = entry

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        async with self._lock:
            entry = self._store.get(memory_id)
            if entry and not entry.is_expired():
                entry.touch()
                return entry
            if entry and entry.is_expired():
                del self._store[memory_id]
            return None

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:
            return self._store.pop(memory_id, None) is not None

    async def list_entries(
        self,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        async with self._lock:
            entries = list(self._store.values())
        results = []
        for e in entries:
            if e.is_expired():
                continue
            if tier and e.tier != tier:
                continue
            if session_id and e.session_id != session_id:
                continue
            if agent_id and e.agent_id != agent_id:
                continue
            if tags and not any(t in e.tags for t in tags):
                continue
            results.append(e)
        results.sort(key=lambda e: e.accessed_at, reverse=True)
        return results[:limit]

    async def clear(self, tier: Optional[MemoryTier] = None) -> int:
        async with self._lock:
            if tier is None:
                count = len(self._store)
                self._store.clear()
                return count
            to_remove = [k for k, v in self._store.items() if v.tier == tier]
            for k in to_remove:
                del self._store[k]
            return len(to_remove)

    async def count(self, tier: Optional[MemoryTier] = None) -> int:
        async with self._lock:
            if tier is None:
                return len(self._store)
            return sum(1 for e in self._store.values() if e.tier == tier)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _RedisBackend:
    """Redis-backed persistent memory storage."""

    def __init__(self, config: MemoryConfig) -> None:
        if not _REDIS_AVAILABLE:
            raise MemoryConnectorError("redis package not installed. pip install redis")
        self.config = config
        self._client: Any = None
        self._prefix = config.redis_prefix

    async def _get_client(self) -> Any:
        if self._client is None:
            self._client = aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _key(self, memory_id: str) -> str:
        return f"{self._prefix}{memory_id}"

    def _index_key(self, tier: MemoryTier) -> str:
        return f"{self._prefix}index:{tier.value}"

    async def store(self, entry: MemoryEntry) -> None:
        client = await self._get_client()
        data = entry.to_dict()
        if entry.embedding:
            data["embedding"] = json.dumps(entry.embedding)
        serialized = json.dumps(data)
        key = self._key(entry.id)
        pipe = client.pipeline()
        if entry.ttl:
            pipe.setex(key, int(entry.ttl), serialized)
        else:
            pipe.set(key, serialized)
        pipe.sadd(self._index_key(entry.tier), entry.id)
        if entry.session_id:
            pipe.sadd(f"{self._prefix}session:{entry.session_id}", entry.id)
        if entry.agent_id:
            pipe.sadd(f"{self._prefix}agent:{entry.agent_id}", entry.id)
        await pipe.execute()

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        client = await self._get_client()
        raw = await client.get(self._key(memory_id))
        if not raw:
            return None
        data = json.loads(raw)
        embedding_raw = data.pop("embedding", None)
        entry = MemoryEntry.from_dict(data)
        if embedding_raw:
            entry.embedding = json.loads(embedding_raw)
        entry.touch()
        await client.set(self._key(memory_id), json.dumps({**entry.to_dict(), "embedding": embedding_raw}))
        return entry

    async def delete(self, memory_id: str) -> bool:
        client = await self._get_client()
        existed = await client.delete(self._key(memory_id))
        for tier in MemoryTier:
            await client.srem(self._index_key(tier), memory_id)
        return bool(existed)

    async def list_entries(
        self,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        client = await self._get_client()
        if session_id:
            ids = await client.smembers(f"{self._prefix}session:{session_id}")
        elif agent_id:
            ids = await client.smembers(f"{self._prefix}agent:{agent_id}")
        elif tier:
            ids = await client.smembers(self._index_key(tier))
        else:
            ids = set()
            for t in MemoryTier:
                ids |= await client.smembers(self._index_key(t))

        entries: List[MemoryEntry] = []
        for mid in list(ids)[:limit * 2]:
            entry = await self.get(mid)
            if entry:
                if tier and entry.tier != tier:
                    continue
                if tags and not any(t in entry.tags for t in tags):
                    continue
                entries.append(entry)
        entries.sort(key=lambda e: e.accessed_at, reverse=True)
        return entries[:limit]

    async def clear(self, tier: Optional[MemoryTier] = None) -> int:
        client = await self._get_client()
        tiers = [tier] if tier else list(MemoryTier)
        count = 0
        for t in tiers:
            ids = await client.smembers(self._index_key(t))
            for mid in ids:
                await client.delete(self._key(mid))
                count += 1
            await client.delete(self._index_key(t))
        return count

    async def count(self, tier: Optional[MemoryTier] = None) -> int:
        client = await self._get_client()
        if tier:
            return await client.scard(self._index_key(tier))
        total = 0
        for t in MemoryTier:
            total += await client.scard(self._index_key(t))
        return total

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class _ChromaBackend:
    """ChromaDB-backed vector storage."""

    def __init__(self, config: MemoryConfig) -> None:
        if not _CHROMA_AVAILABLE:
            raise MemoryConnectorError("chromadb not installed. pip install chromadb")
        self.config = config
        self._client: Any = None
        self._collection: Any = None

    async def _initialize(self) -> None:
        if self._client is not None:
            return
        if self.config.chroma_in_memory:
            self._client = await asyncio.to_thread(chromadb.EphemeralClient)
        else:
            self._client = await asyncio.to_thread(
                chromadb.HttpClient,
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def store(self, entry: MemoryEntry) -> None:
        await self._initialize()
        meta = {
            **entry.to_dict(),
            "tier": entry.tier.value,
            "tags": json.dumps(entry.tags),
            "metadata": json.dumps(entry.metadata),
        }
        # Remove non-scalar fields
        for k in ["embedding"]:
            meta.pop(k, None)
        embedding = entry.embedding or []
        if embedding:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[entry.id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[meta],
            )
        else:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[meta],
            )

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        await self._initialize()
        try:
            result = await asyncio.to_thread(
                self._collection.get,
                ids=[memory_id],
                include=["documents", "metadatas", "embeddings"],
            )
            if not result["ids"]:
                return None
            meta = result["metadatas"][0]
            meta["tier"] = meta.get("tier", MemoryTier.SHORT_TERM.value)
            meta["tags"] = json.loads(meta.get("tags", "[]"))
            meta["metadata"] = json.loads(meta.get("metadata", "{}"))
            entry = MemoryEntry.from_dict(meta)
            entry.content = result["documents"][0]
            embeddings = result.get("embeddings")
            if embeddings and embeddings[0]:
                entry.embedding = list(embeddings[0])
            entry.touch()
            return entry
        except Exception:
            return None

    async def delete(self, memory_id: str) -> bool:
        await self._initialize()
        try:
            await asyncio.to_thread(self._collection.delete, ids=[memory_id])
            return True
        except Exception:
            return False

    async def list_entries(
        self,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        await self._initialize()
        where: Dict[str, Any] = {}
        if tier:
            where["tier"] = tier.value
        if session_id:
            where["session_id"] = session_id
        if agent_id:
            where["agent_id"] = agent_id
        try:
            kwargs: Dict[str, Any] = {
                "include": ["documents", "metadatas"],
                "limit": limit,
            }
            if where:
                kwargs["where"] = where
            result = await asyncio.to_thread(self._collection.get, **kwargs)
            entries = []
            for idx, mid in enumerate(result["ids"]):
                meta = result["metadatas"][idx]
                meta["id"] = mid
                meta["tier"] = meta.get("tier", MemoryTier.SHORT_TERM.value)
                meta["tags"] = json.loads(meta.get("tags", "[]"))
                meta["metadata"] = json.loads(meta.get("metadata", "{}"))
                entry = MemoryEntry.from_dict(meta)
                entry.content = result["documents"][idx]
                if tags and not any(t in entry.tags for t in tags):
                    continue
                entries.append(entry)
            return entries
        except Exception as exc:
            logger.warning("ChromaDB list_entries failed: %s", exc)
            return []

    async def query_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        tier: Optional[MemoryTier] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        await self._initialize()
        filter_dict: Dict[str, Any] = where or {}
        if tier:
            filter_dict["tier"] = tier.value
        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if filter_dict:
                kwargs["where"] = filter_dict
            result = await asyncio.to_thread(self._collection.query, **kwargs)
            entries = []
            for idx, mid in enumerate(result["ids"][0]):
                meta = result["metadatas"][0][idx]
                meta["id"] = mid
                meta["tier"] = meta.get("tier", MemoryTier.SHORT_TERM.value)
                meta["tags"] = json.loads(meta.get("tags", "[]"))
                meta["metadata"] = json.loads(meta.get("metadata", "{}"))
                entry = MemoryEntry.from_dict(meta)
                entry.content = result["documents"][0][idx]
                distance = result["distances"][0][idx]
                score = 1.0 - distance  # cosine distance -> similarity
                entries.append((entry, score))
            return entries
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

    async def clear(self, tier: Optional[MemoryTier] = None) -> int:
        await self._initialize()
        try:
            if tier is None:
                count = self._collection.count()
                self._client.delete_collection(self.config.chroma_collection)
                self._collection = await asyncio.to_thread(
                    self._client.get_or_create_collection,
                    name=self.config.chroma_collection,
                    metadata={"hnsw:space": "cosine"},
                )
                return count
            result = await asyncio.to_thread(
                self._collection.get,
                where={"tier": tier.value},
                include=[],
            )
            ids = result["ids"]
            if ids:
                await asyncio.to_thread(self._collection.delete, ids=ids)
            return len(ids)
        except Exception as exc:
            logger.warning("ChromaDB clear failed: %s", exc)
            return 0

    async def count(self, tier: Optional[MemoryTier] = None) -> int:
        await self._initialize()
        try:
            if tier is None:
                return await asyncio.to_thread(self._collection.count)
            result = await asyncio.to_thread(
                self._collection.get,
                where={"tier": tier.value},
                include=[],
            )
            return len(result["ids"])
        except Exception:
            return 0

    async def health_check(self) -> bool:
        try:
            await self._initialize()
            await asyncio.to_thread(self._collection.count)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        self._client = None
        self._collection = None


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class _Retriever:
    """Multi-strategy retriever over a backend + embedding engine."""

    def __init__(
        self,
        backend: Union[_InMemoryBackend, _RedisBackend, _ChromaBackend],
        embedding_engine: _EmbeddingEngine,
        config: MemoryConfig,
    ) -> None:
        self._backend = backend
        self._emb = embedding_engine
        self._config = config

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        if strategy == RetrievalStrategy.EXACT:
            return await self._retrieve_exact(query, top_k, tier, session_id, agent_id, tags)
        elif strategy == RetrievalStrategy.RECENCY:
            return await self._retrieve_recency(top_k, tier, session_id, agent_id, tags)
        elif strategy == RetrievalStrategy.IMPORTANCE:
            return await self._retrieve_importance(top_k, tier, session_id, agent_id, tags)
        elif strategy == RetrievalStrategy.SEMANTIC:
            return await self._retrieve_semantic(
                query, top_k, tier, session_id, agent_id, tags, min_score
            )
        else:  # HYBRID
            return await self._retrieve_hybrid(
                query, top_k, tier, session_id, agent_id, tags, min_score
            )

    async def _retrieve_exact(
        self,
        query: str,
        top_k: int,
        tier: Optional[MemoryTier],
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[List[str]],
    ) -> List[RetrievalResult]:
        entries = await self._backend.list_entries(
            tier=tier, session_id=session_id, agent_id=agent_id, tags=tags, limit=top_k * 10
        )
        query_lower = query.lower()
        scored = [
            RetrievalResult(
                entry=e,
                score=1.0 if query_lower in e.content.lower() else 0.0,
                strategy=RetrievalStrategy.EXACT,
            )
            for e in entries
        ]
        scored = [r for r in scored if r.score > 0]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def _retrieve_recency(
        self,
        top_k: int,
        tier: Optional[MemoryTier],
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[List[str]],
    ) -> List[RetrievalResult]:
        entries = await self._backend.list_entries(
            tier=tier, session_id=session_id, agent_id=agent_id, tags=tags, limit=top_k
        )
        now = time.time()
        results = []
        for e in entries:
            age = now - e.created_at
            score = 1.0 / (1.0 + age / 3600.0)
            results.append(RetrievalResult(entry=e, score=score, strategy=RetrievalStrategy.RECENCY))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _retrieve_importance(
        self,
        top_k: int,
        tier: Optional[MemoryTier],
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[List[str]],
    ) -> List[RetrievalResult]:
        entries = await self._backend.list_entries(
            tier=tier, session_id=session_id, agent_id=agent_id, tags=tags, limit=top_k * 5
        )
        results = [
            RetrievalResult(entry=e, score=e.importance, strategy=RetrievalStrategy.IMPORTANCE)
            for e in entries
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
        tier: Optional[MemoryTier],
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[List[str]],
        min_score: float,
    ) -> List[RetrievalResult]:
        query_emb = await self._emb.embed(query)

        # ChromaDB native vector search
        if isinstance(self._backend, _ChromaBackend):
            pairs = await self._backend.query_by_embedding(
                embedding=query_emb, top_k=top_k, tier=tier
            )
            results = []
            for entry, score in pairs:
                if score >= min_score:
                    results.append(
                        RetrievalResult(entry=entry, score=score, strategy=RetrievalStrategy.SEMANTIC)
                    )
            return results

        # Fallback: brute-force cosine over all entries
        entries = await self._backend.list_entries(
            tier=tier, session_id=session_id, agent_id=agent_id, tags=tags, limit=10000
        )
        results = []
        for e in entries:
            emb = e.embedding
            if not emb:
                emb = await self._emb.embed(e.content)
                e.embedding = emb
            score = _EmbeddingEngine.cosine_similarity(query_emb, emb)
            if score >= min_score:
                results.append(
                    RetrievalResult(entry=e, score=score, strategy=RetrievalStrategy.SEMANTIC)
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        tier: Optional[MemoryTier],
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[List[str]],
        min_score: float,
    ) -> List[RetrievalResult]:
        semantic_results = await self._retrieve_semantic(
            query, top_k * 2, tier, session_id, agent_id, tags, min_score
        )
        recency_results = await self._retrieve_recency(
            top_k * 2, tier, session_id, agent_id, tags
        )
        seen: Dict[str, float] = {}
        for r in semantic_results:
            seen[r.entry.id] = 0.7 * r.score
        for r in recency_results:
            seen[r.entry.id] = seen.get(r.entry.id, 0.0) + 0.3 * r.score

        all_entries: Dict[str, MemoryEntry] = {}
        for r in semantic_results + recency_results:
            all_entries[r.entry.id] = r.entry

        merged = [
            RetrievalResult(entry=all_entries[mid], score=score, strategy=RetrievalStrategy.HYBRID)
            for mid, score in seen.items()
        ]
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]


# ---------------------------------------------------------------------------
# MemoryConnector – Singleton
# ---------------------------------------------------------------------------

class MemoryConnector:
    """
    Production-grade memory connector.

    Provides short-term, long-term, episodic, semantic, and working memory
    with semantic retrieval, embeddings, and health checks.

    Usage
    -----
    connector = MemoryConnector.get_instance()
    await connector.initialize()
    await connector.store("User prefers dark mode", tier=MemoryTier.LONG_TERM)
    results = await connector.retrieve("user preferences")
    """

    _instance: Optional["MemoryConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config: MemoryConfig = MemoryConfig()
        self._backend: Optional[Union[_InMemoryBackend, _RedisBackend, _ChromaBackend]] = None
        self._embedding_engine: Optional[_EmbeddingEngine] = None
        self._retriever: Optional[_Retriever] = None
        self._stats: MemoryStats = MemoryStats()
        self._initialized = False
        self._hooks: List[Callable[[str, MemoryEntry], None]] = []

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "MemoryConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "MemoryConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self, config: Optional[MemoryConfig] = None) -> "MemoryConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        backend_type = self._config.backend
        if backend_type == MemoryBackend.REDIS:
            self._backend = _RedisBackend(self._config)
        elif backend_type == MemoryBackend.CHROMA:
            self._backend = _ChromaBackend(self._config)
        else:
            self._backend = _InMemoryBackend(self._config)

        self._embedding_engine = _EmbeddingEngine(self._config)
        await self._embedding_engine.initialize()

        self._retriever = _Retriever(self._backend, self._embedding_engine, self._config)
        self._stats.backend = self._config.backend.value
        self._stats.embedding_provider = self._config.embedding_provider.value
        self._initialized = True

        logger.info(
            "MemoryConnector initialized: backend=%s embedding=%s",
            self._config.backend.value,
            self._config.embedding_provider.value,
        )
        return self

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def configure_from_env(self) -> "MemoryConnector":
        backend_str = os.environ.get("MEMORY_BACKEND", MemoryBackend.IN_MEMORY.value)
        try:
            backend = MemoryBackend(backend_str)
        except ValueError:
            backend = MemoryBackend.IN_MEMORY

        emb_str = os.environ.get("MEMORY_EMBEDDING_PROVIDER", EmbeddingProvider.SIMPLE_HASH.value)
        try:
            emb = EmbeddingProvider(emb_str)
        except ValueError:
            emb = EmbeddingProvider.SIMPLE_HASH

        self._config = MemoryConfig(
            backend=backend,
            embedding_provider=emb,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            chroma_host=os.environ.get("CHROMA_HOST", "localhost"),
            chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
            chroma_in_memory=os.environ.get("CHROMA_IN_MEMORY", "true").lower() == "true",
        )
        return self

    # ------------------------------------------------------------------
    # Core store operations
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        ttl: Optional[float] = None,
        embed: bool = True,
    ) -> MemoryEntry:
        await self._ensure_initialized()

        effective_ttl = ttl
        if effective_ttl is None:
            if tier == MemoryTier.SHORT_TERM:
                effective_ttl = self._config.short_term_ttl
            elif tier == MemoryTier.WORKING:
                effective_ttl = 600.0
            else:
                effective_ttl = self._config.long_term_ttl

        entry = MemoryEntry(
            id=memory_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
            tier=tier,
            importance=min(max(importance, 0.0), 1.0),
            tags=tags or [],
            session_id=session_id,
            agent_id=agent_id,
            ttl=effective_ttl,
        )

        if embed and self._embedding_engine:
            try:
                entry.embedding = await self._embedding_engine.embed(content)
            except Exception as exc:
                logger.warning("Embedding failed for store: %s", exc)

        await self._backend.store(entry)  # type: ignore[union-attr]
        self._stats.total_stores += 1
        self._update_tier_count()
        self._invoke_hooks("store", entry)
        logger.debug("Stored memory [%s] tier=%s len=%d", entry.id, tier.value, len(content))
        return entry

    async def store_batch(
        self,
        items: List[Dict[str, Any]],
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        await self._ensure_initialized()
        texts = [item.get("content", "") for item in items]
        embeddings: List[Optional[List[float]]] = []
        if self._embedding_engine:
            try:
                embeddings = await self._embedding_engine.embed_batch(texts)
            except Exception:
                embeddings = [None] * len(texts)
        else:
            embeddings = [None] * len(texts)

        entries: List[MemoryEntry] = []
        for idx, item in enumerate(items):
            entry = await self.store(
                content=item.get("content", ""),
                tier=MemoryTier(item.get("tier", tier.value)),
                metadata=item.get("metadata", {}),
                importance=item.get("importance", 0.5),
                tags=item.get("tags", []),
                session_id=item.get("session_id", session_id),
                agent_id=item.get("agent_id", agent_id),
                embed=False,
            )
            if embeddings[idx]:
                entry.embedding = embeddings[idx]
                await self._backend.store(entry)  # type: ignore[union-attr]
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Core retrieval operations
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        await self._ensure_initialized()
        k = top_k or self._config.default_top_k
        results = await self._retriever.retrieve(  # type: ignore[union-attr]
            query=query,
            top_k=k,
            strategy=strategy,
            tier=tier,
            session_id=session_id,
            agent_id=agent_id,
            tags=tags,
            min_score=min_score,
        )
        self._stats.total_retrievals += 1
        return results

    async def retrieve_short_term(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            tier=MemoryTier.SHORT_TERM,
            session_id=session_id,
            **kwargs,
        )

    async def retrieve_long_term(
        self,
        query: str,
        top_k: int = 5,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            tier=MemoryTier.LONG_TERM,
            agent_id=agent_id,
            **kwargs,
        )

    async def retrieve_all_tiers(
        self,
        query: str,
        top_k_per_tier: int = 3,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[MemoryTier, List[RetrievalResult]]:
        await self._ensure_initialized()
        results: Dict[MemoryTier, List[RetrievalResult]] = {}
        tasks = {
            tier: self.retrieve(
                query=query,
                top_k=top_k_per_tier,
                tier=tier,
                session_id=session_id,
                agent_id=agent_id,
            )
            for tier in MemoryTier
        }
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for tier, outcome in zip(tasks.keys(), outcomes):
            if isinstance(outcome, Exception):
                logger.warning("retrieve_all_tiers tier=%s failed: %s", tier, outcome)
                results[tier] = []
            else:
                results[tier] = outcome  # type: ignore[assignment]
        return results

    # ------------------------------------------------------------------
    # Get / delete / list
    # ------------------------------------------------------------------

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        await self._ensure_initialized()
        entry = await self._backend.get(memory_id)  # type: ignore[union-attr]
        if entry:
            self._stats.cache_hits += 1
        else:
            self._stats.cache_misses += 1
        return entry

    async def delete(self, memory_id: str) -> bool:
        await self._ensure_initialized()
        result = await self._backend.delete(memory_id)  # type: ignore[union-attr]
        if result:
            self._stats.total_deletes += 1
        return result

    async def delete_session(self, session_id: str) -> int:
        await self._ensure_initialized()
        entries = await self._backend.list_entries(session_id=session_id, limit=10000)  # type: ignore[union-attr]
        count = 0
        for e in entries:
            if await self._backend.delete(e.id):  # type: ignore[union-attr]
                count += 1
        self._stats.total_deletes += count
        return count

    async def list_memories(
        self,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        await self._ensure_initialized()
        return await self._backend.list_entries(  # type: ignore[union-attr]
            tier=tier,
            session_id=session_id,
            agent_id=agent_id,
            tags=tags,
            limit=limit,
        )

    async def clear(self, tier: Optional[MemoryTier] = None) -> int:
        await self._ensure_initialized()
        count = await self._backend.clear(tier=tier)  # type: ignore[union-attr]
        self._stats.total_deletes += count
        return count

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        await self._ensure_initialized()
        if not self._embedding_engine:
            raise EmbeddingError("Embedding engine not initialized")
        return await self._embedding_engine.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        await self._ensure_initialized()
        if not self._embedding_engine:
            raise EmbeddingError("Embedding engine not initialized")
        return await self._embedding_engine.embed_batch(texts)

    async def similarity(self, text_a: str, text_b: str) -> float:
        await self._ensure_initialized()
        emb_a, emb_b = await asyncio.gather(self.embed(text_a), self.embed(text_b))
        return _EmbeddingEngine.cosine_similarity(emb_a, emb_b)

    # ------------------------------------------------------------------
    # Memory service compatibility
    # ------------------------------------------------------------------

    async def add_to_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 2000,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories and inject into context window.
        Compatible with context_engine.py.
        """
        results = await self.retrieve(
            query=query,
            top_k=10,
            strategy=RetrievalStrategy.HYBRID,
            session_id=session_id,
            agent_id=agent_id,
        )
        injected_chars = 0
        max_chars = max_tokens * 4
        memory_messages: List[Dict[str, Any]] = []
        for r in results:
            chunk = f"[Memory] {r.entry.content}"
            if injected_chars + len(chunk) > max_chars:
                break
            memory_messages.append({"role": "system", "content": chunk})
            injected_chars += len(chunk)
        return memory_messages + context

    async def summarize_session(
        self,
        session_id: str,
        max_entries: int = 50,
    ) -> str:
        """Return concatenated short-term memories for a session."""
        entries = await self.list_memories(
            tier=MemoryTier.SHORT_TERM,
            session_id=session_id,
            limit=max_entries,
        )
        entries.sort(key=lambda e: e.created_at)
        return "\n".join(e.content for e in entries)

    async def promote_to_long_term(
        self,
        memory_id: str,
        importance: Optional[float] = None,
    ) -> Optional[MemoryEntry]:
        """Promote a short-term memory to long-term storage."""
        entry = await self.get(memory_id)
        if not entry:
            return None
        await self.delete(memory_id)
        entry.tier = MemoryTier.LONG_TERM
        entry.ttl = self._config.long_term_ttl
        if importance is not None:
            entry.importance = importance
        await self._backend.store(entry)  # type: ignore[union-attr]
        logger.debug("Promoted memory %s to long-term", memory_id)
        return entry

    async def decay_importance(self) -> int:
        """Apply importance decay to long-term memories (call periodically)."""
        if not self._config.enable_importance_decay:
            return 0
        entries = await self.list_memories(tier=MemoryTier.LONG_TERM, limit=10000)
        count = 0
        for e in entries:
            new_importance = max(0.0, e.importance - self._config.importance_decay_rate)
            if new_importance != e.importance:
                e.importance = new_importance
                await self._backend.store(e)  # type: ignore[union-attr]
                count += 1
        return count

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def add_hook(self, hook: Callable[[str, MemoryEntry], None]) -> None:
        self._hooks.append(hook)

    def _invoke_hooks(self, event: str, entry: MemoryEntry) -> None:
        for hook in self._hooks:
            try:
                hook(event, entry)
            except Exception as exc:
                logger.warning("Memory hook error: %s", exc)

    # ------------------------------------------------------------------
    # Stats & health
    # ------------------------------------------------------------------

    def _update_tier_count(self) -> None:
        pass  # Deferred to health_check for accuracy

    async def get_stats(self) -> MemoryStats:
        await self._ensure_initialized()
        self._stats.total_entries = await self._backend.count()  # type: ignore[union-attr]
        self._stats.short_term_count = await self._backend.count(MemoryTier.SHORT_TERM)  # type: ignore[union-attr]
        self._stats.long_term_count = await self._backend.count(MemoryTier.LONG_TERM)  # type: ignore[union-attr]
        self._stats.episodic_count = await self._backend.count(MemoryTier.EPISODIC)  # type: ignore[union-attr]
        self._stats.semantic_count = await self._backend.count(MemoryTier.SEMANTIC)  # type: ignore[union-attr]
        self._stats.working_count = await self._backend.count(MemoryTier.WORKING)  # type: ignore[union-attr]
        return self._stats

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        backend_ok = await self._backend.health_check()  # type: ignore[union-attr]
        stats = await self.get_stats()
        return {
            "healthy": backend_ok,
            "backend": self._config.backend.value,
            "embedding_provider": self._config.embedding_provider.value,
            "total_entries": stats.total_entries,
            "short_term_count": stats.short_term_count,
            "long_term_count": stats.long_term_count,
            "total_stores": stats.total_stores,
            "total_retrievals": stats.total_retrievals,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
        }

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MemoryConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._backend:
            await self._backend.close()
        self._initialized = False

    def __repr__(self) -> str:
        return (
            f"MemoryConnector(backend={self._config.backend.value}, "
            f"stores={self._stats.total_stores}, "
            f"retrievals={self._stats.total_retrievals})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_memory_connector() -> MemoryConnector:
    connector = MemoryConnector.get_instance()
    return connector


async def store_memory(
    content: str,
    tier: MemoryTier = MemoryTier.SHORT_TERM,
    **kwargs: Any,
) -> MemoryEntry:
    connector = get_memory_connector()
    await connector._ensure_initialized()
    return await connector.store(content=content, tier=tier, **kwargs)


async def retrieve_memory(
    query: str,
    top_k: int = 5,
    **kwargs: Any,
) -> List[RetrievalResult]:
    connector = get_memory_connector()
    await connector._ensure_initialized()
    return await connector.retrieve(query=query, top_k=top_k, **kwargs)


async def memory_health_check() -> Dict[str, Any]:
    connector = get_memory_connector()
    return await connector.health_check()
