from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(str, Enum):
    SESSION = "session"
    USER = "user"
    GLOBAL = "global"


@dataclass
class MemoryEntry:
    id: str
    content: str
    memory_type: MemoryType
    scope: MemoryScope
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds; None = immortal

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "metadata": self.metadata,
            "score": self.score,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ttl": self.ttl,
        }


@dataclass
class RetrievalResult:
    entry: MemoryEntry
    similarity: float
    rank: int


@dataclass
class MemoryServiceConfig:
    short_term_limit: int = 200
    long_term_limit: int = 10_000
    short_term_ttl: float = 3600.0
    embedding_model: str = "nvidia/nv-embed-v1"
    embedding_dim: int = 1536
    similarity_threshold: float = 0.75
    top_k: int = 5
    storage_path: Optional[str] = None
    enable_persistence: bool = False
    openai_api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    def __init__(self, config: MemoryServiceConfig) -> None:
        self._config = config
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._config.openai_api_key)
            except ImportError:
                self._client = "mock"
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        client = await self._get_client()
        if client == "mock":
            return [self._mock_embed(t) for t in texts]
        try:
            resp = await client.embeddings.create(model=self._config.embedding_model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning("Embedding API error: %s — using mock", exc)
            return [self._mock_embed(t) for t in texts]

    def _mock_embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h]
        pad = self._config.embedding_dim - len(vec)
        vec.extend([0.0] * pad)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


# ---------------------------------------------------------------------------
# Similarity & scoring
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-10
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-10
    return dot / (mag_a * mag_b)


def _score_entry(entry: MemoryEntry, query_sim: float, now: float) -> float:
    recency = 1.0 / (1.0 + (now - entry.last_accessed) / 3600.0)
    frequency = math.log1p(entry.access_count) / 10.0
    return 0.6 * query_sim + 0.25 * recency + 0.15 * frequency


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

    async def add(self, entry: MemoryEntry) -> None:
        async with self._lock:
            self._evict_expired()
            if len(self._entries) >= self._capacity:
                self._evict_lru()
            self._entries[entry.id] = entry

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        async with self._lock:
            e = self._entries.get(entry_id)
            if e and not e.is_expired():
                e.touch()
                return e
            if e:
                del self._entries[entry_id]
            return None

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            return self._entries.pop(entry_id, None) is not None

    async def all(self) -> List[MemoryEntry]:
        async with self._lock:
            self._evict_expired()
            return list(self._entries.values())

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    def _evict_expired(self) -> None:
        expired = [k for k, v in self._entries.items() if v.is_expired()]
        for k in expired:
            del self._entries[k]

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
        del self._entries[lru_key]

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------

class PersistenceLayer:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, entry: MemoryEntry) -> None:
        fp = self._path / f"{entry.id}.json"
        data = entry.to_dict()
        fp.write_text(json.dumps(data, indent=2))

    def load_all(self) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for fp in self._path.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                e = MemoryEntry(
                    id=data["id"],
                    content=data["content"],
                    memory_type=MemoryType(data["memory_type"]),
                    scope=MemoryScope(data["scope"]),
                    metadata=data.get("metadata", {}),
                    score=data.get("score", 0.0),
                    access_count=data.get("access_count", 0),
                    created_at=data.get("created_at", time.time()),
                    last_accessed=data.get("last_accessed", time.time()),
                    ttl=data.get("ttl"),
                )
                entries.append(e)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fp, exc)
        return entries

    def delete(self, entry_id: str) -> None:
        fp = self._path / f"{entry_id}.json"
        fp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MemoryRetriever:
    def __init__(self, embedder: EmbeddingProvider, config: MemoryServiceConfig) -> None:
        self._embedder = embedder
        self._config = config

    async def retrieve(
        self,
        query: str,
        candidates: List[MemoryEntry],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        if not candidates:
            return []
        k = top_k or self._config.top_k
        thresh = threshold or self._config.similarity_threshold
        now = time.time()

        query_emb = (await self._embedder.embed([query]))[0]

        # embed any entries missing embeddings
        missing = [e for e in candidates if e.embedding is None]
        if missing:
            texts = [e.content for e in missing]
            embeddings = await self._embedder.embed(texts)
            for e, emb in zip(missing, embeddings):
                e.embedding = emb

        results: List[RetrievalResult] = []
        for rank, entry in enumerate(candidates):
            if entry.embedding is None:
                continue
            sim = _cosine_similarity(query_emb, entry.embedding)
            if sim < thresh:
                continue
            scored = _score_entry(entry, sim, now)
            results.append(RetrievalResult(entry=entry, similarity=scored, rank=rank))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:k]


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------

class MemoryService:
    """
    Short-term and long-term memory management with embedding-based retrieval,
    scoring, TTL, persistence, and context injection.
    """

    def __init__(self, config: Optional[MemoryServiceConfig] = None) -> None:
        self._config = config or MemoryServiceConfig()
        self._short_term = MemoryStore(self._config.short_term_limit)
        self._long_term = MemoryStore(self._config.long_term_limit)
        self._embedder = EmbeddingProvider(self._config)
        self._retriever = MemoryRetriever(self._embedder, self._config)
        self._persistence: Optional[PersistenceLayer] = (
            PersistenceLayer(self._config.storage_path)
            if self._config.enable_persistence and self._config.storage_path
            else None
        )
        self._session_id: str = str(uuid.uuid4())
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        scope: MemoryScope = MemoryScope.SESSION,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        embed: bool = True,
    ) -> MemoryEntry:
        entry_id = str(uuid.uuid4())
        effective_ttl = ttl if ttl is not None else (
            self._config.short_term_ttl if memory_type == MemoryType.SHORT_TERM else None
        )
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type=memory_type,
            scope=scope,
            metadata=metadata or {},
            ttl=effective_ttl,
        )
        if embed:
            embeddings = await self._embedder.embed([content])
            entry.embedding = embeddings[0]

        if memory_type == MemoryType.SHORT_TERM:
            await self._short_term.add(entry)
        else:
            await self._long_term.add(entry)
            if self._persistence:
                await asyncio.to_thread(self._persistence.save, entry)

        logger.debug("Stored %s memory %s", memory_type.value, entry_id)
        return entry

    async def store_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[MemoryEntry]:
        return await asyncio.gather(*[self.store(**item) for item in items])

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        e = await self._short_term.get(entry_id)
        if e:
            return e
        return await self._long_term.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        deleted = await self._short_term.delete(entry_id) or await self._long_term.delete(entry_id)
        if deleted and self._persistence:
            await asyncio.to_thread(self._persistence.delete, entry_id)
        return deleted

    async def clear_short_term(self) -> None:
        await self._short_term.clear()

    async def clear_all(self) -> None:
        await self._short_term.clear()
        await self._long_term.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        scope: Optional[MemoryScope] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        short = await self._short_term.all()
        long = await self._long_term.all()
        candidates = short + long

        if memory_types:
            candidates = [c for c in candidates if c.memory_type in memory_types]
        if scope:
            candidates = [c for c in candidates if c.scope == scope]

        results = await self._retriever.retrieve(query, candidates, top_k, threshold)
        for r in results:
            r.entry.touch()
        return results

    async def retrieve_recent(
        self,
        n: int = 10,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> List[MemoryEntry]:
        store = self._short_term if memory_type == MemoryType.SHORT_TERM else self._long_term
        entries = await store.all()
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:n]

    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
    ) -> List[MemoryEntry]:
        short = await self._short_term.all()
        long = await self._long_term.all()
        results: List[MemoryEntry] = []
        for e in short + long:
            if all(e.metadata.get(k) == v for k, v in filters.items()):
                results.append(e)
        return results

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    async def promote_to_long_term(self, entry_id: str) -> Optional[MemoryEntry]:
        entry = await self._short_term.get(entry_id)
        if entry is None:
            return None
        await self._short_term.delete(entry_id)
        entry.memory_type = MemoryType.LONG_TERM
        entry.ttl = None
        await self._long_term.add(entry)
        if self._persistence:
            await asyncio.to_thread(self._persistence.save, entry)
        return entry

    # ------------------------------------------------------------------
    # Memory injection (for context_engine / planner compatibility)
    # ------------------------------------------------------------------

    async def inject_memories(
        self,
        query: str,
        system_prompt: str,
        top_k: int = 5,
        separator: str = "\n---\n",
    ) -> str:
        results = await self.retrieve(query, top_k=top_k)
        if not results:
            return system_prompt
        memory_block = separator.join(
            f"[Memory {r.rank + 1} | score={r.similarity:.2f}]\n{r.entry.content}"
            for r in results
        )
        return f"{system_prompt}\n\n## Relevant Memories\n{memory_block}"

    async def build_context_window(
        self,
        query: str,
        max_tokens: int = 2000,
        chars_per_token: int = 4,
    ) -> List[Dict[str, Any]]:
        results = await self.retrieve(query)
        budget = max_tokens * chars_per_token
        context: List[Dict[str, Any]] = []
        used = 0
        for r in results:
            chunk_len = len(r.entry.content)
            if used + chunk_len > budget:
                break
            context.append({
                "id": r.entry.id,
                "content": r.entry.content,
                "similarity": r.similarity,
                "type": r.entry.memory_type.value,
                "metadata": r.entry.metadata,
            })
            used += chunk_len
        return context

    # ------------------------------------------------------------------
    # Persistence restore
    # ------------------------------------------------------------------

    async def restore_from_disk(self) -> int:
        if not self._persistence:
            return 0
        entries = await asyncio.to_thread(self._persistence.load_all)
        count = 0
        for e in entries:
            if e.is_expired():
                continue
            if e.memory_type == MemoryType.SHORT_TERM:
                await self._short_term.add(e)
            else:
                await self._long_term.add(e)
            count += 1
        logger.info("Restored %d memories from disk", count)
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "short_term_count": self._short_term.size,
            "long_term_count": self._long_term.size,
            "session_id": self._session_id,
            "embedding_model": self._config.embedding_model,
            "persistence_enabled": self._persistence is not None,
        }
