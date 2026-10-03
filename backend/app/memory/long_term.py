"""
long_term.py
============
Long-term persistent memory store for the AI Operating System.

Responsibilities
----------------
- Store durable, cross-session memories: preferences, goals, projects,
  facts, and arbitrary structured records.
- Semantic similarity search (vector) and exact/keyword retrieval.
- Full CRUD lifecycle with soft-delete and versioning.
- Session-independent: memories survive across conversations.
- Backend-agnostic: in-memory dict now; PostgreSQL and ChromaDB stubs included.

Future
------
- PostgreSQL persistence via asyncpg.
- Vector search via ChromaDB (or pgvector).
- TTL / expiry policies per memory category.
- Multi-Agent shared memory namespaces.
- Embedding model plug-in (OpenAI / local sentence-transformers).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    FACT = "fact"
    SKILL = "skill"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    CUSTOM = "custom"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """A single long-term memory unit."""

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: MemoryCategory = MemoryCategory.FACT
    key: str = ""                           # human-readable identifier / slug
    value: Any = None                       # primary payload (str, dict, list, …)
    description: str = ""                  # natural-language summary for search
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None # populated by embedding backend
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    accessed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = 0

    def touch(self) -> None:
        self.accessed_at = datetime.now(timezone.utc).isoformat()
        self.access_count += 1

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category.value,
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            category=MemoryCategory(data.get("category", MemoryCategory.FACT)),
            key=data.get("key", ""),
            value=data.get("value"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE)),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            accessed_at=data.get("accessed_at", datetime.now(timezone.utc).isoformat()),
            access_count=data.get("access_count", 0),
        )


@dataclass
class SearchResult:
    record: MemoryRecord
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.record.to_dict()
        d["score"] = self.score
        return d


# ---------------------------------------------------------------------------
# Embedding backend (pluggable)
# ---------------------------------------------------------------------------


class _BaseEmbedder:
    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def similarity(self, a: List[float], b: List[float]) -> float:
        raise NotImplementedError


class _NaiveEmbedder(_BaseEmbedder):
    """
    Bag-of-words TF vector for local similarity without external deps.
    Replace with sentence-transformers or OpenAI embeddings in production.
    """

    def _vectorise(self, text: str) -> Dict[str, int]:
        tokens = re.findall(r"\w+", text.lower())
        vec: Dict[str, int] = {}
        for t in tokens:
            vec[t] = vec.get(t, 0) + 1
        return vec

    async def embed(self, text: str) -> List[float]:
        # Return sparse representation as a stable float list (hash trick, dim=512)
        dim = 512
        vec = [0.0] * dim
        for token, count in self._vectorise(text).items():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
            vec[idx] += float(count)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, min(1.0, dot))


def _build_embedder(name: str = "naive") -> _BaseEmbedder:
    if name == "naive":
        return _NaiveEmbedder()

    if name == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("all-MiniLM-L6-v2")

            class _STEmbedder(_BaseEmbedder):
                async def embed(self, text: str) -> List[float]:
                    loop = asyncio.get_event_loop()
                    vec = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
                    return vec

                def similarity(self, a: List[float], b: List[float]) -> float:
                    dot = sum(x * y for x, y in zip(a, b))
                    na = math.sqrt(sum(x * x for x in a)) or 1.0
                    nb = math.sqrt(sum(x * x for x in b)) or 1.0
                    return max(0.0, min(1.0, dot / (na * nb)))

            return _STEmbedder()
        except ImportError:
            logger.warning("sentence_transformers not installed; falling back to naive embedder.")
            return _NaiveEmbedder()

    raise ValueError(f"Unknown embedder: {name}")


# ---------------------------------------------------------------------------
# Persistence backends
# ---------------------------------------------------------------------------


class _PersistenceBackend:
    """In-memory no-op backend (default)."""

    async def save(self, record: MemoryRecord) -> None: ...

    async def load_all(self) -> List[MemoryRecord]:
        return []

    async def hard_delete(self, memory_id: str) -> None: ...

    async def health(self) -> bool:
        return True


class _PostgreSQLBackend(_PersistenceBackend):
    """
    Async PostgreSQL backend via asyncpg.

    Expected schema
    ---------------
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE long_term_memories (
        memory_id    UUID PRIMARY KEY,
        category     TEXT NOT NULL,
        key          TEXT NOT NULL,
        value        JSONB,
        description  TEXT NOT NULL DEFAULT '',
        tags         TEXT[] NOT NULL DEFAULT '{}',
        metadata     JSONB NOT NULL DEFAULT '{}',
        embedding    FLOAT8[],
        status       TEXT NOT NULL DEFAULT 'active',
        version      INTEGER NOT NULL DEFAULT 1,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        accessed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        access_count INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX ON long_term_memories (category, status);
    CREATE INDEX ON long_term_memories (key, status);
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg  # type: ignore
            import json as _json

            async def _init_conn(conn: Any) -> None:
                await conn.set_type_codec(
                    "jsonb", encoder=_json.dumps, decoder=_json.loads, schema="pg_catalog"
                )

            self._pool = await asyncpg.create_pool(self._dsn, init=_init_conn)
        except ImportError:
            raise RuntimeError("asyncpg not installed. Run: pip install asyncpg")

    async def save(self, record: MemoryRecord) -> None:
        await self._ensure_pool()
        import json as _json

        await self._pool.execute(
            """
            INSERT INTO long_term_memories
                (memory_id, category, key, value, description, tags, metadata,
                 embedding, status, version, created_at, updated_at, accessed_at, access_count)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (memory_id) DO UPDATE SET
                category=EXCLUDED.category, key=EXCLUDED.key, value=EXCLUDED.value,
                description=EXCLUDED.description, tags=EXCLUDED.tags,
                metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding,
                status=EXCLUDED.status, version=EXCLUDED.version,
                updated_at=EXCLUDED.updated_at, accessed_at=EXCLUDED.accessed_at,
                access_count=EXCLUDED.access_count
            """,
            record.memory_id, record.category.value, record.key,
            _json.dumps(record.value), record.description, record.tags,
            record.metadata, record.embedding,
            record.status.value, record.version,
            record.created_at, record.updated_at,
            record.accessed_at, record.access_count,
        )

    async def load_all(self) -> List[MemoryRecord]:
        await self._ensure_pool()
        rows = await self._pool.fetch(
            "SELECT * FROM long_term_memories WHERE status != 'deleted' ORDER BY created_at"
        )
        records = []
        for r in rows:
            import json as _json
            records.append(MemoryRecord(
                memory_id=str(r["memory_id"]),
                category=MemoryCategory(r["category"]),
                key=r["key"],
                value=_json.loads(r["value"]) if isinstance(r["value"], str) else r["value"],
                description=r["description"],
                tags=list(r["tags"] or []),
                metadata=r["metadata"] or {},
                embedding=list(r["embedding"]) if r["embedding"] else None,
                status=MemoryStatus(r["status"]),
                version=r["version"],
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
                accessed_at=str(r["accessed_at"]),
                access_count=r["access_count"],
            ))
        return records

    async def hard_delete(self, memory_id: str) -> None:
        await self._ensure_pool()
        await self._pool.execute(
            "DELETE FROM long_term_memories WHERE memory_id=$1", memory_id
        )

    async def health(self) -> bool:
        try:
            await self._ensure_pool()
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False


class _ChromaDBBackend(_PersistenceBackend):
    """
    Vector-store backend via ChromaDB.
    Embeddings are stored and searched natively by Chroma.
    Scalar fields are persisted in Chroma metadata.

    Future: combine with PostgreSQL for relational queries + vector search.
    """

    def __init__(self, collection_name: str = "long_term_memory", persist_dir: Optional[str] = None) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client: Any = None
        self._collection: Any = None

    async def _ensure_client(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore

            loop = asyncio.get_event_loop()

            def _init() -> Any:
                if self._persist_dir:
                    client = chromadb.PersistentClient(path=self._persist_dir)
                else:
                    client = chromadb.EphemeralClient()
                return client.get_or_create_collection(self._collection_name)

            self._collection = await loop.run_in_executor(None, _init)
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    async def save(self, record: MemoryRecord) -> None:
        await self._ensure_client()
        import json as _json

        loop = asyncio.get_event_loop()
        meta = {
            "category": record.category.value,
            "key": record.key,
            "tags": _json.dumps(record.tags),
            "status": record.status.value,
            "version": record.version,
            "value_json": _json.dumps(record.value),
        }

        def _upsert() -> None:
            kwargs: Dict[str, Any] = dict(
                ids=[record.memory_id],
                documents=[record.description or record.key],
                metadatas=[meta],
            )
            if record.embedding:
                kwargs["embeddings"] = [record.embedding]
            self._collection.upsert(**kwargs)

        await loop.run_in_executor(None, _upsert)

    async def load_all(self) -> List[MemoryRecord]:
        await self._ensure_client()
        import json as _json

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: self._collection.get(include=["metadatas", "embeddings", "documents"]))
        records: List[MemoryRecord] = []
        for i, mid in enumerate(raw.get("ids", [])):
            meta = raw["metadatas"][i] if raw.get("metadatas") else {}
            emb = raw["embeddings"][i] if raw.get("embeddings") else None
            records.append(MemoryRecord(
                memory_id=mid,
                category=MemoryCategory(meta.get("category", "fact")),
                key=meta.get("key", ""),
                value=_json.loads(meta.get("value_json", "null")),
                description=raw["documents"][i] if raw.get("documents") else "",
                tags=_json.loads(meta.get("tags", "[]")),
                status=MemoryStatus(meta.get("status", "active")),
                version=int(meta.get("version", 1)),
                embedding=list(emb) if emb else None,
            ))
        return records

    async def hard_delete(self, memory_id: str) -> None:
        await self._ensure_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete(ids=[memory_id]))

    async def health(self) -> bool:
        try:
            await self._ensure_client()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """
    Persistent, cross-session memory store.

    Compatible with
    ---------------
    - memory_capability.py  →  all public async methods
    - context_engine.py     →  retrieve() / search() for grounding

    Parameters
    ----------
    embedder:    'naive' | 'sentence_transformers'
    persistence: _PersistenceBackend instance
    """

    def __init__(
        self,
        embedder: str = "naive",
        persistence: Optional[_PersistenceBackend] = None,
    ) -> None:
        self._embedder = _build_embedder(embedder)
        self._persistence = persistence or _PersistenceBackend()
        self._store: Dict[str, MemoryRecord] = {}   # memory_id → record
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_record(self, record: MemoryRecord) -> None:
        text = f"{record.key} {record.description} {' '.join(record.tags)}"
        record.embedding = await self._embedder.embed(text)

    def _active(self) -> List[MemoryRecord]:
        return [r for r in self._store.values() if r.status == MemoryStatus.ACTIVE]

    def _keyword_match(self, record: MemoryRecord, query: str) -> bool:
        q = query.lower()
        haystack = " ".join([
            record.key, record.description,
            str(record.value), " ".join(record.tags),
        ]).lower()
        return q in haystack

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(
        self,
        key: str,
        value: Any,
        *,
        category: str | MemoryCategory = MemoryCategory.FACT,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = True,
    ) -> MemoryRecord:
        """
        Persist a new memory or upsert an existing one matched by key+category.

        Parameters
        ----------
        key:         Unique slug within the category (e.g. 'preferred_language').
        value:       Payload — any JSON-serialisable type.
        category:    MemoryCategory value or string.
        description: Free-text description used for semantic search.
        tags:        Searchable tag list.
        metadata:    Arbitrary annotations.
        upsert:      If True and a matching key+category exists, update it.

        Returns
        -------
        The stored MemoryRecord.
        """
        if isinstance(category, str):
            category = MemoryCategory(category)

        async with self._lock:
            existing: Optional[MemoryRecord] = None
            if upsert:
                for r in self._active():
                    if r.key == key and r.category == category:
                        existing = r
                        break

            if existing:
                existing.value = value
                existing.description = description or existing.description
                existing.tags = tags if tags is not None else existing.tags
                existing.metadata.update(metadata or {})
                existing.bump_version()
                record = existing
            else:
                record = MemoryRecord(
                    key=key,
                    value=value,
                    category=category,
                    description=description,
                    tags=tags or [],
                    metadata=metadata or {},
                )

            await self._embed_record(record)
            self._store[record.memory_id] = record

        asyncio.ensure_future(self._persistence.save(record))
        logger.debug(
            "[LTM] store key=%s category=%s memory_id=%s version=%d",
            key, category.value, record.memory_id, record.version,
        )
        return record

    async def retrieve(
        self,
        memory_id: Optional[str] = None,
        *,
        key: Optional[str] = None,
        category: Optional[str | MemoryCategory] = None,
    ) -> Optional[MemoryRecord]:
        """
        Fetch a single record by memory_id or by key+category.

        Returns None if not found or soft-deleted.
        """
        if isinstance(category, str):
            category = MemoryCategory(category)

        async with self._lock:
            if memory_id:
                record = self._store.get(memory_id)
                if record and record.status == MemoryStatus.ACTIVE:
                    record.touch()
                    asyncio.ensure_future(self._persistence.save(record))
                    return record
                return None

            for r in self._active():
                if key and r.key != key:
                    continue
                if category and r.category != category:
                    continue
                r.touch()
                asyncio.ensure_future(self._persistence.save(r))
                return r

        return None

    async def update(
        self,
        memory_id: str,
        *,
        value: Any = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str | MemoryStatus] = None,
    ) -> Optional[MemoryRecord]:
        """
        Partially update an existing record.

        Only supplied (non-None) fields are modified.
        Returns the updated record, or None if not found.
        """
        async with self._lock:
            record = self._store.get(memory_id)
            if record is None or record.status == MemoryStatus.DELETED:
                return None

            if value is not None:
                record.value = value
            if description is not None:
                record.description = description
            if tags is not None:
                record.tags = tags
            if metadata is not None:
                record.metadata.update(metadata)
            if status is not None:
                record.status = MemoryStatus(status) if isinstance(status, str) else status

            record.bump_version()
            await self._embed_record(record)

        asyncio.ensure_future(self._persistence.save(record))
        logger.debug("[LTM] update memory_id=%s version=%d", memory_id, record.version)
        return record

    async def delete(
        self,
        memory_id: str,
        *,
        hard: bool = False,
    ) -> Dict[str, Any]:
        """
        Soft-delete (default) or hard-delete a record.

        Parameters
        ----------
        memory_id: Target record.
        hard:      If True, remove from store and persistence entirely.

        Returns
        -------
        Summary dict.
        """
        async with self._lock:
            record = self._store.get(memory_id)
            if record is None:
                return {"memory_id": memory_id, "deleted": False, "reason": "not_found"}

            if hard:
                del self._store[memory_id]
                asyncio.ensure_future(self._persistence.hard_delete(memory_id))
                logger.info("[LTM] hard_delete memory_id=%s", memory_id)
                return {"memory_id": memory_id, "deleted": True, "hard": True}

            record.status = MemoryStatus.DELETED
            record.bump_version()

        asyncio.ensure_future(self._persistence.save(record))
        logger.info("[LTM] soft_delete memory_id=%s", memory_id)
        return {"memory_id": memory_id, "deleted": True, "hard": False}

    async def search(
        self,
        query: str,
        *,
        category: Optional[str | MemoryCategory] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        min_score: float = 0.0,
        semantic: bool = True,
    ) -> List[SearchResult]:
        """
        Search memories by semantic similarity and/or keyword/tag filters.

        Parameters
        ----------
        query:     Natural-language query string.
        category:  Optional category filter.
        tags:      Optional tag intersection filter.
        top_k:     Maximum results to return.
        min_score: Minimum similarity threshold (0–1).
        semantic:  Use vector similarity (True) or keyword-only (False).

        Returns
        -------
        Ranked list of SearchResult objects (highest score first).
        """
        if isinstance(category, str):
            category = MemoryCategory(category)

        query_embedding: Optional[List[float]] = None
        if semantic:
            query_embedding = await self._embedder.embed(query)

        async with self._lock:
            candidates = self._active()

        # Category filter
        if category:
            candidates = [r for r in candidates if r.category == category]

        # Tag filter (intersection)
        if tags:
            tag_set = set(tags)
            candidates = [r for r in candidates if tag_set.intersection(r.tags)]

        results: List[SearchResult] = []
        for record in candidates:
            score = 0.0
            if semantic and query_embedding and record.embedding:
                score = self._embedder.similarity(query_embedding, record.embedding)
            elif self._keyword_match(record, query):
                score = 0.5
            else:
                continue

            if score >= min_score:
                results.append(SearchResult(record=record, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_k]

        # Touch retrieved records
        async with self._lock:
            for sr in top:
                sr.record.touch()
                asyncio.ensure_future(self._persistence.save(sr.record))

        return top

    # ------------------------------------------------------------------
    # Bulk / utility
    # ------------------------------------------------------------------

    async def list_all(
        self,
        category: Optional[str | MemoryCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryRecord]:
        """Return all active records, optionally filtered."""
        if isinstance(category, str):
            category = MemoryCategory(category)

        async with self._lock:
            records = self._active()

        if category:
            records = [r for r in records if r.category == category]
        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]

        return records

    async def warm_up(self) -> int:
        """Load all persisted records into memory on startup."""
        records = await self._persistence.load_all()
        async with self._lock:
            for r in records:
                self._store[r.memory_id] = r
        logger.info("[LTM] warm_up loaded=%d records", len(records))
        return len(records)

    async def health_check(self) -> Dict[str, Any]:
        backend_ok = await self._persistence.health()
        async with self._lock:
            total = len(self._store)
            active = sum(1 for r in self._store.values() if r.status == MemoryStatus.ACTIVE)
            by_category: Dict[str, int] = {}
            for r in self._store.values():
                if r.status == MemoryStatus.ACTIVE:
                    by_category[r.category.value] = by_category.get(r.category.value, 0) + 1

        return {
            "status": "healthy" if backend_ok else "degraded",
            "total_records": total,
            "active_records": active,
            "records_by_category": by_category,
            "persistence_backend_ok": backend_ok,
            "embedder": type(self._embedder).__name__,
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_long_term_memory(
    *,
    embedder: str = "naive",
    postgres_dsn: Optional[str] = None,
    chromadb_dir: Optional[str] = None,
    chromadb_collection: str = "long_term_memory",
) -> LongTermMemory:
    """
    Instantiate LongTermMemory with the appropriate persistence backend.

    Priority: PostgreSQL > ChromaDB > in-memory.
    """
    backend: _PersistenceBackend

    if postgres_dsn:
        backend = _PostgreSQLBackend(postgres_dsn)
        logger.info("[LTM] Using PostgreSQL persistence backend.")
    elif chromadb_dir is not None:
        backend = _ChromaDBBackend(
            collection_name=chromadb_collection,
            persist_dir=chromadb_dir or None,
        )
        logger.info("[LTM] Using ChromaDB persistence backend.")
    else:
        backend = _PersistenceBackend()
        logger.info("[LTM] Using in-memory (no persistence) backend.")

    return LongTermMemory(embedder=embedder, persistence=backend)
