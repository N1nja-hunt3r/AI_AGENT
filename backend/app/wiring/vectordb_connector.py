"""
vectordb_connector.py - Production-grade vector database connector.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None
    np = None

try:
    import pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    AsyncQdrantClient = None

try:
    import asyncpg
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class VectorProvider(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    QDRANT = "qdrant"

class EmbeddingProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HASH = "hash"

@dataclass
class VectorDBConfig:
    provider: VectorProvider = VectorProvider.CHROMA
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASH
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "nvidia/nv-embed-v1"
    embedding_dim: int = 384
    collection_name: str = "default"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_in_memory: bool = True
    faiss_index_path: str = "./faiss.index"
    faiss_index_type: str = "Flat"
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "agent-memory"
    pinecone_environment: str = "us-east-1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    pgvector_dsn: str = "postgresql://localhost/vectordb"
    pgvector_table: str = "embeddings"
    similarity_threshold: float = 0.5
    default_top_k: int = 5
    cache_ttl: int = 300
    extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VectorDBError(Exception):
    pass

class ProviderNotAvailableError(VectorDBError):
    pass

class EmbeddingError(VectorDBError):
    pass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VectorDocument:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: str = "default"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "collection": self.collection,
            "created_at": self.created_at,
        }

@dataclass
class RetrievalResult:
    document: VectorDocument
    score: float
    provider: str

@dataclass
class IndexStats:
    provider: str
    collection: str
    total_documents: int = 0
    embedding_dim: int = 0
    index_size_bytes: int = 0

# ---------------------------------------------------------------------------
# Embeddings Manager
# ---------------------------------------------------------------------------

class _EmbeddingsManager:
    def __init__(self, config: VectorDBConfig) -> None:
        self._config = config
        self._st_model: Any = None
        self._openai_client: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        p = self._config.embedding_provider
        if p == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            if _ST_AVAILABLE:
                self._st_model = await asyncio.to_thread(
                    SentenceTransformer, self._config.embedding_model
                )
                self._config.embedding_dim = self._st_model.get_sentence_embedding_dimension()
            else:
                logger.warning("sentence-transformers unavailable; falling back to hash")
        elif p == EmbeddingProvider.OPENAI:
            if _OPENAI_AVAILABLE:
                key = self._config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
                self._openai_client = _AsyncOpenAI(api_key=key)
                self._config.embedding_dim = 1536
        self._initialized = True

    async def embed(self, text: str) -> List[float]:
        if not self._initialized:
            await self.initialize()
        p = self._config.embedding_provider
        try:
            if p == EmbeddingProvider.SENTENCE_TRANSFORMERS and self._st_model:
                vec = await asyncio.to_thread(self._st_model.encode, text)
                return vec.tolist() if hasattr(vec, "tolist") else list(vec)
            elif p == EmbeddingProvider.OPENAI and self._openai_client:
                resp = await self._openai_client.embeddings.create(
                    model=self._config.openai_embedding_model, input=text
                )
                return resp.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
        return self._hash_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()
        p = self._config.embedding_provider
        try:
            if p == EmbeddingProvider.SENTENCE_TRANSFORMERS and self._st_model:
                vecs = await asyncio.to_thread(self._st_model.encode, texts)
                return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
            elif p == EmbeddingProvider.OPENAI and self._openai_client:
                resp = await self._openai_client.embeddings.create(
                    model=self._config.openai_embedding_model, input=texts
                )
                return [d.embedding for d in resp.data]
        except Exception as exc:
            logger.warning("Batch embedding failed: %s", exc)
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str, dim: int = 128) -> List[float]:
        import hashlib
        digest = hashlib.sha256(text.encode()).digest()
        ext = (digest * ((dim * 4 // len(digest)) + 2))[: dim * 4]
        floats = [
            int.from_bytes(ext[i: i + 4], "little", signed=True) / 2_147_483_648.0
            for i in range(0, len(ext) - 3, 4)
        ][:dim]
        norm = sum(v * v for v in floats) ** 0.5 or 1.0
        return [v / norm for v in floats]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if (na * nb) else 0.0

# ---------------------------------------------------------------------------
# Store implementations
# ---------------------------------------------------------------------------

class _ChromaStore:
    def __init__(self, config: VectorDBConfig) -> None:
        if not _CHROMA_AVAILABLE:
            raise ProviderNotAvailableError("chromadb not installed")
        self._config = config
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        if self._config.chroma_in_memory:
            self._client = await asyncio.to_thread(chromadb.EphemeralClient)
        else:
            self._client = await asyncio.to_thread(
                chromadb.HttpClient,
                host=self._config.chroma_host,
                port=self._config.chroma_port,
            )
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, doc: VectorDocument) -> None:
        meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in doc.metadata.items()}
        meta["_content"] = doc.content
        meta["_created_at"] = doc.created_at
        kwargs: Dict[str, Any] = {
            "ids": [doc.id],
            "documents": [doc.content],
            "metadatas": [meta],
        }
        if doc.embedding:
            kwargs["embeddings"] = [doc.embedding]
        await asyncio.to_thread(self._collection.upsert, **kwargs)

    async def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for idx, mid in enumerate(result["ids"][0]):
            meta = dict(result["metadatas"][0][idx])
            content = meta.pop("_content", result["documents"][0][idx])
            created_at = float(meta.pop("_created_at", time.time()))
            doc = VectorDocument(id=mid, content=content, metadata=meta, created_at=created_at)
            score = 1.0 - result["distances"][0][idx]
            out.append((doc, score))
        return out

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, ids=[doc_id])

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._collection.count)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass


class _FaissStore:
    def __init__(self, config: VectorDBConfig) -> None:
        if not _FAISS_AVAILABLE:
            raise ProviderNotAvailableError("faiss-cpu not installed")
        self._config = config
        self._index: Any = None
        self._id_map: Dict[int, str] = {}
        self._doc_map: Dict[str, VectorDocument] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        import os
        dim = self._config.embedding_dim
        if os.path.exists(self._config.faiss_index_path):
            self._index = await asyncio.to_thread(faiss.read_index, self._config.faiss_index_path)
        else:
            self._index = faiss.IndexFlatIP(dim)
        logger.info("FAISS store initialized dim=%d", dim)

    async def upsert(self, doc: VectorDocument) -> None:
        if not doc.embedding:
            return
        async with self._lock:
            vec = np.array([doc.embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            self._index.add(vec)
            self._id_map[self._counter] = doc.id
            self._doc_map[doc.id] = doc
            self._counter += 1

    async def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        async with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
            out = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc_id = self._id_map.get(int(idx))
                if doc_id and doc_id in self._doc_map:
                    out.append((self._doc_map[doc_id], float(score)))
            return out

    async def delete(self, doc_id: str) -> None:
        async with self._lock:
            self._doc_map.pop(doc_id, None)

    async def count(self) -> int:
        return self._index.ntotal if self._index else 0

    async def health_check(self) -> bool:
        return self._index is not None

    async def close(self) -> None:
        if self._index and self._config.faiss_index_path:
            await asyncio.to_thread(faiss.write_index, self._index, self._config.faiss_index_path)


class _PineconeStore:
    def __init__(self, config: VectorDBConfig) -> None:
        if not _PINECONE_AVAILABLE:
            raise ProviderNotAvailableError("pinecone-client not installed")
        self._config = config
        self._index: Any = None

    async def initialize(self) -> None:
        api_key = self._config.pinecone_api_key or os.environ.get("PINECONE_API_KEY", "")
        pc = pinecone.Pinecone(api_key=api_key)
        self._index = await asyncio.to_thread(pc.Index, self._config.pinecone_index)
        logger.info("Pinecone store initialized index=%s", self._config.pinecone_index)

    async def upsert(self, doc: VectorDocument) -> None:
        if not doc.embedding:
            return
        meta = {k: str(v) for k, v in doc.metadata.items()}
        meta["content"] = doc.content
        await asyncio.to_thread(
            self._index.upsert,
            vectors=[(doc.id, doc.embedding, meta)],
            namespace=self._config.collection_name,
        )

    async def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        result = await asyncio.to_thread(
            self._index.query,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._config.collection_name,
        )
        out = []
        for match in result.matches:
            meta = dict(match.metadata or {})
            content = meta.pop("content", "")
            doc = VectorDocument(id=match.id, content=content, metadata=meta)
            out.append((doc, float(match.score)))
        return out

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(
            self._index.delete, ids=[doc_id], namespace=self._config.collection_name
        )

    async def count(self) -> int:
        stats = await asyncio.to_thread(self._index.describe_index_stats)
        ns = stats.namespaces.get(self._config.collection_name)
        return ns.vector_count if ns else 0

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._index.describe_index_stats)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass


class _QdrantStore:
    def __init__(self, config: VectorDBConfig) -> None:
        if not _QDRANT_AVAILABLE:
            raise ProviderNotAvailableError("qdrant-client not installed")
        self._config = config
        self._client: Any = None

    async def initialize(self) -> None:
        self._client = AsyncQdrantClient(
            url=self._config.qdrant_url,
            api_key=self._config.qdrant_api_key,
        )
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._config.collection_name not in names:
            await self._client.create_collection(
                collection_name=self._config.collection_name,
                vectors_config=VectorParams(
                    size=self._config.embedding_dim, distance=Distance.COSINE
                ),
            )
        logger.info("Qdrant store initialized collection=%s", self._config.collection_name)

    async def upsert(self, doc: VectorDocument) -> None:
        if not doc.embedding:
            return
        meta = {k: str(v) for k, v in doc.metadata.items()}
        meta["content"] = doc.content
        await self._client.upsert(
            collection_name=self._config.collection_name,
            points=[PointStruct(id=doc.id, vector=doc.embedding, payload=meta)],
        )

    async def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        results = await self._client.search(
            collection_name=self._config.collection_name,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        out = []
        for r in results:
            payload = dict(r.payload or {})
            content = payload.pop("content", "")
            doc = VectorDocument(id=str(r.id), content=content, metadata=payload)
            out.append((doc, float(r.score)))
        return out

    async def delete(self, doc_id: str) -> None:
        from qdrant_client.models import PointIdsList
        await self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=PointIdsList(points=[doc_id]),
        )

    async def count(self) -> int:
        info = await self._client.get_collection(self._config.collection_name)
        return info.points_count or 0

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


class _PgVectorStore:
    def __init__(self, config: VectorDBConfig) -> None:
        if not _PGVECTOR_AVAILABLE:
            raise ProviderNotAvailableError("asyncpg not installed")
        self._config = config
        self._pool: Any = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=self._config.pgvector_dsn)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._config.pgvector_table} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding vector({self._config.embedding_dim}),
                    metadata JSONB,
                    created_at DOUBLE PRECISION
                )
            """)
        logger.info("PgVector store initialized table=%s", self._config.pgvector_table)

    async def upsert(self, doc: VectorDocument) -> None:
        if not doc.embedding:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._config.pgvector_table} (id, content, embedding, metadata, created_at)
                VALUES ($1, $2, $3::vector, $4::jsonb, $5)
                ON CONFLICT (id) DO UPDATE
                SET content=EXCLUDED.content, embedding=EXCLUDED.embedding,
                    metadata=EXCLUDED.metadata
                """,
                doc.id, doc.content, str(doc.embedding), json.dumps(doc.metadata), doc.created_at,
            )

    async def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata, created_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM {self._config.pgvector_table}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(embedding), top_k,
            )
        out = []
        for row in rows:
            doc = VectorDocument(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=row["created_at"],
            )
            out.append((doc, float(row["score"])))
        return out

    async def delete(self, doc_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._config.pgvector_table} WHERE id=$1", doc_id
            )

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {self._config.pgvector_table}")

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

# ---------------------------------------------------------------------------
# Index Manager
# ---------------------------------------------------------------------------

class _IndexManager:
    def __init__(self) -> None:
        self._indexes: Dict[str, Any] = {}

    def register(self, name: str, store: Any) -> None:
        self._indexes[name] = store

    def get(self, name: str) -> Optional[Any]:
        return self._indexes.get(name)

    def list(self) -> List[str]:
        return list(self._indexes.keys())

# ---------------------------------------------------------------------------
# Metadata Store
# ---------------------------------------------------------------------------

class _MetadataStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    async def set(self, doc_id: str, meta: Dict[str, Any]) -> None:
        self._store[doc_id] = meta

    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(doc_id)

    async def delete(self, doc_id: str) -> None:
        self._store.pop(doc_id, None)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _VectorCache:
    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._store: Dict[str, Tuple[Any, float]] = {}

    async def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if item and (time.time() - item[1]) < self._ttl:
            return item[0]
        self._store.pop(key, None)
        return None

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())

    async def clear(self) -> None:
        self._store.clear()

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class _Retriever:
    def __init__(self, store: Any, embeddings: _EmbeddingsManager, cache: _VectorCache, config: VectorDBConfig) -> None:
        self._store = store
        self._emb = embeddings
        self._cache = cache
        self._config = config

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        import hashlib
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        cached = await self._cache.get(cache_key)
        if cached:
            return cached

        embedding = await self._emb.embed(query)
        raw = await self._store.query(embedding, top_k=top_k)
        results = [
            RetrievalResult(document=doc, score=score, provider=self._config.provider.value)
            for doc, score in raw
            if score >= min_score
        ]
        await self._cache.set(cache_key, results)
        return results

# ---------------------------------------------------------------------------
# VectorDBConnector – Singleton
# ---------------------------------------------------------------------------

_StoreType = Union[_ChromaStore, _FaissStore, _PineconeStore, _QdrantStore, _PgVectorStore]

class VectorDBConnector:
    _instance: Optional["VectorDBConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config: VectorDBConfig = VectorDBConfig()
        self._store: Optional[_StoreType] = None
        self._embeddings: Optional[_EmbeddingsManager] = None
        self._retriever: Optional[_Retriever] = None
        self._index_manager: _IndexManager = _IndexManager()
        self._metadata_store: _MetadataStore = _MetadataStore()
        self._cache: _VectorCache = _VectorCache()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "VectorDBConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "VectorDBConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def configure(self, config: VectorDBConfig) -> "VectorDBConnector":
        self._config = config
        return self

    def configure_from_env(self) -> "VectorDBConnector":
        provider_str = os.environ.get("VECTOR_PROVIDER", VectorProvider.CHROMA.value)
        try:
            provider = VectorProvider(provider_str)
        except ValueError:
            provider = VectorProvider.CHROMA
        emb_str = os.environ.get("EMBEDDING_PROVIDER", EmbeddingProvider.HASH.value)
        try:
            emb = EmbeddingProvider(emb_str)
        except ValueError:
            emb = EmbeddingProvider.HASH
        self._config = VectorDBConfig(
            provider=provider,
            embedding_provider=emb,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            chroma_in_memory=os.environ.get("CHROMA_IN_MEMORY", "true").lower() == "true",
        )
        return self

    async def initialize(self, config: Optional[VectorDBConfig] = None) -> "VectorDBConnector":
        if self._initialized:
            return self
        if config:
            self._config = config
        if not self._config:
            self.configure_from_env()

        store_map = {
            VectorProvider.CHROMA: _ChromaStore,
            VectorProvider.FAISS: _FaissStore,
            VectorProvider.PINECONE: _PineconeStore,
            VectorProvider.QDRANT: _QdrantStore,
            VectorProvider.PGVECTOR: _PgVectorStore,
        }
        StoreClass = store_map.get(self._config.provider)
        if StoreClass is None:
            raise VectorDBError(f"Unknown provider: {self._config.provider}")

        self._store = StoreClass(self._config)  # type: ignore[assignment]
        await self._store.initialize()
        self._index_manager.register(self._config.collection_name, self._store)

        self._embeddings = _EmbeddingsManager(self._config)
        await self._embeddings.initialize()

        self._cache = _VectorCache(ttl=self._config.cache_ttl)
        self._retriever = _Retriever(self._store, self._embeddings, self._cache, self._config)
        self._initialized = True
        logger.info("VectorDBConnector initialized provider=%s", self._config.provider.value)
        return self

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    def get_store(self) -> _StoreType:
        if not self._store:
            raise VectorDBError("Not initialized")
        return self._store

    async def embed(self, text: str) -> List[float]:
        await self._ensure()
        return await self._embeddings.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        await self._ensure()
        return await self._embeddings.embed_batch(texts)

    async def upsert(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> VectorDocument:
        await self._ensure()
        emb = embedding or await self._embeddings.embed(content)
        doc = VectorDocument(
            id=doc_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
            embedding=emb,
            collection=self._config.collection_name,
        )
        await self._store.upsert(doc)
        await self._metadata_store.set(doc.id, doc.metadata)
        await self._cache.clear()
        return doc

    async def upsert_batch(self, items: List[Dict[str, Any]]) -> List[VectorDocument]:
        await self._ensure()
        texts = [item.get("content", "") for item in items]
        embeddings = await self._embeddings.embed_batch(texts)
        docs = []
        for idx, item in enumerate(items):
            doc = VectorDocument(
                id=item.get("id", str(uuid.uuid4())),
                content=texts[idx],
                metadata=item.get("metadata", {}),
                embedding=embeddings[idx],
                collection=self._config.collection_name,
            )
            await self._store.upsert(doc)
            docs.append(doc)
        await self._cache.clear()
        return docs

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        await self._ensure()
        k = top_k or self._config.default_top_k
        return await self._retriever.retrieve(query=query, top_k=k, min_score=min_score)

    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        await self._ensure()
        raw = await self._store.query(embedding, top_k=top_k)
        return [
            RetrievalResult(document=doc, score=score, provider=self._config.provider.value)
            for doc, score in raw
            if score >= min_score
        ]

    async def delete(self, doc_id: str) -> None:
        await self._ensure()
        await self._store.delete(doc_id)
        await self._metadata_store.delete(doc_id)
        await self._cache.clear()

    async def get_stats(self) -> IndexStats:
        await self._ensure()
        count = await self._store.count()
        return IndexStats(
            provider=self._config.provider.value,
            collection=self._config.collection_name,
            total_documents=count,
            embedding_dim=self._config.embedding_dim,
        )

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure()
        ok = await self._store.health_check()
        stats = await self.get_stats()
        return {
            "healthy": ok,
            "provider": self._config.provider.value,
            "embedding_provider": self._config.embedding_provider.value,
            "collection": self._config.collection_name,
            "total_documents": stats.total_documents,
        }

    async def shutdown(self) -> None:
        if self._store:
            await self._store.close()
        self._initialized = False

    async def __aenter__(self) -> "VectorDBConnector":
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return f"VectorDBConnector(provider={self._config.provider.value}, initialized={self._initialized})"


def get_vectordb_connector() -> VectorDBConnector:
    return VectorDBConnector.get_instance()
