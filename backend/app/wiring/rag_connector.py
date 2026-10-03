"""
rag_connector.py - Production-grade RAG connector.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore

try:
    import pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False
    pinecone = None  # type: ignore

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _AsyncOpenAI = None  # type: ignore

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"

class VectorDBBackend(str, Enum):
    IN_MEMORY = "in_memory"
    CHROMA = "chroma"
    PINECONE = "pinecone"

class EmbedderBackend(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HASH = "hash"

class RerankerBackend(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    SCORE_FUSION = "score_fusion"
    NONE = "none"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    doc_id: str = ""
    doc_title: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            doc_id=data.get("doc_id", ""),
            doc_title=data.get("doc_title", ""),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 0),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
        )


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    title: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rerank_score: Optional[float] = None
    retrieval_ms: float = 0.0


@dataclass
class RAGResponse:
    query: str
    results: List[RetrievalResult]
    context: str
    total_chunks_searched: int
    retrieval_ms: float
    reranking_ms: float
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RAGConfig:
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    embedder: EmbedderBackend = EmbedderBackend.HASH
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "nvidia/nv-embed-v1"
    openai_api_key: Optional[str] = None
    vector_db: VectorDBBackend = VectorDBBackend.IN_MEMORY
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "rag_chunks"
    chroma_in_memory: bool = True
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "rag-index"
    reranker: RerankerBackend = RerankerBackend.NONE
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 20
    top_k_final: int = 5
    similarity_threshold: float = 0.3
    context_max_chars: int = 8000
    extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RAGConnectorError(Exception):
    pass

class ChunkingError(RAGConnectorError):
    pass

class EmbeddingError(RAGConnectorError):
    pass

class VectorDBError(RAGConnectorError):
    pass

class RerankerError(RAGConnectorError):
    pass

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4

    def _split_fixed(self, text: str) -> List[str]:
        size = self.config.chunk_size * 4
        overlap = self.config.chunk_overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            start += size - overlap
        return chunks

    def _split_sentence(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current = ""
        for s in sentences:
            if len(current) + len(s) <= self.config.chunk_size * 4:
                current += (" " if current else "") + s
            else:
                if current and len(current) >= self.config.min_chunk_size:
                    chunks.append(current.strip())
                current = s
        if current and len(current) >= self.config.min_chunk_size:
            chunks.append(current.strip())
        return chunks

    def _split_paragraph(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) <= self.config.chunk_size * 4:
                current += ("\n\n" if current else "") + p
            else:
                if current and len(current) >= self.config.min_chunk_size:
                    chunks.append(current.strip())
                current = p
        if current and len(current) >= self.config.min_chunk_size:
            chunks.append(current.strip())
        return chunks

    def _split_recursive(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        max_size = self.config.chunk_size * 4
        if len(text) <= max_size:
            return [text] if len(text) >= self.config.min_chunk_size else []
        for sep in separators:
            if sep and sep in text:
                parts = text.split(sep)
                chunks: List[str] = []
                current = ""
                for part in parts:
                    candidate = current + (sep if current else "") + part
                    if len(candidate) <= max_size:
                        current = candidate
                    else:
                        if current and len(current) >= self.config.min_chunk_size:
                            chunks.append(current.strip())
                        if len(part) > max_size:
                            sub = self._split_recursive(part, separators[separators.index(sep) + 1:])
                            chunks.extend(sub)
                            current = ""
                        else:
                            current = part
                if current and len(current) >= self.config.min_chunk_size:
                    chunks.append(current.strip())
                if chunks:
                    return chunks
        return self._split_fixed(text)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        text = doc.content
        strategy = self.config.chunk_strategy
        if strategy == ChunkStrategy.FIXED:
            raw_chunks = self._split_fixed(text)
        elif strategy == ChunkStrategy.SENTENCE:
            raw_chunks = self._split_sentence(text)
        elif strategy == ChunkStrategy.PARAGRAPH:
            raw_chunks = self._split_paragraph(text)
        elif strategy == ChunkStrategy.RECURSIVE:
            raw_chunks = self._split_recursive(text)
        else:
            raw_chunks = self._split_recursive(text)

        chunks: List[Chunk] = []
        pos = 0
        for idx, raw in enumerate(raw_chunks):
            start = text.find(raw, pos)
            if start == -1:
                start = pos
            end = start + len(raw)
            chunk = Chunk(
                doc_id=doc.id,
                doc_title=doc.title,
                content=raw,
                chunk_index=idx,
                total_chunks=len(raw_chunks),
                metadata={**doc.metadata, "source": doc.source},
                token_count=self._count_tokens(raw),
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
            pos = end
        return chunks

    def chunk_text(self, text: str, doc_id: str = "", title: str = "") -> List[Chunk]:
        doc = Document(id=doc_id or str(uuid.uuid4()), content=text, title=title)
        return self.chunk_document(doc)

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._st_model: Any = None
        self._openai_client: Any = None
        self._dim = 128
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        backend = self.config.embedder
        if backend == EmbedderBackend.SENTENCE_TRANSFORMERS:
            if _ST_AVAILABLE:
                self._st_model = await asyncio.to_thread(SentenceTransformer, self.config.embedding_model)
                self._dim = self._st_model.get_sentence_embedding_dimension()
                logger.info("ST embedder loaded: %s dim=%d", self.config.embedding_model, self._dim)
            else:
                logger.warning("sentence-transformers unavailable; using hash embedder")
                self.config.embedder = EmbedderBackend.HASH
        elif backend == EmbedderBackend.OPENAI:
            if _OPENAI_AVAILABLE:
                api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
                self._openai_client = _AsyncOpenAI(api_key=api_key)
                self._dim = 1536
            else:
                logger.warning("openai unavailable; using hash embedder")
                self.config.embedder = EmbedderBackend.HASH
        self._initialized = True

    async def embed(self, text: str) -> List[float]:
        if not self._initialized:
            await self.initialize()
        try:
            if self.config.embedder == EmbedderBackend.SENTENCE_TRANSFORMERS and self._st_model:
                v = await asyncio.to_thread(self._st_model.encode, text)
                return v.tolist() if hasattr(v, "tolist") else list(v)
            elif self.config.embedder == EmbedderBackend.OPENAI and self._openai_client:
                r = await self._openai_client.embeddings.create(
                    model=self.config.openai_embedding_model, input=text
                )
                return r.data[0].embedding
            else:
                return self._hash_embed(text)
        except Exception as exc:
            logger.warning("Embed failed: %s; using hash", exc)
            return self._hash_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()
        try:
            if self.config.embedder == EmbedderBackend.SENTENCE_TRANSFORMERS and self._st_model:
                vecs = await asyncio.to_thread(self._st_model.encode, texts)
                return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
            elif self.config.embedder == EmbedderBackend.OPENAI and self._openai_client:
                r = await self._openai_client.embeddings.create(
                    model=self.config.openai_embedding_model, input=texts
                )
                return [d.embedding for d in r.data]
            else:
                return [self._hash_embed(t) for t in texts]
        except Exception as exc:
            logger.warning("Batch embed failed: %s; using hash", exc)
            return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str, dim: int = 128) -> List[float]:
        digest = hashlib.sha256(text.encode()).digest()
        extended = (digest * ((dim * 4 // len(digest)) + 2))[: dim * 4]
        floats = []
        for i in range(0, len(extended) - 3, 4):
            val = int.from_bytes(extended[i:i+4], "little", signed=True)
            floats.append(val / 2_147_483_648.0)
        vec = floats[:dim]
        norm = (sum(v*v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        if _NUMPY_AVAILABLE:
            va, vb = np.array(a), np.array(b)
            d = float(np.linalg.norm(va) * np.linalg.norm(vb))
            return float(np.dot(va, vb) / d) if d else 0.0
        dot = sum(x*y for x,y in zip(a,b))
        na = sum(x*x for x in a)**0.5
        nb = sum(x*x for x in b)**0.5
        return dot / (na*nb) if na*nb else 0.0

    @property
    def dim(self) -> int:
        return self._dim

# ---------------------------------------------------------------------------
# VectorDB backends
# ---------------------------------------------------------------------------

class _InMemoryVectorDB:
    def __init__(self) -> None:
        self._store: Dict[str, Chunk] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: List[Chunk]) -> None:
        async with self._lock:
            for c in chunks:
                self._store[c.id] = c

    async def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        async with self._lock:
            entries = list(self._store.values())
        scored = []
        for c in entries:
            if not c.embedding:
                continue
            if filters:
                skip = False
                for k, v in filters.items():
                    if c.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            score = Embedder.cosine_similarity(embedding, c.embedding)
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete_by_doc(self, doc_id: str) -> int:
        async with self._lock:
            keys = [k for k, v in self._store.items() if v.doc_id == doc_id]
            for k in keys:
                del self._store[k]
            return len(keys)

    async def count(self) -> int:
        async with self._lock:
            return len(self._store)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _ChromaVectorDB:
    def __init__(self, config: RAGConfig) -> None:
        if not _CHROMA_AVAILABLE:
            raise RAGConnectorError("chromadb not installed")
        self.config = config
        self._client: Any = None
        self._collection: Any = None

    async def _init(self) -> None:
        if self._client:
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

    async def upsert(self, chunks: List[Chunk]) -> None:
        await self._init()
        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding]
        documents = [c.content for c in chunks]
        metadatas = []
        for c in chunks:
            m = c.to_dict()
            m.pop("embedding", None)
            m["metadata"] = json.dumps(m.get("metadata", {}))
            metadatas.append({k: str(v) for k, v in m.items()})
        if embeddings and len(embeddings) == len(ids):
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas,
            )
        else:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids, documents=documents, metadatas=metadatas,
            )

    async def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        await self._init()
        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            result = await asyncio.to_thread(self._collection.query, **kwargs)
            out = []
            for idx, cid in enumerate(result["ids"][0]):
                meta = result["metadatas"][0][idx]
                meta["id"] = cid
                try:
                    meta["metadata"] = json.loads(meta.get("metadata", "{}"))
                except Exception:
                    meta["metadata"] = {}
                chunk = Chunk.from_dict(meta)
                chunk.content = result["documents"][0][idx]
                score = 1.0 - result["distances"][0][idx]
                out.append((chunk, score))
            return out
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

    async def delete_by_doc(self, doc_id: str) -> int:
        await self._init()
        try:
            result = await asyncio.to_thread(
                self._collection.get, where={"doc_id": doc_id}, include=[]
            )
            ids = result["ids"]
            if ids:
                await asyncio.to_thread(self._collection.delete, ids=ids)
            return len(ids)
        except Exception:
            return 0

    async def count(self) -> int:
        await self._init()
        try:
            return await asyncio.to_thread(self._collection.count)
        except Exception:
            return 0

    async def health_check(self) -> bool:
        try:
            await self._init()
            await asyncio.to_thread(self._collection.count)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        self._client = None
        self._collection = None

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._model: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self.config.reranker == RerankerBackend.CROSS_ENCODER:
            if _ST_AVAILABLE and CrossEncoder:
                try:
                    self._model = await asyncio.to_thread(CrossEncoder, self.config.reranker_model)
                    logger.info("CrossEncoder reranker loaded: %s", self.config.reranker_model)
                except Exception as exc:
                    logger.warning("CrossEncoder load failed: %s", exc)
            else:
                logger.warning("sentence-transformers unavailable for reranker")
        self._initialized = True

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        if not self._initialized:
            await self.initialize()
        if self.config.reranker == RerankerBackend.NONE:
            return results[:top_k]
        if self.config.reranker == RerankerBackend.SCORE_FUSION:
            return self._score_fusion(results, top_k)
        if self.config.reranker == RerankerBackend.CROSS_ENCODER and self._model:
            return await self._cross_encoder_rerank(query, results, top_k)
        return results[:top_k]

    def _score_fusion(self, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        for idx, r in enumerate(results):
            rrf_score = 1.0 / (60 + idx + 1)
            r.rerank_score = r.score * 0.7 + rrf_score * 0.3
        results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        return results[:top_k]

    async def _cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        pairs = [[query, r.chunk.content] for r in results]
        try:
            scores = await asyncio.to_thread(self._model.predict, pairs)
            for r, score in zip(results, scores):
                r.rerank_score = float(score)
            results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        except Exception as exc:
            logger.warning("CrossEncoder rerank failed: %s", exc)
        return results[:top_k]

# ---------------------------------------------------------------------------
# RAGConnector – Singleton
# ---------------------------------------------------------------------------

class RAGConnector:
    _instance: Optional["RAGConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = RAGConfig()
        self._chunker: Optional[Chunker] = None
        self._embedder: Optional[Embedder] = None
        self._vector_db: Optional[Union[_InMemoryVectorDB, _ChromaVectorDB]] = None
        self._reranker: Optional[Reranker] = None
        self._initialized = False
        self._stats: Dict[str, int] = {
            "docs_indexed": 0,
            "chunks_indexed": 0,
            "queries": 0,
        }

    @classmethod
    def get_instance(cls) -> "RAGConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "RAGConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    async def initialize(self, config: Optional[RAGConfig] = None) -> "RAGConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        self._chunker = Chunker(self._config)
        self._embedder = Embedder(self._config)
        await self._embedder.initialize()

        if self._config.vector_db == VectorDBBackend.CHROMA:
            self._vector_db = _ChromaVectorDB(self._config)
        else:
            self._vector_db = _InMemoryVectorDB()

        self._reranker = Reranker(self._config)
        await self._reranker.initialize()

        self._initialized = True
        logger.info(
            "RAGConnector initialized: chunker=%s embedder=%s vectordb=%s reranker=%s",
            self._config.chunk_strategy.value,
            self._config.embedder.value,
            self._config.vector_db.value,
            self._config.reranker.value,
        )
        return self

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_document(
        self,
        content: str,
        title: str = "",
        source: str = "",
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        await self._ensure()
        doc = Document(
            id=doc_id or str(uuid.uuid4()),
            content=content,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        chunks = self._chunker.chunk_document(doc)  # type: ignore[union-attr]
        if not chunks:
            return doc.id, 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed_batch(texts)  # type: ignore[union-attr]
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        await self._vector_db.upsert(chunks)  # type: ignore[union-attr]
        self._stats["docs_indexed"] += 1
        self._stats["chunks_indexed"] += len(chunks)
        logger.debug("Indexed doc=%s chunks=%d", doc.id, len(chunks))
        return doc.id, len(chunks)

    async def index_batch(
        self, documents: List[Dict[str, Any]]
    ) -> List[Tuple[str, int]]:
        await self._ensure()
        tasks = [
            self.index_document(
                content=d.get("content", ""),
                title=d.get("title", ""),
                source=d.get("source", ""),
                doc_id=d.get("id"),
                metadata=d.get("metadata"),
            )
            for d in documents
        ]
        return await asyncio.gather(*tasks)

    async def delete_document(self, doc_id: str) -> int:
        await self._ensure()
        return await self._vector_db.delete_by_doc(doc_id)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        use_reranker: bool = True,
    ) -> RAGResponse:
        await self._ensure()
        t0 = time.perf_counter()
        k_retrieval = self._config.top_k_retrieval
        k_final = top_k or self._config.top_k_final

        query_emb = await self._embedder.embed(query)  # type: ignore[union-attr]
        raw_pairs = await self._vector_db.query(  # type: ignore[union-attr]
            embedding=query_emb,
            top_k=k_retrieval,
            filters=filters,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        results = [
            RetrievalResult(chunk=chunk, score=score, retrieval_ms=retrieval_ms)
            for chunk, score in raw_pairs
            if score >= min_score
        ]
        self._stats["queries"] += 1

        tr = time.perf_counter()
        if use_reranker and self._reranker:
            results = await self._reranker.rerank(query, results, top_k=k_final)
        else:
            results = results[:k_final]
        reranking_ms = (time.perf_counter() - tr) * 1000

        context_parts = []
        total_chars = 0
        for r in results:
            segment = f"[Source: {r.chunk.doc_title or r.chunk.doc_id}]\n{r.chunk.content}"
            if total_chars + len(segment) > self._config.context_max_chars:
                break
            context_parts.append(segment)
            total_chars += len(segment)
        context = "\n\n---\n\n".join(context_parts)

        return RAGResponse(
            query=query,
            results=results,
            context=context,
            total_chunks_searched=len(raw_pairs),
            retrieval_ms=retrieval_ms,
            reranking_ms=reranking_ms,
        )

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        response = await self.retrieve(query=query, top_k=top_k, filters=filters)
        return response.context

    # ------------------------------------------------------------------
    # Health & stats
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure()
        db_ok = await self._vector_db.health_check()  # type: ignore[union-attr]
        count = await self._vector_db.count()  # type: ignore[union-attr]
        return {
            "healthy": db_ok,
            "vector_db": self._config.vector_db.value,
            "embedder": self._config.embedder.value,
            "chunk_strategy": self._config.chunk_strategy.value,
            "reranker": self._config.reranker.value,
            "total_chunks": count,
            **self._stats,
        }

    async def __aenter__(self) -> "RAGConnector":
        await self._ensure()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._vector_db:
            await self._vector_db.close()
        self._initialized = False

    def __repr__(self) -> str:
        return (
            f"RAGConnector(db={self._config.vector_db.value}, "
            f"chunks={self._stats['chunks_indexed']}, "
            f"queries={self._stats['queries']})"
        )


def get_rag_connector() -> RAGConnector:
    return RAGConnector.get_instance()


async def index_document(content: str, **kwargs: Any) -> Tuple[str, int]:
    return await get_rag_connector().index_document(content=content, **kwargs)


async def retrieve(query: str, **kwargs: Any) -> RAGResponse:
    return await get_rag_connector().retrieve(query=query, **kwargs)
