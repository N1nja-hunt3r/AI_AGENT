"""Production-quality async RAG service module.

Provides chunk retrieval, an embedding pipeline, retriever integration,
reranking, context preparation, citation preparation, and health checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from app.rag.reranker import Reranker, RerankResult

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class ServiceError(Exception):
    """Base error for RAGService operations."""


class EmbeddingError(ServiceError):
    """Raised when the embedding pipeline fails or times out."""


class RetrievalError(ServiceError):
    """Raised when retriever/vector-store interaction fails."""


class IndexingError(ServiceError):
    """Raised when document indexing fails."""


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: int = 0


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    retrieval_score: float


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedContext:
    context_text: str
    citations: List[Citation]
    token_estimate: int


@dataclass(frozen=True)
class QueryResult:
    query: str
    context: PreparedContext
    chunks: List[RerankResult]
    latency_ms: float


@dataclass(frozen=True)
class IndexResult:
    document_id: str
    chunk_ids: List[str]
    chunk_count: int


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    healthy: bool
    latency_ms: float
    detail: Optional[str] = None


@dataclass(frozen=True)
class HealthCheckResult:
    healthy: bool
    components: List[ComponentHealth]
    timestamp: float


# ----------------------------------------------------------------------
# Dependency protocols
# ----------------------------------------------------------------------

class AsyncEmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]: ...

    async def health(self) -> bool: ...


class TextChunker(Protocol):
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]: ...


class AsyncVectorStore(Protocol):
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]: ...

    async def health(self) -> bool: ...


# ----------------------------------------------------------------------
# Default chunker
# ----------------------------------------------------------------------

class SimpleTextChunker:
    """Fixed-size sliding-window chunker with character overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size).")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        normalized = text.strip()
        if not normalized:
            return []

        meta = dict(metadata or {})
        stride = self.chunk_size - self.chunk_overlap
        chunks: List[Chunk] = []
        position = 0
        start = 0
        length = len(normalized)

        while start < length:
            end = min(start + self.chunk_size, length)
            piece = normalized[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{position}:{uuid.uuid4().hex[:8]}",
                        document_id=document_id,
                        text=piece,
                        metadata=meta,
                        position=position,
                    )
                )
                position += 1
            if end >= length:
                break
            start += stride

        return chunks


# ----------------------------------------------------------------------
# Service configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ServiceConfig:
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    max_context_chars: int = 6000
    request_timeout_s: float = 10.0
    use_mmr: bool = True
    remove_redundancy: bool = True
    lambda_param: Optional[float] = None
    redundancy_threshold: Optional[float] = None


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

class RAGService:
    """Retrieval-augmented generation support service.

    Coordinates chunk retrieval, embedding, vector-store search,
    reranking, and context/citation preparation, with health checks
    for the underlying components.
    """

    def __init__(
        self,
        embedder: AsyncEmbeddingProvider,
        vector_store: AsyncVectorStore,
        chunker: Optional[TextChunker] = None,
        reranker: Optional[Reranker] = None,
        config: Optional[ServiceConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker or SimpleTextChunker()
        self.reranker = reranker or Reranker()
        self.config = config or ServiceConfig()

    # ------------------------------------------------------------------
    # Embedding pipeline
    # ------------------------------------------------------------------

    async def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = await asyncio.wait_for(
                self.embedder.embed(texts), timeout=self.config.request_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise EmbeddingError("Embedding request timed out.") from exc
        except Exception as exc:  # noqa: BLE001 - surface as ServiceError
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Embedder returned {len(embeddings)} vectors for {len(texts)} inputs."
            )
        return embeddings

    # ------------------------------------------------------------------
    # Chunk retrieval / retriever integration
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        embeddings = await self._embed([query])
        query_embedding = embeddings[0]
        try:
            results = await asyncio.wait_for(
                self.vector_store.search(query_embedding, top_k, filters),
                timeout=self.config.request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise RetrievalError("Vector store search timed out.") from exc
        except Exception as exc:  # noqa: BLE001 - surface as ServiceError
            raise RetrievalError(f"Vector store search failed: {exc}") from exc
        return list(results)

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    async def _rerank(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        top_k: int,
    ) -> List[RerankResult]:
        if not retrieved:
            return []
        documents = [item.chunk.text for item in retrieved]
        return await asyncio.to_thread(
            self.reranker.rerank,
            query,
            documents,
            top_k,
            self.config.use_mmr,
            self.config.remove_redundancy,
            self.config.lambda_param,
            self.config.redundancy_threshold,
        )

    # ------------------------------------------------------------------
    # Context preparation / citation preparation
    # ------------------------------------------------------------------

    def prepare_context(
        self,
        query: str,
        reranked: Sequence[RerankResult],
        retrieved: Sequence[RetrievedChunk],
        max_context_chars: Optional[int] = None,
    ) -> PreparedContext:
        """Build a citation-annotated context string from reranked chunks."""
        if not reranked:
            return PreparedContext(context_text="", citations=[], token_estimate=0)

        limit = max_context_chars if max_context_chars is not None else self.config.max_context_chars

        segments: List[str] = []
        citations: List[Citation] = []
        used_chars = 0

        for position, result in enumerate(reranked, start=1):
            if result.index < 0 or result.index >= len(retrieved):
                logger.warning("Rerank result index %d out of bounds; skipping.", result.index)
                continue

            source = retrieved[result.index]
            segment = f"[{position}] {result.document}"
            segment_len = len(segment) + 2  # account for joining separator

            if used_chars + segment_len > limit and segments:
                break

            segments.append(segment)
            used_chars += segment_len
            citations.append(
                Citation(
                    chunk_id=source.chunk.chunk_id,
                    document_id=source.chunk.document_id,
                    text=result.document,
                    score=result.final_score,
                    metadata=source.chunk.metadata,
                )
            )

        context_text = "\n\n".join(segments)
        token_estimate = max(1, len(context_text) // 4) if context_text else 0
        return PreparedContext(
            context_text=context_text,
            citations=citations,
            token_estimate=token_estimate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieval_top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_context_chars: Optional[int] = None,
    ) -> QueryResult:
        """Run the end-to-end retrieve -> rerank -> prepare-context pipeline."""
        start = time.monotonic()

        retrieved = await self._retrieve(
            query,
            retrieval_top_k or self.config.retrieval_top_k,
            filters,
        )

        if not retrieved:
            return QueryResult(
                query=query,
                context=PreparedContext(context_text="", citations=[], token_estimate=0),
                chunks=[],
                latency_ms=(time.monotonic() - start) * 1000.0,
            )

        reranked = await self._rerank(query, retrieved, top_k or self.config.rerank_top_k)
        context = self.prepare_context(query, reranked, retrieved, max_context_chars)

        return QueryResult(
            query=query,
            context=context,
            chunks=list(reranked),
            latency_ms=(time.monotonic() - start) * 1000.0,
        )

    async def index_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexResult:
        """Chunk, embed, and upsert a document into the vector store."""
        chunks = self.chunker.chunk(text, document_id, metadata)
        if not chunks:
            return IndexResult(document_id=document_id, chunk_ids=[], chunk_count=0)

        embeddings = await self._embed([chunk.text for chunk in chunks])

        try:
            await asyncio.wait_for(
                self.vector_store.upsert(chunks, embeddings),
                timeout=self.config.request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise IndexingError("Vector store upsert timed out.") from exc
        except Exception as exc:  # noqa: BLE001 - surface as ServiceError
            raise IndexingError(f"Vector store upsert failed: {exc}") from exc

        return IndexResult(
            document_id=document_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            chunk_count=len(chunks),
        )

    async def health_check(self) -> HealthCheckResult:
        """Check liveness of downstream embedding and vector-store components."""
        components: List[ComponentHealth] = []
        overall_healthy = True

        checks: List[tuple[str, Any]] = [
            ("embedder", self.embedder.health),
            ("vector_store", self.vector_store.health),
        ]

        for name, health_fn in checks:
            t0 = time.monotonic()
            try:
                ok = await asyncio.wait_for(health_fn(), timeout=self.config.request_timeout_s)
                latency_ms = (time.monotonic() - t0) * 1000.0
                healthy = bool(ok)
                components.append(
                    ComponentHealth(name=name, healthy=healthy, latency_ms=latency_ms)
                )
                overall_healthy = overall_healthy and healthy
            except Exception as exc:  # noqa: BLE001 - degrade gracefully per component
                latency_ms = (time.monotonic() - t0) * 1000.0
                components.append(
                    ComponentHealth(
                        name=name,
                        healthy=False,
                        latency_ms=latency_ms,
                        detail=str(exc),
                    )
                )
                overall_healthy = False

        return HealthCheckResult(
            healthy=overall_healthy,
            components=components,
            timestamp=time.time(),
        )
