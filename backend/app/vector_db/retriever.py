"""
retriever.py

Production-grade retrieval engine supporting similarity search, Maximal
Marginal Relevance (MMR) diversification, reranking, metadata filtering,
namespace isolation, top-k retrieval, and citation generation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

Vector = List[float]
MetadataFilter = Dict[str, Any]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class RetrieverError(Exception):
    """Base exception for retrieval failures."""


class NoEmbedderConfiguredError(RetrieverError):
    """Raised when a text query is supplied but no embedder is configured."""


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[Vector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    source: Optional[str] = None


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    rank: int
    rerank_score: Optional[float] = None


@dataclass(frozen=True)
class Citation:
    document_id: str
    source: str
    snippet: str
    score: float
    rank: int


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    backend_healthy: bool
    embedder_healthy: bool
    reranker_healthy: bool
    latency_ms: float
    message: str = ""
    checked_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------

class VectorStoreBackend(Protocol):
    async def search(
        self,
        query_embedding: Vector,
        top_k: int,
        namespace: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> List[Tuple[Document, float]]:
        ...

    async def health_check(self) -> bool:
        ...


class Embedder(Protocol):
    async def embed(self, text: str) -> Vector:
        ...

    async def health_check(self) -> bool:
        ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: Sequence[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        ...

    async def health_check(self) -> bool:
        ...


# --------------------------------------------------------------------------
# Vector math helpers
# --------------------------------------------------------------------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a)) or 1e-12


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


# --------------------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------------------

_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda field_value, target: field_value == target,
    "$ne": lambda field_value, target: field_value != target,
    "$gt": lambda field_value, target: field_value is not None and field_value > target,
    "$gte": lambda field_value, target: field_value is not None and field_value >= target,
    "$lt": lambda field_value, target: field_value is not None and field_value < target,
    "$lte": lambda field_value, target: field_value is not None and field_value <= target,
    "$in": lambda field_value, target: field_value in target,
    "$nin": lambda field_value, target: field_value not in target,
    "$contains": lambda field_value, target: isinstance(field_value, str) and target in field_value,
}


def matches_filter(metadata: Dict[str, Any], filters: Optional[MetadataFilter]) -> bool:
    """Evaluate metadata against a filter spec.

    Supports plain equality (``{"category": "docs"}``) and operator-based
    conditions (``{"score": {"$gte": 0.5}}``).
    """
    if not filters:
        return True
    for key, condition in filters.items():
        field_value = metadata.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                fn = _OPERATORS.get(op)
                if fn is None:
                    raise RetrieverError(f"Unsupported filter operator: {op}")
                if not fn(field_value, target):
                    return False
        else:
            if field_value != condition:
                return False
    return True


# --------------------------------------------------------------------------
# MMR
# --------------------------------------------------------------------------

def maximal_marginal_relevance(
    query_embedding: Vector,
    candidates: Sequence[Tuple[Document, float]],
    top_k: int,
    lambda_mult: float = 0.5,
) -> List[Tuple[Document, float]]:
    """Select a diverse top-k subset of candidates using MMR.

    score = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected))
    """
    if not candidates:
        return []

    pool = list(candidates)
    selected: List[Tuple[Document, float]] = []
    selected_embeddings: List[Vector] = []

    while pool and len(selected) < top_k:
        best_idx = -1
        best_mmr_score = -math.inf

        for idx, (doc, relevance) in enumerate(pool):
            if doc.embedding is None:
                redundancy = 0.0
            else:
                redundancy = max(
                    (cosine_similarity(doc.embedding, sel_emb) for sel_emb in selected_embeddings),
                    default=0.0,
                )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        doc, relevance = pool.pop(best_idx)
        selected.append((doc, relevance))
        if doc.embedding is not None:
            selected_embeddings.append(doc.embedding)

    return selected


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------

class Retriever:
    """Coordinates similarity search, MMR diversification, reranking,
    metadata filtering, and citation generation across a vector store."""

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        default_namespace: str = "default",
        default_top_k: int = 10,
        overfetch_factor: int = 3,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._default_namespace = default_namespace
        self._default_top_k = default_top_k
        self._overfetch_factor = max(overfetch_factor, 1)

    async def _resolve_query_embedding(self, query: Union[str, Vector]) -> Tuple[str, Vector]:
        if isinstance(query, str):
            if self._embedder is None:
                raise NoEmbedderConfiguredError(
                    "A text query was provided but no embedder is configured."
                )
            embedding = await self._embedder.embed(query)
            return query, embedding
        return "", list(query)

    async def similarity_search(
        self,
        query: Union[str, Vector],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> List[RetrievedDocument]:
        """Plain top-k similarity search with namespace and metadata filtering."""
        _, query_embedding = await self._resolve_query_embedding(query)
        effective_top_k = top_k or self._default_top_k
        effective_namespace = namespace or self._default_namespace

        raw_results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            namespace=effective_namespace,
            filters=filters,
        )

        filtered = [(doc, score) for doc, score in raw_results if matches_filter(doc.metadata, filters)]
        filtered.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedDocument(document=doc, score=score, rank=idx)
            for idx, (doc, score) in enumerate(filtered[:effective_top_k])
        ]

    async def retrieve(
        self,
        query: Union[str, Vector],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        rerank: bool = False,
    ) -> List[RetrievedDocument]:
        """Full retrieval pipeline: search -> (optional MMR) -> (optional rerank)."""
        query_text, query_embedding = await self._resolve_query_embedding(query)
        effective_top_k = top_k or self._default_top_k
        effective_namespace = namespace or self._default_namespace
        fetch_k = effective_top_k * self._overfetch_factor if (use_mmr or rerank) else effective_top_k

        raw_results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            namespace=effective_namespace,
            filters=filters,
        )
        candidates = [(doc, score) for doc, score in raw_results if matches_filter(doc.metadata, filters)]
        candidates.sort(key=lambda pair: pair[1], reverse=True)

        if use_mmr:
            candidates = maximal_marginal_relevance(
                query_embedding, candidates, effective_top_k, mmr_lambda
            )

        rerank_scores: Dict[str, float] = {}
        if rerank:
            if self._reranker is None:
                raise RetrieverError("Reranking requested but no reranker is configured.")
            if not query_text:
                raise RetrieverError("Reranking requires a text query, not a raw vector.")
            reranked = await self._reranker.rerank(query_text, candidates)
            rerank_scores = {doc.id: score for doc, score in reranked}
            candidates = reranked

        candidates = candidates[:effective_top_k]

        return [
            RetrievedDocument(
                document=doc,
                score=score,
                rank=idx,
                rerank_score=rerank_scores.get(doc.id),
            )
            for idx, (doc, score) in enumerate(candidates)
        ]

    @staticmethod
    def build_citations(results: Sequence[RetrievedDocument], snippet_length: int = 200) -> List[Citation]:
        """Generate citation records suitable for grounding LLM responses."""
        citations: List[Citation] = []
        for result in results:
            doc = result.document
            snippet = doc.text[:snippet_length].rstrip()
            if len(doc.text) > snippet_length:
                snippet += "..."
            citations.append(
                Citation(
                    document_id=doc.id,
                    source=doc.source or doc.metadata.get("source", "unknown"),
                    snippet=snippet,
                    score=result.rerank_score if result.rerank_score is not None else result.score,
                    rank=result.rank,
                )
            )
        return citations

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        backend_healthy = False
        embedder_healthy = True
        reranker_healthy = True
        messages: List[str] = []

        try:
            backend_healthy = await self._vector_store.health_check()
            if not backend_healthy:
                messages.append("vector store unhealthy")
        except Exception as exc:  # noqa: BLE001
            backend_healthy = False
            messages.append(f"vector store error: {exc}")

        if self._embedder is not None:
            try:
                embedder_healthy = await self._embedder.health_check()
                if not embedder_healthy:
                    messages.append("embedder unhealthy")
            except Exception as exc:  # noqa: BLE001
                embedder_healthy = False
                messages.append(f"embedder error: {exc}")

        if self._reranker is not None:
            try:
                reranker_healthy = await self._reranker.health_check()
                if not reranker_healthy:
                    messages.append("reranker unhealthy")
            except Exception as exc:  # noqa: BLE001
                reranker_healthy = False
                messages.append(f"reranker error: {exc}")

        latency_ms = (time.monotonic() - start) * 1000
        healthy = backend_healthy and embedder_healthy and reranker_healthy
        return HealthStatus(
            healthy=healthy,
            backend_healthy=backend_healthy,
            embedder_healthy=embedder_healthy,
            reranker_healthy=reranker_healthy,
            latency_ms=latency_ms,
            message="; ".join(messages) if messages else "ok",
        )
