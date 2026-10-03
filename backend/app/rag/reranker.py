"""Production-quality reranking module.

Supports cross-encoder relevance scoring, Maximal Marginal Relevance (MMR)
for diversity-aware ranking, diversity scoring, and redundancy removal.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

Vector = Union[Dict[str, float], Sequence[float]]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class CrossEncoderProtocol(Protocol):
    """Interface for cross-encoder models (e.g. sentence-transformers CrossEncoder)."""

    def predict(self, sentence_pairs: Sequence[Tuple[str, str]]) -> Sequence[float]:
        ...


class EmbedderProtocol(Protocol):
    """Interface for embedding models producing dense vectors for texts."""

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True)
class RerankResult:
    """A single reranked document with associated scoring metadata."""

    index: int
    document: str
    relevance_score: float
    diversity_penalty: float = 0.0
    final_score: float = 0.0


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _bow_vector(text: str) -> Dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(c * c for c in counts.values()))
    if norm == 0:
        return {}
    return {term: count / norm for term, count in counts.items()}


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two vectors.

    Supports sparse bag-of-words vectors (Dict[str, float]) and dense
    embedding vectors (Sequence[float]).
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise TypeError("Both vectors must be the same representation type.")
        if not a or not b:
            return 0.0
        shared = set(a.keys()) & set(b.keys())
        numerator = sum(a[k] * b[k] for k in shared)
        denom_a = math.sqrt(sum(v * v for v in a.values()))
        denom_b = math.sqrt(sum(v * v for v in b.values()))
        if denom_a == 0.0 or denom_b == 0.0:
            return 0.0
        return numerator / (denom_a * denom_b)

    denom_a = _norm(a)
    denom_b = _norm(b)
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return _dot(a, b) / (denom_a * denom_b)


class Reranker:
    """Reranks candidate documents against a query.

    Combines optional cross-encoder relevance scoring with Maximal Marginal
    Relevance (MMR) for diversity-aware final ranking, plus utilities for
    diversity scoring and redundancy removal.
    """

    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
        lambda_param: float = 0.5,
        redundancy_threshold: float = 0.92,
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in [0.0, 1.0].")
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be in [0.0, 1.0].")
        self.cross_encoder = cross_encoder
        self.embedder = embedder
        self.lambda_param = lambda_param
        self.redundancy_threshold = redundancy_threshold

    # ------------------------------------------------------------------
    # Vectorization helpers
    # ------------------------------------------------------------------

    def _vectorize(self, texts: Sequence[str]) -> List[Vector]:
        if self.embedder is not None:
            embeddings = self.embedder(texts)
            return [list(vec) for vec in embeddings]
        return [_bow_vector(text) for text in texts]

    def _similarity_matrix(self, documents: Sequence[str]) -> List[List[float]]:
        vectors = self._vectorize(documents)
        n = len(vectors)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = cosine_similarity(vectors[i], vectors[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        """Compute relevance scores between a query and each document.

        Uses the configured cross-encoder if available, falling back to
        bag-of-words cosine similarity otherwise.
        """
        if not documents:
            return []

        if self.cross_encoder is not None:
            pairs = [(query, doc) for doc in documents]
            raw_scores = list(self.cross_encoder.predict(pairs))
            return [float(s) for s in raw_scores]

        query_vector = _bow_vector(query)
        doc_vectors = self._vectorize(documents)
        return [cosine_similarity(query_vector, vec) for vec in doc_vectors]

    # ------------------------------------------------------------------
    # Diversity & redundancy
    # ------------------------------------------------------------------

    def diversity_score(self, documents: Sequence[str]) -> float:
        """Return a diversity score in [0.0, 1.0] for a set of documents.

        Defined as one minus the mean pairwise similarity across all
        document pairs. 1.0 means maximally diverse content; 0.0 means
        all documents are near-duplicates.
        """
        n = len(documents)
        if n <= 1:
            return 1.0
        matrix = self._similarity_matrix(documents)
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += matrix[i][j]
                pairs += 1
        if pairs == 0:
            return 1.0
        mean_similarity = total / pairs
        return max(0.0, min(1.0, 1.0 - mean_similarity))

    def remove_redundant(
        self,
        documents: Sequence[str],
        scores: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[int], List[str], List[float]]:
        """Greedily remove near-duplicate documents.

        Documents are processed in descending score order; a document is
        dropped if its similarity to any already-kept document meets or
        exceeds the redundancy threshold.

        Returns (kept_indices, kept_documents, kept_scores), ordered by
        descending score.
        """
        n = len(documents)
        if n == 0:
            return [], [], []

        effective_threshold = (
            self.redundancy_threshold if threshold is None else threshold
        )
        effective_scores = list(scores) if scores is not None else [0.0] * n
        if len(effective_scores) != n:
            raise ValueError("scores must be the same length as documents.")

        order = sorted(range(n), key=lambda i: effective_scores[i], reverse=True)
        matrix = self._similarity_matrix(documents)

        kept: List[int] = []
        for idx in order:
            is_redundant = any(
                matrix[idx][kept_idx] >= effective_threshold for kept_idx in kept
            )
            if not is_redundant:
                kept.append(idx)

        kept_sorted = sorted(kept, key=lambda i: effective_scores[i], reverse=True)
        kept_documents = [documents[i] for i in kept_sorted]
        kept_scores = [effective_scores[i] for i in kept_sorted]
        return kept_sorted, kept_documents, kept_scores

    # ------------------------------------------------------------------
    # MMR
    # ------------------------------------------------------------------

    def mmr(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        lambda_param: Optional[float] = None,
        relevance_scores: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """Select a diverse, relevant subset of document indices using
        Maximal Marginal Relevance.

        MMR(d) = lambda * Relevance(d) - (1 - lambda) * max_sim(d, selected)
        """
        n = len(documents)
        if n == 0:
            return []

        k = n if top_k is None else min(top_k, n)
        lam = self.lambda_param if lambda_param is None else lambda_param
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lambda_param must be in [0.0, 1.0].")

        relevance = (
            list(relevance_scores)
            if relevance_scores is not None
            else self.score(query, documents)
        )
        if len(relevance) != n:
            raise ValueError("relevance_scores must match documents length.")

        sim_matrix = self._similarity_matrix(documents)

        remaining = set(range(n))
        selected: List[int] = []

        first = max(remaining, key=lambda i: relevance[i])
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < k:
            best_idx: Optional[int] = None
            best_mmr_score = float("-inf")
            for idx in remaining:
                max_sim = max(sim_matrix[idx][s] for s in selected)
                mmr_score = lam * relevance[idx] - (1.0 - lam) * max_sim
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            assert best_idx is not None
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        use_mmr: bool = True,
        remove_redundancy: bool = True,
        lambda_param: Optional[float] = None,
        redundancy_threshold: Optional[float] = None,
    ) -> List[RerankResult]:
        """Run the full reranking pipeline.

        Steps: score documents against the query, optionally remove
        redundant near-duplicates, optionally apply MMR for diversity-aware
        ordering, and return ranked results with metadata.
        """
        if not documents:
            return []

        scores = self.score(query, documents)

        working_indices = list(range(len(documents)))
        working_documents = list(documents)
        working_scores = list(scores)

        if remove_redundancy:
            kept_indices, working_documents, working_scores = self.remove_redundant(
                working_documents,
                scores=working_scores,
                threshold=redundancy_threshold,
            )
            working_indices = [working_indices[i] for i in kept_indices]

        if use_mmr:
            order = self.mmr(
                query,
                working_documents,
                top_k=top_k,
                lambda_param=lambda_param,
                relevance_scores=working_scores,
            )
        else:
            order = sorted(
                range(len(working_documents)),
                key=lambda i: working_scores[i],
                reverse=True,
            )
            if top_k is not None:
                order = order[:top_k]

        effective_lambda = (
            lambda_param if lambda_param is not None else self.lambda_param
        )
        sim_matrix = (
            self._similarity_matrix(working_documents)
            if len(working_documents) > 1
            else [[1.0]]
        )

        results: List[RerankResult] = []
        chosen: List[int] = []
        for pos in order:
            max_sim_to_chosen = max(
                (sim_matrix[pos][c] for c in chosen), default=0.0
            )
            relevance = working_scores[pos]
            final = relevance - max_sim_to_chosen * (1.0 - effective_lambda)
            results.append(
                RerankResult(
                    index=working_indices[pos],
                    document=working_documents[pos],
                    relevance_score=relevance,
                    diversity_penalty=max_sim_to_chosen,
                    final_score=final,
                )
            )
            chosen.append(pos)

        return results
