from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class RerankerType(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    LLM = "llm"
    BM25 = "bm25"
    NONE = "none"


@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    doc_name: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    id: str
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RetrievalResult:
    chunk: Chunk
    similarity: float
    rank: int
    rerank_score: Optional[float] = None


@dataclass
class RAGContext:
    query: str
    chunks: List[RetrievalResult]
    context_text: str
    token_estimate: int
    sources: List[str]
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "context_text": self.context_text,
            "token_estimate": self.token_estimate,
            "sources": self.sources,
            "chunks": [r.chunk.to_dict() for r in self.chunks],
            "latency_ms": self.latency_ms,
        }


@dataclass
class RAGServiceConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    embedding_model: str = "nvidia/nv-embed-v1"
    embedding_dim: int = 1536
    openai_api_key: Optional[str] = None
    top_k: int = 5
    similarity_threshold: float = 0.70
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    mmr_candidates: int = 20
    reranker: RerankerType = RerankerType.BM25
    cohere_api_key: Optional[str] = None
    context_max_tokens: int = 4000
    context_separator: str = "\n\n---\n\n"
    dedup_threshold: float = 0.95


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    def __init__(self, config: RAGServiceConfig) -> None:
        self._config = config

    def chunk(self, doc: Document) -> List[Chunk]:
        strategy = self._config.chunk_strategy
        if strategy == ChunkStrategy.FIXED:
            return self._fixed(doc)
        if strategy == ChunkStrategy.SENTENCE:
            return self._sentence(doc)
        if strategy == ChunkStrategy.PARAGRAPH:
            return self._paragraph(doc)
        return self._recursive(doc)

    def _make_chunk(self, doc: Document, text: str, idx: int, start: int) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            content=text.strip(),
            doc_id=doc.id,
            doc_name=doc.name,
            chunk_index=idx,
            start_char=start,
            end_char=start + len(text),
            metadata=dict(doc.metadata),
        )

    def _fixed(self, doc: Document) -> List[Chunk]:
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        text = doc.content
        chunks: List[Chunk] = []
        i = 0
        idx = 0
        while i < len(text):
            end = min(i + size, len(text))
            chunk_text = text[i:end]
            if chunk_text.strip():
                chunks.append(self._make_chunk(doc, chunk_text, idx, i))
                idx += 1
            i += size - overlap
        return chunks

    def _sentence(self, doc: Document) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks: List[Chunk] = []
        current = ""
        start = 0
        idx = 0
        pos = 0
        for sent in sentences:
            if len(current) + len(sent) > self._config.chunk_size and current:
                chunks.append(self._make_chunk(doc, current, idx, start))
                idx += 1
                overlap_text = current[-self._config.chunk_overlap:] if self._config.chunk_overlap else ""
                start = pos - len(overlap_text)
                current = overlap_text + sent
            else:
                if not current:
                    start = pos
                current += " " + sent if current else sent
            pos += len(sent) + 1
        if current.strip():
            chunks.append(self._make_chunk(doc, current, idx, start))
        return chunks

    def _paragraph(self, doc: Document) -> List[Chunk]:
        paragraphs = re.split(r'\n\s*\n', doc.content)
        chunks: List[Chunk] = []
        current = ""
        start = 0
        idx = 0
        pos = 0
        for para in paragraphs:
            if len(current) + len(para) > self._config.chunk_size and current:
                chunks.append(self._make_chunk(doc, current, idx, start))
                idx += 1
                current = para
                start = pos
            else:
                if not current:
                    start = pos
                current += "\n\n" + para if current else para
            pos += len(para) + 2
        if current.strip():
            chunks.append(self._make_chunk(doc, current, idx, start))
        return chunks

    def _recursive(self, doc: Document) -> List[Chunk]:
        separators = ["\n\n", "\n", ". ", " "]
        return self._split_recursive(doc, doc.content, 0, separators, 0)

    def _split_recursive(
        self, doc: Document, text: str, start: int, separators: List[str], idx_offset: int
    ) -> List[Chunk]:
        if len(text) <= self._config.chunk_size:
            if text.strip():
                return [self._make_chunk(doc, text, idx_offset, start)]
            return []
        sep = separators[0] if separators else " "
        parts = text.split(sep)
        chunks: List[Chunk] = []
        current = ""
        current_start = start
        idx = idx_offset
        pos = start
        for part in parts:
            if len(current) + len(part) + len(sep) > self._config.chunk_size and current:
                if len(current) > self._config.chunk_size and len(separators) > 1:
                    sub = self._split_recursive(doc, current, current_start, separators[1:], idx)
                    chunks.extend(sub)
                    idx += len(sub)
                else:
                    chunks.append(self._make_chunk(doc, current, idx, current_start))
                    idx += 1
                overlap = current[-self._config.chunk_overlap:] if self._config.chunk_overlap else ""
                current_start = pos - len(overlap)
                current = overlap + part
            else:
                if not current:
                    current_start = pos
                current += (sep if current else "") + part
            pos += len(part) + len(sep)
        if current.strip():
            chunks.append(self._make_chunk(doc, current, idx, current_start))
        return chunks


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    def __init__(self, config: RAGServiceConfig) -> None:
        self._config = config
        self._client: Any = None

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._config.openai_api_key)
            except ImportError:
                self._client = "mock"

        if self._client == "mock":
            return [self._mock_embed(t) for t in texts]

        batch_size = 100
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await self._client.embeddings.create(
                model=self._config.embedding_model, input=batch
            )
            all_embeddings.extend(item.embedding for item in resp.data)
        return all_embeddings

    def _mock_embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h]
        pad = self._config.embedding_dim - len(vec)
        vec.extend([0.0] * pad)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-10
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-10
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------

def _mmr(
    query_emb: List[float],
    candidates: List[Chunk],
    top_k: int,
    lmbda: float,
) -> List[Chunk]:
    if not candidates:
        return []
    scores = [_cosine(query_emb, c.embedding or []) for c in candidates]
    selected_indices: List[int] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        if not selected_indices:
            best = max(remaining, key=lambda i: scores[i])
        else:
            def mmr_score(i: int) -> float:
                rel = scores[i]
                redundancy = max(
                    _cosine(candidates[i].embedding or [], candidates[j].embedding or [])
                    for j in selected_indices
                )
                return lmbda * rel - (1 - lmbda) * redundancy
            best = max(remaining, key=mmr_score)
        selected_indices.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# BM25 reranker
# ---------------------------------------------------------------------------

class BM25Reranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        terms = query.lower().split()
        docs = [c.content.lower().split() for c in chunks]
        avg_dl = sum(len(d) for d in docs) / len(docs)
        scored: List[Tuple[float, Chunk]] = []
        for chunk, doc in zip(chunks, docs):
            score = 0.0
            doc_freq = {w: doc.count(w) for w in set(doc)}
            for term in terms:
                tf = doc_freq.get(term, 0)
                df = sum(1 for d in docs if term in d)
                idf = math.log((len(docs) - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self._k1 + 1)) / (
                    tf + self._k1 * (1 - self._b + self._b * len(doc) / avg_dl)
                )
                score += idf * tf_norm
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]


# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------

class CohereReranker:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        try:
            import cohere
            client = cohere.AsyncClient(self._api_key)
            docs = [c.content for c in chunks]
            resp = await client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=len(chunks),
            )
            reordered: List[Chunk] = []
            for hit in resp.results:
                chunk = chunks[hit.index]
                chunk.score = hit.relevance_score
                reordered.append(chunk)
            return reordered
        except Exception as exc:
            logger.warning("Cohere rerank failed: %s", exc)
            return chunks


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    def __init__(self, config: RAGServiceConfig) -> None:
        self._config = config

    def build(self, query: str, results: List[RetrievalResult], latency_ms: float) -> RAGContext:
        chars_budget = self._config.context_max_tokens * 4
        parts: List[str] = []
        sources: List[str] = []
        used = 0
        for r in results:
            chunk_text = f"[Source: {r.chunk.doc_name}, chunk {r.chunk.chunk_index}]\n{r.chunk.content}"
            if used + len(chunk_text) > chars_budget:
                break
            parts.append(chunk_text)
            if r.chunk.doc_name not in sources:
                sources.append(r.chunk.doc_name)
            used += len(chunk_text)

        context_text = self._config.context_separator.join(parts)
        token_estimate = used // 4
        return RAGContext(
            query=query,
            chunks=results,
            context_text=context_text,
            token_estimate=token_estimate,
            sources=sources,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Vector store (in-memory)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}
        self._doc_index: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def add(self, chunks: List[Chunk]) -> None:
        async with self._lock:
            for c in chunks:
                self._chunks[c.id] = c
                self._doc_index.setdefault(c.doc_id, []).append(c.id)

    async def delete_doc(self, doc_id: str) -> int:
        async with self._lock:
            ids = self._doc_index.pop(doc_id, [])
            for cid in ids:
                self._chunks.pop(cid, None)
            return len(ids)

    async def all_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        async with self._lock:
            if doc_id:
                ids = self._doc_index.get(doc_id, [])
                return [self._chunks[i] for i in ids if i in self._chunks]
            return list(self._chunks.values())

    async def search(
        self,
        query_emb: List[float],
        top_k: int,
        threshold: float,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        async with self._lock:
            candidates = list(self._chunks.values())
        if doc_filter:
            candidates = [c for c in candidates if c.doc_id in doc_filter]
        scored: List[Tuple[Chunk, float]] = []
        for chunk in candidates:
            if chunk.embedding is None:
                continue
            sim = _cosine(query_emb, chunk.embedding)
            if sim >= threshold:
                scored.append((chunk, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def doc_count(self) -> int:
        return len(self._doc_index)


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class RAGService:
    """
    Retrieval-Augmented Generation service with chunking, embedding,
    semantic search, MMR, reranking, and context preparation.
    """

    def __init__(self, config: Optional[RAGServiceConfig] = None) -> None:
        self._config = config or RAGServiceConfig()
        self._chunker = DocumentChunker(self._config)
        self._embedder = EmbeddingProvider(self._config)
        self._store = VectorStore()
        self._bm25 = BM25Reranker()
        self._cohere: Optional[CohereReranker] = (
            CohereReranker(self._config.cohere_api_key)
            if self._config.reranker == RerankerType.COHERE and self._config.cohere_api_key
            else None
        )
        self._context_builder = ContextBuilder(self._config)
        self._docs: Dict[str, Document] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def add_document(
        self,
        content: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        doc = Document(
            id=doc_id or str(uuid.uuid4()),
            name=name,
            content=content,
            metadata=metadata or {},
        )
        chunks = self._chunker.chunk(doc)
        if not chunks:
            return doc

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        await self._store.add(chunks)
        self._docs[doc.id] = doc
        logger.info("Indexed document '%s' → %d chunks", name, len(chunks))
        return doc

    async def add_documents_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Document]:
        return list(await asyncio.gather(*[self.add_document(**item) for item in items]))

    async def delete_document(self, doc_id: str) -> int:
        self._docs.pop(doc_id, None)
        return await self._store.delete_doc(doc_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        doc_filter: Optional[List[str]] = None,
        use_mmr: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        k = top_k or self._config.top_k
        thresh = threshold if threshold is not None else self._config.similarity_threshold
        apply_mmr = use_mmr if use_mmr is not None else self._config.use_mmr
        mmr_k = self._config.mmr_candidates if apply_mmr else k

        query_emb = (await self._embedder.embed([query]))[0]
        raw = await self._store.search(query_emb, mmr_k, thresh, doc_filter)

        if not raw:
            return []

        if apply_mmr:
            candidates = [chunk for chunk, _ in raw]
            selected = _mmr(query_emb, candidates, k, self._config.mmr_lambda)
        else:
            selected = [chunk for chunk, _ in raw[:k]]

        selected = self._dedup(selected)
        selected = await self._rerank(query, selected)

        results: List[RetrievalResult] = []
        sim_map = {chunk.id: sim for chunk, sim in raw}
        for rank, chunk in enumerate(selected):
            results.append(RetrievalResult(
                chunk=chunk,
                similarity=sim_map.get(chunk.id, 0.0),
                rank=rank + 1,
                rerank_score=chunk.score if chunk.score else None,
            ))
        return results

    async def retrieve_and_build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        doc_filter: Optional[List[str]] = None,
    ) -> RAGContext:
        t0 = time.monotonic()
        results = await self.retrieve(query, top_k, threshold, doc_filter)
        latency = (time.monotonic() - t0) * 1000
        return self._context_builder.build(query, results, latency)

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        results = await self.retrieve(query, top_k=top_k, threshold=threshold, use_mmr=False)
        return [
            {**r.chunk.to_dict(), "similarity": round(r.similarity, 4)}
            for r in results
        ]

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    async def augment_prompt(
        self,
        query: str,
        system_prompt: str,
        top_k: Optional[int] = None,
    ) -> str:
        ctx = await self.retrieve_and_build_context(query, top_k=top_k)
        if not ctx.chunks:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            f"## Retrieved Context\n{ctx.context_text}\n\n"
            f"## Sources\n" + "\n".join(f"- {s}" for s in ctx.sources)
        )

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {"id": d.id, "name": d.name, "metadata": d.metadata, "created_at": d.created_at}
            for d in self._docs.values()
        ]

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": self._store.doc_count,
            "chunk_count": self._store.chunk_count,
            "embedding_model": self._config.embedding_model,
            "chunk_strategy": self._config.chunk_strategy.value,
            "reranker": self._config.reranker.value,
            "mmr_enabled": self._config.use_mmr,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dedup(self, chunks: List[Chunk]) -> List[Chunk]:
        selected: List[Chunk] = []
        for chunk in chunks:
            duplicate = False
            for kept in selected:
                if chunk.embedding and kept.embedding:
                    sim = _cosine(chunk.embedding, kept.embedding)
                    if sim >= self._config.dedup_threshold:
                        duplicate = True
                        break
            if not duplicate:
                selected.append(chunk)
        return selected

    async def _rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if self._config.reranker == RerankerType.NONE or not chunks:
            return chunks
        if self._config.reranker == RerankerType.COHERE and self._cohere:
            return await self._cohere.rerank(query, chunks)
        return self._bm25.rerank(query, chunks)
