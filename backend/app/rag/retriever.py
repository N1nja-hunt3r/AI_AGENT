from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class VectorEntry:
    id: str
    embedding: List[float]
    content: str
    doc_id: str
    doc_name: str
    chunk_index: int
    namespace: str = "default"
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    content: str
    doc_id: str
    doc_name: str
    chunk_index: int
    score: float
    rank: int
    namespace: str = "default"
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "rank": self.rank,
            "namespace": self.namespace,
            "page_number": self.page_number,
            "section": self.section,
            "metadata": self.metadata,
            "citation": self.citation,
        }


@dataclass
class RetrieverConfig:
    top_k: int = 5
    similarity_threshold: float = 0.70
    default_namespace: str = "default"
    enable_citations: bool = True
    citation_format: str = "[{doc_name}, p.{page}]"
    max_namespace_size: int = 100_000
    score_boost_fields: List[str] = field(default_factory=list)
    score_boost_weight: float = 0.1


class RetrieverError(Exception):
    pass


MetadataFilter = Callable[[Dict[str, Any]], bool]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-10
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-10
    return dot / (mag_a * mag_b)


class NamespaceStore:
    def __init__(self) -> None:
        self._entries: Dict[str, VectorEntry] = {}
        self._lock = asyncio.Lock()

    async def add(self, entry: VectorEntry) -> None:
        async with self._lock:
            self._entries[entry.id] = entry

    async def add_batch(self, entries: List[VectorEntry]) -> None:
        async with self._lock:
            for e in entries:
                self._entries[e.id] = e

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            return self._entries.pop(entry_id, None) is not None

    async def delete_by_doc(self, doc_id: str) -> int:
        async with self._lock:
            ids = [k for k, v in self._entries.items() if v.doc_id == doc_id]
            for k in ids:
                del self._entries[k]
            return len(ids)

    async def all(self) -> List[VectorEntry]:
        async with self._lock:
            return list(self._entries.values())

    async def get(self, entry_id: str) -> Optional[VectorEntry]:
        async with self._lock:
            return self._entries.get(entry_id)

    @property
    def size(self) -> int:
        return len(self._entries)


class Retriever:
    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        self._config = config or RetrieverConfig()
        self._namespaces: Dict[str, NamespaceStore] = {}
        self._ns_lock = asyncio.Lock()
        self._total_queries = 0

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def _get_ns(self, namespace: str) -> NamespaceStore:
        async with self._ns_lock:
            if namespace not in self._namespaces:
                self._namespaces[namespace] = NamespaceStore()
            return self._namespaces[namespace]

    async def add(
        self,
        content: str,
        embedding: List[float],
        doc_id: str,
        doc_name: str,
        chunk_index: int,
        namespace: Optional[str] = None,
        page_number: Optional[int] = None,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        ns = namespace or self._config.default_namespace
        eid = entry_id or str(uuid.uuid4())
        store = await self._get_ns(ns)
        await store.add(VectorEntry(
            id=eid,
            embedding=embedding,
            content=content,
            doc_id=doc_id,
            doc_name=doc_name,
            chunk_index=chunk_index,
            namespace=ns,
            page_number=page_number,
            section=section,
            metadata=metadata or {},
        ))
        return eid

    async def add_batch(
        self,
        entries: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> List[str]:
        ns = namespace or self._config.default_namespace
        store = await self._get_ns(ns)
        vector_entries: List[VectorEntry] = []
        ids: List[str] = []
        for e in entries:
            eid = e.get("id") or str(uuid.uuid4())
            ids.append(eid)
            vector_entries.append(VectorEntry(
                id=eid,
                embedding=e["embedding"],
                content=e["content"],
                doc_id=e["doc_id"],
                doc_name=e["doc_name"],
                chunk_index=e.get("chunk_index", 0),
                namespace=ns,
                page_number=e.get("page_number"),
                section=e.get("section"),
                metadata=e.get("metadata", {}),
            ))
        await store.add_batch(vector_entries)
        return ids

    async def delete(self, entry_id: str, namespace: Optional[str] = None) -> bool:
        ns = namespace or self._config.default_namespace
        store = await self._get_ns(ns)
        return await store.delete(entry_id)

    async def delete_document(
        self, doc_id: str, namespace: Optional[str] = None
    ) -> int:
        ns = namespace or self._config.default_namespace
        store = await self._get_ns(ns)
        return await store.delete_by_doc(doc_id)

    async def delete_namespace(self, namespace: str) -> None:
        async with self._ns_lock:
            self._namespaces.pop(namespace, None)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        threshold: Optional[float] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        doc_filter: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        k = top_k or self._config.top_k
        ns = namespace or self._config.default_namespace
        thresh = threshold if threshold is not None else self._config.similarity_threshold
        store = await self._get_ns(ns)
        candidates = await store.all()
        self._total_queries += 1
        return await self._score_and_rank(
            query_embedding, candidates, k, thresh, metadata_filter, doc_filter
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        namespaces: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        doc_filter: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        ns_list = namespaces or [self._config.default_namespace]
        k = top_k or self._config.top_k
        thresh = threshold if threshold is not None else self._config.similarity_threshold
        candidates: List[VectorEntry] = []
        for ns in ns_list:
            store = await self._get_ns(ns)
            candidates.extend(await store.all())
        self._total_queries += 1
        return await self._score_and_rank(
            query_embedding, candidates, k, thresh, metadata_filter, doc_filter
        )

    async def filter(
        self,
        namespace: Optional[str] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        doc_filter: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        ns = namespace or self._config.default_namespace
        store = await self._get_ns(ns)
        candidates = await store.all()
        if metadata_filter:
            candidates = [c for c in candidates if metadata_filter(c.metadata)]
        if doc_filter:
            candidates = [c for c in candidates if c.doc_id in doc_filter]
        results: List[SearchResult] = []
        for i, entry in enumerate(candidates[:limit]):
            results.append(self._entry_to_result(entry, score=1.0, rank=i + 1))
        return results

    async def score(
        self,
        query_embedding: List[float],
        entry_ids: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, float]:
        ns = namespace or self._config.default_namespace
        store = await self._get_ns(ns)
        scores: Dict[str, float] = {}
        for eid in entry_ids:
            entry = await store.get(eid)
            if entry and entry.embedding:
                scores[eid] = round(_cosine(query_embedding, entry.embedding), 6)
        return scores

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------

    def _build_citation(self, entry: VectorEntry) -> str:
        if not self._config.enable_citations:
            return ""
        page = entry.page_number or "?"
        section = f", §{entry.section}" if entry.section else ""
        return self._config.citation_format.format(
            doc_name=entry.doc_name, page=page
        ) + section

    # ------------------------------------------------------------------
    # Namespace stats
    # ------------------------------------------------------------------

    async def list_namespaces(self) -> List[Dict[str, Any]]:
        async with self._ns_lock:
            return [
                {"namespace": ns, "size": store.size}
                for ns, store in self._namespaces.items()
            ]

    def get_stats(self) -> Dict[str, Any]:
        total = sum(s.size for s in self._namespaces.values())
        return {
            "namespaces": len(self._namespaces),
            "total_entries": total,
            "total_queries": self._total_queries,
            "top_k": self._config.top_k,
            "threshold": self._config.similarity_threshold,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _score_and_rank(
        self,
        query_emb: List[float],
        candidates: List[VectorEntry],
        top_k: int,
        threshold: float,
        metadata_filter: Optional[MetadataFilter],
        doc_filter: Optional[List[str]],
    ) -> List[SearchResult]:
        if metadata_filter:
            candidates = [c for c in candidates if metadata_filter(c.metadata)]
        if doc_filter:
            candidates = [c for c in candidates if c.doc_id in doc_filter]

        scored: List[Tuple[float, VectorEntry]] = []
        for entry in candidates:
            if not entry.embedding:
                continue
            sim = _cosine(query_emb, entry.embedding)
            if sim < threshold:
                continue
            boosted = self._apply_boost(sim, entry)
            scored.append((boosted, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[SearchResult] = []
        for rank, (score, entry) in enumerate(scored[:top_k]):
            results.append(self._entry_to_result(entry, score, rank + 1))
        return results

    def _apply_boost(self, base_score: float, entry: VectorEntry) -> float:
        if not self._config.score_boost_fields:
            return base_score
        boost = 0.0
        for f in self._config.score_boost_fields:
            if entry.metadata.get(f):
                boost += self._config.score_boost_weight
        return min(1.0, base_score + boost)

    def _entry_to_result(self, entry: VectorEntry, score: float, rank: int) -> SearchResult:
        return SearchResult(
            id=entry.id,
            content=entry.content,
            doc_id=entry.doc_id,
            doc_name=entry.doc_name,
            chunk_index=entry.chunk_index,
            score=score,
            rank=rank,
            namespace=entry.namespace,
            page_number=entry.page_number,
            section=entry.section,
            metadata=entry.metadata,
            citation=self._build_citation(entry),
        )
