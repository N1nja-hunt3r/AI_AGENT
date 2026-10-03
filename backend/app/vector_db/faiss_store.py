"""
faiss_store.py

FAISS-backed implementation of VectorStore supporting both flat
(IndexFlatIP / IndexFlatL2) and IVF (IndexIVFFlat) indexes, with
disk persistence (save/load), in-process metadata storage (FAISS itself
has no metadata support), and post-filtering on metadata.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.vector_db.base_vector_store import (
    DeleteResult,
    Document,
    HealthCheckResult,
    HealthStatus,
    SearchQuery,
    SearchResult,
    UpsertResult,
    VectorStore,
    VectorStoreError,
    VectorStoreMetadata,
    VectorStoreValidationError,
)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)

_INT64_MASK = (1 << 63) - 1


class FaissIndexType(str, enum.Enum):
    FLAT = "flat"
    IVF = "ivf"


def _string_id_to_int64(string_id: str) -> int:
    digest = hashlib.sha1(string_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & _INT64_MASK


def _matches_filter(metadata: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for key, expected in flt.items():
        actual = metadata.get(key)
        if isinstance(expected, dict) and ("$in" in expected or "$gte" in expected or "$lte" in expected):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                return False
        else:
            if actual != expected:
                return False
    return True


class FaissStore(VectorStore):
    """VectorStore implementation backed by FAISS, with explicit save/load
    and a lightweight side-table for metadata and document content (FAISS
    only stores raw vectors)."""

    def __init__(
        self,
        *,
        dimension: int,
        index_type: FaissIndexType = FaissIndexType.FLAT,
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 8,
        persist_directory: str = "./faiss_data",
        index_name: str = "documents",
        default_namespace: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        from base_vector_store import DistanceMetric, StorePriority, VectorStoreVersion

        metadata = VectorStoreMetadata(
            name=f"faiss:{index_name}",
            backend="faiss",
            version=VectorStoreVersion(1, 0, 0),
            priority=StorePriority.NORMAL,
            distance_metric=DistanceMetric(metric),
            supports_metadata_filtering=True,
            supports_namespaces=True,
            supports_mmr=False,
            supports_async_native=False,
            embedding_dimension=dimension,
        )
        super().__init__(
            metadata,
            default_namespace=default_namespace or "default",
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._dimension = dimension
        self._index_type = index_type
        self._metric = metric
        self._nlist = nlist
        self._nprobe = nprobe
        self._persist_directory = Path(persist_directory)
        self._index_name = index_name

        # namespace -> faiss.IndexIDMap2
        self._indexes: Dict[str, Any] = {}
        # namespace -> { int_id: Document }
        self._doc_store: Dict[str, Dict[int, Document]] = {}
        # namespace -> { str_id: int_id }
        self._id_map: Dict[str, Dict[str, int]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if faiss is None:
            raise VectorStoreError("faiss is not installed. Install with `pip install faiss-cpu`.")
        if np is None:
            raise VectorStoreError("numpy is required for FaissStore.")
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        await self._load_namespace(self._default_namespace)

    async def _shutdown(self) -> None:
        for namespace in list(self._indexes.keys()):
            await self.save(namespace=namespace)
        self._indexes.clear()
        self._doc_store.clear()
        self._id_map.clear()

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def _health_check(self) -> HealthCheckResult:
        try:
            index = await self._get_or_create_index(self._default_namespace)
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                backend="faiss",
                details={
                    "index_type": self._index_type.value,
                    "dimension": self._dimension,
                    "ntotal": int(index.ntotal),
                    "namespaces": list(self._indexes.keys()),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="faiss", error=str(exc))

    # ------------------------------------------------------------------ #
    # Index construction / persistence
    # ------------------------------------------------------------------ #

    def _index_path(self, namespace: str) -> Path:
        return self._persist_directory / f"{self._index_name}__{namespace}.faiss"

    def _meta_path(self, namespace: str) -> Path:
        return self._persist_directory / f"{self._index_name}__{namespace}.meta.pkl"

    def _build_empty_index(self) -> Any:
        metric_type = faiss.METRIC_INNER_PRODUCT if self._metric == "cosine" else faiss.METRIC_L2
        if self._index_type == FaissIndexType.FLAT:
            base: Any = (
                faiss.IndexFlatIP(self._dimension)
                if metric_type == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(self._dimension)
            )
        elif self._index_type == FaissIndexType.IVF:
            quantizer = (
                faiss.IndexFlatIP(self._dimension)
                if metric_type == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(self._dimension)
            )
            base = faiss.IndexIVFFlat(quantizer, self._dimension, self._nlist, metric_type)
            base.nprobe = self._nprobe
        else:
            raise VectorStoreValidationError(f"Unsupported index type: {self._index_type}")
        return faiss.IndexIDMap2(base)

    async def _get_or_create_index(self, namespace: Optional[str]) -> Any:
        ns = self._resolve_namespace(namespace) or "default"
        if ns not in self._indexes:
            await self._load_namespace(ns)
        return self._indexes[ns]

    async def _load_namespace(self, namespace: str) -> None:
        async with self._lock:
            if namespace in self._indexes:
                return
            index_path = self._index_path(namespace)
            meta_path = self._meta_path(namespace)
            if index_path.exists() and meta_path.exists():
                index = await asyncio.to_thread(faiss.read_index, str(index_path))
                with open(meta_path, "rb") as f:
                    payload = pickle.load(f)
                self._doc_store[namespace] = payload["documents"]
                self._id_map[namespace] = payload["id_map"]
                self._indexes[namespace] = index
            else:
                self._indexes[namespace] = self._build_empty_index()
                self._doc_store[namespace] = {}
                self._id_map[namespace] = {}

    async def save(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or "default"
        if ns not in self._indexes:
            return
        index = self._indexes[ns]
        index_path = self._index_path(ns)
        meta_path = self._meta_path(ns)
        await asyncio.to_thread(faiss.write_index, index, str(index_path))
        payload = {"documents": self._doc_store[ns], "id_map": self._id_map[ns]}
        with open(meta_path, "wb") as f:
            pickle.dump(payload, f)

    async def load(self, *, namespace: Optional[str] = None) -> None:
        ns = self._resolve_namespace(namespace) or "default"
        self._indexes.pop(ns, None)
        self._doc_store.pop(ns, None)
        self._id_map.pop(ns, None)
        await self._load_namespace(ns)

    # ------------------------------------------------------------------ #
    # Vector prep
    # ------------------------------------------------------------------ #

    def _prep_vectors(self, embeddings: Sequence[Sequence[float]]) -> Any:
        arr = np.array(embeddings, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self._dimension:
            raise VectorStoreValidationError(
                f"Embedding dimension {arr.shape[1]} does not match index dimension {self._dimension}"
            )
        if self._metric == "cosine":
            faiss.normalize_L2(arr)
        return arr

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    async def add_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> UpsertResult:
        self._ensure_initialized()
        if not documents:
            return UpsertResult()
        ns = self._resolve_namespace(namespace) or "default"
        index = await self._get_or_create_index(ns)
        result = UpsertResult()

        valid_docs, valid_vecs, int_ids = [], [], []
        for doc in documents:
            doc.validate()
            if doc.embedding is None:
                result.failed_ids.append(doc.id)
                result.errors[doc.id] = "missing embedding"
                continue
            valid_docs.append(doc)
            valid_vecs.append(doc.embedding)
            int_ids.append(_string_id_to_int64(doc.id))

        if not valid_docs:
            return result

        async with self._lock:
            vectors = self._prep_vectors(valid_vecs)
            id_arr = np.array(int_ids, dtype="int64")

            existing = [i for i in int_ids if i in self._doc_store[ns]]
            if existing:
                selector = faiss.IDSelectorBatch(np.array(existing, dtype="int64"))
                await asyncio.to_thread(index.remove_ids, selector)

            if self._index_type == FaissIndexType.IVF and not index.is_trained:
                min_training = max(self._nlist, 1) * 1
                if vectors.shape[0] >= min_training:
                    await asyncio.to_thread(index.train, vectors)
                else:
                    result.failed_ids.extend(d.id for d in valid_docs)
                    for d in valid_docs:
                        result.errors[d.id] = (
                            f"IVF index requires >= {min_training} vectors to train; "
                            f"got {vectors.shape[0]}"
                        )
                    return result

            await asyncio.to_thread(index.add_with_ids, vectors, id_arr)

            for doc, int_id in zip(valid_docs, int_ids):
                self._doc_store[ns][int_id] = doc
                self._id_map[ns][doc.id] = int_id
                result.inserted_ids.append(doc.id)

        return result

    async def update_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        upsert: bool = True,
    ) -> UpsertResult:
        return await self.add_documents(documents, namespace=namespace)

    async def delete_documents(
        self,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> DeleteResult:
        self._ensure_initialized()
        if not ids:
            return DeleteResult()
        ns = self._resolve_namespace(namespace) or "default"
        index = await self._get_or_create_index(ns)
        result = DeleteResult()

        async with self._lock:
            int_ids = []
            for str_id in ids:
                int_id = self._id_map[ns].get(str_id)
                if int_id is None:
                    result.failed_ids.append(str_id)
                    result.errors[str_id] = "id not found"
                    continue
                int_ids.append(int_id)

            if int_ids:
                selector = faiss.IDSelectorBatch(np.array(int_ids, dtype="int64"))
                await asyncio.to_thread(index.remove_ids, selector)
                for str_id, int_id in zip(
                    [s for s in ids if self._id_map[ns].get(s) in int_ids], int_ids
                ):
                    self._doc_store[ns].pop(int_id, None)
                    self._id_map[ns].pop(str_id, None)
                    result.deleted_ids.append(str_id)

        return result

    async def get_document(
        self,
        id: str,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[Document]:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or "default"
        await self._get_or_create_index(ns)
        int_id = self._id_map[ns].get(id)
        if int_id is None:
            return None
        return self._doc_store[ns].get(int_id)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _search_sync(
        self, index: Any, ns: str, query_vec: Any, k: int, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        scores, ids = index.search(query_vec, k)
        hits: List[SearchResult] = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id == -1:
                continue
            doc = self._doc_store[ns].get(int(int_id))
            if doc is None:
                continue
            if metadata_filter and not _matches_filter(doc.metadata, metadata_filter):
                continue
            normalized_score = float(score) if self._metric == "cosine" else 1.0 / (1.0 + float(score))
            hits.append(SearchResult(document=doc, score=normalized_score, distance=float(score)))
        return hits

    async def similarity_search(self, query: SearchQuery) -> List[SearchResult]:
        self._ensure_initialized()
        query.validate()
        if query.query_embedding is None:
            raise VectorStoreValidationError("FaissStore requires query_embedding (no built-in text embedding)")

        ns = self._resolve_namespace(query.namespace) or "default"
        index = await self._get_or_create_index(ns)
        query_vec = self._prep_vectors([query.query_embedding])

        fetch_k = query.top_k * 5 if query.metadata_filter else query.top_k
        fetch_k = min(max(fetch_k, query.top_k), max(index.ntotal, 1))

        hits = await asyncio.to_thread(
            self._search_sync, index, ns, query_vec, fetch_k, query.metadata_filter
        )

        if query.score_threshold is not None:
            hits = [h for h in hits if h.score >= query.score_threshold]

        hits = hits[: query.top_k]
        for rank, hit in enumerate(hits):
            hit.rank = rank
            if not query.include_embeddings:
                hit.document = Document(
                    id=hit.document.id,
                    content=hit.document.content,
                    embedding=None,
                    metadata=hit.document.metadata,
                    namespace=hit.document.namespace,
                )
        return hits

    async def batch_search(
        self,
        queries: Sequence[SearchQuery],
        *,
        max_concurrency: int = 8,
    ) -> List[List[SearchResult]]:
        self._ensure_initialized()
        if not queries:
            return []

        groups: Dict[str, List[int]] = {}
        for i, q in enumerate(queries):
            q.validate()
            ns = self._resolve_namespace(q.namespace) or "default"
            groups.setdefault(ns, []).append(i)

        results: List[Optional[List[SearchResult]]] = [None] * len(queries)

        for ns, indices in groups.items():
            index = await self._get_or_create_index(ns)
            vecs = self._prep_vectors([queries[i].query_embedding for i in indices])  # type: ignore[list-item]
            top_ks = [queries[i].top_k for i in indices]
            max_k = max(top_ks)
            fetch_k = min(max(max_k * 5, max_k), max(index.ntotal, 1))
            scores, ids = await asyncio.to_thread(index.search, vecs, fetch_k)

            for row, query_idx in enumerate(indices):
                q = queries[query_idx]
                hits: List[SearchResult] = []
                for score, int_id in zip(scores[row], ids[row]):
                    if int_id == -1:
                        continue
                    doc = self._doc_store[ns].get(int(int_id))
                    if doc is None:
                        continue
                    if q.metadata_filter and not _matches_filter(doc.metadata, q.metadata_filter):
                        continue
                    normalized_score = (
                        float(score) if self._metric == "cosine" else 1.0 / (1.0 + float(score))
                    )
                    hits.append(SearchResult(document=doc, score=normalized_score, distance=float(score)))
                if q.score_threshold is not None:
                    hits = [h for h in hits if h.score >= q.score_threshold]
                hits = hits[: q.top_k]
                for rank, hit in enumerate(hits):
                    hit.rank = rank
                results[query_idx] = hits

        return [r if r is not None else [] for r in results]

    # ------------------------------------------------------------------ #
    # Aggregate operations
    # ------------------------------------------------------------------ #

    async def count(self, *, namespace: Optional[str] = None) -> int:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or "default"
        index = await self._get_or_create_index(ns)
        return int(index.ntotal)

    async def clear(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or "default"
        async with self._lock:
            self._indexes[ns] = self._build_empty_index()
            self._doc_store[ns] = {}
            self._id_map[ns] = {}
        index_path, meta_path = self._index_path(ns), self._meta_path(ns)
        for path in (index_path, meta_path):
            if path.exists():
                path.unlink()
