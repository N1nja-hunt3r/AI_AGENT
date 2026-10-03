"""
chroma_store.py

ChromaDB-backed implementation of VectorStore using a PersistentClient.
Namespaces are implemented as separate Chroma collections
(f"{base_collection}__{namespace}") so isolation between namespaces is
enforced by the backend itself.
"""

from __future__ import annotations

import asyncio
import logging
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
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore
    Settings = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = "default"


# --------------------------------------------------------------------------- #
# MMR helper
# --------------------------------------------------------------------------- #

def _cosine_similarity_matrix(a: Any, b: Any) -> Any:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def _mmr_select(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    top_k: int,
    lambda_mult: float,
) -> List[int]:
    """Maximal Marginal Relevance selection. Returns selected indices into
    candidate_embeddings, in selection order."""
    if np is None:
        raise VectorStoreError("numpy is required for MMR support")
    if not candidate_embeddings:
        return []

    query_vec = np.array([query_embedding], dtype=float)
    cand = np.array(candidate_embeddings, dtype=float)

    relevance = _cosine_similarity_matrix(query_vec, cand)[0]
    sim_matrix = _cosine_similarity_matrix(cand, cand)

    selected: List[int] = []
    remaining = set(range(len(candidate_embeddings)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            def _score(i: int) -> float:
                redundancy = max(sim_matrix[i][j] for j in selected)
                return lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy

            best = max(remaining, key=_score)
        selected.append(best)
        remaining.remove(best)

    return selected


class ChromaStore(VectorStore):
    """VectorStore implementation backed by a local/persistent ChromaDB
    instance."""

    def __init__(
        self,
        *,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_data",
        distance_metric: str = "cosine",
        embedding_dimension: Optional[int] = None,
        default_namespace: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        client_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        from app.vector_db.base_vector_store import (
            DistanceMetric,
            StorePriority,
            VectorStoreVersion,
        )

        metadata = VectorStoreMetadata(
            name=f"chroma:{collection_name}",
            backend="chromadb",
            version=VectorStoreVersion(1, 0, 0),
            priority=StorePriority.NORMAL,
            distance_metric=DistanceMetric(distance_metric),
            supports_metadata_filtering=True,
            supports_namespaces=True,
            supports_mmr=True,
            supports_async_native=False,
            embedding_dimension=embedding_dimension,
        )
        super().__init__(
            metadata,
            default_namespace=default_namespace or _DEFAULT_NAMESPACE,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._distance_metric = distance_metric
        self._client_settings = client_settings or {}
        self._client: Any = None
        self._collections: Dict[str, Any] = {}
        self._collections_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if chromadb is None:
            raise VectorStoreError(
                "chromadb is not installed. Install with `pip install chromadb`."
            )
        Path(self._persist_directory).mkdir(parents=True, exist_ok=True)

        def _make_client() -> Any:
            return chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False, **self._client_settings)
                if Settings
                else None,
            )

        self._client = await asyncio.to_thread(_make_client)
        # Eagerly create the default namespace collection.
        await self._get_or_create_collection(self._default_namespace)

    async def _shutdown(self) -> None:
        self._collections.clear()
        self._client = None

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def _health_check(self) -> HealthCheckResult:
        if self._client is None:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                backend="chromadb",
                error="client not initialized",
            )
        try:
            heartbeat = await asyncio.to_thread(self._client.heartbeat)
            collection = await self._get_or_create_collection(self._default_namespace)
            count = await asyncio.to_thread(collection.count)
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                backend="chromadb",
                details={
                    "heartbeat_ns": heartbeat,
                    "default_collection_count": count,
                    "persist_directory": self._persist_directory,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, backend="chromadb", error=str(exc)
            )

    # ------------------------------------------------------------------ #
    # Collection / namespace helpers
    # ------------------------------------------------------------------ #

    def _collection_name_for(self, namespace: Optional[str]) -> str:
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        return f"{self._collection_name}__{ns}"

    async def _get_or_create_collection(self, namespace: Optional[str]) -> Any:
        if self._client is None:
            raise VectorStoreError("Chroma client is not initialized")
        coll_name = self._collection_name_for(namespace)
        if coll_name in self._collections:
            return self._collections[coll_name]
        async with self._collections_lock:
            if coll_name in self._collections:
                return self._collections[coll_name]

            def _get_or_create() -> Any:
                return self._client.get_or_create_collection(  # type: ignore[union-attr]
                    name=coll_name,
                    metadata={"hnsw:space": self._distance_metric},
                )

            collection = await asyncio.to_thread(_get_or_create)
            self._collections[coll_name] = collection
            return collection

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
        for doc in documents:
            doc.validate()
            if doc.embedding is None:
                raise VectorStoreValidationError(
                    f"Document '{doc.id}' must have an embedding for ChromaStore.add_documents"
                )

        collection = await self._get_or_create_collection(namespace)
        size = batch_size or self.metadata.max_batch_size
        result = UpsertResult()

        for i in range(0, len(documents), size):
            chunk = documents[i : i + size]
            ids = [d.id for d in chunk]
            embeddings = [d.embedding for d in chunk]
            metadatas = [d.metadata or {} for d in chunk]
            contents = [d.content for d in chunk]
            try:
                await self._with_retry(
                    collection.upsert,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=contents,
                )
                result.inserted_ids.extend(ids)
            except Exception as exc:  # noqa: BLE001
                for doc_id in ids:
                    result.failed_ids.append(doc_id)
                    result.errors[doc_id] = str(exc)
        return result

    async def update_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        upsert: bool = True,
    ) -> UpsertResult:
        # Chroma's upsert natively handles both insert and update semantics.
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
        collection = await self._get_or_create_collection(namespace)
        result = DeleteResult()
        try:
            await self._with_retry(collection.delete, ids=list(ids))
            result.deleted_ids.extend(ids)
        except Exception as exc:  # noqa: BLE001
            for doc_id in ids:
                result.failed_ids.append(doc_id)
                result.errors[doc_id] = str(exc)
        return result

    async def get_document(
        self,
        id: str,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[Document]:
        self._ensure_initialized()
        collection = await self._get_or_create_collection(namespace)
        response = await asyncio.to_thread(
            collection.get,
            ids=[id],
            include=["embeddings", "metadatas", "documents"],
        )
        ids = response.get("ids") or []
        if not ids:
            return None
        embeddings = response.get("embeddings") or [None]
        metadatas = response.get("metadatas") or [{}]
        contents = response.get("documents") or [""]
        return Document(
            id=ids[0],
            content=contents[0] or "",
            embedding=list(embeddings[0]) if embeddings[0] is not None else None,
            metadata=metadatas[0] or {},
            namespace=self._resolve_namespace(namespace),
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    async def similarity_search(self, query: SearchQuery) -> List[SearchResult]:
        self._ensure_initialized()
        query.validate()
        collection = await self._get_or_create_collection(query.namespace)

        fetch_k = query.mmr_fetch_k if query.use_mmr else query.top_k
        fetch_k = max(fetch_k, query.top_k)

        kwargs: Dict[str, Any] = {
            "n_results": fetch_k,
            "include": ["embeddings", "metadatas", "documents", "distances"],
        }
        if query.metadata_filter:
            kwargs["where"] = query.metadata_filter
        if query.query_embedding is not None:
            kwargs["query_embeddings"] = [query.query_embedding]
        else:
            kwargs["query_texts"] = [query.query_text]

        response = await asyncio.to_thread(collection.query, **kwargs)

        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        embeddings = (response.get("embeddings") or [[None] * len(ids)])[0]
        metadatas = (response.get("metadatas") or [[{}] * len(ids)])[0]
        contents = (response.get("documents") or [[""] * len(ids)])[0]

        candidates: List[SearchResult] = []
        for idx, doc_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            embedding = (
                list(embeddings[idx])
                if idx < len(embeddings) and embeddings[idx] is not None
                else None
            )
            document = Document(
                id=doc_id,
                content=contents[idx] if idx < len(contents) else "",
                embedding=embedding if query.include_embeddings else None,
                metadata=metadatas[idx] if idx < len(metadatas) else {},
                namespace=self._resolve_namespace(query.namespace),
            )
            candidates.append(SearchResult(document=document, score=score, distance=distance))

        if query.score_threshold is not None:
            candidates = [c for c in candidates if c.score >= query.score_threshold]

        if query.use_mmr:
            if query.query_embedding is None:
                raise VectorStoreValidationError(
                    "MMR search requires query_embedding to be provided"
                )
            cand_embeddings = [
                c.document.embedding if c.document.embedding is not None else embeddings[i]
                for i, c in enumerate(candidates)
            ]
            cand_embeddings = [e for e in cand_embeddings if e is not None]
            selected_idx = _mmr_select(
                query.query_embedding,
                [list(e) for e in cand_embeddings],
                top_k=query.top_k,
                lambda_mult=query.mmr_lambda,
            )
            candidates = [candidates[i] for i in selected_idx]
        else:
            candidates = candidates[: query.top_k]

        for rank, result in enumerate(candidates):
            result.rank = rank

        return candidates

    async def batch_search(
        self,
        queries: Sequence[SearchQuery],
        *,
        max_concurrency: int = 8,
    ) -> List[List[SearchResult]]:
        self._ensure_initialized()
        if not queries:
            return []

        # Group queries that share namespace/filter/top_k settings so we
        # can use Chroma's native batched query_embeddings where possible.
        simple_groups: Dict[Any, List[int]] = {}
        for i, q in enumerate(queries):
            q.validate()
            if q.use_mmr or q.query_embedding is None:
                simple_groups.setdefault("__fallback__", []).append(i)
                continue
            key = (
                self._resolve_namespace(q.namespace),
                tuple(sorted((q.metadata_filter or {}).items())),
                q.top_k,
                q.include_embeddings,
            )
            simple_groups.setdefault(key, []).append(i)

        results: List[Optional[List[SearchResult]]] = [None] * len(queries)

        for key, indices in simple_groups.items():
            if key == "__fallback__":
                continue
            namespace, filter_items, top_k, include_embeddings = key
            collection = await self._get_or_create_collection(namespace)
            embeddings = [queries[i].query_embedding for i in indices]
            kwargs: Dict[str, Any] = {
                "query_embeddings": embeddings,
                "n_results": top_k,
                "include": ["embeddings", "metadatas", "documents", "distances"],
            }
            if filter_items:
                kwargs["where"] = dict(filter_items)
            response = await asyncio.to_thread(collection.query, **kwargs)

            for batch_pos, query_idx in enumerate(indices):
                ids = (response.get("ids") or [[]])[batch_pos]
                distances = (response.get("distances") or [[]])[batch_pos]
                embeds = (response.get("embeddings") or [[None] * len(ids)])[batch_pos]
                metadatas = (response.get("metadatas") or [[{}] * len(ids)])[batch_pos]
                contents = (response.get("documents") or [[""] * len(ids)])[batch_pos]

                hits: List[SearchResult] = []
                for idx, doc_id in enumerate(ids):
                    distance = distances[idx] if idx < len(distances) else None
                    score = 1.0 / (1.0 + distance) if distance is not None else 0.0
                    document = Document(
                        id=doc_id,
                        content=contents[idx] if idx < len(contents) else "",
                        embedding=list(embeds[idx])
                        if include_embeddings and idx < len(embeds) and embeds[idx] is not None
                        else None,
                        metadata=metadatas[idx] if idx < len(metadatas) else {},
                        namespace=namespace,
                    )
                    hits.append(SearchResult(document=document, score=score, distance=distance, rank=idx))
                results[query_idx] = hits

        fallback_indices = simple_groups.get("__fallback__", [])
        if fallback_indices:
            semaphore = asyncio.Semaphore(max(1, max_concurrency))

            async def _run(idx: int) -> None:
                async with semaphore:
                    results[idx] = await self.similarity_search(queries[idx])

            await asyncio.gather(*(_run(i) for i in fallback_indices))

        return [r if r is not None else [] for r in results]

    # ------------------------------------------------------------------ #
    # Aggregate operations
    # ------------------------------------------------------------------ #

    async def count(self, *, namespace: Optional[str] = None) -> int:
        self._ensure_initialized()
        collection = await self._get_or_create_collection(namespace)
        return await asyncio.to_thread(collection.count)

    async def clear(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        coll_name = self._collection_name_for(namespace)
        if self._client is None:
            raise VectorStoreError("Chroma client is not initialized")
        try:
            await asyncio.to_thread(self._client.delete_collection, coll_name)
        except Exception:  # noqa: BLE001
            pass
        self._collections.pop(coll_name, None)
        await self._get_or_create_collection(namespace)

    async def list_namespaces(self) -> List[str]:
        self._ensure_initialized()
        if self._client is None:
            raise VectorStoreError("Chroma client is not initialized")
        collections = await asyncio.to_thread(self._client.list_collections)
        prefix = f"{self._collection_name}__"
        namespaces = []
        for coll in collections:
            coll_name = getattr(coll, "name", coll)
            if isinstance(coll_name, str) and coll_name.startswith(prefix):
                namespaces.append(coll_name[len(prefix) :])
        return namespaces
