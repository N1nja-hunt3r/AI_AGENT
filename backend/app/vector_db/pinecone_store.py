"""
pinecone_store.py

Pinecone-backed implementation of VectorStore. The Pinecone Python SDK is
synchronous, so all calls are dispatched via asyncio.to_thread to provide a
non-blocking async interface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Sequence

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
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = ""  # Pinecone's default namespace is the empty string


class PineconeStore(VectorStore):
    """VectorStore implementation backed by the Pinecone managed vector
    database."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        distance_metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        default_namespace: Optional[str] = None,
        create_if_missing: bool = True,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        from base_vector_store import DistanceMetric, StorePriority, VectorStoreVersion

        metadata = VectorStoreMetadata(
            name=f"pinecone:{index_name}",
            backend="pinecone",
            version=VectorStoreVersion(1, 0, 0),
            priority=StorePriority.HIGH,
            distance_metric=DistanceMetric(distance_metric),
            supports_metadata_filtering=True,
            supports_namespaces=True,
            supports_mmr=False,
            supports_async_native=False,
            embedding_dimension=dimension,
            max_batch_size=100,
        )
        super().__init__(
            metadata,
            default_namespace=default_namespace
            if default_namespace is not None
            else _DEFAULT_NAMESPACE,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._api_key = api_key
        self._index_name = index_name
        self._dimension = dimension
        self._distance_metric = distance_metric
        self._cloud = cloud
        self._region = region
        self._create_if_missing = create_if_missing
        self._client: Any = None
        self._index: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if Pinecone is None:
            raise VectorStoreError("pinecone is not installed. Install with `pip install pinecone`.")

        self._client = await asyncio.to_thread(Pinecone, api_key=self._api_key)

        existing = await asyncio.to_thread(self._client.list_indexes)
        existing_names = {idx["name"] for idx in existing}

        if self._index_name not in existing_names:
            if not self._create_if_missing:
                raise VectorStoreError(f"Pinecone index '{self._index_name}' does not exist")
            await asyncio.to_thread(
                self._client.create_index,
                name=self._index_name,
                dimension=self._dimension,
                metric=self._distance_metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )

        self._index = await asyncio.to_thread(self._client.Index, self._index_name)

    async def _shutdown(self) -> None:
        self._index = None
        self._client = None

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def _health_check(self) -> HealthCheckResult:
        if self._index is None:
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="pinecone", error="index not initialized")
        try:
            stats = await asyncio.to_thread(self._index.describe_index_stats)
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                backend="pinecone",
                details={
                    "index_name": self._index_name,
                    "total_vector_count": stats.get("total_vector_count", 0),
                    "namespaces": list((stats.get("namespaces") or {}).keys()),
                    "dimension": stats.get("dimension", self._dimension),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="pinecone", error=str(exc))

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
        ns = self._resolve_namespace(namespace)
        size = batch_size or self.metadata.max_batch_size
        result = UpsertResult()

        for i in range(0, len(documents), size):
            chunk = documents[i : i + size]
            vectors = []
            chunk_ids = []
            for doc in chunk:
                doc.validate()
                if doc.embedding is None:
                    result.failed_ids.append(doc.id)
                    result.errors[doc.id] = "missing embedding"
                    continue
                meta = dict(doc.metadata or {})
                meta["_content"] = doc.content
                vectors.append({"id": doc.id, "values": doc.embedding, "metadata": meta})
                chunk_ids.append(doc.id)

            if not vectors:
                continue
            try:
                await self._with_retry(self._index.upsert, vectors=vectors, namespace=ns)  # type: ignore[union-attr]
                result.inserted_ids.extend(chunk_ids)
            except Exception as exc:  # noqa: BLE001
                for doc_id in chunk_ids:
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
        # Pinecone's upsert is natively idempotent for both insert/update.
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
        ns = self._resolve_namespace(namespace)
        result = DeleteResult()
        try:
            await self._with_retry(self._index.delete, ids=list(ids), namespace=ns)  # type: ignore[union-attr]
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
        ns = self._resolve_namespace(namespace)
        response = await asyncio.to_thread(self._index.fetch, ids=[id], namespace=ns)  # type: ignore[union-attr]
        vectors = response.get("vectors") if isinstance(response, dict) else response.vectors
        if not vectors or id not in vectors:
            return None
        record = vectors[id]
        meta = dict(record.get("metadata") or {}) if isinstance(record, dict) else dict(record.metadata or {})
        content = meta.pop("_content", "")
        values = record.get("values") if isinstance(record, dict) else record.values
        return Document(id=id, content=content, embedding=list(values) if values else None, metadata=meta, namespace=ns)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _parse_match(self, match: Any, namespace: Optional[str], include_embeddings: bool) -> SearchResult:
        match_id = match.get("id") if isinstance(match, dict) else match.id
        score = match.get("score") if isinstance(match, dict) else match.score
        meta = dict((match.get("metadata") if isinstance(match, dict) else match.metadata) or {})
        content = meta.pop("_content", "")
        values = match.get("values") if isinstance(match, dict) else getattr(match, "values", None)
        document = Document(
            id=match_id,
            content=content,
            embedding=list(values) if include_embeddings and values else None,
            metadata=meta,
            namespace=namespace,
        )
        return SearchResult(document=document, score=float(score), distance=None)

    async def similarity_search(self, query: SearchQuery) -> List[SearchResult]:
        self._ensure_initialized()
        query.validate()
        if query.query_embedding is None:
            raise VectorStoreValidationError("PineconeStore requires query_embedding")

        ns = self._resolve_namespace(query.namespace)
        response = await asyncio.to_thread(
            self._index.query,  # type: ignore[union-attr]
            vector=query.query_embedding,
            top_k=query.top_k,
            namespace=ns,
            filter=query.metadata_filter or None,
            include_metadata=True,
            include_values=query.include_embeddings,
        )
        matches = response.get("matches") if isinstance(response, dict) else response.matches
        results = [self._parse_match(m, ns, query.include_embeddings) for m in (matches or [])]
        if query.score_threshold is not None:
            results = [r for r in results if r.score >= query.score_threshold]
        for rank, r in enumerate(results):
            r.rank = rank
        return results

    async def batch_search(
        self,
        queries: Sequence[SearchQuery],
        *,
        max_concurrency: int = 8,
    ) -> List[List[SearchResult]]:
        self._ensure_initialized()
        if not queries:
            return []
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _run(q: SearchQuery) -> List[SearchResult]:
            async with semaphore:
                return await self.similarity_search(q)

        return await asyncio.gather(*(_run(q) for q in queries))

    # ------------------------------------------------------------------ #
    # Aggregate operations
    # ------------------------------------------------------------------ #

    async def count(self, *, namespace: Optional[str] = None) -> int:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace)
        stats = await asyncio.to_thread(self._index.describe_index_stats)  # type: ignore[union-attr]
        namespaces = stats.get("namespaces") or {}
        if ns in namespaces:
            return int(namespaces[ns].get("vector_count", 0))
        if ns == _DEFAULT_NAMESPACE or ns is None:
            return int(stats.get("total_vector_count", 0))
        return 0

    async def clear(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace)
        await asyncio.to_thread(self._index.delete, delete_all=True, namespace=ns)  # type: ignore[union-attr]
