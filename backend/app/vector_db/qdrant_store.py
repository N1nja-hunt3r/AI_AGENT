"""
qdrant_store.py

Qdrant-backed implementation of VectorStore using qdrant-client's native
AsyncQdrantClient.
"""

from __future__ import annotations

import logging
import uuid
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
    from qdrant_client import AsyncQdrantClient, models as qmodels
except ImportError:  # pragma: no cover
    AsyncQdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = "default"
_DOC_ID_PAYLOAD_KEY = "_doc_id"
_NAMESPACE_PAYLOAD_KEY = "_namespace"
_CONTENT_PAYLOAD_KEY = "_content"


def _is_valid_qdrant_id(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return value.isdigit()


def _to_point_id(doc_id: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned ints. Arbitrary string ids
    are deterministically mapped to a UUID5, with the original id preserved
    in the payload for lookups."""
    if _is_valid_qdrant_id(doc_id):
        return doc_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _build_filter(metadata_filter: Optional[Dict[str, Any]], namespace: Optional[str]) -> Optional[Any]:
    if qmodels is None:
        return None
    conditions: List[Any] = []
    if namespace is not None:
        conditions.append(
            qmodels.FieldCondition(key=_NAMESPACE_PAYLOAD_KEY, match=qmodels.MatchValue(value=namespace))
        )
    if metadata_filter:
        for key, value in metadata_filter.items():
            if isinstance(value, dict) and "$in" in value:
                conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=value["$in"])))
            elif isinstance(value, dict) and ("$gte" in value or "$lte" in value):
                conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        range=qmodels.Range(gte=value.get("$gte"), lte=value.get("$lte")),
                    )
                )
            else:
                conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


class QdrantStore(VectorStore):
    """VectorStore implementation backed by Qdrant."""

    def __init__(
        self,
        *,
        collection_name: str = "documents",
        dimension: int,
        distance_metric: str = "cosine",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        default_namespace: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        from base_vector_store import DistanceMetric, StorePriority, VectorStoreVersion

        metadata = VectorStoreMetadata(
            name=f"qdrant:{collection_name}",
            backend="qdrant",
            version=VectorStoreVersion(1, 0, 0),
            priority=StorePriority.HIGH,
            distance_metric=DistanceMetric(distance_metric),
            supports_metadata_filtering=True,
            supports_namespaces=True,
            supports_mmr=False,
            supports_async_native=True,
            embedding_dimension=dimension,
        )
        super().__init__(
            metadata,
            default_namespace=default_namespace or _DEFAULT_NAMESPACE,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._collection_name = collection_name
        self._dimension = dimension
        self._distance_metric = distance_metric
        self._url = url
        self._api_key = api_key
        self._client: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if AsyncQdrantClient is None:
            raise VectorStoreError("qdrant-client is not installed. Install with `pip install qdrant-client`.")

        self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)

        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "euclidean": qmodels.Distance.EUCLID,
            "dot_product": qmodels.Distance.DOT,
            "manhattan": qmodels.Distance.MANHATTAN,
        }
        existing = await self._client.collection_exists(self._collection_name)
        if not existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._dimension,
                    distance=distance_map.get(self._distance_metric, qmodels.Distance.COSINE),
                ),
            )
            for field_name in (_DOC_ID_PAYLOAD_KEY, _NAMESPACE_PAYLOAD_KEY):
                try:
                    await self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                except Exception:  # noqa: BLE001
                    pass

    async def _shutdown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def _health_check(self) -> HealthCheckResult:
        if self._client is None:
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="qdrant", error="client not initialized")
        try:
            info = await self._client.get_collection(self._collection_name)
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                backend="qdrant",
                details={
                    "collection": self._collection_name,
                    "points_count": info.points_count,
                    "status": str(info.status),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="qdrant", error=str(exc))

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
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        size = batch_size or self.metadata.max_batch_size
        result = UpsertResult()

        for i in range(0, len(documents), size):
            chunk = documents[i : i + size]
            points = []
            chunk_ids = []
            for doc in chunk:
                doc.validate()
                if doc.embedding is None:
                    result.failed_ids.append(doc.id)
                    result.errors[doc.id] = "missing embedding"
                    continue
                payload = dict(doc.metadata or {})
                payload[_DOC_ID_PAYLOAD_KEY] = doc.id
                payload[_NAMESPACE_PAYLOAD_KEY] = ns
                payload[_CONTENT_PAYLOAD_KEY] = doc.content
                points.append(
                    qmodels.PointStruct(id=_to_point_id(doc.id), vector=doc.embedding, payload=payload)
                )
                chunk_ids.append(doc.id)

            if not points:
                continue
            try:
                await self._with_retry(
                    self._client.upsert, collection_name=self._collection_name, points=points  # type: ignore[union-attr]
                )
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
        result = DeleteResult()
        try:
            point_ids = [_to_point_id(i) for i in ids]
            await self._with_retry(
                self._client.delete,  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
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
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        points = await self._client.retrieve(  # type: ignore[union-attr]
            collection_name=self._collection_name,
            ids=[_to_point_id(id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        point = points[0]
        payload = dict(point.payload or {})
        return Document(
            id=payload.get(_DOC_ID_PAYLOAD_KEY, str(point.id)),
            content=payload.pop(_CONTENT_PAYLOAD_KEY, ""),
            embedding=list(point.vector) if point.vector else None,
            metadata={
                k: v
                for k, v in payload.items()
                if k not in (_DOC_ID_PAYLOAD_KEY, _NAMESPACE_PAYLOAD_KEY, _CONTENT_PAYLOAD_KEY)
            },
            namespace=payload.get(_NAMESPACE_PAYLOAD_KEY, ns),
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _hit_to_result(self, hit: Any, namespace: str, include_embeddings: bool) -> SearchResult:
        payload = dict(hit.payload or {})
        content = payload.pop(_CONTENT_PAYLOAD_KEY, "")
        doc_id = payload.pop(_DOC_ID_PAYLOAD_KEY, str(hit.id))
        ns = payload.pop(_NAMESPACE_PAYLOAD_KEY, namespace)
        document = Document(
            id=doc_id,
            content=content,
            embedding=list(hit.vector) if include_embeddings and getattr(hit, "vector", None) else None,
            metadata=payload,
            namespace=ns,
        )
        return SearchResult(document=document, score=float(hit.score), distance=None)

    async def similarity_search(self, query: SearchQuery) -> List[SearchResult]:
        self._ensure_initialized()
        query.validate()
        if query.query_embedding is None:
            raise VectorStoreValidationError("QdrantStore requires query_embedding")

        ns = self._resolve_namespace(query.namespace) or _DEFAULT_NAMESPACE
        q_filter = _build_filter(query.metadata_filter, ns)

        response = await self._client.query_points(  # type: ignore[union-attr]
            collection_name=self._collection_name,
            query=query.query_embedding,
            limit=query.top_k,
            query_filter=q_filter,
            with_payload=True,
            with_vectors=query.include_embeddings,
            score_threshold=query.score_threshold,
        )
        hits = response.points
        results = [self._hit_to_result(h, ns, query.include_embeddings) for h in hits]
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

        requests = []
        for q in queries:
            q.validate()
            if q.query_embedding is None:
                raise VectorStoreValidationError("QdrantStore requires query_embedding")
            ns = self._resolve_namespace(q.namespace) or _DEFAULT_NAMESPACE
            requests.append(
                qmodels.QueryRequest(
                    query=q.query_embedding,
                    limit=q.top_k,
                    filter=_build_filter(q.metadata_filter, ns),
                    with_payload=True,
                    with_vector=q.include_embeddings,
                    score_threshold=q.score_threshold,
                )
            )

        responses = await self._client.query_batch_points(  # type: ignore[union-attr]
            collection_name=self._collection_name, requests=requests
        )

        all_results: List[List[SearchResult]] = []
        for q, response in zip(queries, responses):
            ns = self._resolve_namespace(q.namespace) or _DEFAULT_NAMESPACE
            hits = response.points if hasattr(response, "points") else response
            results = [self._hit_to_result(h, ns, q.include_embeddings) for h in hits]
            for rank, r in enumerate(results):
                r.rank = rank
            all_results.append(results)
        return all_results

    # ------------------------------------------------------------------ #
    # Aggregate operations
    # ------------------------------------------------------------------ #

    async def count(self, *, namespace: Optional[str] = None) -> int:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        result = await self._client.count(  # type: ignore[union-attr]
            collection_name=self._collection_name,
            count_filter=_build_filter(None, ns),
            exact=True,
        )
        return int(result.count)

    async def clear(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        await self._client.delete(  # type: ignore[union-attr]
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(filter=_build_filter(None, ns)),
        )
