"""
pgvector_store.py

PostgreSQL + pgvector backed implementation of VectorStore, using asyncpg
for native async connection pooling.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = "default"

_DISTANCE_OPERATORS = {
    "cosine": "<=>",
    "euclidean": "<->",
    "dot_product": "<#>",
    "manhattan": "<+>",
}


def _build_where_clause(
    namespace: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    start_idx: int,
) -> "tuple[str, List[Any]]":
    clauses: List[str] = []
    params: List[Any] = []
    idx = start_idx

    if namespace is not None:
        clauses.append(f"namespace = ${idx}")
        params.append(namespace)
        idx += 1

    if metadata_filter:
        for key, value in metadata_filter.items():
            if isinstance(value, dict) and "$in" in value:
                placeholders = ", ".join(f"${idx + j}" for j in range(len(value["$in"])))
                clauses.append(f"metadata->>'{key}' IN ({placeholders})")
                params.extend(str(v) for v in value["$in"])
                idx += len(value["$in"])
            elif isinstance(value, dict):
                range_clauses = []
                if "$gte" in value:
                    range_clauses.append(f"CAST(metadata->>'{key}' AS FLOAT) >= ${idx}")
                    params.append(float(value["$gte"]))
                    idx += 1
                if "$lte" in value:
                    range_clauses.append(f"CAST(metadata->>'{key}' AS FLOAT) <= ${idx}")
                    params.append(float(value["$lte"]))
                    idx += 1
                if "$gt" in value:
                    range_clauses.append(f"CAST(metadata->>'{key}' AS FLOAT) > ${idx}")
                    params.append(float(value["$gt"]))
                    idx += 1
                if "$lt" in value:
                    range_clauses.append(f"CAST(metadata->>'{key}' AS FLOAT) < ${idx}")
                    params.append(float(value["$lt"]))
                    idx += 1
                if "$ne" in value:
                    range_clauses.append(f"metadata->>'{key}' <> ${idx}")
                    params.append(str(value["$ne"]))
                    idx += 1
                if range_clauses:
                    clauses.append("(" + " AND ".join(range_clauses) + ")")
            else:
                clauses.append(f"metadata->>'{key}' = ${idx}")
                params.append(str(value))
                idx += 1

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params


class PgVectorStore(VectorStore):
    """VectorStore implementation backed by PostgreSQL with the pgvector
    extension."""

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str = "documents",
        dimension: int,
        distance_metric: str = "cosine",
        default_namespace: Optional[str] = None,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        index_method: str = "hnsw",
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        from base_vector_store import DistanceMetric, StorePriority, VectorStoreVersion

        if distance_metric not in _DISTANCE_OPERATORS:
            raise VectorStoreValidationError(f"Unsupported distance metric: {distance_metric}")

        metadata = VectorStoreMetadata(
            name=f"pgvector:{table_name}",
            backend="pgvector",
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
        self._dsn = dsn
        self._table_name = table_name
        self._dimension = dimension
        self._distance_metric = distance_metric
        self._distance_op = _DISTANCE_OPERATORS[distance_metric]
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._index_method = index_method
        self._pool: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if asyncpg is None:
            raise VectorStoreError("asyncpg is not installed. Install with `pip install asyncpg`.")

        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT '{_DEFAULT_NAMESPACE}',
                    content TEXT NOT NULL DEFAULT '',
                    embedding VECTOR({self._dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (id, namespace)
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_metadata_gin "
                f"ON {self._table_name} USING GIN (metadata);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_namespace_idx "
                f"ON {self._table_name} (namespace);"
            )
            op_class = {
                "cosine": "vector_cosine_ops",
                "euclidean": "vector_l2_ops",
                "dot_product": "vector_ip_ops",
                "manhattan": "vector_l1_ops",
            }[self._distance_metric]
            try:
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx "
                    f"ON {self._table_name} USING {self._index_method} (embedding {op_class});"
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Could not create vector index (%s); continuing without it: %s",
                    self._index_method,
                    exc,
                )

    async def _shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def _health_check(self) -> HealthCheckResult:
        if self._pool is None:
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="pgvector", error="pool not initialized")
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1;")
                total = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table_name};")
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                backend="pgvector",
                details={
                    "table": self._table_name,
                    "total_rows": int(total) if total is not None else 0,
                    "pool_size": self._pool.get_size(),
                    "pool_free": self._pool.get_idle_size(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(status=HealthStatus.UNHEALTHY, backend="pgvector", error=str(exc))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _vec_literal(self, embedding: Sequence[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in embedding) + "]"

    def _row_to_document(self, row: Any) -> Document:
        raw_meta = row["metadata"]
        metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
        embedding = row.get("embedding") if "embedding" in row.keys() else None
        return Document(
            id=row["id"],
            content=row["content"] or "",
            embedding=list(embedding) if embedding is not None else None,
            metadata=metadata,
            namespace=row["namespace"],
        )

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

        assert self._pool is not None
        for i in range(0, len(documents), size):
            chunk = documents[i : i + size]
            for doc in chunk:
                doc.validate()
                if doc.embedding is None:
                    result.failed_ids.append(doc.id)
                    result.errors[doc.id] = "missing embedding"
            valid = [d for d in chunk if d.embedding is not None]
            if not valid:
                continue

            try:
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        for doc in valid:
                            await conn.execute(
                                f"""
                                INSERT INTO {self._table_name}
                                    (id, namespace, content, embedding, metadata, updated_at)
                                VALUES ($1, $2, $3, $4::vector, $5::jsonb, now())
                                ON CONFLICT (id, namespace) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    embedding = EXCLUDED.embedding,
                                    metadata = EXCLUDED.metadata,
                                    updated_at = now();
                                """,
                                doc.id,
                                ns,
                                doc.content,
                                self._vec_literal(doc.embedding),  # type: ignore[arg-type]
                                json.dumps(doc.metadata or {}),
                            )
                result.inserted_ids.extend(d.id for d in valid)
            except Exception as exc:  # noqa: BLE001
                for doc in valid:
                    result.failed_ids.append(doc.id)
                    result.errors[doc.id] = str(exc)

        return result

    async def update_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        upsert: bool = True,
    ) -> UpsertResult:
        if upsert:
            return await self.add_documents(documents, namespace=namespace)

        self._ensure_initialized()
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        result = UpsertResult()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            for doc in documents:
                doc.validate()
                set_clauses, params = [], []
                idx = 1
                if doc.content is not None:
                    set_clauses.append(f"content = ${idx}")
                    params.append(doc.content)
                    idx += 1
                if doc.embedding is not None:
                    set_clauses.append(f"embedding = ${idx}::vector")
                    params.append(self._vec_literal(doc.embedding))
                    idx += 1
                if doc.metadata is not None:
                    set_clauses.append(f"metadata = ${idx}::jsonb")
                    params.append(json.dumps(doc.metadata))
                    idx += 1
                set_clauses.append("updated_at = now()")
                params.extend([doc.id, ns])
                query = (
                    f"UPDATE {self._table_name} SET {', '.join(set_clauses)} "
                    f"WHERE id = ${idx} AND namespace = ${idx + 1};"
                )
                try:
                    status = await conn.execute(query, *params)
                    if status.endswith("0"):
                        result.failed_ids.append(doc.id)
                        result.errors[doc.id] = "document not found"
                    else:
                        result.updated_ids.append(doc.id)
                except Exception as exc:  # noqa: BLE001
                    result.failed_ids.append(doc.id)
                    result.errors[doc.id] = str(exc)
        return result

    async def delete_documents(
        self,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> DeleteResult:
        self._ensure_initialized()
        if not ids:
            return DeleteResult()
        ns = self._resolve_namespace(namespace) or _DEFAULT_NAMESPACE
        result = DeleteResult()
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id = ANY($1::text[]) AND namespace = $2;",
                    list(ids),
                    ns,
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
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT id, namespace, content, embedding, metadata FROM {self._table_name} "
                f"WHERE id = $1 AND namespace = $2;",
                id,
                ns,
            )
        if row is None:
            return None
        return self._row_to_document(row)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    async def similarity_search(self, query: SearchQuery) -> List[SearchResult]:
        self._ensure_initialized()
        query.validate()
        if query.query_embedding is None:
            raise VectorStoreValidationError("PgVectorStore requires query_embedding")

        ns = self._resolve_namespace(query.namespace)
        where_sql, params = _build_where_clause(ns, query.metadata_filter, start_idx=2)
        embed_param = self._vec_literal(query.query_embedding)
        select_embedding = ", embedding" if query.include_embeddings else ""

        sql = (
            f"SELECT id, namespace, content, metadata{select_embedding}, "
            f"embedding {self._distance_op} $1::vector AS distance "
            f"FROM {self._table_name} {where_sql} "
            f"ORDER BY distance ASC LIMIT {int(query.top_k)};"
        )
        params = [embed_param] + params

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results: List[SearchResult] = []
        for rank, row in enumerate(rows):
            distance = float(row["distance"])
            score = 1.0 - distance if self._distance_metric == "cosine" else 1.0 / (1.0 + distance)
            document = self._row_to_document(row)
            if query.score_threshold is not None and score < query.score_threshold:
                continue
            results.append(SearchResult(document=document, score=score, distance=distance, rank=rank))
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
        assert self._pool is not None
        ns = self._resolve_namespace(namespace)
        async with self._pool.acquire() as conn:
            if ns is not None:
                total = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self._table_name} WHERE namespace = $1;", ns
                )
            else:
                total = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table_name};")
        return int(total or 0)

    async def list_namespaces(self) -> List[str]:
        self._ensure_initialized()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT DISTINCT namespace FROM {self._table_name} ORDER BY namespace;"
            )
            return [row["namespace"] for row in rows]

    async def clear(self, *, namespace: Optional[str] = None) -> None:
        self._ensure_initialized()
        assert self._pool is not None
        ns = self._resolve_namespace(namespace)
        async with self._pool.acquire() as conn:
            if ns is not None:
                await conn.execute(f"DELETE FROM {self._table_name} WHERE namespace = $1;", ns)
            else:
                await conn.execute(f"TRUNCATE TABLE {self._table_name};")
