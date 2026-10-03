"""
base_vector_store.py

Abstract base class defining the contract for all vector store backends.
"""

from __future__ import annotations

import abc
import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class VectorStoreError(Exception):
    """Base exception for all vector store errors."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when a vector store cannot establish or maintain a connection."""


class VectorStoreNotInitializedError(VectorStoreError):
    """Raised when an operation is attempted before initialize() has completed."""


class DocumentNotFoundError(VectorStoreError):
    """Raised when a requested document id does not exist."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when input data fails validation."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class HealthStatus(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DistanceMetric(str, enum.Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class StorePriority(int, enum.Enum):
    """Relative priority used by orchestration/registry layers when selecting
    among multiple available backends."""
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100


# --------------------------------------------------------------------------- #
# Versioning / Metadata
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VectorStoreVersion:
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


@dataclass
class VectorStoreMetadata:
    """Descriptive metadata about a vector store backend implementation."""
    name: str
    backend: str
    version: VectorStoreVersion = field(default_factory=VectorStoreVersion)
    priority: StorePriority = StorePriority.NORMAL
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    supports_metadata_filtering: bool = True
    supports_namespaces: bool = False
    supports_mmr: bool = False
    supports_async_native: bool = True
    max_batch_size: int = 1000
    embedding_dimension: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    """A single document to be stored in / retrieved from a vector store."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: Optional[str] = None

    def validate(self) -> None:
        if not self.id:
            raise VectorStoreValidationError("Document.id must be non-empty")
        if self.embedding is not None and len(self.embedding) == 0:
            raise VectorStoreValidationError(
                "Document.embedding must not be empty when provided"
            )


@dataclass
class SearchResult:
    """A single similarity search hit."""
    document: Document
    score: float
    distance: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class SearchQuery:
    """Parameters governing a similarity search."""
    query_embedding: Optional[List[float]] = None
    query_text: Optional[str] = None
    top_k: int = 10
    namespace: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 20
    score_threshold: Optional[float] = None
    include_embeddings: bool = False

    def validate(self) -> None:
        if self.query_embedding is None and self.query_text is None:
            raise VectorStoreValidationError(
                "SearchQuery requires either query_embedding or query_text"
            )
        if self.top_k <= 0:
            raise VectorStoreValidationError("top_k must be a positive integer")
        if not (0.0 <= self.mmr_lambda <= 1.0):
            raise VectorStoreValidationError("mmr_lambda must be within [0, 1]")


@dataclass
class HealthCheckResult:
    status: HealthStatus
    backend: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass
class UpsertResult:
    inserted_ids: List[str] = field(default_factory=list)
    updated_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return len(self.inserted_ids) + len(self.updated_ids)

    @property
    def failure_count(self) -> int:
        return len(self.failed_ids)


@dataclass
class DeleteResult:
    deleted_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Abstract base class
# --------------------------------------------------------------------------- #

class VectorStore(abc.ABC):
    """
    Abstract base class for all vector store backend implementations.

    Concrete subclasses (Chroma, FAISS, pgvector, Pinecone, Qdrant, ...)
    must implement every abstract method. The class provides:
      * lifecycle management (initialize / shutdown)
      * health monitoring
      * CRUD over documents (add / update / delete / get)
      * similarity search (single + batch)
      * count / clear utilities
      * a default safe-call wrapper with retry/backoff helpers subclasses
        may reuse.
    """

    def __init__(
        self,
        metadata: VectorStoreMetadata,
        *,
        default_namespace: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._metadata = metadata
        self._default_namespace = default_namespace
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{metadata.name}")

    # ------------------------------------------------------------------ #
    # Metadata / introspection
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> VectorStoreMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def version(self) -> VectorStoreVersion:
        return self._metadata.version

    @property
    def priority(self) -> StorePriority:
        return self._metadata.priority

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise VectorStoreNotInitializedError(
                f"VectorStore '{self.name}' has not been initialized. Call "
                f"`await store.initialize()` first."
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Idempotently initialize underlying resources (connections,
        clients, collections, indexes, ...). Subclasses override
        `_initialize`."""
        async with self._init_lock:
            if self._initialized:
                return
            self._logger.info("Initializing vector store '%s'", self.name)
            await self._initialize()
            self._initialized = True
            self._logger.info("Vector store '%s' initialized", self.name)

    async def shutdown(self) -> None:
        """Idempotently release underlying resources."""
        async with self._init_lock:
            if not self._initialized:
                return
            self._logger.info("Shutting down vector store '%s'", self.name)
            await self._shutdown()
            self._initialized = False
            self._logger.info("Vector store '%s' shut down", self.name)

    async def __aenter__(self) -> "VectorStore":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown()

    @abc.abstractmethod
    async def _initialize(self) -> None:
        """Backend-specific initialization logic."""
        raise NotImplementedError

    @abc.abstractmethod
    async def _shutdown(self) -> None:
        """Backend-specific teardown logic."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def health_check(self) -> HealthCheckResult:
        """Run a backend-specific health check, capturing latency and
        normalizing failures into a HealthCheckResult instead of raising."""
        start = time.perf_counter()
        try:
            if not self._initialized:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    backend=self._metadata.backend,
                    latency_ms=0.0,
                    error="store not initialized",
                )
            result = await self._health_check()
            result.latency_ms = (time.perf_counter() - start) * 1000.0
            return result
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Health check failed for '%s'", self.name)
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                backend=self._metadata.backend,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error=str(exc),
            )

    @abc.abstractmethod
    async def _health_check(self) -> HealthCheckResult:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def add_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> UpsertResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_documents(
        self,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> DeleteResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def update_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: Optional[str] = None,
        upsert: bool = True,
    ) -> UpsertResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_document(
        self,
        id: str,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[Document]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def similarity_search(
        self,
        query: SearchQuery,
    ) -> List[SearchResult]:
        raise NotImplementedError

    async def batch_search(
        self,
        queries: Sequence[SearchQuery],
        *,
        max_concurrency: int = 8,
    ) -> List[List[SearchResult]]:
        """Default batch implementation: runs similarity_search concurrently
        with a semaphore. Subclasses may override for native batch APIs."""
        self._ensure_initialized()
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _run(q: SearchQuery) -> List[SearchResult]:
            async with semaphore:
                return await self.similarity_search(q)

        return await asyncio.gather(*(_run(q) for q in queries))

    # ------------------------------------------------------------------ #
    # Aggregate operations
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def count(self, *, namespace: Optional[str] = None) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def clear(self, *, namespace: Optional[str] = None) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared helpers for subclasses
    # ------------------------------------------------------------------ #

    def _resolve_namespace(self, namespace: Optional[str]) -> Optional[str]:
        return namespace if namespace is not None else self._default_namespace

    async def _with_retry(
        self,
        func: Any,
        *args: Any,
        retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Generic exponential-backoff retry wrapper for transient backend
        errors. Subclasses can call this around network calls."""
        attempts = retries if retries is not None else self._max_retries
        delay = (
            backoff_seconds
            if backoff_seconds is not None
            else self._retry_backoff_seconds
        )
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._logger.warning(
                    "Attempt %d/%d failed for '%s': %s",
                    attempt,
                    attempts,
                    self.name,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(delay * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise VectorStoreError(
            f"Operation failed after {attempts} attempts: {last_exc}"
        ) from last_exc

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"backend={self._metadata.backend!r}, version={self.version}, "
            f"priority={self.priority.name}, initialized={self._initialized})"
        )
