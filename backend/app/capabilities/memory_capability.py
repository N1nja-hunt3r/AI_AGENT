"""
Memory Capability - Unified memory management for the AI Agent Platform.

This module provides the MemoryCapability class which handles:
- Short-term memory (session-scoped, auto-expiring)
- Long-term memory (persistent across sessions)
- Semantic memory (vector-searchable knowledge)
- Episodic memory (event/interaction history)
- Working memory (active context window)

Supports multiple backends:
- InMemory (default, for development/testing)
- ChromaDB (future, for vector search)
- PostgreSQL (future, for persistence)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

if TYPE_CHECKING:
    pass

# Import from kernel modules
from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CapabilityType,
)
from app.agents.exceptions import (
    ValidationError,
)


# =============================================================================
# ENUMS
# =============================================================================


class MemoryType(Enum):
    """Types of memory supported by the capability."""

    SHORT_TERM = "short_term"  # Session-scoped, auto-expires
    LONG_TERM = "long_term"  # Persistent across sessions
    SEMANTIC = "semantic"  # Knowledge/facts, vector-searchable
    EPISODIC = "episodic"  # Events/interactions history
    WORKING = "working"  # Active context window


class MemoryAction(Enum):
    """Actions supported by the memory capability."""

    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    INJECT = "inject"
    CLEAR = "clear"
    LIST = "list"
    STATS = "stats"


class MemoryPriority(Enum):
    """Priority levels for memory entries."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry.

    Attributes:
        id: Unique identifier
        content: The memory content (text or structured data)
        memory_type: Type classification
        metadata: Additional metadata
        embedding: Vector embedding for semantic search
        priority: Importance level
        access_count: Number of times accessed
        created_at: Creation timestamp
        updated_at: Last update timestamp
        expires_at: Expiration timestamp (optional)
        agent_id: Owning agent ID (for multi-agent)
        session_id: Session scope (for short-term)
        tags: Searchable tags
        source: Origin of the memory
        confidence: Confidence score (0-1)
    """

    id: str
    content: str | dict[str, Any]
    memory_type: MemoryType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    priority: MemoryPriority = MemoryPriority.NORMAL
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    source: str = "user"
    confidence: float = 1.0

    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def content_hash(self) -> str:
        """Get hash of content for deduplication."""
        content_str = str(self.content)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "priority": self.priority.value,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tags": list(self.tags),
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            priority=MemoryPriority(data.get("priority", 5)),
            access_count=data.get("access_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc)),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now(timezone.utc)),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            tags=tuple(data.get("tags", [])),
            source=data.get("source", "user"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class MemoryQuery:
    """
    Query parameters for memory retrieval/search.

    Attributes:
        query_text: Text to search for
        memory_types: Filter by memory types
        tags: Filter by tags
        agent_id: Filter by agent
        session_id: Filter by session
        min_confidence: Minimum confidence threshold
        min_priority: Minimum priority level
        limit: Maximum results to return
        offset: Pagination offset
        include_expired: Include expired memories
        sort_by: Sort field
        sort_order: Sort direction
        time_range: Filter by time range
        embedding: Query embedding for semantic search
    """

    query_text: str | None = None
    memory_types: tuple[MemoryType, ...] | None = None
    tags: tuple[str, ...] | None = None
    agent_id: str | None = None
    session_id: str | None = None
    min_confidence: float = 0.0
    min_priority: MemoryPriority = MemoryPriority.LOW
    limit: int = 10
    offset: int = 0
    include_expired: bool = False
    sort_by: str = "created_at"
    sort_order: str = "desc"
    time_range: tuple[datetime, datetime] | None = None
    embedding: list[float] | None = None


@dataclass
class MemoryResult:
    """
    Result of a memory operation.

    Attributes:
        success: Whether operation succeeded
        entries: Retrieved/affected entries
        total_count: Total matching entries (for pagination)
        message: Status message
        metadata: Additional result metadata
    """

    success: bool
    entries: list[MemoryEntry] = field(default_factory=list)
    total_count: int = 0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Statistics about memory storage."""

    total_entries: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_agent: dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    oldest_entry: datetime | None = None
    newest_entry: datetime | None = None
    expired_count: int = 0


# =============================================================================
# MEMORY INDEX (Simple Similarity Search)
# =============================================================================


class MemoryIndex:
    """
    Simple in-memory index for similarity search.

    Uses cosine similarity for vector comparisons.
    Will be replaced by ChromaDB for production.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[MemoryEntry, list[float]]] = {}

    def add(self, entry: MemoryEntry, embedding: list[float]) -> None:
        """Add entry to index."""
        self._entries[entry.id] = (entry, embedding)

    def remove(self, entry_id: str) -> None:
        """Remove entry from index."""
        self._entries.pop(entry_id, None)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search for similar entries.

        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self._entries or not query_embedding:
            return []

        results: list[tuple[MemoryEntry, float]] = []

        for entry, embedding in self._entries.values():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                results.append((entry, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all entries from index."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Get number of indexed entries."""
        return len(self._entries)


# =============================================================================
# MEMORY BACKEND (Abstract)
# =============================================================================


class MemoryBackend(ABC):
    """
    Abstract base class for memory storage backends.

    Implementations:
    - InMemoryBackend: Default, for development/testing
    - ChromaDBBackend: Future, for vector search
    - PostgreSQLBackend: Future, for persistence
    """

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        ...

    @abstractmethod
    async def retrieve(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        ...

    @abstractmethod
    async def update(self, entry: MemoryEntry) -> bool:
        """Update an existing memory entry."""
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        ...

    @abstractmethod
    async def query(self, query: MemoryQuery) -> MemoryResult:
        """Query memories based on criteria."""
        ...

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        memory_types: tuple[MemoryType, ...] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic search using embeddings."""
        ...

    @abstractmethod
    async def clear(
        self,
        memory_type: MemoryType | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear memories matching criteria. Returns count deleted."""
        ...

    @abstractmethod
    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health."""
        ...

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count deleted."""
        ...


# =============================================================================
# IN-MEMORY BACKEND
# =============================================================================


class InMemoryBackend(MemoryBackend):
    """
    In-memory storage backend for development and testing.

    Features:
    - Fast access
    - No external dependencies
    - Automatic expiration cleanup
    - Simple similarity search
    """

    def __init__(
        self,
        max_entries: int = 10000,
        cleanup_interval: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize in-memory backend.

        Args:
            max_entries: Maximum entries to store
            cleanup_interval: Seconds between cleanup runs
            logger: Optional logger
        """
        self._entries: dict[str, MemoryEntry] = {}
        self._index = MemoryIndex()
        self._max_entries = max_entries
        self._cleanup_interval = cleanup_interval
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.debug("InMemoryBackend started")

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._logger.debug("InMemoryBackend stopped")

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                count = await self.cleanup_expired()
                if count > 0:
                    self._logger.debug(f"Cleaned up {count} expired memories")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup error: {e}")

    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        async with self._lock:
            # Check capacity
            if len(self._entries) >= self._max_entries:
                # Evict oldest low-priority entry
                await self._evict_one()

            self._entries[entry.id] = entry

            # Index if has embedding
            if entry.embedding:
                self._index.add(entry, entry.embedding)

            return True

    async def _evict_one(self) -> None:
        """Evict one entry based on priority and age."""
        if not self._entries:
            return

        # Find lowest priority, oldest entry
        candidates = sorted(
            self._entries.values(),
            key=lambda e: (e.priority.value, -e.created_at.timestamp()),
        )

        if candidates:
            entry = candidates[0]
            del self._entries[entry.id]
            self._index.remove(entry.id)
            self._logger.debug(f"Evicted memory {entry.id}")

    async def retrieve(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        async with self._lock:
            entry = self._entries.get(entry_id)

            if entry and not entry.is_expired:
                # Update access count
                entry.access_count += 1
                return entry

            return None

    async def update(self, entry: MemoryEntry) -> bool:
        """Update an existing memory entry."""
        async with self._lock:
            if entry.id not in self._entries:
                return False

            entry.updated_at = datetime.now(timezone.utc)
            self._entries[entry.id] = entry

            # Update index
            if entry.embedding:
                self._index.add(entry, entry.embedding)

            return True

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        async with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                self._index.remove(entry_id)
                return True
            return False

    async def query(self, query: MemoryQuery) -> MemoryResult:
        """Query memories based on criteria."""
        async with self._lock:
            results: list[MemoryEntry] = []

            for entry in self._entries.values():
                if self._matches_query(entry, query):
                    results.append(entry)

            # Sort
            reverse = query.sort_order == "desc"
            if query.sort_by == "created_at":
                results.sort(key=lambda e: e.created_at, reverse=reverse)
            elif query.sort_by == "updated_at":
                results.sort(key=lambda e: e.updated_at, reverse=reverse)
            elif query.sort_by == "priority":
                results.sort(key=lambda e: e.priority.value, reverse=reverse)
            elif query.sort_by == "access_count":
                results.sort(key=lambda e: e.access_count, reverse=reverse)

            total = len(results)

            # Paginate
            results = results[query.offset : query.offset + query.limit]

            # Update access counts
            for entry in results:
                entry.access_count += 1

            return MemoryResult(
                success=True,
                entries=results,
                total_count=total,
                message=f"Found {total} matching memories",
            )

    def _matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        """Check if entry matches query criteria."""
        # Check expiration
        if not query.include_expired and entry.is_expired:
            return False

        # Check memory type
        if query.memory_types and entry.memory_type not in query.memory_types:
            return False

        # Check agent
        if query.agent_id and entry.agent_id != query.agent_id:
            return False

        # Check session
        if query.session_id and entry.session_id != query.session_id:
            return False

        # Check confidence
        if entry.confidence < query.min_confidence:
            return False

        # Check priority
        if entry.priority.value < query.min_priority.value:
            return False

        # Check tags
        if query.tags:
            if not any(tag in entry.tags for tag in query.tags):
                return False

        # Check time range
        if query.time_range:
            start, end = query.time_range
            if not (start <= entry.created_at <= end):
                return False

        # Check text query (simple substring match)
        if query.query_text:
            content_str = str(entry.content).lower()
            if query.query_text.lower() not in content_str:
                return False

        return True

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        memory_types: tuple[MemoryType, ...] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic search using embeddings."""
        async with self._lock:
            results = self._index.search(embedding, limit=limit * 2)

            # Filter by memory type
            if memory_types:
                results = [
                    (entry, score)
                    for entry, score in results
                    if entry.memory_type in memory_types
                ]

            # Filter expired
            results = [
                (entry, score)
                for entry, score in results
                if not entry.is_expired
            ]

            return results[:limit]

    async def clear(
        self,
        memory_type: MemoryType | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear memories matching criteria."""
        async with self._lock:
            to_delete: list[str] = []

            for entry_id, entry in self._entries.items():
                should_delete = True

                if memory_type and entry.memory_type != memory_type:
                    should_delete = False
                if agent_id and entry.agent_id != agent_id:
                    should_delete = False
                if session_id and entry.session_id != session_id:
                    should_delete = False

                if should_delete:
                    to_delete.append(entry_id)

            for entry_id in to_delete:
                del self._entries[entry_id]
                self._index.remove(entry_id)

            return len(to_delete)

    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        async with self._lock:
            stats = MemoryStats()
            stats.total_entries = len(self._entries)

            by_type: dict[str, int] = {}
            by_agent: dict[str, int] = {}
            expired_count = 0
            oldest: datetime | None = None
            newest: datetime | None = None

            for entry in self._entries.values():
                # By type
                type_key = entry.memory_type.value
                by_type[type_key] = by_type.get(type_key, 0) + 1

                # By agent
                if entry.agent_id:
                    by_agent[entry.agent_id] = by_agent.get(entry.agent_id, 0) + 1

                # Expired
                if entry.is_expired:
                    expired_count += 1

                # Time range
                if oldest is None or entry.created_at < oldest:
                    oldest = entry.created_at
                if newest is None or entry.created_at > newest:
                    newest = entry.created_at

            stats.by_type = by_type
            stats.by_agent = by_agent
            stats.expired_count = expired_count
            stats.oldest_entry = oldest
            stats.newest_entry = newest

            return stats

    async def health_check(self) -> bool:
        """Check backend health."""
        return True

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            to_delete = [
                entry_id
                for entry_id, entry in self._entries.items()
                if entry.is_expired
            ]

            for entry_id in to_delete:
                del self._entries[entry_id]
                self._index.remove(entry_id)

            return len(to_delete)


# =============================================================================
# MEMORY CAPABILITY
# =============================================================================


class MemoryCapability(Capability):
    """
    Memory management capability for the AI Agent Platform.

    Provides unified interface for:
    - Storing memories (short-term, long-term, semantic)
    - Retrieving memories by ID or query
    - Semantic search across memories
    - Memory injection into context
    - Memory lifecycle management

    Compatible with:
    - CapabilityRegistry for registration
    - Executor for action execution
    - ContextEngine for context preparation
    """

    # Default TTLs for memory types (in seconds)
    DEFAULT_TTLS: dict[MemoryType, int | None] = {
        MemoryType.SHORT_TERM: 3600,  # 1 hour
        MemoryType.LONG_TERM: None,  # Never expires
        MemoryType.SEMANTIC: None,  # Never expires
        MemoryType.EPISODIC: 86400 * 30,  # 30 days
        MemoryType.WORKING: 300,  # 5 minutes
    }

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        default_agent_id: str | None = None,
        default_session_id: str | None = None,
        max_context_memories: int = 10,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize memory capability.

        Args:
            backend: Storage backend (default: InMemoryBackend)
            embedding_fn: Function to generate embeddings
            default_agent_id: Default agent ID for entries
            default_session_id: Default session ID for entries
            max_context_memories: Max memories to inject into context
            logger: Optional logger
        """
        self._backend = backend or InMemoryBackend(logger=logger)
        self._embedding_fn = embedding_fn
        self._default_agent_id = default_agent_id
        self._default_session_id = default_session_id
        self._max_context_memories = max_context_memories
        self._logger = logger or logging.getLogger(__name__)
        self._initialized = False

        # Action handlers
        self._action_handlers: dict[MemoryAction, Callable[..., Any]] = {
            MemoryAction.STORE: self._handle_store,
            MemoryAction.RETRIEVE: self._handle_retrieve,
            MemoryAction.UPDATE: self._handle_update,
            MemoryAction.DELETE: self._handle_delete,
            MemoryAction.SEARCH: self._handle_search,
            MemoryAction.INJECT: self._handle_inject,
            MemoryAction.CLEAR: self._handle_clear,
            MemoryAction.LIST: self._handle_list,
            MemoryAction.STATS: self._handle_stats,
        }

    # -------------------------------------------------------------------------
    # Capability Interface
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def _do_initialize(self) -> None:
        self._logger.debug("MemoryCapability._do_initialize")

    async def _do_shutdown(self) -> None:
        self._logger.debug("MemoryCapability._do_shutdown")

    async def _do_execute(self, context) -> CapabilityResult:
        return CapabilityResult(success=True, status=CapabilityStatus.SUCCESS)  # type: ignore[attr-defined]

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return CapabilityMetadata(
            name="memory",
            version="1.0.0",
            capability_type=CapabilityType.MEMORY,
            description="Memory management capability for storing, retrieving, and searching memories",
            actions=tuple(action.value for action in MemoryAction),
            required_permissions=("memory:read", "memory:write"),
            config_schema={
                "type": "object",
                "properties": {
                    "backend_type": {"type": "string", "enum": ["memory", "chromadb", "postgresql"]},
                    "max_entries": {"type": "integer", "minimum": 100},
                    "default_ttl": {"type": "integer", "minimum": 0},
                },
            },
            tags=("memory", "storage", "retrieval", "semantic"),
        )

    async def initialize(self) -> None:
        """Initialize the capability."""
        if self._initialized:
            return

        # Start backend if it has a start method
        if hasattr(self._backend, "start"):
            await self._backend.start()

        self._initialized = True
        self._logger.info("MemoryCapability initialized")

    async def shutdown(self) -> None:
        """Shutdown the capability."""
        if not self._initialized:
            return

        # Stop backend if it has a stop method
        if hasattr(self._backend, "stop"):
            await self._backend.stop()

        self._initialized = False
        self._logger.info("MemoryCapability shutdown")

    async def health_check(self) -> bool:
        """Check capability health."""
        try:
            return await self._backend.health_check()
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    async def execute(  # type: ignore[override]
        self,
        action: str,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> CapabilityResult:
        """
        Execute a memory action.

        Args:
            action: Action to perform (store, retrieve, update, delete, search, inject, clear)
            parameters: Action parameters
            context: Execution context

        Returns:
            CapabilityResult with operation outcome
        """
        start_time = time.time()
        context = context or {}

        try:
            # Validate action
            try:
                memory_action = MemoryAction(action)
            except ValueError:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Get handler
            handler = self._action_handlers.get(memory_action)
            if handler is None:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"No handler for action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Execute handler
            result = await handler(parameters, context)

            return CapabilityResult(
                success=result.success,
                status=CapabilityStatus.COMPLETED if result.success else CapabilityStatus.FAILED,
                data={
                    "entries": [e.to_dict() for e in result.entries],
                    "total_count": result.total_count,
                    "message": result.message,
                    "metadata": result.metadata,
                },
                execution_time=time.time() - start_time,
            )

        except ValidationError as e:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=f"Validation error: {e}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            self._logger.exception(f"Memory action failed: {e}")
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    async def _handle_store(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle store action."""
        # Validate required parameters
        content = parameters.get("content")
        if content is None:
            raise ValidationError("content is required")

        # Parse memory type
        memory_type_str = parameters.get("memory_type", "long_term")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            raise ValidationError(f"Invalid memory_type: {memory_type_str}")

        # Create entry
        entry = await self.store(
            content=content,
            memory_type=memory_type,
            metadata=parameters.get("metadata", {}),
            tags=tuple(parameters.get("tags", [])),
            priority=MemoryPriority(parameters.get("priority", 5)),
            agent_id=parameters.get("agent_id") or context.get("agent_id") or self._default_agent_id,
            session_id=parameters.get("session_id") or context.get("session_id") or self._default_session_id,
            source=parameters.get("source", "user"),
            confidence=parameters.get("confidence", 1.0),
            ttl=parameters.get("ttl"),
        )

        return MemoryResult(
            success=True,
            entries=[entry],
            total_count=1,
            message=f"Stored memory {entry.id}",
        )

    async def _handle_retrieve(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle retrieve action."""
        entry_id = parameters.get("id")
        if entry_id:
            entry = await self.retrieve(entry_id)
            if entry:
                return MemoryResult(
                    success=True,
                    entries=[entry],
                    total_count=1,
                    message=f"Retrieved memory {entry_id}",
                )
            else:
                return MemoryResult(
                    success=False,
                    message=f"Memory {entry_id} not found",
                )

        # Build query from parameters
        query = self._build_query(parameters, context)
        return await self._backend.query(query)

    async def _handle_update(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle update action."""
        entry_id = parameters.get("id")
        if not entry_id:
            raise ValidationError("id is required for update")

        # Get existing entry
        entry = await self.retrieve(entry_id)
        if not entry:
            return MemoryResult(
                success=False,
                message=f"Memory {entry_id} not found",
            )

        # Update fields
        if "content" in parameters:
            entry.content = parameters["content"]
            # Regenerate embedding if content changed
            if self._embedding_fn:
                content_str = str(entry.content)
                entry.embedding = self._embedding_fn(content_str)

        if "metadata" in parameters:
            entry.metadata.update(parameters["metadata"])

        if "tags" in parameters:
            entry.tags = tuple(parameters["tags"])

        if "priority" in parameters:
            entry.priority = MemoryPriority(parameters["priority"])

        if "confidence" in parameters:
            entry.confidence = parameters["confidence"]

        success = await self._backend.update(entry)

        return MemoryResult(
            success=success,
            entries=[entry] if success else [],
            total_count=1 if success else 0,
            message=f"Updated memory {entry_id}" if success else f"Failed to update {entry_id}",
        )

    async def _handle_delete(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle delete action."""
        entry_id = parameters.get("id")
        if not entry_id:
            raise ValidationError("id is required for delete")

        success = await self._backend.delete(entry_id)

        return MemoryResult(
            success=success,
            message=f"Deleted memory {entry_id}" if success else f"Memory {entry_id} not found",
        )

    async def _handle_search(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle search action."""
        query_text = parameters.get("query")
        if not query_text:
            raise ValidationError("query is required for search")

        # Generate embedding for semantic search
        embedding: list[float] | None = None
        if self._embedding_fn:
            embedding = self._embedding_fn(query_text)

        # Parse memory types
        memory_types: tuple[MemoryType, ...] | None = None
        if "memory_types" in parameters:
            memory_types = tuple(
                MemoryType(t) for t in parameters["memory_types"]
            )

        limit = parameters.get("limit", 10)

        if embedding:
            # Semantic search
            results = await self._backend.search(
                embedding=embedding,
                limit=limit,
                memory_types=memory_types,
            )

            entries = [entry for entry, _ in results]
            scores = {entry.id: score for entry, score in results}

            return MemoryResult(
                success=True,
                entries=entries,
                total_count=len(entries),
                message=f"Found {len(entries)} similar memories",
                metadata={"scores": scores},
            )
        else:
            # Text search fallback
            query = self._build_query(parameters, context)
            query.query_text = query_text
            return await self._backend.query(query)

    async def _handle_inject(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle inject action - prepare memories for context injection."""
        query_text = parameters.get("query")
        max_memories = parameters.get("max_memories", self._max_context_memories)

        # Get relevant memories
        if query_text and self._embedding_fn:
            embedding = self._embedding_fn(query_text)
            results = await self._backend.search(
                embedding=embedding,
                limit=max_memories,
            )
            entries = [entry for entry, _ in results]
        else:
            # Get recent memories
            query = MemoryQuery(
                agent_id=context.get("agent_id") or self._default_agent_id,
                session_id=context.get("session_id") or self._default_session_id,
                limit=max_memories,
                sort_by="updated_at",
                sort_order="desc",
            )
            result = await self._backend.query(query)
            entries = result.entries

        # Format for injection
        injection_text = self._format_for_injection(entries)

        return MemoryResult(
            success=True,
            entries=entries,
            total_count=len(entries),
            message=f"Prepared {len(entries)} memories for injection",
            metadata={"injection_text": injection_text},
        )

    async def _handle_clear(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle clear action."""
        memory_type: MemoryType | None = None
        if "memory_type" in parameters:
            memory_type = MemoryType(parameters["memory_type"])

        agent_id = parameters.get("agent_id") or context.get("agent_id")
        session_id = parameters.get("session_id") or context.get("session_id")

        count = await self._backend.clear(
            memory_type=memory_type,
            agent_id=agent_id,
            session_id=session_id,
        )

        return MemoryResult(
            success=True,
            total_count=count,
            message=f"Cleared {count} memories",
        )

    async def _handle_list(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle list action."""
        query = self._build_query(parameters, context)
        return await self._backend.query(query)

    async def _handle_stats(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryResult:
        """Handle stats action."""
        stats = await self._backend.get_stats()

        return MemoryResult(
            success=True,
            message="Memory statistics retrieved",
            metadata={
                "total_entries": stats.total_entries,
                "by_type": stats.by_type,
                "by_agent": stats.by_agent,
                "expired_count": stats.expired_count,
                "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
                "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
            },
        )

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    async def store(
        self,
        content: str | dict[str, Any],
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: dict[str, Any] | None = None,
        tags: tuple[str, ...] = (),
        priority: MemoryPriority = MemoryPriority.NORMAL,
        agent_id: str | None = None,
        session_id: str | None = None,
        source: str = "user",
        confidence: float = 1.0,
        ttl: int | None = None,
    ) -> MemoryEntry:
        """
        Store a new memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Additional metadata
            tags: Searchable tags
            priority: Importance level
            agent_id: Owning agent
            session_id: Session scope
            source: Origin of memory
            confidence: Confidence score
            ttl: Time-to-live in seconds (overrides default)

        Returns:
            Created MemoryEntry
        """
        # Generate ID
        entry_id = str(uuid.uuid4())

        # Calculate expiration
        expires_at: datetime | None = None
        effective_ttl = ttl if ttl is not None else self.DEFAULT_TTLS.get(memory_type)
        if effective_ttl is not None:
            from datetime import timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=effective_ttl)

        # Generate embedding
        embedding: list[float] | None = None
        if self._embedding_fn:
            content_str = str(content)
            embedding = self._embedding_fn(content_str)

        # Create entry
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            embedding=embedding,
            priority=priority,
            agent_id=agent_id or self._default_agent_id,
            session_id=session_id or self._default_session_id,
            tags=tags,
            source=source,
            confidence=confidence,
            expires_at=expires_at,
        )

        # Store
        await self._backend.store(entry)

        self._logger.debug(f"Stored memory {entry_id} ({memory_type.value})")

        return entry

    async def retrieve(self, entry_id: str) -> MemoryEntry | None:
        """
        Retrieve a memory by ID.

        Args:
            entry_id: Memory ID

        Returns:
            MemoryEntry or None if not found
        """
        return await self._backend.retrieve(entry_id)

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_types: tuple[MemoryType, ...] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Semantic search for memories.

        Args:
            query: Search query
            limit: Maximum results
            memory_types: Filter by types
            min_score: Minimum similarity score

        Returns:
            List of (entry, score) tuples
        """
        if not self._embedding_fn:
            self._logger.warning("No embedding function configured, search unavailable")
            return []

        embedding = self._embedding_fn(query)

        results = await self._backend.search(
            embedding=embedding,
            limit=limit,
            memory_types=memory_types,
        )

        # Filter by minimum score
        return [(entry, score) for entry, score in results if score >= min_score]

    async def get_context(
        self,
        query: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        max_memories: int | None = None,
    ) -> dict[str, Any]:
        """
        Get memory context for context engine integration.

        Args:
            query: Optional query for relevance filtering
            agent_id: Filter by agent
            session_id: Filter by session
            max_memories: Maximum memories to include

        Returns:
            Context dictionary for injection
        """
        max_memories = max_memories or self._max_context_memories

        if query and self._embedding_fn:
            # Semantic retrieval
            results = await self.search(query, limit=max_memories)
            entries = [entry for entry, _ in results]
        else:
            # Recent memories
            mem_query = MemoryQuery(
                agent_id=agent_id or self._default_agent_id,
                session_id=session_id or self._default_session_id,
                limit=max_memories,
                sort_by="updated_at",
                sort_order="desc",
            )
            result = await self._backend.query(mem_query)
            entries = result.entries

        return {
            "memories": [e.to_dict() for e in entries],
            "memory_text": self._format_for_injection(entries),
            "memory_count": len(entries),
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_query(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> MemoryQuery:
        """Build MemoryQuery from parameters."""
        memory_types: tuple[MemoryType, ...] | None = None
        if "memory_types" in parameters:
            memory_types = tuple(
                MemoryType(t) for t in parameters["memory_types"]
            )

        return MemoryQuery(
            query_text=parameters.get("query_text"),
            memory_types=memory_types,
            tags=tuple(parameters.get("tags", [])) or None,
            agent_id=parameters.get("agent_id") or context.get("agent_id") or self._default_agent_id,
            session_id=parameters.get("session_id") or context.get("session_id") or self._default_session_id,
            min_confidence=parameters.get("min_confidence", 0.0),
            min_priority=MemoryPriority(parameters.get("min_priority", 1)),
            limit=parameters.get("limit", 10),
            offset=parameters.get("offset", 0),
            include_expired=parameters.get("include_expired", False),
            sort_by=parameters.get("sort_by", "created_at"),
            sort_order=parameters.get("sort_order", "desc"),
        )

    def _format_for_injection(self, entries: list[MemoryEntry]) -> str:
        """Format memories for context injection."""
        if not entries:
            return "No relevant memories found."

        lines: list[str] = ["## Relevant Memories", ""]

        for i, entry in enumerate(entries, 1):
            content = entry.content
            if isinstance(content, dict):
                content = str(content)

            # Truncate long content
            if len(content) > 500:
                content = content[:497] + "..."

            lines.append(f"[{i}] ({entry.memory_type.value}) {content}")

            if entry.tags:
                lines.append(f"    Tags: {', '.join(entry.tags)}")

            lines.append("")

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_memory_capability(
    backend_type: str = "memory",
    embedding_fn: Callable[[str], list[float]] | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> MemoryCapability:
    """
    Factory function to create MemoryCapability with specified backend.

    Args:
        backend_type: Backend type ("memory", "chromadb", "postgresql")
        embedding_fn: Function to generate embeddings
        logger: Optional logger
        **kwargs: Additional backend configuration

    Returns:
        Configured MemoryCapability
    """
    logger = logger or logging.getLogger(__name__)

    if backend_type == "memory":
        backend = InMemoryBackend(
            max_entries=kwargs.get("max_entries", 10000),
            cleanup_interval=kwargs.get("cleanup_interval", 60.0),
            logger=logger,
        )
    elif backend_type == "chromadb":
        # Future: ChromaDB backend
        raise NotImplementedError("ChromaDB backend not yet implemented")
    elif backend_type == "postgresql":
        # Future: PostgreSQL backend
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return MemoryCapability(
        backend=backend,
        embedding_fn=embedding_fn,
        default_agent_id=kwargs.get("default_agent_id"),
        default_session_id=kwargs.get("default_session_id"),
        max_context_memories=kwargs.get("max_context_memories", 10),
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MemoryType",
    "MemoryAction",
    "MemoryPriority",
    # Data Classes
    "MemoryEntry",
    "MemoryQuery",
    "MemoryResult",
    "MemoryStats",
    # Backend
    "MemoryBackend",
    "InMemoryBackend",
    "MemoryIndex",
    # Capability
    "MemoryCapability",
    # Factory
    "create_memory_capability",
]
