"""
State Manager - Session state lifecycle management for the AI Agent Platform.

This module handles all aspects of session state management including:
- Loading and creating sessions
- In-memory caching with TTL
- Persistence to configurable backends
- Checkpointing during execution
- Session cleanup and expiration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from app.agents.models import (
    AgentRequest,
    AgentResponse,
    ExecutionPlan,
    PlanStep,
    SessionState,
    StepResult,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# PROTOCOLS & ABSTRACT BASES
# =============================================================================


@runtime_checkable
class StateBackend(Protocol):
    """
    Protocol defining the interface for state persistence backends.

    Implementations must provide async methods for CRUD operations
    on session state. Examples: Redis, PostgreSQL, filesystem.
    """

    async def load(self, session_id: str) -> SessionState | None:
        """
        Load session state from storage.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState if found, None otherwise
        """
        ...

    async def save(self, state: SessionState) -> None:
        """
        Save session state to storage.

        Args:
            state: Session state to persist
        """
        ...

    async def delete(self, session_id: str) -> None:
        """
        Delete session state from storage.

        Args:
            session_id: Session to delete
        """
        ...

    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists in storage.

        Args:
            session_id: Session to check

        Returns:
            True if session exists
        """
        ...


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================


@dataclass
class CacheEntry:
    """
    Single entry in the session cache.

    Attributes:
        state: The cached session state
        created_at: When entry was created
        last_accessed: Last access timestamp
        access_count: Number of times accessed
    """

    state: SessionState
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = time.monotonic()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Return age of entry in seconds."""
        return time.monotonic() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Return time since last access in seconds."""
        return time.monotonic() - self.last_accessed


class SessionCache:
    """
    In-memory LRU cache for session states with TTL support.

    Provides fast access to frequently used sessions while
    respecting memory limits and expiration policies.

    Attributes:
        max_size: Maximum number of cached sessions
        ttl_seconds: Time-to-live for cache entries
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the session cache.

        Args:
            max_size: Maximum cached sessions (default: 1000)
            ttl_seconds: Entry TTL in seconds (default: 1 hour)
            logger: Optional logger instance
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self._hits + self._misses
        return (self._hits / total * 100) if total > 0 else 0.0

    async def get(self, session_id: str) -> SessionState | None:
        """
        Retrieve session from cache.

        Args:
            session_id: Session to retrieve

        Returns:
            SessionState if found and not expired, None otherwise
        """
        async with self._lock:
            entry = self._cache.get(session_id)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if entry.age_seconds > self._ttl_seconds:
                del self._cache[session_id]
                self._misses += 1
                self._logger.debug(f"Cache entry expired: {session_id}")
                return None

            entry.touch()
            self._hits += 1
            return entry.state

    async def set(self, state: SessionState) -> None:
        """
        Store session in cache.

        Args:
            state: Session state to cache
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                await self._evict_lru()

            self._cache[state.session_id] = CacheEntry(state=state)

    async def invalidate(self, session_id: str) -> bool:
        """
        Remove session from cache.

        Args:
            session_id: Session to invalidate

        Returns:
            True if session was in cache
        """
        async with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cached sessions.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._logger.info(f"Cache cleared: {count} entries removed")
            return count

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = time.monotonic()
            expired = [
                sid
                for sid, entry in self._cache.items()
                if (now - entry.created_at) > self._ttl_seconds
            ]

            for sid in expired:
                del self._cache[sid]

            if expired:
                self._logger.debug(f"Cleaned up {len(expired)} expired cache entries")

            return len(expired)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_id = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_id]
        self._logger.debug(f"Evicted LRU cache entry: {lru_id}")

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# BACKEND IMPLEMENTATIONS
# =============================================================================


class InMemoryBackend:
    """
    Simple in-memory state backend for development and testing.

    WARNING: Data is lost on process restart. Use only for
    development or single-instance deployments.
    """

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str) -> SessionState | None:
        """Load session from memory."""
        async with self._lock:
            data = self._storage.get(session_id)
            if data is None:
                return None
            return SessionState.from_dict(data)

    async def save(self, state: SessionState) -> None:
        """Save session to memory."""
        async with self._lock:
            self._storage[state.session_id] = state.to_dict()

    async def delete(self, session_id: str) -> None:
        """Delete session from memory."""
        async with self._lock:
            self._storage.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self._storage

    async def clear_all(self) -> int:
        """Clear all sessions (testing utility)."""
        async with self._lock:
            count = len(self._storage)
            self._storage.clear()
            return count


class FileSystemBackend:
    """
    File-based state backend for simple persistent storage.

    Stores each session as a JSON file. Suitable for low-traffic
    deployments or development environments.
    """

    def __init__(
        self,
        storage_path: str = "./sessions",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize filesystem backend.

        Args:
            storage_path: Directory for session files
            logger: Optional logger instance
        """
        import os

        self._storage_path = storage_path
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

    def _get_file_path(self, session_id: str) -> str:
        """Get file path for session."""
        import os

        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c == "-")
        return os.path.join(self._storage_path, f"{safe_id}.json")

    async def load(self, session_id: str) -> SessionState | None:
        """Load session from file."""
        import os

        file_path = self._get_file_path(session_id)

        if not os.path.exists(file_path):
            return None

        try:
            async with self._lock:
                # Run file I/O in thread pool
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, self._read_file, file_path
                )
                return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self._logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def save(self, state: SessionState) -> None:
        """Save session to file."""
        file_path = self._get_file_path(state.session_id)

        try:
            async with self._lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._write_file, file_path, state.to_dict()
                )
        except OSError as e:
            self._logger.error(f"Failed to save session {state.session_id}: {e}")
            raise

    async def delete(self, session_id: str) -> None:
        """Delete session file."""
        import os

        file_path = self._get_file_path(session_id)

        try:
            async with self._lock:
                if os.path.exists(file_path):
                    os.remove(file_path)
        except OSError as e:
            self._logger.error(f"Failed to delete session {session_id}: {e}")

    async def exists(self, session_id: str) -> bool:
        """Check if session file exists."""
        import os

        return os.path.exists(self._get_file_path(session_id))

    def _read_file(self, path: str) -> dict[str, Any]:
        """Synchronous file read (for executor)."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_file(self, path: str, data: dict[str, Any]) -> None:
        """Synchronous file write (for executor)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# STATE MANAGER
# =============================================================================


@dataclass
class StateManagerConfig:
    """
    Configuration for StateManager.

    Attributes:
        cache_enabled: Whether to use in-memory caching
        cache_max_size: Maximum cached sessions
        cache_ttl_seconds: Cache entry TTL
        checkpoint_enabled: Whether to checkpoint during execution
        checkpoint_async: Whether checkpoints are fire-and-forget
        cleanup_interval_seconds: Interval for cleanup task
    """

    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600
    checkpoint_enabled: bool = True
    checkpoint_async: bool = True
    cleanup_interval_seconds: int = 300


class StateManager:
    """
    Manages session state lifecycle for the agent platform.

    Coordinates between cache and persistent backend to provide
    fast access with durability guarantees.

    Responsibilities:
        - Load existing sessions or create new ones
        - Cache frequently accessed sessions
        - Persist state changes to backend
        - Checkpoint during plan execution
        - Clean up expired sessions
    """

    def __init__(
        self,
        backend: StateBackend,
        config: StateManagerConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the state manager.

        Args:
            backend: Persistence backend implementation
            config: Optional configuration
            logger: Optional logger instance
        """
        self._backend = backend
        self._config = config or StateManagerConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Initialize cache if enabled
        self._cache: SessionCache | None = None
        if self._config.cache_enabled:
            self._cache = SessionCache(
                max_size=self._config.cache_max_size,
                ttl_seconds=self._config.cache_ttl_seconds,
                logger=self._logger,
            )

        # Background tasks
        self._cleanup_task: asyncio.Task[None] | None = None
        self._pending_checkpoints: set[asyncio.Task[None]] = set()

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    async def load_or_create(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        """
        Load existing session or create a new one.

        Checks cache first, then backend. Creates new session
        if not found in either location.

        Args:
            session_id: Session identifier
            metadata: Optional metadata for new sessions

        Returns:
            Existing or newly created SessionState
        """
        # Try cache first
        if self._cache:
            cached = await self._cache.get(session_id)
            if cached:
                self._logger.debug(f"Session loaded from cache: {session_id}")
                return cached

        # Try backend
        state = await self._backend.load(session_id)

        if state is None:
            # Create new session
            state = SessionState.create(
                session_id=session_id,
                metadata=metadata,
            )
            self._logger.info(f"Created new session: {session_id}")

            # Persist immediately
            await self._backend.save(state)
        else:
            self._logger.debug(f"Session loaded from backend: {session_id}")

        # Update cache
        if self._cache:
            await self._cache.set(state)

        return state

    async def load(self, session_id: str) -> SessionState | None:
        """
        Load session without creating if not found.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        # Try cache
        if self._cache:
            cached = await self._cache.get(session_id)
            if cached:
                return cached

        # Try backend
        state = await self._backend.load(session_id)

        if state and self._cache:
            await self._cache.set(state)

        return state

    async def save(self, state: SessionState) -> None:
        """
        Save session state to cache and backend.

        Args:
            state: Session state to save
        """
        state.touch()

        # Update cache
        if self._cache:
            await self._cache.set(state)

        # Persist to backend
        await self._backend.save(state)

        self._logger.debug(f"Session saved: {state.session_id}")

    async def persist(self, state: SessionState) -> None:
        """
        Persist session state (alias for save with logging).

        Args:
            state: Session state to persist
        """
        await self.save(state)
        self._logger.info(
            f"Session persisted: {state.session_id} "
            f"(history: {state.history_size} entries)"
        )

    async def delete(self, session_id: str) -> bool:
        """
        Delete session from cache and backend.

        Args:
            session_id: Session to delete

        Returns:
            True if session existed
        """
        existed = await self._backend.exists(session_id)

        # Remove from cache
        if self._cache:
            await self._cache.invalidate(session_id)

        # Remove from backend
        await self._backend.delete(session_id)

        if existed:
            self._logger.info(f"Session deleted: {session_id}")

        return existed

    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists.

        Args:
            session_id: Session to check

        Returns:
            True if session exists
        """
        # Check cache first
        if self._cache:
            cached = await self._cache.get(session_id)
            if cached:
                return True

        return await self._backend.exists(session_id)

    # -------------------------------------------------------------------------
    # Session Updates
    # -------------------------------------------------------------------------

    async def update_history(
        self,
        session_id: str,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SessionState:
        """
        Add request/response pair to session history.

        Args:
            session_id: Session to update
            request: The user's request
            response: The agent's response

        Returns:
            Updated SessionState

        Raises:
            ValueError: If session not found
        """
        state = await self.load(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        state.add_to_history(request, response)
        await self.save(state)

        return state

    async def set_active_plan(
        self,
        session_id: str,
        plan: ExecutionPlan,
    ) -> SessionState:
        """
        Set the active execution plan for a session.

        Args:
            session_id: Session to update
            plan: Execution plan to set

        Returns:
            Updated SessionState

        Raises:
            ValueError: If session not found
        """
        state = await self.load(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        state.set_active_plan(plan)
        await self.save(state)

        self._logger.debug(
            f"Active plan set for session {session_id}: {plan.plan_id}"
        )

        return state

    async def clear_active_plan(self, session_id: str) -> SessionState:
        """
        Clear the active execution plan for a session.

        Args:
            session_id: Session to update

        Returns:
            Updated SessionState

        Raises:
            ValueError: If session not found
        """
        state = await self.load(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        state.clear_active_plan()
        await self.save(state)

        return state

    async def add_memory_ref(
        self,
        session_id: str,
        memory_ref: str,
    ) -> SessionState:
        """
        Add a memory reference to the session.

        Args:
            session_id: Session to update
            memory_ref: Memory reference to add

        Returns:
            Updated SessionState

        Raises:
            ValueError: If session not found
        """
        state = await self.load(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        state.add_memory_ref(memory_ref)
        await self.save(state)

        return state

    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> SessionState:
        """
        Update session metadata.

        Args:
            session_id: Session to update
            metadata: Metadata to set/merge
            merge: If True, merge with existing; if False, replace

        Returns:
            Updated SessionState

        Raises:
            ValueError: If session not found
        """
        state = await self.load(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}")

        if merge:
            state.metadata.update(metadata)
        else:
            state.metadata = metadata

        await self.save(state)

        return state

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    async def checkpoint(
        self,
        state: SessionState,
        step: PlanStep,
        result: StepResult,
    ) -> None:
        """
        Save a checkpoint after step completion.

        Checkpoints provide recovery points during plan execution.
        Can be synchronous or async based on configuration.

        Args:
            state: Current session state
            step: Completed step
            result: Step result
        """
        if not self._config.checkpoint_enabled:
            return

        state.touch()

        if self._config.checkpoint_async:
            # Fire and forget
            task = asyncio.create_task(
                self._async_checkpoint(state, step.step_id)
            )
            self._pending_checkpoints.add(task)
            task.add_done_callback(self._pending_checkpoints.discard)
        else:
            # Synchronous checkpoint
            await self._backend.save(state)
            self._logger.debug(
                f"Checkpoint saved: session={state.session_id}, step={step.step_id}"
            )

    async def _async_checkpoint(
        self,
        state: SessionState,
        step_id: str,
    ) -> None:
        """Async checkpoint implementation."""
        try:
            await self._backend.save(state)
            self._logger.debug(
                f"Async checkpoint saved: session={state.session_id}, step={step_id}"
            )
        except Exception as e:
            self._logger.warning(
                f"Async checkpoint failed: session={state.session_id}, "
                f"step={step_id}, error={e}"
            )

    async def wait_for_checkpoints(self) -> None:
        """Wait for all pending checkpoints to complete."""
        if self._pending_checkpoints:
            await asyncio.gather(*self._pending_checkpoints, return_exceptions=True)

    # -------------------------------------------------------------------------
    # Cleanup & Maintenance
    # -------------------------------------------------------------------------

    async def cleanup_expired(
        self,
        max_age_seconds: int = 86400,
    ) -> int:
        """
        Clean up expired sessions from cache.

        Note: Backend cleanup depends on backend implementation.

        Args:
            max_age_seconds: Maximum session age (default: 24 hours)

        Returns:
            Number of cache entries cleaned
        """
        cleaned = 0

        if self._cache:
            cleaned = await self._cache.cleanup_expired()

        return cleaned

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("Started background cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is None:
            return

        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass

        self._cleanup_task = None
        self._logger.info("Stopped background cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup task error: {e}")

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    async def invalidate_cache(self, session_id: str) -> bool:
        """
        Invalidate a specific session in cache.

        Args:
            session_id: Session to invalidate

        Returns:
            True if session was in cache
        """
        if self._cache:
            return await self._cache.invalidate(session_id)
        return False

    async def clear_cache(self) -> int:
        """
        Clear all cached sessions.

        Returns:
            Number of entries cleared
        """
        if self._cache:
            return await self._cache.clear()
        return 0

    async def refresh_cache(self, session_id: str) -> SessionState | None:
        """
        Refresh cache entry from backend.

        Args:
            session_id: Session to refresh

        Returns:
            Refreshed SessionState or None
        """
        # Invalidate current cache entry
        if self._cache:
            await self._cache.invalidate(session_id)

        # Load fresh from backend
        state = await self._backend.load(session_id)

        # Update cache
        if state and self._cache:
            await self._cache.set(state)

        return state

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """
        Get state manager statistics.

        Returns:
            Dictionary of statistics
        """
        stats: dict[str, Any] = {
            "cache_enabled": self._config.cache_enabled,
            "checkpoint_enabled": self._config.checkpoint_enabled,
            "pending_checkpoints": len(self._pending_checkpoints),
        }

        if self._cache:
            stats["cache"] = self._cache.get_stats()

        return stats

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of state manager components.

        Returns:
            Dictionary mapping component to health status
        """
        health: dict[str, bool] = {}

        # Check backend
        try:
            test_id = "__health_check__"
            test_state = SessionState.create(session_id=test_id)
            await self._backend.save(test_state)
            loaded = await self._backend.load(test_id)
            await self._backend.delete(test_id)
            health["backend"] = loaded is not None
        except Exception:
            health["backend"] = False

        # Check cache
        health["cache"] = self._cache is not None

        return health


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_state_manager(
    backend_type: str = "memory",
    backend_config: dict[str, Any] | None = None,
    manager_config: StateManagerConfig | None = None,
    logger: logging.Logger | None = None,
) -> StateManager:
    """
    Factory function to create a configured StateManager.

    Args:
        backend_type: Type of backend ("memory", "filesystem")
        backend_config: Backend-specific configuration
        manager_config: StateManager configuration
        logger: Optional logger instance

    Returns:
        Configured StateManager instance

    Raises:
        ValueError: If backend_type is unknown
    """
    backend_config = backend_config or {}

    if backend_type == "memory":
        backend: StateBackend = InMemoryBackend()
    elif backend_type == "filesystem":
        backend = FileSystemBackend(
            storage_path=backend_config.get("storage_path", "./sessions"),
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return StateManager(
        backend=backend,
        config=manager_config,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

def create_agent() -> StateManager:
    """Factory function for agent_loader compatibility."""
    return create_state_manager(backend_type="memory")


__all__ = [
    # Protocols
    "StateBackend",
    # Cache
    "CacheEntry",
    "SessionCache",
    # Backends
    "InMemoryBackend",
    "FileSystemBackend",
    # Manager
    "StateManagerConfig",
    "StateManager",
    # Factory
    "create_state_manager",
]
