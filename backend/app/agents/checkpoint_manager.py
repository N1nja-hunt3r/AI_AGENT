"""
Checkpoint Manager - State checkpointing and recovery for the AI Agent Platform.

This module provides comprehensive checkpoint management including:
- Save and load execution checkpoints
- Rollback to previous states
- Recovery from failures
- Execution snapshots with versioning
- Incremental and full checkpoints
- Checkpoint compression and storage
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
StateT = TypeVar("StateT")


# =============================================================================
# ENUMS
# =============================================================================


class CheckpointType(Enum):
    """Types of checkpoints."""

    FULL = "full"  # Complete state snapshot
    INCREMENTAL = "incremental"  # Delta from previous
    AUTOMATIC = "automatic"  # System-triggered
    MANUAL = "manual"  # User-triggered
    RECOVERY = "recovery"  # Created during recovery
    MILESTONE = "milestone"  # Significant progress point


class CheckpointStatus(Enum):
    """Status of a checkpoint."""

    PENDING = auto()
    CREATING = auto()
    VALID = auto()
    CORRUPTED = auto()
    EXPIRED = auto()
    DELETED = auto()


class RecoveryStrategy(Enum):
    """Strategies for recovery."""

    LATEST = auto()  # Recover from latest checkpoint
    LATEST_VALID = auto()  # Recover from latest valid checkpoint
    SPECIFIC = auto()  # Recover from specific checkpoint
    BEST_EFFORT = auto()  # Try multiple checkpoints


class CompressionLevel(Enum):
    """Compression levels for checkpoints."""

    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CheckpointError(Exception):
    """Base exception for checkpoint errors."""

    def __init__(self, message: str, checkpoint_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.checkpoint_id = checkpoint_id


class CheckpointNotFoundError(CheckpointError):
    """Checkpoint not found."""

    pass


class CheckpointCorruptedError(CheckpointError):
    """Checkpoint data is corrupted."""

    pass


class RecoveryError(CheckpointError):
    """Recovery operation failed."""

    pass


class RollbackError(CheckpointError):
    """Rollback operation failed."""

    pass


# =============================================================================
# CHECKPOINT DATA STRUCTURES
# =============================================================================


@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        checkpoint_type: Type of checkpoint
        version: Checkpoint version number
        created_at: Creation timestamp
        session_id: Associated session
        agent_id: Associated agent
        step_index: Execution step index
        parent_id: Parent checkpoint (for incremental)
        description: Human-readable description
        tags: Searchable tags
        size_bytes: Checkpoint size
        checksum: Data checksum
        status: Current status
        expires_at: Expiration timestamp
        custom_metadata: Additional metadata
    """

    checkpoint_id: str
    checkpoint_type: CheckpointType
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None
    agent_id: str | None = None
    step_index: int = 0
    parent_id: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    size_bytes: int = 0
    checksum: str = ""
    status: CheckpointStatus = CheckpointStatus.VALID
    expires_at: datetime | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if checkpoint has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if checkpoint is valid."""
        return self.status == CheckpointStatus.VALID and not self.is_expired

    @property
    def age_seconds(self) -> float:
        """Get checkpoint age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "step_index": self.step_index,
            "parent_id": self.parent_id,
            "description": self.description,
            "tags": list(self.tags),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "status": self.status.name,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            checkpoint_type=CheckpointType(data.get("checkpoint_type", "full")),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            step_index=data.get("step_index", 0),
            parent_id=data.get("parent_id"),
            description=data.get("description", ""),
            tags=tuple(data.get("tags", [])),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            status=CheckpointStatus[data.get("status", "VALID")],
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class ExecutionSnapshot:
    """
    Snapshot of execution state.

    Attributes:
        state: Main state data
        context: Execution context
        plan_state: Current plan state
        step_results: Results from completed steps
        memory_state: Memory system state
        tool_state: Tool system state
        variables: Execution variables
        stack_trace: Execution stack for debugging
    """

    state: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    plan_state: dict[str, Any] | None = None
    step_results: list[dict[str, Any]] = field(default_factory=list)
    memory_state: dict[str, Any] | None = None
    tool_state: dict[str, Any] | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    stack_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "context": self.context,
            "plan_state": self.plan_state,
            "step_results": self.step_results,
            "memory_state": self.memory_state,
            "tool_state": self.tool_state,
            "variables": self.variables,
            "stack_trace": self.stack_trace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionSnapshot:
        """Create from dictionary."""
        return cls(
            state=data.get("state", {}),
            context=data.get("context", {}),
            plan_state=data.get("plan_state"),
            step_results=data.get("step_results", []),
            memory_state=data.get("memory_state"),
            tool_state=data.get("tool_state"),
            variables=data.get("variables", {}),
            stack_trace=data.get("stack_trace", []),
        )

    def clone(self) -> ExecutionSnapshot:
        """Create a deep copy."""
        return ExecutionSnapshot(
            state=copy.deepcopy(self.state),
            context=copy.deepcopy(self.context),
            plan_state=copy.deepcopy(self.plan_state) if self.plan_state else None,
            step_results=copy.deepcopy(self.step_results),
            memory_state=copy.deepcopy(self.memory_state) if self.memory_state else None,
            tool_state=copy.deepcopy(self.tool_state) if self.tool_state else None,
            variables=copy.deepcopy(self.variables),
            stack_trace=list(self.stack_trace),
        )


@dataclass
class Checkpoint:
    """
    Complete checkpoint with metadata and data.

    Attributes:
        metadata: Checkpoint metadata
        snapshot: Execution snapshot
        delta: Delta data for incremental checkpoints
    """

    metadata: CheckpointMetadata
    snapshot: ExecutionSnapshot
    delta: dict[str, Any] | None = None

    @property
    def checkpoint_id(self) -> str:
        return self.metadata.checkpoint_id

    @property
    def is_incremental(self) -> bool:
        return self.metadata.checkpoint_type == CheckpointType.INCREMENTAL

    @property
    def is_valid(self) -> bool:
        return self.metadata.is_valid


# =============================================================================
# SERIALIZATION
# =============================================================================


class CheckpointSerializer(Protocol):
    """Protocol for checkpoint serialization."""

    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint to bytes."""
        ...

    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize checkpoint from bytes."""
        ...


class JSONSerializer:
    """JSON-based checkpoint serializer."""

    def __init__(self, indent: int | None = None) -> None:
        self._indent = indent

    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize to JSON bytes."""
        data = {
            "metadata": checkpoint.metadata.to_dict(),
            "snapshot": checkpoint.snapshot.to_dict(),
            "delta": checkpoint.delta,
        }
        return json.dumps(data, indent=self._indent, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize from JSON bytes."""
        parsed = json.loads(data.decode("utf-8"))
        return Checkpoint(
            metadata=CheckpointMetadata.from_dict(parsed["metadata"]),
            snapshot=ExecutionSnapshot.from_dict(parsed["snapshot"]),
            delta=parsed.get("delta"),
        )


class PickleSerializer:
    """Pickle-based checkpoint serializer."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        self._protocol = protocol

    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize to pickle bytes."""
        return pickle.dumps(checkpoint, protocol=self._protocol)

    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize from pickle bytes."""
        return pickle.loads(data)


# =============================================================================
# COMPRESSION
# =============================================================================


class Compressor(Protocol):
    """Protocol for data compression."""

    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        ...

    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        ...


class GzipCompressor:
    """Gzip compression."""

    def __init__(self, level: CompressionLevel = CompressionLevel.BALANCED) -> None:
        self._level = level.value

    def compress(self, data: bytes) -> bytes:
        """Compress with gzip."""
        if self._level == 0:
            return data
        return gzip.compress(data, compresslevel=self._level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        try:
            return gzip.decompress(data)
        except gzip.BadGzipFile:
            # Data might not be compressed
            return data


class NoCompressor:
    """No compression (passthrough)."""

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


# =============================================================================
# STORAGE BACKENDS
# =============================================================================


class CheckpointStorage(ABC):
    """Abstract base for checkpoint storage."""

    @abstractmethod
    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> None:
        """Save checkpoint data."""
        ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> tuple[bytes, CheckpointMetadata] | None:
        """Load checkpoint data and metadata."""
        ...

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint."""
        ...

    @abstractmethod
    async def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        ...

    @abstractmethod
    async def list_checkpoints(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[CheckpointMetadata]:
        """List checkpoint metadata."""
        ...

    @abstractmethod
    async def get_metadata(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Get checkpoint metadata only."""
        ...


class InMemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage."""

    def __init__(self, max_checkpoints: int = 1000) -> None:
        self._data: dict[str, bytes] = {}
        self._metadata: dict[str, CheckpointMetadata] = {}
        self._max_checkpoints = max_checkpoints
        self._lock = asyncio.Lock()

    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> None:
        async with self._lock:
            # Enforce limit
            if len(self._data) >= self._max_checkpoints and checkpoint_id not in self._data:
                # Remove oldest
                oldest_id = min(
                    self._metadata.keys(),
                    key=lambda k: self._metadata[k].created_at,
                )
                del self._data[oldest_id]
                del self._metadata[oldest_id]

            self._data[checkpoint_id] = data
            self._metadata[checkpoint_id] = metadata

    async def load(self, checkpoint_id: str) -> tuple[bytes, CheckpointMetadata] | None:
        if checkpoint_id not in self._data:
            return None
        return self._data[checkpoint_id], self._metadata[checkpoint_id]

    async def delete(self, checkpoint_id: str) -> bool:
        async with self._lock:
            if checkpoint_id in self._data:
                del self._data[checkpoint_id]
                del self._metadata[checkpoint_id]
                return True
            return False

    async def exists(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self._data

    async def list_checkpoints(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[CheckpointMetadata]:
        results = list(self._metadata.values())

        if session_id:
            results = [m for m in results if m.session_id == session_id]

        if agent_id:
            results = [m for m in results if m.agent_id == agent_id]

        # Sort by creation time (newest first)
        results.sort(key=lambda m: m.created_at, reverse=True)

        return results

    async def get_metadata(self, checkpoint_id: str) -> CheckpointMetadata | None:
        return self._metadata.get(checkpoint_id)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def total_bytes(self) -> int:
        return sum(len(d) for d in self._data.values())


class FileSystemCheckpointStorage(CheckpointStorage):
    """Filesystem-based checkpoint storage."""

    def __init__(
        self,
        base_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base_path = Path(base_path)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

        # Create directories
        self._data_path = self._base_path / "data"
        self._metadata_path = self._base_path / "metadata"
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._metadata_path.mkdir(parents=True, exist_ok=True)

    def _get_data_path(self, checkpoint_id: str) -> Path:
        """Get path for checkpoint data."""
        return self._data_path / f"{checkpoint_id}.bin"

    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        """Get path for checkpoint metadata."""
        return self._metadata_path / f"{checkpoint_id}.json"

    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> None:
        async with self._lock:
            try:
                # Save data
                data_path = self._get_data_path(checkpoint_id)
                temp_path = data_path.with_suffix(".tmp")
                temp_path.write_bytes(data)
                temp_path.rename(data_path)

                # Save metadata
                metadata_path = self._get_metadata_path(checkpoint_id)
                temp_path = metadata_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(metadata.to_dict(), indent=2))
                temp_path.rename(metadata_path)

            except OSError as e:
                self._logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
                raise CheckpointError(f"Failed to save checkpoint: {e}", checkpoint_id)

    async def load(self, checkpoint_id: str) -> tuple[bytes, CheckpointMetadata] | None:
        data_path = self._get_data_path(checkpoint_id)
        metadata_path = self._get_metadata_path(checkpoint_id)

        if not data_path.exists() or not metadata_path.exists():
            return None

        try:
            data = data_path.read_bytes()
            metadata_dict = json.loads(metadata_path.read_text())
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            return data, metadata

        except (OSError, json.JSONDecodeError) as e:
            self._logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def delete(self, checkpoint_id: str) -> bool:
        async with self._lock:
            data_path = self._get_data_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)

            deleted = False
            if data_path.exists():
                data_path.unlink()
                deleted = True
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            return deleted

    async def exists(self, checkpoint_id: str) -> bool:
        return self._get_data_path(checkpoint_id).exists()

    async def list_checkpoints(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[CheckpointMetadata]:
        results: list[CheckpointMetadata] = []

        for metadata_file in self._metadata_path.glob("*.json"):
            try:
                metadata_dict = json.loads(metadata_file.read_text())
                metadata = CheckpointMetadata.from_dict(metadata_dict)

                if session_id and metadata.session_id != session_id:
                    continue
                if agent_id and metadata.agent_id != agent_id:
                    continue

                results.append(metadata)

            except (OSError, json.JSONDecodeError) as e:
                self._logger.warning(f"Failed to read metadata {metadata_file}: {e}")

        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    async def get_metadata(self, checkpoint_id: str) -> CheckpointMetadata | None:
        metadata_path = self._get_metadata_path(checkpoint_id)

        if not metadata_path.exists():
            return None

        try:
            metadata_dict = json.loads(metadata_path.read_text())
            return CheckpointMetadata.from_dict(metadata_dict)
        except (OSError, json.JSONDecodeError):
            return None


# =============================================================================
# DELTA CALCULATOR
# =============================================================================


class DeltaCalculator:
    """Calculates deltas between snapshots for incremental checkpoints."""

    def calculate_delta(
        self,
        old_snapshot: ExecutionSnapshot,
        new_snapshot: ExecutionSnapshot,
    ) -> dict[str, Any]:
        """
        Calculate delta between two snapshots.

        Args:
            old_snapshot: Previous snapshot
            new_snapshot: Current snapshot

        Returns:
            Delta dictionary
        """
        delta: dict[str, Any] = {
            "changes": {},
            "additions": {},
            "deletions": [],
        }

        # Compare state
        delta["changes"]["state"] = self._diff_dict(
            old_snapshot.state,
            new_snapshot.state,
        )

        # Compare context
        delta["changes"]["context"] = self._diff_dict(
            old_snapshot.context,
            new_snapshot.context,
        )

        # Compare variables
        delta["changes"]["variables"] = self._diff_dict(
            old_snapshot.variables,
            new_snapshot.variables,
        )

        # New step results
        old_count = len(old_snapshot.step_results)
        if len(new_snapshot.step_results) > old_count:
            delta["additions"]["step_results"] = new_snapshot.step_results[old_count:]

        # Plan state changes
        if new_snapshot.plan_state != old_snapshot.plan_state:
            delta["changes"]["plan_state"] = new_snapshot.plan_state

        # Memory state changes
        if new_snapshot.memory_state != old_snapshot.memory_state:
            delta["changes"]["memory_state"] = new_snapshot.memory_state

        # Tool state changes
        if new_snapshot.tool_state != old_snapshot.tool_state:
            delta["changes"]["tool_state"] = new_snapshot.tool_state

        return delta

    def apply_delta(
        self,
        base_snapshot: ExecutionSnapshot,
        delta: dict[str, Any],
    ) -> ExecutionSnapshot:
        """
        Apply delta to base snapshot.

        Args:
            base_snapshot: Base snapshot
            delta: Delta to apply

        Returns:
            New snapshot with delta applied
        """
        new_snapshot = base_snapshot.clone()

        changes = delta.get("changes", {})
        additions = delta.get("additions", {})

        # Apply state changes
        if "state" in changes:
            self._apply_diff(new_snapshot.state, changes["state"])

        # Apply context changes
        if "context" in changes:
            self._apply_diff(new_snapshot.context, changes["context"])

        # Apply variable changes
        if "variables" in changes:
            self._apply_diff(new_snapshot.variables, changes["variables"])

        # Add new step results
        if "step_results" in additions:
            new_snapshot.step_results.extend(additions["step_results"])

        # Apply plan state
        if "plan_state" in changes:
            new_snapshot.plan_state = changes["plan_state"]

        # Apply memory state
        if "memory_state" in changes:
            new_snapshot.memory_state = changes["memory_state"]

        # Apply tool state
        if "tool_state" in changes:
            new_snapshot.tool_state = changes["tool_state"]

        return new_snapshot

    def _diff_dict(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate diff between two dictionaries."""
        diff: dict[str, Any] = {
            "changed": {},
            "added": {},
            "removed": [],
        }

        # Find changed and added
        for key, value in new.items():
            if key not in old:
                diff["added"][key] = value
            elif old[key] != value:
                diff["changed"][key] = value

        # Find removed
        for key in old:
            if key not in new:
                diff["removed"].append(key)

        return diff

    def _apply_diff(self, target: dict[str, Any], diff: dict[str, Any]) -> None:
        """Apply diff to target dictionary."""
        # Apply changes
        for key, value in diff.get("changed", {}).items():
            target[key] = value

        # Apply additions
        for key, value in diff.get("added", {}).items():
            target[key] = value

        # Apply removals
        for key in diff.get("removed", []):
            target.pop(key, None)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


@dataclass
class CheckpointManagerConfig:
    """
    Configuration for CheckpointManager.

    Attributes:
        auto_checkpoint_interval: Steps between auto checkpoints
        max_checkpoints_per_session: Maximum checkpoints per session
        retention_hours: Hours to retain checkpoints
        enable_incremental: Enable incremental checkpoints
        compression_level: Compression level
        verify_on_load: Verify checksum on load
    """

    auto_checkpoint_interval: int = 10
    max_checkpoints_per_session: int = 100
    retention_hours: int = 24
    enable_incremental: bool = True
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    verify_on_load: bool = True


class CheckpointManager:
    """
    Central checkpoint management system.

    Provides save, load, rollback, and recovery operations
    for execution state checkpoints.
    """

    def __init__(
        self,
        config: CheckpointManagerConfig | None = None,
        storage: CheckpointStorage | None = None,
        serializer: CheckpointSerializer | None = None,
        compressor: Compressor | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            config: Manager configuration
            storage: Checkpoint storage backend
            serializer: Checkpoint serializer
            compressor: Data compressor
            logger: Optional logger
        """
        self._config = config or CheckpointManagerConfig()
        self._storage = storage or InMemoryCheckpointStorage()
        self._serializer = serializer or JSONSerializer()
        self._compressor = compressor or GzipCompressor(self._config.compression_level)
        self._logger = logger or logging.getLogger(__name__)

        # Components
        self._delta_calculator = DeltaCalculator()

        # State tracking
        self._last_checkpoint: dict[str, Checkpoint] = {}  # session_id -> checkpoint
        self._step_counters: dict[str, int] = {}  # session_id -> step count
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Save Operations
    # -------------------------------------------------------------------------

    async def save(
        self,
        snapshot: ExecutionSnapshot,
        session_id: str,
        agent_id: str | None = None,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        description: str = "",
        tags: list[str] | None = None,
        ttl_hours: int | None = None,
        force_full: bool = False,
    ) -> Checkpoint:
        """
        Save a checkpoint.

        Args:
            snapshot: Execution snapshot to save
            session_id: Session identifier
            agent_id: Agent identifier
            checkpoint_type: Type of checkpoint
            description: Human-readable description
            tags: Searchable tags
            ttl_hours: Time-to-live in hours
            force_full: Force full checkpoint (not incremental)

        Returns:
            Created Checkpoint
        """
        checkpoint_id = str(uuid4())
        step_index = self._step_counters.get(session_id, 0)

        # Determine if incremental
        use_incremental = (
            self._config.enable_incremental
            and not force_full
            and session_id in self._last_checkpoint
            and checkpoint_type not in {CheckpointType.FULL, CheckpointType.MILESTONE}
        )

        delta: dict[str, Any] | None = None
        parent_id: str | None = None

        if use_incremental:
            last_checkpoint = self._last_checkpoint[session_id]
            delta = self._delta_calculator.calculate_delta(
                last_checkpoint.snapshot,
                snapshot,
            )
            parent_id = last_checkpoint.checkpoint_id
            checkpoint_type = CheckpointType.INCREMENTAL

        # Calculate expiration
        expires_at = None
        if ttl_hours or self._config.retention_hours:
            hours = ttl_hours or self._config.retention_hours
            expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)

        # Create checkpoint
        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                version=1,
                session_id=session_id,
                agent_id=agent_id,
                step_index=step_index,
                parent_id=parent_id,
                description=description,
                tags=tuple(tags or []),
                expires_at=expires_at,
            ),
            snapshot=snapshot.clone(),
            delta=delta,
        )

        # Serialize and compress
        data = self._serializer.serialize(checkpoint)
        compressed = self._compressor.compress(data)

        # Calculate checksum
        checksum = hashlib.sha256(compressed).hexdigest()

        # Update metadata
        checkpoint.metadata.size_bytes = len(compressed)
        checkpoint.metadata.checksum = checksum

        # Save to storage
        await self._storage.save(checkpoint_id, compressed, checkpoint.metadata)

        # Update tracking
        async with self._lock:
            self._last_checkpoint[session_id] = checkpoint

        self._logger.info(
            f"Saved checkpoint {checkpoint_id[:8]} "
            f"({checkpoint_type.value}, {len(compressed)} bytes)"
        )

        # Cleanup old checkpoints
        await self._cleanup_session_checkpoints(session_id)

        return checkpoint

    async def save_milestone(
        self,
        snapshot: ExecutionSnapshot,
        session_id: str,
        description: str,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Checkpoint:
        """Save a milestone checkpoint (always full)."""
        return await self.save(
            snapshot=snapshot,
            session_id=session_id,
            agent_id=agent_id,
            checkpoint_type=CheckpointType.MILESTONE,
            description=description,
            tags=tags,
            force_full=True,
        )

    async def auto_checkpoint(
        self,
        snapshot: ExecutionSnapshot,
        session_id: str,
        agent_id: str | None = None,
    ) -> Checkpoint | None:
        """
        Automatically checkpoint based on step interval.

        Args:
            snapshot: Current execution snapshot
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            Checkpoint if created, None otherwise
        """
        async with self._lock:
            step_count = self._step_counters.get(session_id, 0) + 1
            self._step_counters[session_id] = step_count

        if step_count % self._config.auto_checkpoint_interval == 0:
            return await self.save(
                snapshot=snapshot,
                session_id=session_id,
                agent_id=agent_id,
                checkpoint_type=CheckpointType.AUTOMATIC,
                description=f"Auto checkpoint at step {step_count}",
            )

        return None

    # -------------------------------------------------------------------------
    # Load Operations
    # -------------------------------------------------------------------------

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Checkpoint or None if not found
        """
        result = await self._storage.load(checkpoint_id)
        if result is None:
            return None

        compressed, metadata = result

        # Verify checksum
        if self._config.verify_on_load:
            checksum = hashlib.sha256(compressed).hexdigest()
            if checksum != metadata.checksum:
                self._logger.error(f"Checksum mismatch for checkpoint {checkpoint_id}")
                metadata.status = CheckpointStatus.CORRUPTED
                raise CheckpointCorruptedError(
                    f"Checkpoint {checkpoint_id} is corrupted",
                    checkpoint_id,
                )

        # Decompress and deserialize
        try:
            data = self._compressor.decompress(compressed)
            checkpoint = self._serializer.deserialize(data)

            # Resolve incremental checkpoint
            if checkpoint.is_incremental:
                checkpoint = await self._resolve_incremental(checkpoint)

            return checkpoint

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise CheckpointCorruptedError(
                f"Failed to deserialize checkpoint: {e}",
                checkpoint_id,
            )

    async def load_latest(
        self,
        session_id: str,
        agent_id: str | None = None,
        valid_only: bool = True,
    ) -> Checkpoint | None:
        """
        Load the latest checkpoint for a session.

        Args:
            session_id: Session identifier
            agent_id: Optional agent filter
            valid_only: Only return valid checkpoints

        Returns:
            Latest Checkpoint or None
        """
        checkpoints = await self._storage.list_checkpoints(
            session_id=session_id,
            agent_id=agent_id,
        )

        for metadata in checkpoints:
            if valid_only and not metadata.is_valid:
                continue

            checkpoint = await self.load(metadata.checkpoint_id)
            if checkpoint:
                return checkpoint

        return None

    async def load_by_step(
        self,
        session_id: str,
        step_index: int,
    ) -> Checkpoint | None:
        """
        Load checkpoint closest to a step index.

        Args:
            session_id: Session identifier
            step_index: Target step index

        Returns:
            Checkpoint closest to step or None
        """
        checkpoints = await self._storage.list_checkpoints(session_id=session_id)

        # Find checkpoint at or before step
        best_match: CheckpointMetadata | None = None
        for metadata in checkpoints:
            if metadata.step_index <= step_index:
                if best_match is None or metadata.step_index > best_match.step_index:
                    best_match = metadata

        if best_match:
            return await self.load(best_match.checkpoint_id)

        return None

    async def _resolve_incremental(self, checkpoint: Checkpoint) -> Checkpoint:
        """Resolve incremental checkpoint to full snapshot."""
        if not checkpoint.is_incremental or checkpoint.delta is None:
            return checkpoint

        # Load parent checkpoint
        parent_id = checkpoint.metadata.parent_id
        if parent_id is None:
            raise CheckpointCorruptedError(
                "Incremental checkpoint missing parent",
                checkpoint.checkpoint_id,
            )

        parent = await self.load(parent_id)
        if parent is None:
            raise CheckpointCorruptedError(
                f"Parent checkpoint {parent_id} not found",
                checkpoint.checkpoint_id,
            )

        # Apply delta
        resolved_snapshot = self._delta_calculator.apply_delta(
            parent.snapshot,
            checkpoint.delta,
        )

        return Checkpoint(
            metadata=checkpoint.metadata,
            snapshot=resolved_snapshot,
            delta=None,
        )

    # -------------------------------------------------------------------------
    # Rollback Operations
    # -------------------------------------------------------------------------

    async def rollback(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
        steps_back: int | None = None,
    ) -> Checkpoint:
        """
        Rollback to a previous checkpoint.

        Args:
            session_id: Session identifier
            checkpoint_id: Specific checkpoint to rollback to
            steps_back: Number of checkpoints to go back

        Returns:
            Checkpoint rolled back to

        Raises:
            RollbackError: If rollback fails
        """
        target_checkpoint: Checkpoint | None = None

        if checkpoint_id:
            # Rollback to specific checkpoint
            target_checkpoint = await self.load(checkpoint_id)
            if target_checkpoint is None:
                raise RollbackError(
                    f"Checkpoint {checkpoint_id} not found",
                    checkpoint_id,
                )

        elif steps_back:
            # Rollback N checkpoints
            checkpoints = await self._storage.list_checkpoints(session_id=session_id)
            valid_checkpoints = [m for m in checkpoints if m.is_valid]

            if steps_back >= len(valid_checkpoints):
                raise RollbackError(
                    f"Cannot rollback {steps_back} steps, only {len(valid_checkpoints)} available"
                )

            target_metadata = valid_checkpoints[steps_back]
            target_checkpoint = await self.load(target_metadata.checkpoint_id)

        else:
            # Rollback to previous checkpoint
            checkpoints = await self._storage.list_checkpoints(session_id=session_id)
            valid_checkpoints = [m for m in checkpoints if m.is_valid]

            if len(valid_checkpoints) < 2:
                raise RollbackError("No previous checkpoint available for rollback")

            target_metadata = valid_checkpoints[1]  # Second newest
            target_checkpoint = await self.load(target_metadata.checkpoint_id)

        if target_checkpoint is None:
            raise RollbackError("Failed to load target checkpoint")

        # Update tracking
        async with self._lock:
            self._last_checkpoint[session_id] = target_checkpoint
            self._step_counters[session_id] = target_checkpoint.metadata.step_index

        self._logger.info(
            f"Rolled back session {session_id} to checkpoint "
            f"{target_checkpoint.checkpoint_id[:8]} (step {target_checkpoint.metadata.step_index})"
        )

        return target_checkpoint

    # -------------------------------------------------------------------------
    # Recovery Operations
    # -------------------------------------------------------------------------

    async def recover(
        self,
        session_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.LATEST_VALID,
        checkpoint_id: str | None = None,
    ) -> Checkpoint:
        """
        Recover session state from checkpoints.

        Args:
            session_id: Session identifier
            strategy: Recovery strategy
            checkpoint_id: Specific checkpoint for SPECIFIC strategy

        Returns:
            Recovered Checkpoint

        Raises:
            RecoveryError: If recovery fails
        """
        self._logger.info(f"Starting recovery for session {session_id} ({strategy.name})")

        if strategy == RecoveryStrategy.SPECIFIC:
            if checkpoint_id is None:
                raise RecoveryError("checkpoint_id required for SPECIFIC strategy")

            checkpoint = await self.load(checkpoint_id)
            if checkpoint is None:
                raise RecoveryError(f"Checkpoint {checkpoint_id} not found")

            return await self._finalize_recovery(session_id, checkpoint)

        checkpoints = await self._storage.list_checkpoints(session_id=session_id)

        if not checkpoints:
            raise RecoveryError(f"No checkpoints found for session {session_id}")

        if strategy == RecoveryStrategy.LATEST:
            checkpoint = await self.load(checkpoints[0].checkpoint_id)
            if checkpoint:
                return await self._finalize_recovery(session_id, checkpoint)
            raise RecoveryError("Failed to load latest checkpoint")

        if strategy == RecoveryStrategy.LATEST_VALID:
            for metadata in checkpoints:
                if metadata.is_valid:
                    checkpoint = await self.load(metadata.checkpoint_id)
                    if checkpoint:
                        return await self._finalize_recovery(session_id, checkpoint)

            raise RecoveryError("No valid checkpoints found")

        if strategy == RecoveryStrategy.BEST_EFFORT:
            # Try each checkpoint until one works
            for metadata in checkpoints:
                try:
                    checkpoint = await self.load(metadata.checkpoint_id)
                    if checkpoint:
                        return await self._finalize_recovery(session_id, checkpoint)
                except CheckpointError:
                    continue

            raise RecoveryError("All recovery attempts failed")

        raise RecoveryError(f"Unknown recovery strategy: {strategy}")

    async def _finalize_recovery(
        self,
        session_id: str,
        checkpoint: Checkpoint,
    ) -> Checkpoint:
        """Finalize recovery by updating tracking state."""
        # Create recovery checkpoint
        recovery_checkpoint = await self.save(
            snapshot=checkpoint.snapshot,
            session_id=session_id,
            agent_id=checkpoint.metadata.agent_id,
            checkpoint_type=CheckpointType.RECOVERY,
            description=f"Recovery from {checkpoint.checkpoint_id[:8]}",
            force_full=True,
        )

        self._logger.info(
            f"Recovery complete for session {session_id}, "
            f"restored to step {checkpoint.metadata.step_index}"
        )

        return recovery_checkpoint

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    async def list_checkpoints(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        checkpoint_type: CheckpointType | None = None,
        valid_only: bool = False,
    ) -> list[CheckpointMetadata]:
        """
        List checkpoints matching criteria.

        Args:
            session_id: Filter by session
            agent_id: Filter by agent
            checkpoint_type: Filter by type
            valid_only: Only return valid checkpoints

        Returns:
            List of checkpoint metadata
        """
        checkpoints = await self._storage.list_checkpoints(
            session_id=session_id,
            agent_id=agent_id,
        )

        if checkpoint_type:
            checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]

        if valid_only:
            checkpoints = [c for c in checkpoints if c.is_valid]

        return checkpoints

    async def get_checkpoint_chain(
        self,
        checkpoint_id: str,
    ) -> list[CheckpointMetadata]:
        """
        Get the chain of checkpoints leading to this one.

        Args:
            checkpoint_id: Starting checkpoint

        Returns:
            List of checkpoint metadata from oldest to newest
        """
        chain: list[CheckpointMetadata] = []
        current_id: str | None = checkpoint_id

        while current_id:
            metadata = await self._storage.get_metadata(current_id)
            if metadata is None:
                break

            chain.append(metadata)
            current_id = metadata.parent_id

        chain.reverse()
        return chain

    async def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        return await self._storage.exists(checkpoint_id)

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted
        """
        # Check if any checkpoints depend on this one
        all_checkpoints = await self._storage.list_checkpoints()
        dependents = [c for c in all_checkpoints if c.parent_id == checkpoint_id]

        if dependents:
            self._logger.warning(
                f"Cannot delete checkpoint {checkpoint_id}: "
                f"{len(dependents)} checkpoints depend on it"
            )
            return False

        deleted = await self._storage.delete(checkpoint_id)

        if deleted:
            self._logger.info(f"Deleted checkpoint {checkpoint_id[:8]}")

        return deleted

    async def delete_session_checkpoints(
        self,
        session_id: str,
        keep_latest: int = 0,
    ) -> int:
        """
        Delete checkpoints for a session.

        Args:
            session_id: Session identifier
            keep_latest: Number of latest checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self._storage.list_checkpoints(session_id=session_id)

        # Keep the latest N
        to_delete = checkpoints[keep_latest:] if keep_latest > 0 else checkpoints

        deleted = 0
        for metadata in to_delete:
            if await self._storage.delete(metadata.checkpoint_id):
                deleted += 1

        # Clear tracking
        if keep_latest == 0:
            async with self._lock:
                self._last_checkpoint.pop(session_id, None)
                self._step_counters.pop(session_id, None)

        return deleted

    async def _cleanup_session_checkpoints(self, session_id: str) -> int:
        """Cleanup old checkpoints for a session."""
        checkpoints = await self._storage.list_checkpoints(session_id=session_id)

        if len(checkpoints) <= self._config.max_checkpoints_per_session:
            return 0

        # Delete oldest checkpoints beyond limit
        to_delete = checkpoints[self._config.max_checkpoints_per_session:]

        deleted = 0
        for metadata in to_delete:
            # Don't delete if other checkpoints depend on it
            dependents = [c for c in checkpoints if c.parent_id == metadata.checkpoint_id]
            if not dependents:
                if await self._storage.delete(metadata.checkpoint_id):
                    deleted += 1

        return deleted

    async def cleanup_expired(self) -> int:
        """
        Cleanup expired checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        all_checkpoints = await self._storage.list_checkpoints()
        deleted = 0

        for metadata in all_checkpoints:
            if metadata.is_expired:
                # Check dependencies
                dependents = [c for c in all_checkpoints if c.parent_id == metadata.checkpoint_id]
                if not dependents:
                    if await self._storage.delete(metadata.checkpoint_id):
                        deleted += 1

        if deleted > 0:
            self._logger.info(f"Cleaned up {deleted} expired checkpoints")

        return deleted

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics."""
        all_checkpoints = await self._storage.list_checkpoints()

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_size = 0

        for metadata in all_checkpoints:
            type_name = metadata.checkpoint_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            status_name = metadata.status.name
            by_status[status_name] = by_status.get(status_name, 0) + 1

            total_size += metadata.size_bytes

        sessions = set(m.session_id for m in all_checkpoints if m.session_id)

        return {
            "total_checkpoints": len(all_checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "by_status": by_status,
            "sessions_with_checkpoints": len(sessions),
            "incremental_enabled": self._config.enable_incremental,
            "auto_interval": self._config.auto_checkpoint_interval,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_checkpoint_manager(
    storage_path: str | Path | None = None,
    auto_interval: int = 10,
    max_per_session: int = 100,
    retention_hours: int = 24,
    enable_incremental: bool = True,
    compression: CompressionLevel = CompressionLevel.BALANCED,
    in_memory: bool = False,
    logger: logging.Logger | None = None,
) -> CheckpointManager:
    """
    Factory function to create configured CheckpointManager.

    Args:
        storage_path: Path for file storage
        auto_interval: Steps between auto checkpoints
        max_per_session: Maximum checkpoints per session
        retention_hours: Hours to retain checkpoints
        enable_incremental: Enable incremental checkpoints
        compression: Compression level
        in_memory: Use in-memory storage
        logger: Optional logger

    Returns:
        Configured CheckpointManager
    """
    config = CheckpointManagerConfig(
        auto_checkpoint_interval=auto_interval,
        max_checkpoints_per_session=max_per_session,
        retention_hours=retention_hours,
        enable_incremental=enable_incremental,
        compression_level=compression,
    )

    storage: CheckpointStorage
    if in_memory or storage_path is None:
        storage = InMemoryCheckpointStorage()
    else:
        storage = FileSystemCheckpointStorage(storage_path, logger)

    compressor = GzipCompressor(compression)

    return CheckpointManager(
        config=config,
        storage=storage,
        compressor=compressor,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CheckpointType",
    "CheckpointStatus",
    "RecoveryStrategy",
    "CompressionLevel",
    # Exceptions
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptedError",
    "RecoveryError",
    "RollbackError",
    # Data Structures
    "CheckpointMetadata",
    "ExecutionSnapshot",
    "Checkpoint",
    # Serialization
    "CheckpointSerializer",
    "JSONSerializer",
    "PickleSerializer",
    # Compression
    "Compressor",
    "GzipCompressor",
    "NoCompressor",
    # Storage
    "CheckpointStorage",
    "InMemoryCheckpointStorage",
    "FileSystemCheckpointStorage",
    # Delta
    "DeltaCalculator",
    # Manager
    "CheckpointManagerConfig",
    "CheckpointManager",
    # Factory
    "create_checkpoint_manager",
]
