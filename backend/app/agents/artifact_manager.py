"""
Artifact Manager - Artifact storage and management for the AI Agent Platform.

This module provides comprehensive artifact management including:
- Storage and retrieval of various artifact types
- Versioning with semantic version support
- Support for PDFs, images, reports, code, logs
- Metadata tracking and search
- Cleanup and retention policies
- Multiple storage backends (filesystem, memory, S3-ready)
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import logging
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class ArtifactType(Enum):
    """Types of artifacts in the system."""

    PDF = "pdf"
    IMAGE = "image"
    REPORT = "report"
    CODE = "code"
    LOG = "log"
    DATA = "data"
    CONFIG = "config"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class ArtifactStatus(Enum):
    """Status of an artifact."""

    ACTIVE = auto()
    ARCHIVED = auto()
    PENDING_DELETION = auto()
    DELETED = auto()
    CORRUPTED = auto()


class CompressionType(Enum):
    """Compression types for artifacts."""

    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class RetentionPolicy(Enum):
    """Retention policies for artifacts."""

    KEEP_FOREVER = "keep_forever"
    KEEP_LATEST = "keep_latest"
    KEEP_N_VERSIONS = "keep_n_versions"
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ArtifactError(Exception):
    """Base exception for artifact errors."""

    def __init__(self, message: str, artifact_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.artifact_id = artifact_id


class ArtifactNotFoundError(ArtifactError):
    """Artifact not found."""

    pass


class ArtifactExistsError(ArtifactError):
    """Artifact already exists."""

    pass


class ArtifactCorruptedError(ArtifactError):
    """Artifact data is corrupted."""

    pass


class StorageError(ArtifactError):
    """Storage operation failed."""

    pass


# =============================================================================
# ARTIFACT METADATA
# =============================================================================


@dataclass
class ArtifactMetadata:
    """
    Metadata for an artifact.

    Attributes:
        artifact_id: Unique artifact identifier
        name: Human-readable name
        artifact_type: Type classification
        version: Semantic version string
        mime_type: MIME type
        size_bytes: Size in bytes
        checksum: Content checksum (SHA-256)
        compression: Compression type used
        created_at: Creation timestamp
        updated_at: Last update timestamp
        created_by: Creator identifier
        tags: Searchable tags
        labels: Key-value labels
        description: Human-readable description
        source: Source of the artifact
        parent_id: Parent artifact ID (for versions)
        session_id: Associated session
        request_id: Associated request
        status: Current status
        retention_until: Retention expiry
        custom_metadata: Additional metadata
    """

    artifact_id: str
    name: str
    artifact_type: ArtifactType
    version: str = "1.0.0"
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0
    checksum: str = ""
    compression: CompressionType = CompressionType.NONE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    tags: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)
    description: str = ""
    source: str = ""
    parent_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    retention_until: datetime | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version_tuple(self) -> tuple[int, ...]:
        """Parse version into tuple for comparison."""
        try:
            return tuple(int(x) for x in self.version.split("."))
        except ValueError:
            return (0, 0, 0)

    @property
    def extension(self) -> str:
        """Get file extension from MIME type."""
        ext = mimetypes.guess_extension(self.mime_type)
        return ext or ""

    @property
    def is_expired(self) -> bool:
        """Check if artifact has expired."""
        if self.retention_until is None:
            return False
        return datetime.now(timezone.utc) > self.retention_until

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "version": self.version,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "compression": self.compression.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": list(self.tags),
            "labels": self.labels,
            "description": self.description,
            "source": self.source,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "status": self.status.name,
            "retention_until": self.retention_until.isoformat() if self.retention_until else None,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Create from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            name=data["name"],
            artifact_type=ArtifactType(data.get("artifact_type", "unknown")),
            version=data.get("version", "1.0.0"),
            mime_type=data.get("mime_type", "application/octet-stream"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            compression=CompressionType(data.get("compression", "none")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
            created_by=data.get("created_by", ""),
            tags=tuple(data.get("tags", [])),
            labels=data.get("labels", {}),
            description=data.get("description", ""),
            source=data.get("source", ""),
            parent_id=data.get("parent_id"),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            status=ArtifactStatus[data.get("status", "ACTIVE")],
            retention_until=datetime.fromisoformat(data["retention_until"]) if data.get("retention_until") else None,
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class Artifact:
    """
    Complete artifact with metadata and content.

    Attributes:
        metadata: Artifact metadata
        content: Raw content bytes
    """

    metadata: ArtifactMetadata
    content: bytes

    @property
    def artifact_id(self) -> str:
        return self.metadata.artifact_id

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def artifact_type(self) -> ArtifactType:
        return self.metadata.artifact_type

    @property
    def size(self) -> int:
        return len(self.content)

    def get_text(self, encoding: str = "utf-8") -> str:
        """Get content as text."""
        return self.content.decode(encoding)

    def get_json(self) -> Any:
        """Get content as JSON."""
        return json.loads(self.content.decode("utf-8"))

    def get_base64(self) -> str:
        """Get content as base64."""
        return base64.b64encode(self.content).decode("ascii")


# =============================================================================
# TYPE DETECTION
# =============================================================================


class TypeDetector:
    """Detects artifact type from content and metadata."""

    # MIME type to artifact type mapping
    MIME_TYPE_MAP: dict[str, ArtifactType] = {
        "application/pdf": ArtifactType.PDF,
        "image/png": ArtifactType.IMAGE,
        "image/jpeg": ArtifactType.IMAGE,
        "image/gif": ArtifactType.IMAGE,
        "image/webp": ArtifactType.IMAGE,
        "image/svg+xml": ArtifactType.IMAGE,
        "text/plain": ArtifactType.TEXT,
        "text/html": ArtifactType.REPORT,
        "text/markdown": ArtifactType.REPORT,
        "application/json": ArtifactType.DATA,
        "application/xml": ArtifactType.DATA,
        "text/csv": ArtifactType.DATA,
        "text/x-python": ArtifactType.CODE,
        "text/javascript": ArtifactType.CODE,
        "application/x-python-code": ArtifactType.CODE,
        "audio/mpeg": ArtifactType.AUDIO,
        "audio/wav": ArtifactType.AUDIO,
        "video/mp4": ArtifactType.VIDEO,
        "video/webm": ArtifactType.VIDEO,
        "application/zip": ArtifactType.ARCHIVE,
        "application/x-tar": ArtifactType.ARCHIVE,
        "application/gzip": ArtifactType.ARCHIVE,
    }

    # Extension to artifact type mapping
    EXTENSION_MAP: dict[str, ArtifactType] = {
        ".pdf": ArtifactType.PDF,
        ".png": ArtifactType.IMAGE,
        ".jpg": ArtifactType.IMAGE,
        ".jpeg": ArtifactType.IMAGE,
        ".gif": ArtifactType.IMAGE,
        ".webp": ArtifactType.IMAGE,
        ".svg": ArtifactType.IMAGE,
        ".py": ArtifactType.CODE,
        ".js": ArtifactType.CODE,
        ".ts": ArtifactType.CODE,
        ".java": ArtifactType.CODE,
        ".go": ArtifactType.CODE,
        ".rs": ArtifactType.CODE,
        ".cpp": ArtifactType.CODE,
        ".c": ArtifactType.CODE,
        ".h": ArtifactType.CODE,
        ".rb": ArtifactType.CODE,
        ".php": ArtifactType.CODE,
        ".html": ArtifactType.REPORT,
        ".md": ArtifactType.REPORT,
        ".rst": ArtifactType.REPORT,
        ".json": ArtifactType.DATA,
        ".xml": ArtifactType.DATA,
        ".csv": ArtifactType.DATA,
        ".yaml": ArtifactType.CONFIG,
        ".yml": ArtifactType.CONFIG,
        ".toml": ArtifactType.CONFIG,
        ".ini": ArtifactType.CONFIG,
        ".log": ArtifactType.LOG,
        ".txt": ArtifactType.TEXT,
        ".zip": ArtifactType.ARCHIVE,
        ".tar": ArtifactType.ARCHIVE,
        ".gz": ArtifactType.ARCHIVE,
        ".mp3": ArtifactType.AUDIO,
        ".wav": ArtifactType.AUDIO,
        ".mp4": ArtifactType.VIDEO,
        ".webm": ArtifactType.VIDEO,
        ".pt": ArtifactType.MODEL,
        ".pth": ArtifactType.MODEL,
        ".onnx": ArtifactType.MODEL,
        ".ckpt": ArtifactType.CHECKPOINT,
    }

    # Magic bytes for type detection
    MAGIC_BYTES: dict[bytes, ArtifactType] = {
        b"%PDF": ArtifactType.PDF,
        b"\x89PNG": ArtifactType.IMAGE,
        b"\xff\xd8\xff": ArtifactType.IMAGE,  # JPEG
        b"GIF8": ArtifactType.IMAGE,
        b"RIFF": ArtifactType.IMAGE,  # WebP (needs further check)
        b"PK\x03\x04": ArtifactType.ARCHIVE,  # ZIP
        b"\x1f\x8b": ArtifactType.ARCHIVE,  # GZIP
    }

    @classmethod
    def detect_type(
        cls,
        content: bytes | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> ArtifactType:
        """
        Detect artifact type from available information.

        Args:
            content: Raw content bytes
            filename: Original filename
            mime_type: MIME type if known

        Returns:
            Detected ArtifactType
        """
        # Try MIME type first
        if mime_type and mime_type in cls.MIME_TYPE_MAP:
            return cls.MIME_TYPE_MAP[mime_type]

        # Try extension
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in cls.EXTENSION_MAP:
                return cls.EXTENSION_MAP[ext]

        # Try magic bytes
        if content and len(content) >= 4:
            for magic, artifact_type in cls.MAGIC_BYTES.items():
                if content.startswith(magic):
                    return artifact_type

        return ArtifactType.UNKNOWN

    @classmethod
    def detect_mime_type(
        cls,
        content: bytes | None = None,
        filename: str | None = None,
    ) -> str:
        """Detect MIME type from content or filename."""
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                return mime_type

        if content:
            # Check magic bytes
            if content.startswith(b"%PDF"):
                return "application/pdf"
            elif content.startswith(b"\x89PNG"):
                return "image/png"
            elif content.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            elif content.startswith(b"GIF8"):
                return "image/gif"
            elif content.startswith(b"PK\x03\x04"):
                return "application/zip"
            elif content.startswith(b"\x1f\x8b"):
                return "application/gzip"

        return "application/octet-stream"


# =============================================================================
# STORAGE BACKENDS
# =============================================================================


class StorageBackend(ABC):
    """Abstract base for storage backends."""

    @abstractmethod
    async def store(self, artifact_id: str, content: bytes) -> None:
        """Store artifact content."""
        ...

    @abstractmethod
    async def retrieve(self, artifact_id: str) -> bytes | None:
        """Retrieve artifact content."""
        ...

    @abstractmethod
    async def delete(self, artifact_id: str) -> bool:
        """Delete artifact content."""
        ...

    @abstractmethod
    async def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        ...

    @abstractmethod
    async def list_ids(self) -> list[str]:
        """List all artifact IDs."""
        ...

    async def get_size(self, artifact_id: str) -> int | None:
        """Get artifact size in bytes."""
        content = await self.retrieve(artifact_id)
        return len(content) if content else None


class InMemoryStorageBackend(StorageBackend):
    """
    In-memory storage backend.

    Suitable for testing and small-scale usage.
    """

    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024) -> None:
        """
        Initialize in-memory storage.

        Args:
            max_size_bytes: Maximum total storage size
        """
        self._storage: dict[str, bytes] = {}
        self._max_size = max_size_bytes
        self._current_size = 0
        self._lock = asyncio.Lock()

    async def store(self, artifact_id: str, content: bytes) -> None:
        async with self._lock:
            # Check size limit
            new_size = self._current_size + len(content)
            if artifact_id in self._storage:
                new_size -= len(self._storage[artifact_id])

            if new_size > self._max_size:
                raise StorageError(
                    f"Storage limit exceeded: {new_size} > {self._max_size}",
                    artifact_id=artifact_id,
                )

            if artifact_id in self._storage:
                self._current_size -= len(self._storage[artifact_id])

            self._storage[artifact_id] = content
            self._current_size += len(content)

    async def retrieve(self, artifact_id: str) -> bytes | None:
        return self._storage.get(artifact_id)

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock:
            if artifact_id in self._storage:
                self._current_size -= len(self._storage[artifact_id])
                del self._storage[artifact_id]
                return True
            return False

    async def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._storage

    async def list_ids(self) -> list[str]:
        return list(self._storage.keys())

    @property
    def current_size(self) -> int:
        return self._current_size


class FileSystemStorageBackend(StorageBackend):
    """
    Filesystem-based storage backend.

    Stores artifacts as files with optional compression.
    """

    def __init__(
        self,
        base_path: str | Path,
        compression: CompressionType = CompressionType.NONE,
        create_dirs: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize filesystem storage.

        Args:
            base_path: Base directory for storage
            compression: Compression to apply
            create_dirs: Create directories if missing
            logger: Optional logger
        """
        self._base_path = Path(base_path)
        self._compression = compression
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

        if create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, artifact_id: str) -> Path:
        """Get file path for artifact."""
        # Use first 2 chars as subdirectory for better distribution
        subdir = artifact_id[:2] if len(artifact_id) >= 2 else "00"
        return self._base_path / subdir / artifact_id

    def _compress(self, content: bytes) -> bytes:
        """Compress content."""
        if self._compression == CompressionType.GZIP:
            return gzip.compress(content)
        return content

    def _decompress(self, content: bytes) -> bytes:
        """Decompress content."""
        if self._compression == CompressionType.GZIP:
            try:
                return gzip.decompress(content)
            except gzip.BadGzipFile:
                return content
        return content

    async def store(self, artifact_id: str, content: bytes) -> None:
        path = self._get_path(artifact_id)

        async with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                compressed = self._compress(content)

                # Write atomically
                temp_path = path.with_suffix(".tmp")
                temp_path.write_bytes(compressed)
                temp_path.rename(path)

            except OSError as e:
                raise StorageError(
                    f"Failed to store artifact: {e}",
                    artifact_id=artifact_id,
                )

    async def retrieve(self, artifact_id: str) -> bytes | None:
        path = self._get_path(artifact_id)

        if not path.exists():
            return None

        try:
            content = path.read_bytes()
            return self._decompress(content)
        except OSError as e:
            self._logger.error(f"Failed to retrieve {artifact_id}: {e}")
            return None

    async def delete(self, artifact_id: str) -> bool:
        path = self._get_path(artifact_id)

        async with self._lock:
            if path.exists():
                try:
                    path.unlink()
                    # Clean up empty parent directory
                    if not any(path.parent.iterdir()):
                        path.parent.rmdir()
                    return True
                except OSError as e:
                    self._logger.error(f"Failed to delete {artifact_id}: {e}")
                    return False
            return False

    async def exists(self, artifact_id: str) -> bool:
        return self._get_path(artifact_id).exists()

    async def list_ids(self) -> list[str]:
        ids: list[str] = []

        if not self._base_path.exists():
            return ids

        for subdir in self._base_path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file() and not file_path.suffix == ".tmp":
                        ids.append(file_path.name)

        return ids

    async def get_size(self, artifact_id: str) -> int | None:
        path = self._get_path(artifact_id)
        if path.exists():
            return path.stat().st_size
        return None

    def get_total_size(self) -> int:
        """Get total storage size."""
        total = 0
        if self._base_path.exists():
            for path in self._base_path.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        return total


# =============================================================================
# METADATA STORE
# =============================================================================


class MetadataStore(ABC):
    """Abstract base for metadata storage."""

    @abstractmethod
    async def save(self, metadata: ArtifactMetadata) -> None:
        """Save artifact metadata."""
        ...

    @abstractmethod
    async def load(self, artifact_id: str) -> ArtifactMetadata | None:
        """Load artifact metadata."""
        ...

    @abstractmethod
    async def delete(self, artifact_id: str) -> bool:
        """Delete artifact metadata."""
        ...

    @abstractmethod
    async def search(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        session_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        status: ArtifactStatus | None = None,
    ) -> list[ArtifactMetadata]:
        """Search for artifacts matching criteria."""
        ...

    @abstractmethod
    async def list_all(self) -> list[ArtifactMetadata]:
        """List all artifact metadata."""
        ...


class InMemoryMetadataStore(MetadataStore):
    """In-memory metadata store."""

    def __init__(self) -> None:
        self._metadata: dict[str, ArtifactMetadata] = {}
        self._lock = asyncio.Lock()

    async def save(self, metadata: ArtifactMetadata) -> None:
        async with self._lock:
            self._metadata[metadata.artifact_id] = metadata

    async def load(self, artifact_id: str) -> ArtifactMetadata | None:
        return self._metadata.get(artifact_id)

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock:
            if artifact_id in self._metadata:
                del self._metadata[artifact_id]
                return True
            return False

    async def search(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        session_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        status: ArtifactStatus | None = None,
    ) -> list[ArtifactMetadata]:
        results: list[ArtifactMetadata] = []

        for metadata in self._metadata.values():
            # Filter by type
            if artifact_type and metadata.artifact_type != artifact_type:
                continue

            # Filter by tags
            if tags and not all(t in metadata.tags for t in tags):
                continue

            # Filter by labels
            if labels:
                match = all(
                    metadata.labels.get(k) == v
                    for k, v in labels.items()
                )
                if not match:
                    continue

            # Filter by session
            if session_id and metadata.session_id != session_id:
                continue

            # Filter by creation time
            if created_after and metadata.created_at < created_after:
                continue
            if created_before and metadata.created_at > created_before:
                continue

            # Filter by status
            if status and metadata.status != status:
                continue

            results.append(metadata)

        return results

    async def list_all(self) -> list[ArtifactMetadata]:
        return list(self._metadata.values())


class FileMetadataStore(MetadataStore):
    """File-based metadata store using JSON."""

    def __init__(
        self,
        base_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base_path = Path(base_path)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, artifact_id: str) -> Path:
        return self._base_path / f"{artifact_id}.json"

    async def save(self, metadata: ArtifactMetadata) -> None:
        path = self._get_path(metadata.artifact_id)

        async with self._lock:
            try:
                data = metadata.to_dict()
                temp_path = path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(data, indent=2))
                temp_path.rename(path)
            except OSError as e:
                self._logger.error(f"Failed to save metadata: {e}")
                raise StorageError(f"Failed to save metadata: {e}")

    async def load(self, artifact_id: str) -> ArtifactMetadata | None:
        path = self._get_path(artifact_id)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return ArtifactMetadata.from_dict(data)
        except (OSError, json.JSONDecodeError) as e:
            self._logger.error(f"Failed to load metadata: {e}")
            return None

    async def delete(self, artifact_id: str) -> bool:
        path = self._get_path(artifact_id)

        async with self._lock:
            if path.exists():
                try:
                    path.unlink()
                    return True
                except OSError:
                    return False
            return False

    async def search(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        session_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        status: ArtifactStatus | None = None,
    ) -> list[ArtifactMetadata]:
        all_metadata = await self.list_all()
        results: list[ArtifactMetadata] = []

        for metadata in all_metadata:
            if artifact_type and metadata.artifact_type != artifact_type:
                continue
            if tags and not all(t in metadata.tags for t in tags):
                continue
            if labels and not all(metadata.labels.get(k) == v for k, v in labels.items()):
                continue
            if session_id and metadata.session_id != session_id:
                continue
            if created_after and metadata.created_at < created_after:
                continue
            if created_before and metadata.created_at > created_before:
                continue
            if status and metadata.status != status:
                continue

            results.append(metadata)

        return results

    async def list_all(self) -> list[ArtifactMetadata]:
        results: list[ArtifactMetadata] = []

        for path in self._base_path.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                results.append(ArtifactMetadata.from_dict(data))
            except (OSError, json.JSONDecodeError) as e:
                self._logger.warning(f"Failed to load {path}: {e}")

        return results


# =============================================================================
# VERSION MANAGER
# =============================================================================


class VersionManager:
    """
    Manages artifact versioning.

    Supports semantic versioning and version history.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._logger = logger or logging.getLogger(__name__)

    async def get_versions(self, name: str) -> list[ArtifactMetadata]:
        """Get all versions of an artifact by name."""
        all_metadata = await self._metadata_store.list_all()

        versions = [m for m in all_metadata if m.name == name]
        versions.sort(key=lambda m: m.version_tuple, reverse=True)

        return versions

    async def get_latest_version(self, name: str) -> ArtifactMetadata | None:
        """Get latest version of an artifact."""
        versions = await self.get_versions(name)
        return versions[0] if versions else None

    async def get_version(self, name: str, version: str) -> ArtifactMetadata | None:
        """Get specific version of an artifact."""
        versions = await self.get_versions(name)

        for metadata in versions:
            if metadata.version == version:
                return metadata

        return None

    def increment_version(
        self,
        current_version: str,
        bump: str = "patch",
    ) -> str:
        """
        Increment version number.

        Args:
            current_version: Current version string
            bump: Version component to bump (major, minor, patch)

        Returns:
            New version string
        """
        try:
            parts = [int(x) for x in current_version.split(".")]
            while len(parts) < 3:
                parts.append(0)

            if bump == "major":
                parts[0] += 1
                parts[1] = 0
                parts[2] = 0
            elif bump == "minor":
                parts[1] += 1
                parts[2] = 0
            else:  # patch
                parts[2] += 1

            return ".".join(str(p) for p in parts)

        except ValueError:
            return "1.0.0"

    async def create_new_version(
        self,
        name: str,
        bump: str = "patch",
    ) -> str:
        """
        Create new version number for artifact.

        Args:
            name: Artifact name
            bump: Version component to bump

        Returns:
            New version string
        """
        latest = await self.get_latest_version(name)

        if latest is None:
            return "1.0.0"

        return self.increment_version(latest.version, bump)


# =============================================================================
# CLEANUP MANAGER
# =============================================================================


@dataclass
class CleanupConfig:
    """
    Configuration for artifact cleanup.

    Attributes:
        retention_policy: Default retention policy
        max_versions: Maximum versions to keep
        max_age_days: Maximum age in days
        max_total_size_bytes: Maximum total storage size
        cleanup_interval_hours: Hours between cleanup runs
    """

    retention_policy: RetentionPolicy = RetentionPolicy.KEEP_N_VERSIONS
    max_versions: int = 10
    max_age_days: int = 90
    max_total_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
    cleanup_interval_hours: int = 24


class CleanupManager:
    """
    Manages artifact cleanup and retention.

    Enforces retention policies and storage limits.
    """

    def __init__(
        self,
        storage: StorageBackend,
        metadata_store: MetadataStore,
        config: CleanupConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._storage = storage
        self._metadata_store = metadata_store
        self._config = config or CleanupConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._last_cleanup: datetime | None = None

    async def cleanup(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Run cleanup based on retention policies.

        Args:
            dry_run: If True, don't actually delete

        Returns:
            Cleanup statistics
        """
        stats = {
            "checked": 0,
            "deleted": 0,
            "bytes_freed": 0,
            "errors": 0,
            "candidates": [],
        }

        all_metadata = await self._metadata_store.list_all()
        stats["checked"] = len(all_metadata)

        # Group by name for version-based cleanup
        by_name: dict[str, list[ArtifactMetadata]] = {}
        for metadata in all_metadata:
            if metadata.name not in by_name:
                by_name[metadata.name] = []
            by_name[metadata.name].append(metadata)

        # Sort each group by version
        for name in by_name:
            by_name[name].sort(key=lambda m: m.version_tuple, reverse=True)

        candidates: list[ArtifactMetadata] = []

        # Apply retention policies
        if self._config.retention_policy == RetentionPolicy.KEEP_N_VERSIONS:
            for name, versions in by_name.items():
                if len(versions) > self._config.max_versions:
                    candidates.extend(versions[self._config.max_versions:])

        elif self._config.retention_policy == RetentionPolicy.TIME_BASED:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._config.max_age_days)
            for metadata in all_metadata:
                if metadata.created_at < cutoff:
                    candidates.append(metadata)

        elif self._config.retention_policy == RetentionPolicy.KEEP_LATEST:
            for name, versions in by_name.items():
                if len(versions) > 1:
                    candidates.extend(versions[1:])

        # Check expired artifacts
        for metadata in all_metadata:
            if metadata.is_expired and metadata not in candidates:
                candidates.append(metadata)

        # Check pending deletion
        for metadata in all_metadata:
            if metadata.status == ArtifactStatus.PENDING_DELETION and metadata not in candidates:
                candidates.append(metadata)

        stats["candidates"] = [m.artifact_id for m in candidates]

        # Delete candidates
        if not dry_run:
            for metadata in candidates:
                try:
                    size = await self._storage.get_size(metadata.artifact_id)
                    deleted = await self._storage.delete(metadata.artifact_id)

                    if deleted:
                        await self._metadata_store.delete(metadata.artifact_id)
                        stats["deleted"] += 1  # type: ignore[operator]
                        stats["bytes_freed"] += size or 0  # type: ignore[operator]
                except Exception as e:
                    self._logger.error(f"Cleanup error for {metadata.artifact_id}: {e}")
                    stats["errors"] += 1  # type: ignore[operator]

        self._last_cleanup = datetime.now(timezone.utc)
        return stats

    async def cleanup_by_size(
        self,
        target_size_bytes: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Cleanup to reach target storage size.

        Deletes oldest artifacts first.
        """
        target = target_size_bytes or self._config.max_total_size_bytes

        stats = {
            "initial_size": 0,
            "final_size": 0,
            "deleted": 0,
            "bytes_freed": 0,
        }

        # Get all metadata sorted by creation time (oldest first)
        all_metadata = await self._metadata_store.list_all()
        all_metadata.sort(key=lambda m: m.created_at)

        # Calculate current size
        current_size = sum(m.size_bytes for m in all_metadata)
        stats["initial_size"] = current_size

        if current_size <= target:
            stats["final_size"] = current_size
            return stats

        # Delete oldest until under target
        for metadata in all_metadata:
            if current_size <= target:
                break

            if not dry_run:
                try:
                    await self._storage.delete(metadata.artifact_id)
                    await self._metadata_store.delete(metadata.artifact_id)
                except Exception as e:
                    self._logger.error(f"Cleanup error: {e}")
                    continue

            current_size -= metadata.size_bytes
            stats["deleted"] += 1
            stats["bytes_freed"] += metadata.size_bytes

        stats["final_size"] = current_size
        return stats

    async def mark_for_deletion(self, artifact_id: str) -> bool:
        """Mark artifact for deletion in next cleanup."""
        metadata = await self._metadata_store.load(artifact_id)
        if metadata:
            metadata.status = ArtifactStatus.PENDING_DELETION
            metadata.updated_at = datetime.now(timezone.utc)
            await self._metadata_store.save(metadata)
            return True
        return False

    def should_run_cleanup(self) -> bool:
        """Check if cleanup should run based on interval."""
        if self._last_cleanup is None:
            return True

        elapsed = datetime.now(timezone.utc) - self._last_cleanup
        return elapsed.total_seconds() >= self._config.cleanup_interval_hours * 3600


# =============================================================================
# ARTIFACT MANAGER
# =============================================================================


@dataclass
class ArtifactManagerConfig:
    """
    Configuration for ArtifactManager.

    Attributes:
        storage_path: Path for file storage
        use_compression: Enable compression
        compression_type: Type of compression
        auto_detect_type: Auto-detect artifact types
        enable_versioning: Enable versioning
        cleanup_config: Cleanup configuration
    """

    storage_path: str | Path = "./artifacts"
    use_compression: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    auto_detect_type: bool = True
    enable_versioning: bool = True
    cleanup_config: CleanupConfig = field(default_factory=CleanupConfig)


class ArtifactManager:
    """
    Central artifact management system.

    Provides storage, retrieval, versioning, and cleanup
    of artifacts across the agent platform.
    """

    def __init__(
        self,
        config: ArtifactManagerConfig | None = None,
        storage: StorageBackend | None = None,
        metadata_store: MetadataStore | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize artifact manager.

        Args:
            config: Manager configuration
            storage: Custom storage backend
            metadata_store: Custom metadata store
            logger: Optional logger
        """
        self._config = config or ArtifactManagerConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Initialize storage
        if storage:
            self._storage = storage
        else:
            self._storage = FileSystemStorageBackend(
                base_path=Path(self._config.storage_path) / "data",
                compression=self._config.compression_type if self._config.use_compression else CompressionType.NONE,
                logger=self._logger,
            )

        # Initialize metadata store
        if metadata_store:
            self._metadata_store = metadata_store
        else:
            self._metadata_store = FileMetadataStore(
                base_path=Path(self._config.storage_path) / "metadata",
                logger=self._logger,
            )

        # Initialize components
        self._version_manager = VersionManager(
            self._metadata_store,
            logger=self._logger,
        )
        self._cleanup_manager = CleanupManager(
            self._storage,
            self._metadata_store,
            config=self._config.cleanup_config,
            logger=self._logger,
        )
        self._type_detector = TypeDetector()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Store Operations
    # -------------------------------------------------------------------------

    async def store(
        self,
        content: bytes | str,
        name: str,
        artifact_type: ArtifactType | None = None,
        version: str | None = None,
        mime_type: str | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        description: str = "",
        source: str = "",
        session_id: str | None = None,
        request_id: str | None = None,
        created_by: str = "",
        retention_days: int | None = None,
        custom_metadata: dict[str, Any] | None = None,
        version_bump: str = "patch",
    ) -> Artifact:
        """
        Store a new artifact.

        Args:
            content: Artifact content (bytes or string)
            name: Artifact name
            artifact_type: Type (auto-detected if None)
            version: Version (auto-incremented if None)
            mime_type: MIME type (auto-detected if None)
            tags: Searchable tags
            labels: Key-value labels
            description: Human-readable description
            source: Source of artifact
            session_id: Associated session
            request_id: Associated request
            created_by: Creator identifier
            retention_days: Days to retain (None = forever)
            custom_metadata: Additional metadata
            version_bump: Version bump type if auto-versioning

        Returns:
            Stored Artifact
        """
        # Convert string to bytes
        if isinstance(content, str):
            content = content.encode("utf-8")

        # Generate artifact ID
        artifact_id = str(uuid4())

        # Auto-detect type
        if artifact_type is None and self._config.auto_detect_type:
            artifact_type = self._type_detector.detect_type(
                content=content,
                filename=name,
                mime_type=mime_type,
            )
        artifact_type = artifact_type or ArtifactType.UNKNOWN

        # Auto-detect MIME type
        if mime_type is None:
            mime_type = self._type_detector.detect_mime_type(
                content=content,
                filename=name,
            )

        # Auto-version
        if version is None and self._config.enable_versioning:
            version = await self._version_manager.create_new_version(name, version_bump)
        version = version or "1.0.0"

        # Get parent ID for versioning
        parent_id = None
        if self._config.enable_versioning:
            latest = await self._version_manager.get_latest_version(name)
            if latest:
                parent_id = latest.artifact_id

        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()

        # Calculate retention
        retention_until = None
        if retention_days:
            retention_until = datetime.now(timezone.utc) + timedelta(days=retention_days)

        # Create metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            version=version,
            mime_type=mime_type,
            size_bytes=len(content),
            checksum=checksum,
            compression=self._config.compression_type if self._config.use_compression else CompressionType.NONE,
            created_by=created_by,
            tags=tuple(tags or []),
            labels=labels or {},
            description=description,
            source=source,
            parent_id=parent_id,
            session_id=session_id,
            request_id=request_id,
            retention_until=retention_until,
            custom_metadata=custom_metadata or {},
        )

        # Store content and metadata
        async with self._lock:
            await self._storage.store(artifact_id, content)
            await self._metadata_store.save(metadata)

        self._logger.info(f"Stored artifact: {name} v{version} ({artifact_id})")

        return Artifact(metadata=metadata, content=content)

    async def store_pdf(
        self,
        content: bytes,
        name: str,
        **kwargs: Any,
    ) -> Artifact:
        """Store a PDF artifact."""
        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.PDF,
            mime_type="application/pdf",
            **kwargs,
        )

    async def store_image(
        self,
        content: bytes,
        name: str,
        mime_type: str = "image/png",
        **kwargs: Any,
    ) -> Artifact:
        """Store an image artifact."""
        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.IMAGE,
            mime_type=mime_type,
            **kwargs,
        )

    async def store_report(
        self,
        content: str,
        name: str,
        mime_type: str = "text/markdown",
        **kwargs: Any,
    ) -> Artifact:
        """Store a report artifact."""
        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.REPORT,
            mime_type=mime_type,
            **kwargs,
        )

    async def store_code(
        self,
        content: str,
        name: str,
        language: str = "python",
        **kwargs: Any,
    ) -> Artifact:
        """Store a code artifact."""
        mime_map = {
            "python": "text/x-python",
            "javascript": "text/javascript",
            "typescript": "text/typescript",
            "java": "text/x-java",
            "go": "text/x-go",
            "rust": "text/x-rust",
        }
        mime_type = mime_map.get(language, "text/plain")

        labels = kwargs.pop("labels", {})
        labels["language"] = language

        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.CODE,
            mime_type=mime_type,
            labels=labels,
            **kwargs,
        )

    async def store_log(
        self,
        content: str,
        name: str,
        **kwargs: Any,
    ) -> Artifact:
        """Store a log artifact."""
        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.LOG,
            mime_type="text/plain",
            **kwargs,
        )

    async def store_data(
        self,
        data: Any,
        name: str,
        **kwargs: Any,
    ) -> Artifact:
        """Store a data artifact (JSON serializable)."""
        content = json.dumps(data, indent=2, default=str)
        return await self.store(
            content=content,
            name=name,
            artifact_type=ArtifactType.DATA,
            mime_type="application/json",
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Retrieve Operations
    # -------------------------------------------------------------------------

    async def retrieve(self, artifact_id: str) -> Artifact | None:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Artifact or None if not found
        """
        metadata = await self._metadata_store.load(artifact_id)
        if metadata is None:
            return None

        content = await self._storage.retrieve(artifact_id)
        if content is None:
            return None

        # Verify checksum
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != metadata.checksum:
            self._logger.warning(f"Checksum mismatch for {artifact_id}")
            metadata.status = ArtifactStatus.CORRUPTED
            await self._metadata_store.save(metadata)

        return Artifact(metadata=metadata, content=content)

    async def retrieve_by_name(
        self,
        name: str,
        version: str | None = None,
    ) -> Artifact | None:
        """
        Retrieve artifact by name and optional version.

        Args:
            name: Artifact name
            version: Specific version (None = latest)

        Returns:
            Artifact or None if not found
        """
        if version:
            metadata = await self._version_manager.get_version(name, version)
        else:
            metadata = await self._version_manager.get_latest_version(name)

        if metadata is None:
            return None

        return await self.retrieve(metadata.artifact_id)

    async def retrieve_content(self, artifact_id: str) -> bytes | None:
        """Retrieve only artifact content."""
        return await self._storage.retrieve(artifact_id)

    async def retrieve_metadata(self, artifact_id: str) -> ArtifactMetadata | None:
        """Retrieve only artifact metadata."""
        return await self._metadata_store.load(artifact_id)

    async def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        return await self._storage.exists(artifact_id)

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    async def search(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        session_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        status: ArtifactStatus | None = None,
    ) -> list[ArtifactMetadata]:
        """
        Search for artifacts matching criteria.

        Args:
            artifact_type: Filter by type
            tags: Filter by tags (all must match)
            labels: Filter by labels (all must match)
            session_id: Filter by session
            created_after: Filter by creation time
            created_before: Filter by creation time
            status: Filter by status

        Returns:
            List of matching metadata
        """
        return await self._metadata_store.search(
            artifact_type=artifact_type,
            tags=tags,
            labels=labels,
            session_id=session_id,
            created_after=created_after,
            created_before=created_before,
            status=status,
        )

    async def list_all(self) -> list[ArtifactMetadata]:
        """List all artifact metadata."""
        return await self._metadata_store.list_all()

    async def list_by_type(self, artifact_type: ArtifactType) -> list[ArtifactMetadata]:
        """List artifacts by type."""
        return await self.search(artifact_type=artifact_type)

    async def list_by_session(self, session_id: str) -> list[ArtifactMetadata]:
        """List artifacts by session."""
        return await self.search(session_id=session_id)

    # -------------------------------------------------------------------------
    # Version Operations
    # -------------------------------------------------------------------------

    async def get_versions(self, name: str) -> list[ArtifactMetadata]:
        """Get all versions of an artifact."""
        return await self._version_manager.get_versions(name)

    async def get_latest_version(self, name: str) -> ArtifactMetadata | None:
        """Get latest version metadata."""
        return await self._version_manager.get_latest_version(name)

    async def get_version_history(self, name: str) -> list[dict[str, Any]]:
        """Get version history with summary info."""
        versions = await self._version_manager.get_versions(name)

        return [
            {
                "artifact_id": v.artifact_id,
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "size_bytes": v.size_bytes,
                "checksum": v.checksum[:12],
            }
            for v in versions
        ]

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    async def delete(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Artifact identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            storage_deleted = await self._storage.delete(artifact_id)
            metadata_deleted = await self._metadata_store.delete(artifact_id)

            if storage_deleted or metadata_deleted:
                self._logger.info(f"Deleted artifact: {artifact_id}")
                return True

            return False

    async def delete_by_name(
        self,
        name: str,
        version: str | None = None,
        all_versions: bool = False,
    ) -> int:
        """
        Delete artifact(s) by name.

        Args:
            name: Artifact name
            version: Specific version to delete
            all_versions: Delete all versions

        Returns:
            Number of artifacts deleted
        """
        deleted = 0

        if all_versions:
            versions = await self._version_manager.get_versions(name)
            for metadata in versions:
                if await self.delete(metadata.artifact_id):
                    deleted += 1
        elif version:
            metadata = await self._version_manager.get_version(name, version)
            if metadata and await self.delete(metadata.artifact_id):
                deleted += 1
        else:
            # Delete latest only
            metadata = await self._version_manager.get_latest_version(name)
            if metadata and await self.delete(metadata.artifact_id):
                deleted += 1

        return deleted

    async def mark_for_deletion(self, artifact_id: str) -> bool:
        """Mark artifact for deletion in next cleanup."""
        return await self._cleanup_manager.mark_for_deletion(artifact_id)

    # -------------------------------------------------------------------------
    # Cleanup Operations
    # -------------------------------------------------------------------------

    async def cleanup(self, dry_run: bool = False) -> dict[str, Any]:
        """Run cleanup based on retention policies."""
        return await self._cleanup_manager.cleanup(dry_run=dry_run)

    async def cleanup_by_size(
        self,
        target_size_bytes: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Cleanup to reach target storage size."""
        return await self._cleanup_manager.cleanup_by_size(
            target_size_bytes=target_size_bytes,
            dry_run=dry_run,
        )

    async def cleanup_session(self, session_id: str) -> int:
        """Delete all artifacts for a session."""
        artifacts = await self.list_by_session(session_id)
        deleted = 0

        for metadata in artifacts:
            if await self.delete(metadata.artifact_id):
                deleted += 1

        return deleted

    async def cleanup_expired(self) -> int:
        """Delete all expired artifacts."""
        all_metadata = await self._metadata_store.list_all()
        deleted = 0

        for metadata in all_metadata:
            if metadata.is_expired:
                if await self.delete(metadata.artifact_id):
                    deleted += 1

        return deleted

    # -------------------------------------------------------------------------
    # Utility Operations
    # -------------------------------------------------------------------------

    async def update_metadata(
        self,
        artifact_id: str,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        description: str | None = None,
        retention_days: int | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata | None:
        """
        Update artifact metadata.

        Args:
            artifact_id: Artifact identifier
            tags: New tags (replaces existing)
            labels: Labels to add/update
            description: New description
            retention_days: New retention period
            custom_metadata: Metadata to add/update

        Returns:
            Updated metadata or None if not found
        """
        metadata = await self._metadata_store.load(artifact_id)
        if metadata is None:
            return None

        if tags is not None:
            metadata.tags = tuple(tags)

        if labels is not None:
            metadata.labels.update(labels)

        if description is not None:
            metadata.description = description

        if retention_days is not None:
            metadata.retention_until = datetime.now(timezone.utc) + timedelta(days=retention_days)

        if custom_metadata is not None:
            metadata.custom_metadata.update(custom_metadata)

        metadata.updated_at = datetime.now(timezone.utc)

        await self._metadata_store.save(metadata)
        return metadata

    async def copy(
        self,
        artifact_id: str,
        new_name: str | None = None,
        new_version: str | None = None,
    ) -> Artifact | None:
        """
        Copy an artifact.

        Args:
            artifact_id: Source artifact ID
            new_name: New name (None = same name)
            new_version: New version (None = auto-increment)

        Returns:
            New Artifact or None if source not found
        """
        source = await self.retrieve(artifact_id)
        if source is None:
            return None

        name = new_name or source.metadata.name

        return await self.store(
            content=source.content,
            name=name,
            artifact_type=source.metadata.artifact_type,
            version=new_version,
            mime_type=source.metadata.mime_type,
            tags=list(source.metadata.tags),
            labels=dict(source.metadata.labels),
            description=source.metadata.description,
            source=f"copy:{artifact_id}",
            custom_metadata=dict(source.metadata.custom_metadata),
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        all_metadata = await self._metadata_store.list_all()

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_size = 0

        for metadata in all_metadata:
            type_name = metadata.artifact_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            status_name = metadata.status.name
            by_status[status_name] = by_status.get(status_name, 0) + 1

            total_size += metadata.size_bytes

        return {
            "total_artifacts": len(all_metadata),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "by_status": by_status,
            "storage_path": str(self._config.storage_path),
            "compression_enabled": self._config.use_compression,
            "versioning_enabled": self._config.enable_versioning,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_artifact_manager(
    storage_path: str | Path = "./artifacts",
    use_compression: bool = True,
    enable_versioning: bool = True,
    max_versions: int = 10,
    retention_days: int = 90,
    in_memory: bool = False,
    logger: logging.Logger | None = None,
) -> ArtifactManager:
    """
    Factory function to create configured ArtifactManager.

    Args:
        storage_path: Path for file storage
        use_compression: Enable compression
        enable_versioning: Enable versioning
        max_versions: Maximum versions to keep
        retention_days: Default retention period
        in_memory: Use in-memory storage
        logger: Optional logger

    Returns:
        Configured ArtifactManager
    """
    cleanup_config = CleanupConfig(
        max_versions=max_versions,
        max_age_days=retention_days,
    )

    config = ArtifactManagerConfig(
        storage_path=storage_path,
        use_compression=use_compression,
        enable_versioning=enable_versioning,
        cleanup_config=cleanup_config,
    )

    storage: StorageBackend | None = None
    metadata_store: MetadataStore | None = None

    if in_memory:
        storage = InMemoryStorageBackend()
        metadata_store = InMemoryMetadataStore()

    return ArtifactManager(
        config=config,
        storage=storage,
        metadata_store=metadata_store,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ArtifactType",
    "ArtifactStatus",
    "CompressionType",
    "RetentionPolicy",
    # Exceptions
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactExistsError",
    "ArtifactCorruptedError",
    "StorageError",
    # Metadata
    "ArtifactMetadata",
    "Artifact",
    # Type Detection
    "TypeDetector",
    # Storage
    "StorageBackend",
    "InMemoryStorageBackend",
    "FileSystemStorageBackend",
    # Metadata Store
    "MetadataStore",
    "InMemoryMetadataStore",
    "FileMetadataStore",
    # Version Manager
    "VersionManager",
    # Cleanup
    "CleanupConfig",
    "CleanupManager",
    # Manager
    "ArtifactManagerConfig",
    "ArtifactManager",
    # Factory
    "create_artifact_manager",
]
