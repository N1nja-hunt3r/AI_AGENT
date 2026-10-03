from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class ArtifactHandlerError(Exception):
    pass


class ArtifactNotFoundError(ArtifactHandlerError):
    pass


class ArtifactVersionNotFoundError(ArtifactHandlerError):
    pass


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArtifactVersion:
    version: int
    artifact_id: str
    storage_path: str
    size_bytes: int
    checksum: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "artifact_id": self.artifact_id, "storage_path": self.storage_path,
            "size_bytes": self.size_bytes, "checksum": self.checksum, "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class ArtifactRecord:
    artifact_id: str
    name: str
    latest_version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    owner: Optional[str] = None
    versions: list[ArtifactVersion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id, "name": self.name, "latest_version": self.latest_version,
            "created_at": self.created_at, "updated_at": self.updated_at, "tags": self.tags,
            "owner": self.owner, "version_count": len(self.versions),
        }


class ArtifactHandler:
    """Stores and retrieves versioned artifacts on disk with metadata and cleanup support."""

    def __init__(self, *, storage_dir: Optional[str | Path] = None, max_versions_per_artifact: int = 20) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("/tmp/artifact_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions_per_artifact = max_versions_per_artifact
        self._records: dict[str, ArtifactRecord] = {}
        self._name_index: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._save_count = 0
        self._load_count = 0
        self._delete_count = 0
        self._cleanup_count = 0

    def _artifact_dir(self, artifact_id: str) -> Path:
        d = self.storage_dir / artifact_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def save(
        self,
        name: str,
        data: bytes | str,
        *,
        artifact_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        owner: Optional[str] = None,
    ) -> ArtifactVersion:
        self._save_count += 1
        raw = data.encode("utf-8") if isinstance(data, str) else data

        aid = artifact_id or self._name_index.get(name) or str(uuid.uuid4())
        record = self._records.get(aid)
        if record is None:
            record = ArtifactRecord(artifact_id=aid, name=name, tags=tags or [], owner=owner)
            self._records[aid] = record
            self._name_index[name] = aid

        new_version_num = record.latest_version + 1
        artifact_dir = self._artifact_dir(aid)
        storage_path = artifact_dir / f"v{new_version_num}.bin"

        await asyncio.to_thread(storage_path.write_bytes, raw)
        checksum = hashlib.sha256(raw).hexdigest()

        version = ArtifactVersion(
            version=new_version_num, artifact_id=aid, storage_path=str(storage_path),
            size_bytes=len(raw), checksum=checksum, metadata=metadata or {},
        )
        record.versions.append(version)
        record.latest_version = new_version_num
        record.updated_at = time.time()
        if tags:
            record.tags = list(set(record.tags) | set(tags))

        await self._enforce_version_limit(record)
        return version

    async def _enforce_version_limit(self, record: ArtifactRecord) -> None:
        if len(record.versions) <= self.max_versions_per_artifact:
            return
        excess = len(record.versions) - self.max_versions_per_artifact
        to_remove = record.versions[:excess]
        record.versions = record.versions[excess:]
        for version in to_remove:
            try:
                await asyncio.to_thread(Path(version.storage_path).unlink, True)
            except OSError:
                pass

    async def load(self, artifact_id: str, *, version: Optional[int] = None) -> bytes:
        self._load_count += 1
        record = self._records.get(artifact_id)
        if record is None:
            raise ArtifactNotFoundError(f"no artifact with id {artifact_id}")

        target_version_num = version if version is not None else record.latest_version
        target = next((v for v in record.versions if v.version == target_version_num), None)
        if target is None:
            raise ArtifactVersionNotFoundError(f"no version {target_version_num} for artifact '{artifact_id}'")

        path = Path(target.storage_path)
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact data missing on disk for '{artifact_id}' v{target_version_num}")
        return await asyncio.to_thread(path.read_bytes)

    async def load_by_name(self, name: str, *, version: Optional[int] = None) -> bytes:
        aid = self._name_index.get(name)
        if aid is None:
            raise ArtifactNotFoundError(f"no artifact registered under name '{name}'")
        return await self.load(aid, version=version)

    def get_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        return self._records.get(artifact_id)

    def get_version_metadata(self, artifact_id: str, version: int) -> Optional[ArtifactVersion]:
        record = self._records.get(artifact_id)
        if record is None:
            return None
        return next((v for v in record.versions if v.version == version), None)

    def list_artifacts(self, *, owner: Optional[str] = None, tag: Optional[str] = None) -> list[ArtifactRecord]:
        items = list(self._records.values())
        if owner is not None:
            items = [r for r in items if r.owner == owner]
        if tag is not None:
            items = [r for r in items if tag in r.tags]
        return items

    def list_versions(self, artifact_id: str) -> list[ArtifactVersion]:
        record = self._records.get(artifact_id)
        if record is None:
            raise ArtifactNotFoundError(f"no artifact with id {artifact_id}")
        return list(record.versions)

    async def delete(self, artifact_id: str, *, version: Optional[int] = None) -> bool:
        self._delete_count += 1
        record = self._records.get(artifact_id)
        if record is None:
            raise ArtifactNotFoundError(f"no artifact with id {artifact_id}")

        if version is None:
            for v in record.versions:
                try:
                    await asyncio.to_thread(Path(v.storage_path).unlink, True)
                except OSError:
                    pass
            self._records.pop(artifact_id, None)
            self._name_index = {k: v for k, v in self._name_index.items() if v != artifact_id}
            artifact_dir = self.storage_dir / artifact_id
            if artifact_dir.exists():
                await asyncio.to_thread(shutil.rmtree, str(artifact_dir), True)
            return True

        target = next((v for v in record.versions if v.version == version), None)
        if target is None:
            raise ArtifactVersionNotFoundError(f"no version {version} for artifact '{artifact_id}'")
        try:
            await asyncio.to_thread(Path(target.storage_path).unlink, True)
        except OSError:
            pass
        record.versions = [v for v in record.versions if v.version != version]
        return True

    async def cleanup(self, *, older_than_seconds: Optional[float] = None, keep_latest_n: Optional[int] = None) -> int:
        self._cleanup_count += 1
        now = time.time()
        removed = 0

        for record in list(self._records.values()):
            versions_to_check = sorted(record.versions, key=lambda v: v.version)

            if keep_latest_n is not None and len(versions_to_check) > keep_latest_n:
                excess = versions_to_check[: len(versions_to_check) - keep_latest_n]
                for v in excess:
                    if older_than_seconds is None or (now - v.created_at) >= older_than_seconds:
                        try:
                            await asyncio.to_thread(Path(v.storage_path).unlink, True)
                        except OSError:
                            pass
                        record.versions.remove(v)
                        removed += 1
            elif older_than_seconds is not None:
                stale = [v for v in versions_to_check if (now - v.created_at) >= older_than_seconds and v.version != record.latest_version]
                for v in stale:
                    try:
                        await asyncio.to_thread(Path(v.storage_path).unlink, True)
                    except OSError:
                        pass
                    record.versions.remove(v)
                    removed += 1

            if not record.versions:
                self._records.pop(record.artifact_id, None)
                self._name_index = {k: v for k, v in self._name_index.items() if v != record.artifact_id}

        return removed

    def health_check(self) -> HealthStatus:
        try:
            total_versions = sum(len(r.versions) for r in self._records.values())
            total_bytes = sum(v.size_bytes for r in self._records.values() for v in r.versions)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "storage_dir": str(self.storage_dir),
                "artifact_count": len(self._records),
                "total_version_count": total_versions,
                "total_bytes": total_bytes,
                "save_count": self._save_count,
                "load_count": self._load_count,
                "delete_count": self._delete_count,
                "cleanup_count": self._cleanup_count,
            }
            storage_writable = self.storage_dir.exists() and self.storage_dir.is_dir()
            return HealthStatus(healthy=storage_writable, component="artifact_handler", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="artifact_handler", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "ArtifactHandler",
    "ArtifactRecord",
    "ArtifactVersion",
    "ArtifactHandlerError",
    "ArtifactNotFoundError",
    "ArtifactVersionNotFoundError",
    "HealthStatus",
]
