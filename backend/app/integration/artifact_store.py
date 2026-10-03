"""
artifact_store.py

In-memory artifact storage for the artifact router CRUD operations.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryArtifactStore:
    """Thread-safe in-memory store for artifact metadata and content."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()

    async def create_artifact(
        self,
        user_id: str,
        name: str,
        content: bytes,
        content_type: str,
        tags: List[str],
    ) -> Dict[str, Any]:
        artifact_id = str(uuid.uuid4())
        now = _now_iso()
        checksum = hashlib.sha256(content).hexdigest()
        entry: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "name": name,
            "content_type": content_type,
            "current_version": 1,
            "versions": [
                {
                    "version": 1,
                    "size_bytes": len(content),
                    "checksum": checksum,
                    "created_at": now,
                    "created_by": user_id,
                }
            ],
            "tags": tags,
            "created_at": now,
            "updated_at": now,
            "_content": {1: content},
        }
        with self._lock:
            self._artifacts[artifact_id] = entry
        return self._strip_internal(entry)

    async def add_version(
        self,
        artifact_id: str,
        user_id: str,
        content: bytes,
        content_type: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._artifacts.get(artifact_id)
            if entry is None:
                return None
            version = entry["current_version"] + 1
            now = _now_iso()
            checksum = hashlib.sha256(content).hexdigest()
            entry["versions"].append(
                {
                    "version": version,
                    "size_bytes": len(content),
                    "checksum": checksum,
                    "created_at": now,
                    "created_by": user_id,
                }
            )
            entry["current_version"] = version
            entry["updated_at"] = now
            entry["_content"][version] = content
            entry["content_type"] = content_type
        return self._strip_internal(entry)

    async def list_artifacts(
        self,
        user_id: str,
        tag: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            results = [
                self._strip_internal(a)
                for a in self._artifacts.values()
                if tag is None or tag in a.get("tags", [])
            ]
            total = len(results)
            return results[offset: offset + limit], total

    async def get_metadata(self, artifact_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._artifacts.get(artifact_id)
            if entry is None:
                return None
            return self._strip_internal(entry)

    async def get_content(
        self,
        artifact_id: str,
        user_id: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._artifacts.get(artifact_id)
            if entry is None:
                return None
            v = version if version is not None else entry["current_version"]
            content = entry["_content"].get(v)
            if content is None:
                return None
            return {
                "content": content,
                "filename": entry["name"],
                "content_type": entry["content_type"],
            }

    async def delete_artifact(
        self,
        artifact_id: str,
        user_id: str,
        version: Optional[int] = None,
    ) -> bool:
        with self._lock:
            entry = self._artifacts.get(artifact_id)
            if entry is None:
                return False
            if version is not None:
                if version not in entry["_content"]:
                    return False
                del entry["_content"][version]
                entry["versions"] = [v for v in entry["versions"] if v["version"] != version]
                if not entry["versions"]:
                    del self._artifacts[artifact_id]
                else:
                    entry["current_version"] = max(v["version"] for v in entry["versions"])
            else:
                del self._artifacts[artifact_id]
            return True

    def _strip_internal(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in entry.items() if not k.startswith("_")}
