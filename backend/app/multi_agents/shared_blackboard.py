from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BlackboardError(Exception):
    pass


class KeyNotFoundError(BlackboardError):
    pass


class VersionConflictError(BlackboardError):
    pass


class TaskState(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class VersionedValue:
    value: Any
    version: int
    updated_at_epoch: float
    updated_by: str = ""


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    name: str
    content: Any
    content_type: str
    created_by: str
    version: int
    created_at_epoch: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlackboardMessage:
    message_id: str
    sender: str
    recipient: Optional[str]
    topic: str
    content: Any
    sent_at_epoch: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskStatusEntry:
    task_id: str
    state: TaskState
    assigned_to: Optional[str]
    progress: float
    updated_at_epoch: float
    detail: str = ""
    history: list[tuple[float, TaskState, str]] = field(default_factory=list)


class SharedBlackboard:
    def __init__(self) -> None:
        self._state: dict[str, VersionedValue] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._artifact_versions: dict[str, list[Artifact]] = {}
        self._messages: list[BlackboardMessage] = []
        self._task_statuses: dict[str, TaskStatusEntry] = {}
        self._memory_context: dict[str, Any] = {}
        self._global_lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._sequence = 0

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}-{int(time.time() * 1000)}"

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = asyncio.Lock()
            return self._key_locks[key]

    async def acquire_lock(self, key: str) -> asyncio.Lock:
        lock = await self._get_key_lock(key)
        await lock.acquire()
        return lock

    def release_lock(self, lock: asyncio.Lock) -> None:
        if lock.locked():
            lock.release()

    async def set_state(
        self,
        key: str,
        value: Any,
        updated_by: str = "",
        expected_version: Optional[int] = None,
    ) -> VersionedValue:
        lock = await self._get_key_lock(key)
        async with lock:
            current = self._state.get(key)
            if expected_version is not None:
                current_version = current.version if current else 0
                if current_version != expected_version:
                    raise VersionConflictError(
                        f"key '{key}' has version {current_version}, expected {expected_version}"
                    )
            new_version = (current.version if current else 0) + 1
            new_value = VersionedValue(
                value=value,
                version=new_version,
                updated_at_epoch=time.time(),
                updated_by=updated_by,
            )
            self._state[key] = new_value
            return new_value

    async def get_state(self, key: str) -> VersionedValue:
        async with self._global_lock:
            entry = self._state.get(key)
            if entry is None:
                raise KeyNotFoundError(f"state key '{key}' not found")
            return entry

    async def get_state_or_default(self, key: str, default: Any = None) -> Any:
        async with self._global_lock:
            entry = self._state.get(key)
            return entry.value if entry is not None else default

    async def delete_state(self, key: str) -> None:
        lock = await self._get_key_lock(key)
        async with lock:
            self._state.pop(key, None)

    async def list_state_keys(self) -> list[str]:
        async with self._global_lock:
            return list(self._state.keys())

    async def put_artifact(
        self,
        name: str,
        content: Any,
        content_type: str,
        created_by: str,
        artifact_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        async with self._global_lock:
            resolved_id = artifact_id or self._next_id("artifact")
            existing = self._artifact_versions.get(resolved_id, [])
            new_version = len(existing) + 1
            artifact = Artifact(
                artifact_id=resolved_id,
                name=name,
                content=content,
                content_type=content_type,
                created_by=created_by,
                version=new_version,
                created_at_epoch=time.time(),
                metadata=metadata or {},
            )
            self._artifacts[resolved_id] = artifact
            self._artifact_versions.setdefault(resolved_id, []).append(artifact)
            return artifact

    async def get_artifact(self, artifact_id: str, version: Optional[int] = None) -> Artifact:
        async with self._global_lock:
            if version is None:
                artifact = self._artifacts.get(artifact_id)
                if artifact is None:
                    raise KeyNotFoundError(f"artifact '{artifact_id}' not found")
                return artifact
            versions = self._artifact_versions.get(artifact_id, [])
            for artifact in versions:
                if artifact.version == version:
                    return artifact
            raise KeyNotFoundError(f"artifact '{artifact_id}' version {version} not found")

    async def list_artifact_versions(self, artifact_id: str) -> list[Artifact]:
        async with self._global_lock:
            return list(self._artifact_versions.get(artifact_id, []))

    async def list_artifacts(self) -> list[Artifact]:
        async with self._global_lock:
            return list(self._artifacts.values())

    async def delete_artifact(self, artifact_id: str) -> None:
        async with self._global_lock:
            self._artifacts.pop(artifact_id, None)
            self._artifact_versions.pop(artifact_id, None)

    async def post_message(
        self,
        sender: str,
        topic: str,
        content: Any,
        recipient: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BlackboardMessage:
        async with self._global_lock:
            message = BlackboardMessage(
                message_id=self._next_id("msg"),
                sender=sender,
                recipient=recipient,
                topic=topic,
                content=content,
                sent_at_epoch=time.time(),
                metadata=metadata or {},
            )
            self._messages.append(message)
            return message

    async def get_messages(
        self,
        topic: Optional[str] = None,
        recipient: Optional[str] = None,
        since_epoch: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[BlackboardMessage]:
        async with self._global_lock:
            results = self._messages
            if topic is not None:
                results = [m for m in results if m.topic == topic]
            if recipient is not None:
                results = [m for m in results if m.recipient in (None, recipient)]
            if since_epoch is not None:
                results = [m for m in results if m.sent_at_epoch >= since_epoch]
            if limit is not None:
                results = results[-limit:]
            return list(results)

    async def set_task_status(
        self,
        task_id: str,
        state: TaskState,
        assigned_to: Optional[str] = None,
        progress: float = 0.0,
        detail: str = "",
    ) -> TaskStatusEntry:
        lock = await self._get_key_lock(f"task:{task_id}")
        async with lock:
            existing = self._task_statuses.get(task_id)
            history = existing.history if existing else []
            history.append((time.time(), state, detail))
            entry = TaskStatusEntry(
                task_id=task_id,
                state=state,
                assigned_to=assigned_to if assigned_to is not None else (existing.assigned_to if existing else None),
                progress=progress,
                updated_at_epoch=time.time(),
                detail=detail,
                history=history,
            )
            self._task_statuses[task_id] = entry
            return entry

    async def get_task_status(self, task_id: str) -> TaskStatusEntry:
        async with self._global_lock:
            entry = self._task_statuses.get(task_id)
            if entry is None:
                raise KeyNotFoundError(f"task '{task_id}' not found")
            return entry

    async def list_task_statuses(
        self, state: Optional[TaskState] = None
    ) -> list[TaskStatusEntry]:
        async with self._global_lock:
            entries = list(self._task_statuses.values())
            if state is not None:
                entries = [e for e in entries if e.state == state]
            return entries

    async def inject_memory_context(self, key: str, value: Any) -> None:
        async with self._global_lock:
            self._memory_context[key] = value

    async def get_memory_context(self, key: str) -> Any:
        async with self._global_lock:
            if key not in self._memory_context:
                raise KeyNotFoundError(f"memory context key '{key}' not found")
            return self._memory_context[key]

    async def get_all_memory_context(self) -> dict[str, Any]:
        async with self._global_lock:
            return dict(self._memory_context)

    async def clear_memory_context(self, key: Optional[str] = None) -> None:
        async with self._global_lock:
            if key is None:
                self._memory_context.clear()
            else:
                self._memory_context.pop(key, None)
