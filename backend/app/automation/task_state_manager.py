from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskState(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERED = "recovered"


class TaskStateManagerError(Exception):
    pass


class TaskNotFoundError(TaskStateManagerError):
    pass


class InvalidTransitionError(TaskStateManagerError):
    pass


class CheckpointNotFoundError(TaskStateManagerError):
    pass


_VALID_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.CREATED: frozenset({TaskState.INITIALIZING, TaskState.CANCELLED}),
    TaskState.INITIALIZING: frozenset({TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.RUNNING: frozenset({TaskState.PAUSED, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.PAUSED: frozenset({TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.FAILED: frozenset({TaskState.RECOVERED, TaskState.CANCELLED}),
    TaskState.RECOVERED: frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskStateRecord:
    task_id: str
    state: TaskState = TaskState.CREATED
    data: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "state": self.state.value, "data": self.data, "updated_at": self.updated_at, "created_at": self.created_at}


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    state: TaskState
    data_snapshot: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    label: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id, "task_id": self.task_id, "state": self.state.value,
            "data_snapshot": self.data_snapshot, "created_at": self.created_at, "label": self.label,
        }


class TaskStateManager:
    """Manages task state, transitions, checkpoints, and crash recovery with SQLite persistence."""

    def __init__(self, *, db_path: Optional[str | Path] = None, max_checkpoints_per_task: int = 100) -> None:
        self.db_path = str(db_path) if db_path else ":memory:"
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints_per_task = max_checkpoints_per_task
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._save_count = 0
        self._restore_count = 0
        self._transition_count = 0

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_states (
                task_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                state TEXT NOT NULL,
                data_snapshot TEXT NOT NULL,
                created_at REAL NOT NULL,
                label TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id)")
        self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> TaskStateRecord:
        return TaskStateRecord(
            task_id=row["task_id"], state=TaskState(row["state"]), data=json.loads(row["data"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def save(self, task_id: Optional[str] = None, *, state: TaskState = TaskState.CREATED, data: Optional[dict[str, Any]] = None) -> TaskStateRecord:
        self._save_count += 1
        tid = task_id or str(uuid.uuid4())
        now = time.time()

        existing = self.get(tid)
        record = TaskStateRecord(
            task_id=tid, state=state, data=data if data is not None else (existing.data if existing else {}),
            created_at=existing.created_at if existing else now, updated_at=now,
        )

        self._conn.execute(
            """
            INSERT INTO task_states (task_id, state, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET state=excluded.state, data=excluded.data, updated_at=excluded.updated_at
            """,
            (record.task_id, record.state.value, json.dumps(record.data, default=str), record.created_at, record.updated_at),
        )
        self._conn.commit()
        return record

    def get(self, task_id: str) -> Optional[TaskStateRecord]:
        cursor = self._conn.execute("SELECT * FROM task_states WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        return self._row_to_record(row) if row is not None else None

    def transition(self, task_id: str, new_state: TaskState, *, data: Optional[dict[str, Any]] = None) -> TaskStateRecord:
        self._transition_count += 1
        record = self.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"no task state record for task_id {task_id}")

        allowed = _VALID_TRANSITIONS.get(record.state, frozenset())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"task '{task_id}' cannot transition from {record.state.value} to {new_state.value}"
            )

        merged_data = {**record.data, **(data or {})}
        return self.save(task_id, state=new_state, data=merged_data)

    def checkpoint(self, task_id: str, *, label: Optional[str] = None) -> Checkpoint:
        record = self.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"no task state record for task_id {task_id}")

        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4()), task_id=task_id, state=record.state,
            data_snapshot=dict(record.data), label=label,
        )
        self._conn.execute(
            "INSERT INTO checkpoints (checkpoint_id, task_id, state, data_snapshot, created_at, label) VALUES (?, ?, ?, ?, ?, ?)",
            (cp.checkpoint_id, cp.task_id, cp.state.value, json.dumps(cp.data_snapshot, default=str), cp.created_at, cp.label),
        )
        self._conn.commit()

        self._enforce_checkpoint_limit(task_id)
        return cp

    def _enforce_checkpoint_limit(self, task_id: str) -> None:
        cursor = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE task_id = ? ORDER BY created_at ASC", (task_id,)
        )
        ids = [row["checkpoint_id"] for row in cursor.fetchall()]
        if len(ids) > self.max_checkpoints_per_task:
            to_delete = ids[: len(ids) - self.max_checkpoints_per_task]
            self._conn.executemany("DELETE FROM checkpoints WHERE checkpoint_id = ?", [(cid,) for cid in to_delete])
            self._conn.commit()

    def list_checkpoints(self, task_id: str, *, limit: int = 100) -> list[Checkpoint]:
        cursor = self._conn.execute(
            "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY created_at DESC LIMIT ?", (task_id, limit)
        )
        return [
            Checkpoint(
                checkpoint_id=row["checkpoint_id"], task_id=row["task_id"], state=TaskState(row["state"]),
                data_snapshot=json.loads(row["data_snapshot"]), created_at=row["created_at"], label=row["label"],
            )
            for row in cursor.fetchall()
        ]

    def restore(self, task_id: str, *, checkpoint_id: Optional[str] = None) -> TaskStateRecord:
        self._restore_count += 1
        if checkpoint_id is not None:
            cursor = self._conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            if row is None:
                raise CheckpointNotFoundError(f"no checkpoint with id {checkpoint_id}")
        else:
            cursor = self._conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY created_at DESC LIMIT 1", (task_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise CheckpointNotFoundError(f"no checkpoints available for task_id {task_id}")

        data_snapshot = json.loads(row["data_snapshot"])
        return self.save(task_id, state=TaskState.RECOVERED, data=data_snapshot)

    def delete(self, task_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM task_states WHERE task_id = ?", (task_id,))
        self._conn.execute("DELETE FROM checkpoints WHERE task_id = ?", (task_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_tasks(self, *, state: Optional[TaskState] = None, limit: int = 500) -> list[TaskStateRecord]:
        if state is not None:
            cursor = self._conn.execute(
                "SELECT * FROM task_states WHERE state = ? ORDER BY updated_at DESC LIMIT ?", (state.value, limit)
            )
        else:
            cursor = self._conn.execute("SELECT * FROM task_states ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [self._row_to_record(row) for row in cursor.fetchall()]

    async def save_async(self, task_id: Optional[str] = None, **kwargs: Any) -> TaskStateRecord:
        async with self._lock:
            return await asyncio.to_thread(lambda: self.save(task_id, **kwargs))

    async def restore_async(self, task_id: str, **kwargs: Any) -> TaskStateRecord:
        async with self._lock:
            return await asyncio.to_thread(lambda: self.restore(task_id, **kwargs))

    async def transition_async(self, task_id: str, new_state: TaskState, **kwargs: Any) -> TaskStateRecord:
        async with self._lock:
            return await asyncio.to_thread(lambda: self.transition(task_id, new_state, **kwargs))

    def health_check(self) -> HealthStatus:
        try:
            self._conn.execute("SELECT 1").fetchone()
            total = self._conn.execute("SELECT COUNT(*) as cnt FROM task_states").fetchone()["cnt"]
            checkpoints = self._conn.execute("SELECT COUNT(*) as cnt FROM checkpoints").fetchone()["cnt"]
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "db_path": self.db_path,
                "task_count": total,
                "checkpoint_count": checkpoints,
                "save_count": self._save_count,
                "restore_count": self._restore_count,
                "transition_count": self._transition_count,
            }
            return HealthStatus(healthy=True, component="task_state_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="task_state_manager", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return await asyncio.to_thread(self.health_check)

    def close(self) -> None:
        self._conn.close()


__all__ = [
    "TaskStateManager",
    "TaskStateRecord",
    "TaskState",
    "Checkpoint",
    "TaskStateManagerError",
    "TaskNotFoundError",
    "InvalidTransitionError",
    "CheckpointNotFoundError",
    "HealthStatus",
]
