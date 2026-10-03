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


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStoreError(Exception):
    pass


class JobNotFoundError(JobStoreError):
    pass


@dataclass
class Job:
    job_id: str
    name: str
    state: JobState = JobState.PENDING
    payload: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    owner: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "name": self.name, "state": self.state.value,
            "payload": self.payload, "result": self.result, "error": self.error,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "tags": self.tags, "owner": self.owner,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Job":
        return Job(
            job_id=data["job_id"], name=data["name"], state=JobState(data["state"]),
            payload=data.get("payload", {}), result=data.get("result"), error=data.get("error"),
            created_at=data.get("created_at", time.time()), updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", []), owner=data.get("owner"),
        )


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PageResult:
    items: list[Job]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return max(1, (self.total + self.page_size - 1) // self.page_size)


class JobStore:
    """Persistent job storage backed by SQLite, with filtering, pagination, and history tracking."""

    def __init__(self, *, db_path: Optional[str | Path] = None) -> None:
        self.db_path = str(db_path) if db_path else ":memory:"
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._save_count = 0
        self._load_count = 0
        self._delete_count = 0

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                payload TEXT NOT NULL,
                result TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                tags TEXT NOT NULL,
                owner TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_history (
                history_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                state TEXT NOT NULL,
                recorded_at REAL NOT NULL,
                snapshot TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_history_job ON job_history(job_id)")
        self._conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"], name=row["name"], state=JobState(row["state"]),
            payload=json.loads(row["payload"]), result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
            tags=json.loads(row["tags"]), owner=row["owner"],
        )

    def save(self, job: Optional[Job] = None, **kwargs: Any) -> Job:
        self._save_count += 1
        if job is None:
            job = Job(job_id=kwargs.pop("job_id", str(uuid.uuid4())), name=kwargs.pop("name", "unnamed_job"), **kwargs)
        job.updated_at = time.time()

        self._conn.execute(
            """
            INSERT INTO jobs (job_id, name, state, payload, result, error, created_at, updated_at, tags, owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                name=excluded.name, state=excluded.state, payload=excluded.payload,
                result=excluded.result, error=excluded.error, updated_at=excluded.updated_at,
                tags=excluded.tags, owner=excluded.owner
            """,
            (
                job.job_id, job.name, job.state.value, json.dumps(job.payload, default=str),
                json.dumps(job.result, default=str) if job.result is not None else None,
                job.error, job.created_at, job.updated_at, json.dumps(job.tags), job.owner,
            ),
        )
        self._conn.execute(
            "INSERT INTO job_history (history_id, job_id, state, recorded_at, snapshot) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), job.job_id, job.state.value, job.updated_at, json.dumps(job.to_dict(), default=str)),
        )
        self._conn.commit()
        return job

    def load(self, job_id: str) -> Job:
        self._load_count += 1
        cursor = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row is None:
            raise JobNotFoundError(f"no job with id {job_id}")
        return self._row_to_job(row)

    def try_load(self, job_id: str) -> Optional[Job]:
        try:
            return self.load(job_id)
        except JobNotFoundError:
            return None

    def delete(self, job_id: str) -> bool:
        self._delete_count += 1
        cursor = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        self._conn.execute("DELETE FROM job_history WHERE job_id = ?", (job_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def update_state(self, job_id: str, state: JobState, *, result: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> Job:
        job = self.load(job_id)
        job.state = state
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        return self.save(job)

    def history(self, job_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT snapshot FROM job_history WHERE job_id = ? ORDER BY recorded_at DESC LIMIT ?",
            (job_id, limit),
        )
        return [json.loads(row["snapshot"]) for row in cursor.fetchall()]

    def list_jobs(
        self,
        *,
        state: Optional[JobState] = None,
        owner: Optional[str] = None,
        tag: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> PageResult:
        if page < 1:
            raise JobStoreError("page must be >= 1")
        if page_size < 1:
            raise JobStoreError("page_size must be >= 1")

        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if tag is not None:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        count_cursor = self._conn.execute(f"SELECT COUNT(*) as cnt FROM jobs {where_clause}", params)
        total = count_cursor.fetchone()["cnt"]

        offset = (page - 1) * page_size
        rows_cursor = self._conn.execute(
            f"SELECT * FROM jobs {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, page_size, offset),
        )
        items = [self._row_to_job(row) for row in rows_cursor.fetchall()]
        return PageResult(items=items, total=total, page=page, page_size=page_size)

    def count(self, *, state: Optional[JobState] = None) -> int:
        if state is not None:
            cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM jobs WHERE state = ?", (state.value,))
        else:
            cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM jobs")
        return cursor.fetchone()["cnt"]

    async def save_async(self, job: Optional[Job] = None, **kwargs: Any) -> Job:
        async with self._lock:
            return await asyncio.to_thread(self.save, job, **kwargs)

    async def load_async(self, job_id: str) -> Job:
        async with self._lock:
            return await asyncio.to_thread(self.load, job_id)

    async def delete_async(self, job_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self.delete, job_id)

    async def list_jobs_async(self, **kwargs: Any) -> PageResult:
        async with self._lock:
            return await asyncio.to_thread(lambda: self.list_jobs(**kwargs))

    def health_check(self) -> HealthStatus:
        try:
            self._conn.execute("SELECT 1").fetchone()
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "db_path": self.db_path,
                "job_count": self.count(),
                "pending_count": self.count(state=JobState.PENDING),
                "running_count": self.count(state=JobState.RUNNING),
                "failed_count": self.count(state=JobState.FAILED),
                "save_count": self._save_count,
                "load_count": self._load_count,
                "delete_count": self._delete_count,
            }
            return HealthStatus(healthy=True, component="job_store", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="job_store", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return await asyncio.to_thread(self.health_check)

    def close(self) -> None:
        self._conn.close()


__all__ = [
    "JobStore",
    "Job",
    "JobState",
    "PageResult",
    "JobStoreError",
    "JobNotFoundError",
    "HealthStatus",
]
