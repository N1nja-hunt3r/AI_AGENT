from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional


class TraceKind(str, Enum):
    EXECUTION = "execution"
    STEP = "step"
    CAPABILITY = "capability"
    AGENT = "agent"


class TraceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    CANCELLED = "cancelled"


class TracesError(Exception):
    pass


class TraceRecordNotFoundError(TracesError):
    pass


@dataclass
class TraceRecord:
    trace_id: str
    kind: TraceKind
    name: str
    started_at: float
    parent_id: Optional[str] = None
    agent_id: Optional[str] = None
    capability_name: Optional[str] = None
    outcome: TraceOutcome = TraceOutcome.RUNNING
    ended_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    children: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind.value,
            "name": self.name,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "capability_name": self.capability_name,
            "outcome": self.outcome.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "error": self.error,
            "children": list(self.children),
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Traces:
    """Records hierarchical execution/step/capability/agent traces with export support."""

    def __init__(self, *, max_records: int = 20_000) -> None:
        self.max_records = max_records
        self._records: dict[str, TraceRecord] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._trace_count = 0
        self._record_count = 0

    def trace(
        self,
        name: str,
        *,
        kind: TraceKind = TraceKind.EXECUTION,
        parent_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceRecord:
        self._trace_count += 1
        trace_id = str(uuid.uuid4())
        record = TraceRecord(
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=time.time(),
            parent_id=parent_id,
            agent_id=agent_id,
            capability_name=capability_name,
            metadata=dict(metadata or {}),
        )
        self._records[trace_id] = record
        self._order.append(trace_id)
        if parent_id is not None and parent_id in self._records:
            self._records[parent_id].children.append(trace_id)

        self._enforce_limit()
        return record

    def _enforce_limit(self) -> None:
        while len(self._order) > self.max_records:
            oldest_id = self._order.pop(0)
            self._records.pop(oldest_id, None)

    def record(
        self,
        trace_id: str,
        *,
        outcome: TraceOutcome = TraceOutcome.SUCCESS,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceRecord:
        self._record_count += 1
        rec = self._records.get(trace_id)
        if rec is None:
            raise TraceRecordNotFoundError(f"no trace record with id {trace_id}")
        rec.outcome = outcome
        rec.ended_at = time.time()
        rec.error = error
        if metadata:
            rec.metadata.update(metadata)
        return rec

    @contextmanager
    def trace_execution(
        self,
        name: str,
        *,
        kind: TraceKind = TraceKind.EXECUTION,
        parent_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[TraceRecord]:
        record = self.trace(
            name, kind=kind, parent_id=parent_id, agent_id=agent_id,
            capability_name=capability_name, metadata=metadata,
        )
        try:
            yield record
            self.record(record.trace_id, outcome=TraceOutcome.SUCCESS)
        except Exception as exc:
            self.record(record.trace_id, outcome=TraceOutcome.FAILURE, error=str(exc))
            raise

    def get(self, trace_id: str) -> Optional[TraceRecord]:
        return self._records.get(trace_id)

    def get_children(self, trace_id: str) -> list[TraceRecord]:
        rec = self._records.get(trace_id)
        if rec is None:
            return []
        return [self._records[cid] for cid in rec.children if cid in self._records]

    def list_traces(
        self,
        *,
        kind: Optional[TraceKind] = None,
        agent_id: Optional[str] = None,
        capability_name: Optional[str] = None,
        outcome: Optional[TraceOutcome] = None,
        limit: int = 200,
    ) -> list[TraceRecord]:
        items = [self._records[tid] for tid in self._order if tid in self._records]
        if kind is not None:
            items = [r for r in items if r.kind == kind]
        if agent_id is not None:
            items = [r for r in items if r.agent_id == agent_id]
        if capability_name is not None:
            items = [r for r in items if r.capability_name == capability_name]
        if outcome is not None:
            items = [r for r in items if r.outcome == outcome]
        return items[-limit:]

    def export(self, *, fmt: str = "json", limit: int = 1000) -> str:
        records = self.list_traces(limit=limit)
        data = [r.to_dict() for r in records]
        if fmt == "json":
            return json.dumps(data, default=str, sort_keys=True)
        if fmt == "jsonl":
            return "\n".join(json.dumps(d, default=str, sort_keys=True) for d in data)
        raise TracesError(f"unsupported export format: {fmt}")

    async def trace_async(self, name: str, **kwargs: Any) -> TraceRecord:
        async with self._lock:
            return self.trace(name, **kwargs)

    async def record_async(self, trace_id: str, **kwargs: Any) -> TraceRecord:
        async with self._lock:
            return self.record(trace_id, **kwargs)

    async def export_async(self, *, fmt: str = "json", limit: int = 1000) -> str:
        async with self._lock:
            return self.export(fmt=fmt, limit=limit)

    def health_check(self) -> HealthStatus:
        try:
            probe = self.trace("__health_check__", kind=TraceKind.EXECUTION)
            self.record(probe.trace_id, outcome=TraceOutcome.SUCCESS)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "active_record_count": len(self._records),
                "trace_count": self._trace_count,
                "record_count": self._record_count,
                "max_records": self.max_records,
            }
            return HealthStatus(healthy=True, component="traces", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="traces", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "Traces",
    "TraceRecord",
    "TraceKind",
    "TraceOutcome",
    "TracesError",
    "TraceRecordNotFoundError",
    "HealthStatus",
]
