from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalHandlerError(Exception):
    pass


class ApprovalRequestNotFoundError(ApprovalHandlerError):
    pass


class InvalidApprovalStateError(ApprovalHandlerError):
    pass


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuditEntry:
    entry_id: str
    request_id: str
    action: str
    actor: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id, "request_id": self.request_id, "action": self.action,
            "actor": self.actor, "timestamp": self.timestamp, "details": self.details,
        }


@dataclass
class ApprovalRequest:
    request_id: str
    action: str
    requested_by: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    reason: Optional[str] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time.time()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "action": self.action, "requested_by": self.requested_by,
            "payload": self.payload, "status": self.status.value, "created_at": self.created_at,
            "expires_at": self.expires_at, "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by, "reason": self.reason,
        }


class ApprovalHandler:
    """Human-in-the-loop approval workflow with expiration, audit logging, and full history."""

    def __init__(self, *, default_ttl_seconds: float = 3600.0, max_history: int = 10_000, max_audit_log: int = 20_000) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.max_history = max_history
        self.max_audit_log = max_audit_log
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._audit_log: list[AuditEntry] = []
        self._listeners: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._request_count = 0
        self._approve_count = 0
        self._reject_count = 0
        self._expire_count = 0

    def _audit(self, request_id: str, action: str, actor: str, *, details: Optional[dict[str, Any]] = None) -> AuditEntry:
        entry = AuditEntry(entry_id=str(uuid.uuid4()), request_id=request_id, action=action, actor=actor, details=details or {})
        self._audit_log.append(entry)
        if len(self._audit_log) > self.max_audit_log:
            self._audit_log = self._audit_log[-self.max_audit_log :]
        return entry

    def _sweep_expired(self) -> None:
        now = time.time()
        for rid, req in list(self._pending.items()):
            if req.is_expired(now):
                req.status = ApprovalStatus.EXPIRED
                req.resolved_at = now
                self._pending.pop(rid, None)
                self._archive(req)
                self._expire_count += 1
                self._audit(rid, "expired", "system")
                event = self._listeners.get(rid)
                if event is not None:
                    event.set()

    def _archive(self, request: ApprovalRequest) -> None:
        self._history.append(request)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def request(
        self,
        action: str,
        requested_by: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> ApprovalRequest:
        self._sweep_expired()
        self._request_count += 1
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        req = ApprovalRequest(
            request_id=str(uuid.uuid4()), action=action, requested_by=requested_by,
            payload=payload or {}, expires_at=time.time() + ttl if ttl > 0 else None,
        )
        self._pending[req.request_id] = req
        self._listeners[req.request_id] = asyncio.Event()
        self._audit(req.request_id, "requested", requested_by, details={"action": action})
        return req

    def approve(self, request_id: str, *, approved_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        self._sweep_expired()
        req = self._pending.get(request_id)
        if req is None:
            existing = next((r for r in self._history if r.request_id == request_id), None)
            if existing is not None:
                raise InvalidApprovalStateError(f"request {request_id} already resolved as {existing.status.value}")
            raise ApprovalRequestNotFoundError(f"no pending request with id {request_id}")

        req.status = ApprovalStatus.APPROVED
        req.resolved_at = time.time()
        req.resolved_by = approved_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        self._approve_count += 1
        self._audit(request_id, "approved", approved_by, details={"reason": reason})
        event = self._listeners.get(request_id)
        if event is not None:
            event.set()
        return req

    def reject(self, request_id: str, *, rejected_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        self._sweep_expired()
        req = self._pending.get(request_id)
        if req is None:
            existing = next((r for r in self._history if r.request_id == request_id), None)
            if existing is not None:
                raise InvalidApprovalStateError(f"request {request_id} already resolved as {existing.status.value}")
            raise ApprovalRequestNotFoundError(f"no pending request with id {request_id}")

        req.status = ApprovalStatus.REJECTED
        req.resolved_at = time.time()
        req.resolved_by = rejected_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        self._reject_count += 1
        self._audit(request_id, "rejected", rejected_by, details={"reason": reason})
        event = self._listeners.get(request_id)
        if event is not None:
            event.set()
        return req

    def cancel(self, request_id: str, *, cancelled_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        req = self._pending.get(request_id)
        if req is None:
            raise ApprovalRequestNotFoundError(f"no pending request with id {request_id}")
        req.status = ApprovalStatus.CANCELLED
        req.resolved_at = time.time()
        req.resolved_by = cancelled_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        self._audit(request_id, "cancelled", cancelled_by, details={"reason": reason})
        event = self._listeners.get(request_id)
        if event is not None:
            event.set()
        return req

    def pending(self, *, requested_by: Optional[str] = None, action: Optional[str] = None) -> list[ApprovalRequest]:
        self._sweep_expired()
        items = list(self._pending.values())
        if requested_by is not None:
            items = [r for r in items if r.requested_by == requested_by]
        if action is not None:
            items = [r for r in items if r.action == action]
        return sorted(items, key=lambda r: r.created_at)

    def get(self, request_id: str) -> Optional[ApprovalRequest]:
        self._sweep_expired()
        if request_id in self._pending:
            return self._pending[request_id]
        return next((r for r in self._history if r.request_id == request_id), None)

    def history(self, limit: int = 200, *, status: Optional[ApprovalStatus] = None) -> list[ApprovalRequest]:
        items = self._history if status is None else [r for r in self._history if r.status == status]
        return items[-limit:]

    def audit_log(self, *, request_id: Optional[str] = None, limit: int = 500) -> list[AuditEntry]:
        items = self._audit_log
        if request_id is not None:
            items = [e for e in items if e.request_id == request_id]
        return items[-limit:]

    async def wait_for_resolution(self, request_id: str, *, timeout: Optional[float] = None) -> ApprovalRequest:
        event = self._listeners.get(request_id)
        if event is None:
            existing = self.get(request_id)
            if existing is None:
                raise ApprovalRequestNotFoundError(f"no request with id {request_id}")
            return existing
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        result = self.get(request_id)
        if result is None:
            raise ApprovalRequestNotFoundError(f"no request with id {request_id}")
        return result

    async def request_async(self, action: str, requested_by: str, **kwargs: Any) -> ApprovalRequest:
        async with self._lock:
            return self.request(action, requested_by, **kwargs)

    async def approve_async(self, request_id: str, *, approved_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        async with self._lock:
            return self.approve(request_id, approved_by=approved_by, reason=reason)

    async def reject_async(self, request_id: str, *, rejected_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        async with self._lock:
            return self.reject(request_id, rejected_by=rejected_by, reason=reason)

    async def pending_async(self, **kwargs: Any) -> list[ApprovalRequest]:
        async with self._lock:
            return self.pending(**kwargs)

    def health_check(self) -> HealthStatus:
        try:
            self._sweep_expired()
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "pending_count": len(self._pending),
                "history_count": len(self._history),
                "audit_log_size": len(self._audit_log),
                "request_count": self._request_count,
                "approve_count": self._approve_count,
                "reject_count": self._reject_count,
                "expire_count": self._expire_count,
            }
            return HealthStatus(healthy=True, component="approval_handler", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="approval_handler", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalStatus",
    "AuditEntry",
    "ApprovalHandlerError",
    "ApprovalRequestNotFoundError",
    "InvalidApprovalStateError",
    "HealthStatus",
]
