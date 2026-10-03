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


class ApprovalGateError(Exception):
    pass


class RequestNotFoundError(ApprovalGateError):
    pass


class InvalidStateTransitionError(ApprovalGateError):
    pass


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
            "request_id": self.request_id,
            "action": self.action,
            "requested_by": self.requested_by,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ApprovalGate:
    """Human-in-the-loop approval workflow with expiration and full history."""

    def __init__(self, *, default_ttl_seconds: float = 3600.0, max_history: int = 10_000) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.max_history = max_history
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._created_at = time.time()
        self._lock = asyncio.Lock()
        self._listeners: dict[str, asyncio.Event] = {}
        self._request_count = 0
        self._approve_count = 0
        self._reject_count = 0
        self._expire_count = 0

    def _sweep_expired(self) -> None:
        now = time.time()
        for rid, req in list(self._pending.items()):
            if req.is_expired(now):
                req.status = ApprovalStatus.EXPIRED
                req.resolved_at = now
                self._pending.pop(rid, None)
                self._archive(req)
                self._expire_count += 1
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
            request_id=str(uuid.uuid4()),
            action=action,
            requested_by=requested_by,
            payload=payload or {},
            expires_at=time.time() + ttl if ttl > 0 else None,
        )
        self._pending[req.request_id] = req
        self._listeners[req.request_id] = asyncio.Event()
        return req

    def approve(self, request_id: str, *, approved_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        self._sweep_expired()
        req = self._pending.get(request_id)
        if req is None:
            existing = next((r for r in self._history if r.request_id == request_id), None)
            if existing is not None:
                raise InvalidStateTransitionError(f"request {request_id} already resolved as {existing.status.value}")
            raise RequestNotFoundError(f"no pending request with id {request_id}")

        req.status = ApprovalStatus.APPROVED
        req.resolved_at = time.time()
        req.resolved_by = approved_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        self._approve_count += 1
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
                raise InvalidStateTransitionError(f"request {request_id} already resolved as {existing.status.value}")
            raise RequestNotFoundError(f"no pending request with id {request_id}")

        req.status = ApprovalStatus.REJECTED
        req.resolved_at = time.time()
        req.resolved_by = rejected_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        self._reject_count += 1
        event = self._listeners.get(request_id)
        if event is not None:
            event.set()
        return req

    def cancel(self, request_id: str, *, cancelled_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        req = self._pending.get(request_id)
        if req is None:
            raise RequestNotFoundError(f"no pending request with id {request_id}")
        req.status = ApprovalStatus.CANCELLED
        req.resolved_at = time.time()
        req.resolved_by = cancelled_by
        req.reason = reason
        self._pending.pop(request_id, None)
        self._archive(req)
        event = self._listeners.get(request_id)
        if event is not None:
            event.set()
        return req

    def list_pending(self, *, requested_by: Optional[str] = None, action: Optional[str] = None) -> list[ApprovalRequest]:
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

    def history(self, limit: int = 100, *, status: Optional[ApprovalStatus] = None) -> list[ApprovalRequest]:
        items = self._history if status is None else [r for r in self._history if r.status == status]
        return items[-limit:]

    async def request_async(
        self,
        action: str,
        requested_by: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> ApprovalRequest:
        async with self._lock:
            return self.request(action, requested_by, payload=payload, ttl_seconds=ttl_seconds)

    async def approve_async(self, request_id: str, *, approved_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        async with self._lock:
            return self.approve(request_id, approved_by=approved_by, reason=reason)

    async def reject_async(self, request_id: str, *, rejected_by: str, reason: Optional[str] = None) -> ApprovalRequest:
        async with self._lock:
            return self.reject(request_id, rejected_by=rejected_by, reason=reason)

    async def list_pending_async(
        self, *, requested_by: Optional[str] = None, action: Optional[str] = None
    ) -> list[ApprovalRequest]:
        async with self._lock:
            return self.list_pending(requested_by=requested_by, action=action)

    async def wait_for_resolution(self, request_id: str, *, timeout: Optional[float] = None) -> ApprovalRequest:
        event = self._listeners.get(request_id)
        if event is None:
            existing = self.get(request_id)
            if existing is None:
                raise RequestNotFoundError(f"no request with id {request_id}")
            return existing
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        result = self.get(request_id)
        if result is None:
            raise RequestNotFoundError(f"no request with id {request_id}")
        return result

    def health_check(self) -> HealthStatus:
        try:
            self._sweep_expired()
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "pending_count": len(self._pending),
                "history_count": len(self._history),
                "request_count": self._request_count,
                "approve_count": self._approve_count,
                "reject_count": self._reject_count,
                "expire_count": self._expire_count,
            }
            return HealthStatus(healthy=True, component="approval_gate", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="approval_gate", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalGateError",
    "RequestNotFoundError",
    "InvalidStateTransitionError",
    "HealthStatus",
]
