"""
security.py

FastAPI router for security administration: permission checks,
action approvals, audit log retrieval, and role management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["security"])

VALID_ROLES = {"admin", "owner", "member", "developer", "viewer", "guest", "service_account"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PermissionCheckRequest(BaseModel):
    user_id: Optional[str] = None
    action: str = Field(..., min_length=1)
    resource: Optional[str] = None


class PermissionCheckResponse(BaseModel):
    allowed: bool
    action: str
    resource: Optional[str] = None
    reason: Optional[str] = None


class ApprovalRequest(BaseModel):
    session_id: Optional[str] = None
    action: str = Field(..., min_length=1)
    requested_by: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    expires_in_seconds: int = Field(default=300, ge=10, le=86_400)


class ApprovalRecord(BaseModel):
    approval_id: str
    action: str
    session_id: Optional[str] = None
    status: str
    requested_by: Optional[str] = None
    created_at: str
    expires_at: str


class ApprovalDecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|deny)$")
    reason: Optional[str] = None


class ApprovalListResponse(BaseModel):
    approvals: List[ApprovalRecord]
    total: int


class AuditLogEntry(BaseModel):
    audit_id: str
    user_id: Optional[str]
    action: str
    resource: Optional[str]
    result: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    entries: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


class RoleAssignmentRequest(BaseModel):
    user_id: str
    role: str = Field(..., min_length=1)


class RoleInfo(BaseModel):
    user_id: str
    role: str
    assigned_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") not in {"admin", "owner"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user


def get_security_service(request: Request) -> Any:
    service = getattr(request.app.state, "security_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Security service unavailable")
    return service


# ---------------------------------------------------------------------------
# POST /security/permissions/check
# ---------------------------------------------------------------------------
@router.post("/permissions/check", response_model=PermissionCheckResponse)
async def check_permission(
    payload: PermissionCheckRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> PermissionCheckResponse:
    """Check whether a user is permitted to perform an action on a resource."""
    target_user_id = payload.user_id or user.get("id")
    try:
        result = await security_service.check_permission(
            user_id=target_user_id, action=payload.action, resource=payload.resource
        )
        return PermissionCheckResponse(
            allowed=result.get("allowed", False),
            action=payload.action,
            resource=payload.resource,
            reason=result.get("reason"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("check_permission failed")
        raise HTTPException(status_code=500, detail=f"Permission check failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /security/approvals
# ---------------------------------------------------------------------------
@router.post("/approvals", response_model=ApprovalRecord, status_code=status.HTTP_201_CREATED)
async def request_approval(
    payload: ApprovalRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> ApprovalRecord:
    """Submit a request for human approval of a sensitive action."""
    try:
        record = await security_service.request_approval(
            user_id=user.get("id"),
            session_id=payload.session_id,
            action=payload.action,
            context=payload.context or {},
            expires_in_seconds=payload.expires_in_seconds,
        )
        return ApprovalRecord(
            approval_id=record["approval_id"],
            action=payload.action,
            session_id=payload.session_id,
            status=record.get("status", "pending"),
            requested_by=user.get("id"),
            created_at=record.get("created_at", _now_iso()),
            expires_at=record.get("expires_at", _now_iso()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("request_approval failed")
        raise HTTPException(status_code=500, detail=f"Failed to create approval request: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /security/approvals
# ---------------------------------------------------------------------------
@router.get("/approvals", response_model=ApprovalListResponse)
async def list_approvals(
    status_filter: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> ApprovalListResponse:
    """List pending or historical approval requests."""
    try:
        raw_approvals = await security_service.list_approvals(
            user_id=user.get("id"), status_filter=status_filter, session_id=session_id
        )
        approvals = [
            ApprovalRecord(
                approval_id=a["approval_id"],
                action=a.get("action", ""),
                session_id=a.get("session_id"),
                status=a.get("status", "pending"),
                requested_by=a.get("requested_by"),
                created_at=a.get("created_at", _now_iso()),
                expires_at=a.get("expires_at", _now_iso()),
            )
            for a in raw_approvals
        ]
        return ApprovalListResponse(approvals=approvals, total=len(approvals))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_approvals failed")
        raise HTTPException(status_code=500, detail=f"Failed to list approvals: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /security/approvals/{approval_id}/decision
# ---------------------------------------------------------------------------
@router.post("/approvals/{approval_id}/decision", response_model=ApprovalRecord)
async def decide_approval(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> ApprovalRecord:
    """Approve or deny a pending approval request."""
    try:
        record = await security_service.decide_approval(
            approval_id=approval_id,
            decided_by=user.get("id"),
            decision=payload.decision,
            reason=payload.reason,
        )
        if record is None:
            raise HTTPException(status_code=404, detail=f"Approval not found: {approval_id}")
        return ApprovalRecord(
            approval_id=approval_id,
            action=record.get("action", ""),
            session_id=record.get("session_id"),
            status=record.get("status", payload.decision),
            requested_by=record.get("requested_by"),
            created_at=record.get("created_at", _now_iso()),
            expires_at=record.get("expires_at", _now_iso()),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("decide_approval failed for approval_id=%s", approval_id)
        raise HTTPException(status_code=500, detail=f"Failed to record approval decision: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /security/audit
# ---------------------------------------------------------------------------
@router.get("/audit", response_model=AuditLogResponse)
async def get_audit_log(
    user_id: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    admin: Dict[str, Any] = Depends(require_admin),
    security_service: Any = Depends(get_security_service),
) -> AuditLogResponse:
    """Retrieve paginated audit log entries (admin only)."""
    try:
        offset = (page - 1) * page_size
        raw_entries, total = await security_service.query_audit_log(
            user_id=user_id, action=action, offset=offset, limit=page_size
        )
        entries = [
            AuditLogEntry(
                audit_id=e["audit_id"],
                user_id=e.get("user_id"),
                action=e.get("action", ""),
                resource=e.get("resource"),
                result=e.get("result", "unknown"),
                timestamp=e.get("timestamp", _now_iso()),
                metadata=e.get("metadata", {}),
            )
            for e in raw_entries
        ]
        return AuditLogResponse(entries=entries, total=total, page=page, page_size=page_size)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_audit_log failed")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /security/roles
# ---------------------------------------------------------------------------
@router.post("/roles", response_model=RoleInfo)
async def assign_role(
    payload: RoleAssignmentRequest,
    admin: Dict[str, Any] = Depends(require_admin),
    security_service: Any = Depends(get_security_service),
) -> RoleInfo:
    """Assign a role to a user (admin only)."""
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {payload.role}")
    try:
        result = await security_service.assign_role(
            user_id=payload.user_id, role=payload.role, assigned_by=admin.get("id")
        )
        return RoleInfo(
            user_id=payload.user_id,
            role=payload.role,
            assigned_at=result.get("assigned_at", _now_iso()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("assign_role failed")
        raise HTTPException(status_code=500, detail=f"Failed to assign role: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /security/roles/{user_id}
# ---------------------------------------------------------------------------
@router.get("/roles/{user_id}", response_model=RoleInfo)
async def get_user_role(
    user_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> RoleInfo:
    """Retrieve the role assigned to a given user."""
    try:
        result = await security_service.get_role(user_id=user_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No role found for user: {user_id}")
        return RoleInfo(
            user_id=user_id,
            role=result.get("role", "viewer"),
            assigned_at=result.get("assigned_at", _now_iso()),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_user_role failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve role: {exc}") from exc
