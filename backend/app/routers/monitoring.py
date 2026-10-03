"""
monitoring.py

FastAPI router exposing observability endpoints: metrics, dashboard
summaries, log queries, profiler snapshots, and alert management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DashboardSummary(BaseModel):
    active_agents: int
    active_sessions: int
    requests_per_minute: float
    error_rate: float
    avg_latency_ms: float
    cost_today_usd: float
    generated_at: str


class LogEntry(BaseModel):
    timestamp: str
    level: str
    logger: str
    message: str
    request_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class LogQueryResponse(BaseModel):
    entries: List[LogEntry]
    total: int
    page: int
    page_size: int


class ProfilerSnapshot(BaseModel):
    component: str
    cpu_percent: float
    memory_mb: float
    open_connections: int
    captured_at: str


class Alert(BaseModel):
    alert_id: str
    severity: str
    title: str
    description: str
    component: str
    triggered_at: str
    resolved: bool = False
    resolved_at: Optional[str] = None


class AlertListResponse(BaseModel):
    alerts: List[Alert]
    total: int


class AlertResolveResponse(BaseModel):
    alert_id: str
    resolved: bool
    resolved_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    role = user.get("role")
    if role not in {"admin", "owner"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user


def get_monitoring_service(request: Request) -> Any:
    service = getattr(request.app.state, "monitoring_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Monitoring service unavailable")
    return service


# ---------------------------------------------------------------------------
# GET /monitoring/metrics (Prometheus exposition format)
# ---------------------------------------------------------------------------
@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics(
    monitoring_service: Any = Depends(get_monitoring_service),
) -> str:
    """Expose metrics in Prometheus text exposition format."""
    try:
        return await monitoring_service.export_prometheus_metrics()
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_metrics failed")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /monitoring/dashboard
# ---------------------------------------------------------------------------
@router.get("/dashboard", response_model=DashboardSummary)
async def get_dashboard(
    user: Dict[str, Any] = Depends(get_current_user),
    monitoring_service: Any = Depends(get_monitoring_service),
) -> DashboardSummary:
    """Return a summarized snapshot of system activity for dashboards."""
    try:
        data = await monitoring_service.get_dashboard_summary()
        return DashboardSummary(
            active_agents=data.get("active_agents", 0),
            active_sessions=data.get("active_sessions", 0),
            requests_per_minute=data.get("requests_per_minute", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            cost_today_usd=data.get("cost_today_usd", 0.0),
            generated_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_dashboard failed")
        raise HTTPException(status_code=500, detail=f"Failed to build dashboard: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /monitoring/logs
# ---------------------------------------------------------------------------
@router.get("/logs", response_model=LogQueryResponse)
async def query_logs(
    level: Optional[str] = Query(default=None),
    request_id: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    user: Dict[str, Any] = Depends(require_admin),
    monitoring_service: Any = Depends(get_monitoring_service),
) -> LogQueryResponse:
    """Query recent log entries with optional filtering and pagination."""
    try:
        offset = (page - 1) * page_size
        raw_entries, total = await monitoring_service.query_logs(
            level=level, request_id=request_id, search=search, offset=offset, limit=page_size
        )
        entries = [
            LogEntry(
                timestamp=e.get("timestamp", _now_iso()),
                level=e.get("level", "INFO"),
                logger=e.get("logger", ""),
                message=e.get("message", ""),
                request_id=e.get("request_id"),
                extra=e.get("extra", {}),
            )
            for e in raw_entries
        ]
        return LogQueryResponse(entries=entries, total=total, page=page, page_size=page_size)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query_logs failed")
        raise HTTPException(status_code=500, detail=f"Failed to query logs: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /monitoring/profiler
# ---------------------------------------------------------------------------
@router.get("/profiler", response_model=List[ProfilerSnapshot])
async def get_profiler_snapshot(
    component: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(require_admin),
    monitoring_service: Any = Depends(get_monitoring_service),
) -> List[ProfilerSnapshot]:
    """Return current resource-usage profiler snapshots per component."""
    try:
        raw_snapshots = await monitoring_service.get_profiler_snapshots(component=component)
        return [
            ProfilerSnapshot(
                component=s["component"],
                cpu_percent=s.get("cpu_percent", 0.0),
                memory_mb=s.get("memory_mb", 0.0),
                open_connections=s.get("open_connections", 0),
                captured_at=_now_iso(),
            )
            for s in raw_snapshots
        ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_profiler_snapshot failed")
        raise HTTPException(status_code=500, detail=f"Failed to capture profiler snapshot: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /monitoring/alerts
# ---------------------------------------------------------------------------
@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(
    severity: Optional[str] = Query(default=None),
    resolved: Optional[bool] = Query(default=None),
    user: Dict[str, Any] = Depends(require_admin),
    monitoring_service: Any = Depends(get_monitoring_service),
) -> AlertListResponse:
    """List active and historical alerts, optionally filtered."""
    try:
        raw_alerts = await monitoring_service.list_alerts(severity=severity, resolved=resolved)
        alerts = [
            Alert(
                alert_id=a["alert_id"],
                severity=a.get("severity", "info"),
                title=a.get("title", ""),
                description=a.get("description", ""),
                component=a.get("component", ""),
                triggered_at=a.get("triggered_at", _now_iso()),
                resolved=a.get("resolved", False),
                resolved_at=a.get("resolved_at"),
            )
            for a in raw_alerts
        ]
        return AlertListResponse(alerts=alerts, total=len(alerts))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_alerts failed")
        raise HTTPException(status_code=500, detail=f"Failed to list alerts: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /monitoring/alerts/{alert_id}/resolve
# ---------------------------------------------------------------------------
@router.post("/alerts/{alert_id}/resolve", response_model=AlertResolveResponse)
async def resolve_alert(
    alert_id: str,
    user: Dict[str, Any] = Depends(require_admin),
    monitoring_service: Any = Depends(get_monitoring_service),
) -> AlertResolveResponse:
    """Mark an alert as resolved."""
    try:
        result = await monitoring_service.resolve_alert(alert_id=alert_id, resolved_by=user.get("id"))
        if not result.get("found", True):
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
        return AlertResolveResponse(alert_id=alert_id, resolved=True, resolved_at=_now_iso())
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("resolve_alert failed for alert_id=%s", alert_id)
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {exc}") from exc
