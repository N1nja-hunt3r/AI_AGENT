"""
health.py

FastAPI router exposing layered health checks: overall system status,
individual services, capability modules, agent fleet, and security
subsystem.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

SERVICE_STATE_ATTRS: Dict[str, str] = {
    "controller": "controller",
    "planner": "planner",
    "executor": "executor",
    "memory_capability": "memory_capability",
    "rag_service": "rag_service",
    "tool_registry": "tool_registry",
    "computer_service": "computer_service",
    "agent_manager": "agent_manager",
    "automation_service": "automation_service",
    "monitoring_service": "monitoring_service",
    "security_service": "security_service",
    "auth_service": "auth_service",
    "settings_service": "settings_service",
    "database": "database",
    "redis": "redis",
    "vectordb": "vectordb",
}

CAPABILITY_ATTRS: Dict[str, str] = {
    "memory": "memory_capability",
    "rag": "rag_service",
    "tools": "tool_registry",
    "computer_use": "computer_service",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ComponentHealth(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None
    latency_ms: Optional[float] = None


class OverallHealthResponse(BaseModel):
    status: str
    services: List[ComponentHealth]
    capabilities: List[ComponentHealth]
    agents: AgentsHealthResponse
    security: SecurityHealthResponse
    checked_at: str


class ServicesHealthResponse(BaseModel):
    status: str
    services: List[ComponentHealth]
    checked_at: str


class CapabilitiesHealthResponse(BaseModel):
    status: str
    capabilities: List[ComponentHealth]
    checked_at: str


class AgentsHealthResponse(BaseModel):
    status: str
    active_count: int
    error_count: int
    detail: Optional[str] = None
    checked_at: str


class SecurityHealthResponse(BaseModel):
    status: str
    detail: Optional[str] = None
    checked_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _check_component(request: Request, attr_name: str) -> ComponentHealth:
    import time

    service = getattr(request.app.state, attr_name, None)
    if service is None:
        return ComponentHealth(name=attr_name, status="unavailable", detail="Not configured")

    start = time.monotonic()
    healthy = True
    detail = None
    try:
        if hasattr(service, "health_check"):
            result = service.health_check()
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                if "status" in result:
                    healthy = result["status"] in ("healthy", "active", "degraded")
                    detail = result.get("detail")
                else:
                    healthy = all(
                        (v.value if hasattr(v, "value") else v) in ("healthy", "active", True)
                        for v in result.values()
                    ) if result else True
            else:
                healthy = bool(result)
        else:
            detail = "No health_check method; assumed healthy"
        latency_ms = (time.monotonic() - start) * 1000
        return ComponentHealth(
            name=attr_name,
            status="healthy" if healthy else "unhealthy",
            detail=detail,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start) * 1000
        logger.exception("Health check failed for component=%s", attr_name)
        return ComponentHealth(
            name=attr_name, status="unhealthy", detail=str(exc), latency_ms=round(latency_ms, 2)
        )


def _aggregate_status(components: List[ComponentHealth]) -> str:
    if not components:
        return "unknown"
    if any(c.status == "unhealthy" for c in components):
        return "degraded"
    if any(c.status == "unavailable" for c in components):
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("", response_model=OverallHealthResponse)
async def overall_health(request: Request) -> OverallHealthResponse:
    """Aggregate health across services, capabilities, agents, and security."""
    services = await asyncio.gather(
        *(_check_component(request, attr) for attr in SERVICE_STATE_ATTRS.values())
    )
    capabilities = await asyncio.gather(
        *(_check_component(request, attr) for attr in CAPABILITY_ATTRS.values())
    )

    agents_health = await _agents_health_internal(request)
    security_health = await _security_health_internal(request)

    overall_status = _aggregate_status(list(services) + list(capabilities) + [
        ComponentHealth(name="agents", status=agents_health.status),
        ComponentHealth(name="security", status=security_health.status),
    ])

    return OverallHealthResponse(
        status=overall_status,
        services=list(services),
        capabilities=list(capabilities),
        agents=agents_health,
        security=security_health,
        checked_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# GET /health/services
# ---------------------------------------------------------------------------
@router.get("/services", response_model=ServicesHealthResponse)
async def services_health(request: Request) -> ServicesHealthResponse:
    """Report health for all backend services."""
    services = await asyncio.gather(
        *(_check_component(request, attr) for attr in SERVICE_STATE_ATTRS.values())
    )
    return ServicesHealthResponse(
        status=_aggregate_status(list(services)), services=list(services), checked_at=_now_iso()
    )


# ---------------------------------------------------------------------------
# GET /health/capabilities
# ---------------------------------------------------------------------------
@router.get("/capabilities", response_model=CapabilitiesHealthResponse)
async def capabilities_health(request: Request) -> CapabilitiesHealthResponse:
    """Report health for capability modules (memory, RAG, tools, computer-use)."""
    capabilities = await asyncio.gather(
        *(_check_component(request, attr) for attr in CAPABILITY_ATTRS.values())
    )
    return CapabilitiesHealthResponse(
        status=_aggregate_status(list(capabilities)),
        capabilities=list(capabilities),
        checked_at=_now_iso(),
    )


async def _agents_health_internal(request: Request) -> AgentsHealthResponse:
    agent_manager = getattr(request.app.state, "agent_manager", None)
    if agent_manager is None:
        return AgentsHealthResponse(
            status="unavailable", active_count=0, error_count=0,
            detail="Agent manager not configured", checked_at=_now_iso(),
        )
    try:
        if hasattr(agent_manager, "list_agents"):
            agents = agent_manager.list_agents()
            if asyncio.iscoroutine(agents):
                agents = await agents
        else:
            agents = []
        error_count = 0
        for a in agents:
            if hasattr(a, "state") and hasattr(a.state, "value") and a.state.value == "error":
                error_count += 1
            elif isinstance(a, dict) and a.get("status") == "error":
                error_count += 1
        status_value = "degraded" if error_count > 0 else "healthy"
        return AgentsHealthResponse(
            status=status_value, active_count=len(agents), error_count=error_count, checked_at=_now_iso()
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents_health check failed")
        return AgentsHealthResponse(
            status="unhealthy", active_count=0, error_count=0, detail=str(exc), checked_at=_now_iso()
        )


@router.get("/agents", response_model=AgentsHealthResponse)
async def agents_health(request: Request) -> AgentsHealthResponse:
    """Report aggregate health of the running agent fleet."""
    return await _agents_health_internal(request)


async def _security_health_internal(request: Request) -> SecurityHealthResponse:
    security_service = getattr(request.app.state, "security_service", None)
    if security_service is None:
        return SecurityHealthResponse(
            status="unavailable", detail="Security service not configured", checked_at=_now_iso()
        )
    try:
        if hasattr(security_service, "health_check"):
            result = security_service.health_check()
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                healthy = result.get("status") in ("healthy", "active", "degraded")
            else:
                healthy = bool(result)
        else:
            healthy = True
        return SecurityHealthResponse(
            status="healthy" if healthy else "unhealthy", checked_at=_now_iso()
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("security_health check failed")
        return SecurityHealthResponse(status="unhealthy", detail=str(exc), checked_at=_now_iso())


@router.get("/security", response_model=SecurityHealthResponse)
async def security_health(request: Request) -> SecurityHealthResponse:
    """Report health of the security subsystem."""
    return await _security_health_internal(request)


# ---------------------------------------------------------------------------
# GET /health/live, /health/ready (standard k8s-style probes)
# ---------------------------------------------------------------------------
@router.get("/live")
async def liveness_probe() -> Dict[str, str]:
    """Simple liveness probe; returns 200 if the process is running."""
    return {"status": "alive", "checked_at": _now_iso()}


@router.get("/ready")
async def readiness_probe(request: Request) -> Dict[str, Any]:
    """Readiness probe; fails if critical services (DB, Redis) are unavailable."""
    critical_attrs = ["database", "redis"]
    results = await asyncio.gather(*(_check_component(request, attr) for attr in critical_attrs))
    not_ready = [r for r in results if r.status != "healthy"]
    if not_ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "failed_components": [r.name for r in not_ready]},
        )
    return {"status": "ready", "checked_at": _now_iso()}
