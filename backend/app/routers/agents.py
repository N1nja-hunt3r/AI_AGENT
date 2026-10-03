"""
agents.py

FastAPI router for agent lifecycle management: listing, spawning,
shutting down, status reporting, and task delegation between agents.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AgentInfo(BaseModel):
    agent_id: str
    name: str
    status: str
    capabilities: List[str] = Field(default_factory=list)
    parent_agent_id: Optional[str] = None
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentListResponse(BaseModel):
    agents: List[AgentInfo]
    total: int


class SpawnAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    agent_type: str = Field(default="generic")
    capabilities: List[str] = Field(default_factory=list)
    parent_agent_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class SpawnAgentResponse(BaseModel):
    agent: AgentInfo


class ShutdownResponse(BaseModel):
    agent_id: str
    status: str
    shutdown_at: str


class DelegateTaskRequest(BaseModel):
    from_agent_id: str
    to_agent_id: str
    task_description: str = Field(..., min_length=1, max_length=50_000)
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(default=0, ge=0, le=10)


class DelegateTaskResponse(BaseModel):
    delegation_id: str
    from_agent_id: str
    to_agent_id: str
    status: str
    created_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_agent_manager(request: Request) -> Any:
    manager = getattr(request.app.state, "agent_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Agent manager unavailable")
    return manager


def _to_agent_info(raw: Dict[str, Any]) -> AgentInfo:
    return AgentInfo(
        agent_id=raw["agent_id"],
        name=raw.get("name", ""),
        status=raw.get("status", "unknown"),
        capabilities=raw.get("capabilities", []),
        parent_agent_id=raw.get("parent_agent_id"),
        created_at=raw.get("created_at", _now_iso()),
        metadata=raw.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# GET /agents
# ---------------------------------------------------------------------------
@router.get("", response_model=AgentListResponse)
async def list_agents(
    status_filter: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
    agent_manager: Any = Depends(get_agent_manager),
) -> AgentListResponse:
    """List all active agents, optionally filtered by status."""
    try:
        raw_agents = await agent_manager.list_agents(
            user_id=user.get("id"), status_filter=status_filter
        )
        agents = [_to_agent_info(a) for a in raw_agents]
        return AgentListResponse(agents=agents, total=len(agents))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_agents failed")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /agents/{agent_id}
# ---------------------------------------------------------------------------
@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent_status(
    agent_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    agent_manager: Any = Depends(get_agent_manager),
) -> AgentInfo:
    """Retrieve current status and metadata for a single agent."""
    try:
        raw = await agent_manager.get_agent(agent_id=agent_id, user_id=user.get("id"))
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return _to_agent_info(raw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_agent_status failed for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent status: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /agents/spawn
# ---------------------------------------------------------------------------
@router.post("/spawn", response_model=SpawnAgentResponse, status_code=status.HTTP_201_CREATED)
async def spawn_agent(
    payload: SpawnAgentRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    agent_manager: Any = Depends(get_agent_manager),
) -> SpawnAgentResponse:
    """Spawn a new agent instance, optionally as a child of an existing agent."""
    try:
        raw = await agent_manager.spawn_agent(
            user_id=user.get("id"),
            name=payload.name,
            agent_type=payload.agent_type,
            capabilities=payload.capabilities,
            parent_agent_id=payload.parent_agent_id,
            config=payload.config or {},
            session_id=payload.session_id,
        )
        return SpawnAgentResponse(agent=_to_agent_info(raw))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("spawn_agent failed")
        raise HTTPException(status_code=500, detail=f"Failed to spawn agent: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /agents/{agent_id}/shutdown
# ---------------------------------------------------------------------------
@router.post("/{agent_id}/shutdown", response_model=ShutdownResponse)
async def shutdown_agent(
    agent_id: str,
    force: bool = False,
    user: Dict[str, Any] = Depends(get_current_user),
    agent_manager: Any = Depends(get_agent_manager),
) -> ShutdownResponse:
    """Gracefully (or forcefully) shut down a running agent."""
    try:
        result = await agent_manager.shutdown_agent(
            agent_id=agent_id, user_id=user.get("id"), force=force
        )
        if not result.get("found", True):
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return ShutdownResponse(
            agent_id=agent_id,
            status=result.get("status", "terminated"),
            shutdown_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("shutdown_agent failed for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to shut down agent: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /agents/delegate
# ---------------------------------------------------------------------------
@router.post("/delegate", response_model=DelegateTaskResponse, status_code=status.HTTP_201_CREATED)
async def delegate_task(
    payload: DelegateTaskRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    agent_manager: Any = Depends(get_agent_manager),
) -> DelegateTaskResponse:
    """Delegate a task from one agent to another."""
    if payload.from_agent_id == payload.to_agent_id:
        raise HTTPException(status_code=400, detail="Cannot delegate a task to the same agent")
    try:
        result = await agent_manager.delegate_task(
            user_id=user.get("id"),
            from_agent_id=payload.from_agent_id,
            to_agent_id=payload.to_agent_id,
            task_description=payload.task_description,
            context=payload.context or {},
            priority=payload.priority,
        )
        return DelegateTaskResponse(
            delegation_id=result["delegation_id"],
            from_agent_id=payload.from_agent_id,
            to_agent_id=payload.to_agent_id,
            status=result.get("status", "queued"),
            created_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("delegate_task failed")
        raise HTTPException(status_code=500, detail=f"Task delegation failed: {exc}") from exc
