"""
tools.py

FastAPI router for tool discovery, execution, and health checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    requires_approval: bool = False


class ToolListResponse(BaseModel):
    tools: List[ToolSchema]
    total: int


class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)


class ToolExecuteResponse(BaseModel):
    tool_name: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float
    executed_at: str


class ToolHealthItem(BaseModel):
    tool_name: str
    status: str
    detail: Optional[str] = None


class ToolHealthResponse(BaseModel):
    status: str
    tools: List[ToolHealthItem]
    checked_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_tool_registry(request: Request) -> Any:
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Tool registry unavailable")
    return registry


# ---------------------------------------------------------------------------
# GET /tools
# ---------------------------------------------------------------------------
@router.get("", response_model=ToolListResponse)
async def list_tools(
    tag: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
    tool_registry: Any = Depends(get_tool_registry),
) -> ToolListResponse:
    """List all registered tools, optionally filtered by tag."""
    try:
        raw_tools = await tool_registry.list_tools(tag=tag)
        tools = [
            ToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
                version=t.get("version", "1.0.0"),
                tags=t.get("tags", []),
                requires_approval=t.get("requires_approval", False),
            )
            for t in raw_tools
        ]
        return ToolListResponse(tools=tools, total=len(tools))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_tools failed")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /tools/{tool_name}
# ---------------------------------------------------------------------------
@router.get("/{tool_name}", response_model=ToolSchema)
async def get_tool(
    tool_name: str,
    user: Dict[str, Any] = Depends(get_current_user),
    tool_registry: Any = Depends(get_tool_registry),
) -> ToolSchema:
    """Retrieve the schema and metadata for a single tool."""
    try:
        tool = await tool_registry.get_tool(tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
        return ToolSchema(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}),
            version=tool.get("version", "1.0.0"),
            tags=tool.get("tags", []),
            requires_approval=tool.get("requires_approval", False),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_tool failed for tool_name=%s", tool_name)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tool: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /tools/execute
# ---------------------------------------------------------------------------
@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(
    payload: ToolExecuteRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    tool_registry: Any = Depends(get_tool_registry),
) -> ToolExecuteResponse:
    """Execute a registered tool with the given arguments."""
    import time

    tool = await tool_registry.get_tool(payload.tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {payload.tool_name}")

    if tool.get("requires_approval"):
        approval_service = getattr(request.app.state, "security_service", None)
        approved = False
        if approval_service is not None:
            approved = await approval_service.is_approved(
                user_id=user.get("id"),
                action=f"tool:{payload.tool_name}",
                session_id=payload.session_id,
            )
        if not approved:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tool '{payload.tool_name}' requires approval before execution",
            )

    start = time.monotonic()
    try:
        import asyncio

        output = await asyncio.wait_for(
            tool_registry.execute(
                tool_name=payload.tool_name,
                arguments=payload.arguments,
                user_id=user.get("id"),
                session_id=payload.session_id,
            ),
            timeout=payload.timeout_seconds,
        )
        duration_ms = (time.monotonic() - start) * 1000
        return ToolExecuteResponse(
            tool_name=payload.tool_name,
            status="succeeded",
            output=output,
            error=None,
            duration_ms=round(duration_ms, 2),
            executed_at=_now_iso(),
        )
    except asyncio.TimeoutError:
        duration_ms = (time.monotonic() - start) * 1000
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "tool_name": payload.tool_name,
                "status": "timed_out",
                "error": f"Execution exceeded {payload.timeout_seconds}s",
                "duration_ms": round(duration_ms, 2),
                "executed_at": _now_iso(),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.monotonic() - start) * 1000
        logger.exception("execute_tool failed for tool_name=%s", payload.tool_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "tool_name": payload.tool_name,
                "status": "failed",
                "output": None,
                "error": str(exc),
                "duration_ms": round(duration_ms, 2),
                "executed_at": _now_iso(),
            },
        )


# ---------------------------------------------------------------------------
# GET /tools/health
# ---------------------------------------------------------------------------
@router.get("/health/check", response_model=ToolHealthResponse)
async def tools_health_check(
    tool_registry: Any = Depends(get_tool_registry),
) -> ToolHealthResponse:
    """Run health checks across all registered tools."""
    try:
        report = await tool_registry.health_check()
        items = [
            ToolHealthItem(
                tool_name=name,
                status=detail.get("status", "unknown"),
                detail=detail.get("error"),
            )
            for name, detail in report.get("details", {}).items()
        ]
        return ToolHealthResponse(
            status=report.get("status", "unknown"),
            tools=items,
            checked_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("tools_health_check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}") from exc
