"""
automation.py

FastAPI router for automation task lifecycle management: creation,
deletion, pausing, resuming, and status reporting.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automation"])

# ---------------------------------------------------------------------------
# In-memory fallback task store (when AutomationEngine lacks CRUD methods)
# ---------------------------------------------------------------------------
_tasks: Dict[str, Dict[str, Any]] = {}
_task_lock: threading.Lock = threading.Lock()


def _get_task_store(service: Any) -> Any:
    """Return the service if it supports CRUD, otherwise fall back to in-memory store."""
    if hasattr(service, "create_task") and callable(service.create_task):
        return service
    return None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AutomationTaskInfo(BaseModel):
    task_id: str
    name: str
    status: str
    schedule: Optional[str] = None
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateTaskRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    schedule: Optional[str] = Field(default=None, description="Cron expression, if recurring")
    payload: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class CreateTaskResponse(BaseModel):
    task: AutomationTaskInfo


class TaskListResponse(BaseModel):
    tasks: List[AutomationTaskInfo]
    total: int


class DeleteTaskResponse(BaseModel):
    task_id: str
    deleted: bool


class TaskActionResponse(BaseModel):
    task_id: str
    status: str
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_automation_service(request: Request) -> Any:
    service = getattr(request.app.state, "automation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Automation service unavailable")
    return service


def _to_task_info(raw: Dict[str, Any]) -> AutomationTaskInfo:
    return AutomationTaskInfo(
        task_id=raw["task_id"],
        name=raw.get("name", ""),
        status=raw.get("status", "unknown"),
        schedule=raw.get("schedule"),
        created_at=raw.get("created_at", _now_iso()),
        updated_at=raw.get("updated_at", _now_iso()),
        metadata=raw.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# POST /automation/tasks
# ---------------------------------------------------------------------------
@router.post("/tasks", response_model=CreateTaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: CreateTaskRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> CreateTaskResponse:
    """Create a new automation task, optionally recurring via cron schedule."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            raw = await store.create_task(
                user_id=user.get("id"),
                name=payload.name,
                description=payload.description,
                schedule=payload.schedule,
                payload=payload.payload,
                enabled=payload.enabled,
            )
        else:
            task_id = str(uuid.uuid4())
            now = _now_iso()
            raw = {
                "task_id": task_id,
                "name": payload.name,
                "status": "created",
                "schedule": payload.schedule,
                "created_at": now,
                "updated_at": now,
                "metadata": {"description": payload.description, "user_id": user.get("id"), "enabled": payload.enabled},
            }
            with _task_lock:
                _tasks[task_id] = raw
        return CreateTaskResponse(task=_to_task_info(raw))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("create_task failed")
        raise HTTPException(status_code=500, detail=f"Failed to create task: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /automation/tasks
# ---------------------------------------------------------------------------
@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status_filter: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> TaskListResponse:
    """List automation tasks, optionally filtered by status."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            raw_tasks = await store.list_tasks(
                user_id=user.get("id"), status_filter=status_filter
            )
        else:
            with _task_lock:
                raw_tasks = [
                    t for t in _tasks.values()
                    if t.get("metadata", {}).get("user_id") == user.get("id")
                    and (status_filter is None or t.get("status") == status_filter)
                ]
        tasks = [_to_task_info(t) for t in raw_tasks]
        return TaskListResponse(tasks=tasks, total=len(tasks))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_tasks failed")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /automation/tasks/{task_id}
# ---------------------------------------------------------------------------
@router.get("/tasks/{task_id}", response_model=AutomationTaskInfo)
async def get_task_status(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> AutomationTaskInfo:
    """Retrieve the current status of an automation task."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            raw = await store.get_task(task_id=task_id, user_id=user.get("id"))
        else:
            with _task_lock:
                raw = _tasks.get(task_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return _to_task_info(raw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_task_status failed for task_id=%s", task_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /automation/tasks/{task_id}
# ---------------------------------------------------------------------------
@router.delete("/tasks/{task_id}", response_model=DeleteTaskResponse)
async def delete_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> DeleteTaskResponse:
    """Permanently delete an automation task."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            deleted = await store.delete_task(task_id=task_id, user_id=user.get("id"))
        else:
            with _task_lock:
                deleted = task_id in _tasks
                if deleted:
                    del _tasks[task_id]
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return DeleteTaskResponse(task_id=task_id, deleted=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("delete_task failed for task_id=%s", task_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /automation/tasks/{task_id}/pause
# ---------------------------------------------------------------------------
@router.post("/tasks/{task_id}/pause", response_model=TaskActionResponse)
async def pause_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> TaskActionResponse:
    """Pause a running or scheduled automation task."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            result = await store.pause_task(task_id=task_id, user_id=user.get("id"))
        else:
            with _task_lock:
                task = _tasks.get(task_id)
                if task is None:
                    result = {"found": False}
                else:
                    task["status"] = "paused"
                    task["updated_at"] = _now_iso()
                    result = {"found": True, "status": "paused"}
        if not result.get("found", True):
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return TaskActionResponse(
            task_id=task_id, status=result.get("status", "paused"), updated_at=_now_iso()
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("pause_task failed for task_id=%s", task_id)
        raise HTTPException(status_code=500, detail=f"Failed to pause task: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /automation/tasks/{task_id}/resume
# ---------------------------------------------------------------------------
@router.post("/tasks/{task_id}/resume", response_model=TaskActionResponse)
async def resume_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    automation_service: Any = Depends(get_automation_service),
) -> TaskActionResponse:
    """Resume a previously paused automation task."""
    try:
        store = _get_task_store(automation_service)
        if store is not None:
            result = await store.resume_task(task_id=task_id, user_id=user.get("id"))
        else:
            with _task_lock:
                task = _tasks.get(task_id)
                if task is None:
                    result = {"found": False}
                else:
                    task["status"] = "running"
                    task["updated_at"] = _now_iso()
                    result = {"found": True, "status": "running"}
        if not result.get("found", True):
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return TaskActionResponse(
            task_id=task_id, status=result.get("status", "running"), updated_at=_now_iso()
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("resume_task failed for task_id=%s", task_id)
        raise HTTPException(status_code=500, detail=f"Failed to resume task: {exc}") from exc
