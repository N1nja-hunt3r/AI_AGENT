"""
memory.py

FastAPI router for managing agent/session memories: create, list,
search, and delete, with pagination support.

Primary endpoints at /api/v1/memories/* (matching frontend expectations).
Backward-compatible endpoints at /api/v1/memories/memory/*.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Router loader uses this to set the prefix instead of the default /api/v1/memory
PREFIX = "/api/v1/memories"

router = APIRouter(tags=["memory"])


# ---------------------------------------------------------------------------
# Legacy Schemas (backward compatibility)
# ---------------------------------------------------------------------------
class MemoryCreateRequest(BaseModel):
    session_id: str
    content: str = Field(..., min_length=1, max_length=100_000)
    memory_type: str = Field(default="short_term")
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = Field(default=None, ge=1)


class MemoryItem(BaseModel):
    memory_id: str
    session_id: str
    user_id: Optional[str] = None
    content: str
    memory_type: str
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    score: Optional[float] = None


class MemoryListResponse(BaseModel):
    items: List[MemoryItem]
    total: int
    page: int
    page_size: int
    has_more: bool


class MemorySearchRequest(BaseModel):
    session_id: Optional[str] = None
    query: str = Field(..., min_length=1, max_length=10_000)
    memory_type: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class MemoryDeleteResponse(BaseModel):
    deleted: bool
    memory_id: str


class BulkDeleteRequest(BaseModel):
    session_id: Optional[str] = None
    memory_ids: Optional[List[str]] = None
    delete_all_for_session: bool = False


class BulkDeleteResponse(BaseModel):
    deleted_count: int


# ---------------------------------------------------------------------------
# New Schemas (frontend API)
# ---------------------------------------------------------------------------
class CreateMemoryRequest(BaseModel):
    type: str
    content: str = Field(..., min_length=1, max_length=100_000)
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    conversationId: Optional[str] = None
    agentId: Optional[str] = None
    expiresAt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateMemoryRequest(BaseModel):
    content: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    expiresAt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    type: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    threshold: Optional[float] = None
    agentId: Optional[str] = None
    conversationId: Optional[str] = None


class PurgeRequest(BaseModel):
    type: str


class FrontendMemoryItem(BaseModel):
    id: str
    type: str
    status: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    conversationId: Optional[str] = None
    agentId: Optional[str] = None
    createdAt: str
    updatedAt: str
    expiresAt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    memory: FrontendMemoryItem


class MemoryListResponseFE(BaseModel):
    memories: List[FrontendMemoryItem]
    total: int
    page: int
    limit: int


class SearchResult(BaseModel):
    memory: FrontendMemoryItem
    similarity: float
    relevanceScore: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


class DeleteMemoryResponseFE(BaseModel):
    success: bool
    id: str
    deletedAt: str


class PurgeResponse(BaseModel):
    success: bool
    purgedCount: int
    purgedAt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _compute_ttl_seconds(expires_at_str: str | None) -> int | None:
    """Compute ttl_seconds from an ISO-8601 expiration timestamp."""
    if expires_at_str is None:
        return None
    dt = _parse_datetime(expires_at_str)
    if dt is None:
        return None
    delta = int((dt - datetime.now(timezone.utc)).total_seconds())
    return max(delta, 1) if delta > 0 else None


def _record_to_frontend(
    record: dict[str, Any],
    score: float | None = None,
) -> FrontendMemoryItem:
    """Convert a legacy memory record dict to the frontend MemoryItem schema."""
    return FrontendMemoryItem(
        id=record.get("memory_id", record.get("id", "")),
        type=record.get("memory_type", record.get("type", "short_term")),
        status=record.get("status", "active"),
        content=record.get("content", ""),
        summary=record.get("summary") or (record.get("metadata") or {}).get("summary"),
        tags=list(record.get("tags", [])),
        score=score if score is not None else record.get("score") or record.get("confidence"),
        conversationId=record.get("session_id") or record.get("conversationId"),
        agentId=record.get("agent_id") or record.get("agentId"),
        createdAt=record.get("created_at", record.get("createdAt", _now_iso())),
        updatedAt=record.get("updated_at", record.get("updatedAt", record.get("created_at", _now_iso()))),
        expiresAt=record.get("expires_at", record.get("expiresAt")),
        metadata=record.get("metadata"),
    )


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_memory_capability(request: Request) -> Any:
    memory = getattr(request.app.state, "memory_capability", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory capability unavailable")
    return memory


# ===========================================================================
# NEW FRONTEND-FACING ENDPOINTS  (under /api/v1/memories)
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------
@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_frontend_memory(
    payload: CreateMemoryRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryResponse:
    """Create a new memory entry (frontend API)."""
    try:
        merged_metadata: dict[str, Any] = dict(payload.metadata or {})
        if payload.summary:
            merged_metadata["summary"] = payload.summary
        if payload.agentId:
            merged_metadata["agent_id"] = payload.agentId
        if payload.tags:
            merged_metadata["tags"] = payload.tags

        record = await memory_capability.store(
            session_id=payload.conversationId or user.get("id"),
            user_id=user.get("id"),
            content=payload.content,
            memory_type=payload.type,
            role=None,
            metadata=merged_metadata,
            ttl_seconds=_compute_ttl_seconds(payload.expiresAt),
        )
        return MemoryResponse(
            memory=_record_to_frontend(
                record if isinstance(record, dict) else record.to_dict() if hasattr(record, "to_dict") else {"memory_id": str(record)},
            )
        )
    except Exception as exc:
        logger.exception("create_frontend_memory failed")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get("", response_model=MemoryListResponseFE)
async def list_frontend_memories(
    type: Optional[str] = Query(default=None, alias="type"),
    status: Optional[str] = Query(default=None),
    tags: Optional[str] = Query(default=None),
    agentId: Optional[str] = Query(default=None),
    conversationId: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryListResponseFE:
    """List memories with optional filtering and pagination (frontend API)."""
    try:
        offset = (page - 1) * limit
        records, total = await memory_capability.list(
            user_id=user.get("id"),
            session_id=conversationId,
            memory_type=type,
            offset=offset,
            limit=limit,
        )
        memories = [_record_to_frontend(r) for r in records]
        return MemoryListResponseFE(
            memories=memories,
            total=total,
            page=page,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("list_frontend_memories failed")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /{id}
# ---------------------------------------------------------------------------
@router.get("/{id}", response_model=MemoryResponse)
async def get_frontend_memory(
    id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryResponse:
    """Get a single memory by ID (frontend API)."""
    try:
        record = await memory_capability.retrieve(memory_id=id, user_id=user.get("id"))
        if record is None:
            raise HTTPException(status_code=404, detail=f"Memory not found: {id}")
        record_dict = record if isinstance(record, dict) else record.to_dict() if hasattr(record, "to_dict") else {"memory_id": id}
        if "memory_id" not in record_dict and "id" in record_dict:
            record_dict["memory_id"] = record_dict["id"]
        return MemoryResponse(memory=_record_to_frontend(record_dict))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_frontend_memory failed for id=%s", id)
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {exc}") from exc


# ---------------------------------------------------------------------------
# PATCH /{id}
# ---------------------------------------------------------------------------
@router.patch("/{id}", response_model=MemoryResponse)
async def update_frontend_memory(
    id: str,
    payload: UpdateMemoryRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryResponse:
    """Update a memory entry (frontend API)."""
    try:
        existing = await memory_capability.retrieve(memory_id=id, user_id=user.get("id"))
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Memory not found: {id}")

        existing_dict = existing if isinstance(existing, dict) else existing.to_dict() if hasattr(existing, "to_dict") else {}

        merged_metadata: dict[str, Any] = dict(existing_dict.get("metadata", {}) or {})
        if payload.summary is not None:
            merged_metadata["summary"] = payload.summary
        if payload.tags is not None:
            merged_metadata["tags"] = payload.tags
        if payload.status is not None:
            merged_metadata["status"] = payload.status
        if payload.metadata is not None:
            merged_metadata.update(payload.metadata)

        update_params: dict[str, Any] = {
            "entry_id": id,
            "user_id": user.get("id"),
            "metadata": merged_metadata,
        }
        if payload.content is not None:
            update_params["content"] = payload.content
        if payload.expiresAt is not None:
            update_params["expires_at"] = payload.expiresAt

        updated = await memory_capability.update(**update_params)
        updated_dict = updated if isinstance(updated, dict) else updated.to_dict() if hasattr(updated, "to_dict") else {"memory_id": id}

        return MemoryResponse(memory=_record_to_frontend(updated_dict))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("update_frontend_memory failed for id=%s", id)
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /purge  (must be before /{id} to avoid path conflict)
# ---------------------------------------------------------------------------
@router.delete("/purge", response_model=PurgeResponse)
async def purge_frontend_memories(
    payload: PurgeRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> PurgeResponse:
    """Purge memories by type (frontend API)."""
    try:
        count = await memory_capability.delete_session(
            session_id=payload.type,
            user_id=user.get("id"),
        )
        return PurgeResponse(success=True, purgedCount=count, purgedAt=_now_iso())
    except Exception as exc:
        logger.exception("purge_frontend_memories failed")
        raise HTTPException(status_code=500, detail=f"Purge failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /{id}
# ---------------------------------------------------------------------------
@router.delete("/{id}", response_model=DeleteMemoryResponseFE)
async def delete_frontend_memory(
    id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> DeleteMemoryResponseFE:
    """Delete a single memory entry (frontend API)."""
    try:
        deleted = await memory_capability.delete(memory_id=id, user_id=user.get("id"))
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory not found: {id}")
        return DeleteMemoryResponseFE(success=True, id=id, deletedAt=_now_iso())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("delete_frontend_memory failed for id=%s", id)
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
async def search_frontend_memories(
    payload: SearchMemoryRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> SearchResponse:
    """Semantically search memories (frontend API)."""
    try:
        records = await memory_capability.search(
            user_id=user.get("id"),
            session_id=payload.conversationId,
            query=payload.query,
            memory_type=payload.type,
            top_k=payload.limit,
        )
        results: list[SearchResult] = []
        for r in records:
            if isinstance(r, tuple):
                entry, score = r
                entry_dict = entry.to_dict() if hasattr(entry, "to_dict") else {}
            elif isinstance(r, dict):
                entry_dict = r
                score = r.get("score", 0.0)
            else:
                entry_dict = {}
                score = 0.0
            memory_item = _record_to_frontend(entry_dict, score=score)
            results.append(SearchResult(memory=memory_item, similarity=score, relevanceScore=score))
        return SearchResponse(results=results, total=len(results), query=payload.query)
    except Exception as exc:
        logger.exception("search_frontend_memories failed")
        raise HTTPException(status_code=500, detail=f"Memory search failed: {exc}") from exc


# ===========================================================================
# BACKWARD-COMPATIBLE ENDPOINTS  (under /api/v1/memories/memory)
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /memory
# ---------------------------------------------------------------------------
@router.post("/memory", response_model=MemoryItem, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryItem:
    """Persist a new memory entry (legacy)."""
    try:
        record = await memory_capability.store(
            session_id=payload.session_id,
            user_id=user.get("id"),
            content=payload.content,
            memory_type=payload.memory_type,
            role=payload.role,
            metadata=payload.metadata or {},
            ttl_seconds=payload.ttl_seconds,
        )
        return MemoryItem(
            memory_id=record["memory_id"],
            session_id=payload.session_id,
            user_id=user.get("id"),
            content=payload.content,
            memory_type=payload.memory_type,
            role=payload.role,
            metadata=payload.metadata or {},
            created_at=record.get("created_at", _now_iso()),
        )
    except Exception as exc:
        logger.exception("create_memory failed")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /memory
# ---------------------------------------------------------------------------
@router.get("/memory", response_model=MemoryListResponse)
async def list_memories(
    session_id: Optional[str] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryListResponse:
    """List memories with optional filtering and pagination (legacy)."""
    try:
        offset = (page - 1) * page_size
        records, total = await memory_capability.list(
            user_id=user.get("id"),
            session_id=session_id,
            memory_type=memory_type,
            offset=offset,
            limit=page_size,
        )
        items = [
            MemoryItem(
                memory_id=r["memory_id"],
                session_id=r["session_id"],
                user_id=r.get("user_id"),
                content=r["content"],
                memory_type=r.get("memory_type", "short_term"),
                role=r.get("role"),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at", _now_iso()),
            )
            for r in records
        ]
        return MemoryListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=offset + len(items) < total,
        )
    except Exception as exc:
        logger.exception("list_memories failed")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /memory/search
# ---------------------------------------------------------------------------
@router.post("/memory/search", response_model=MemoryListResponse)
async def search_memories(
    payload: MemorySearchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryListResponse:
    """Semantically search memories relevant to a query (legacy)."""
    try:
        records = await memory_capability.search(
            user_id=user.get("id"),
            session_id=payload.session_id,
            query=payload.query,
            memory_type=payload.memory_type,
            top_k=payload.top_k,
        )
        items = [
            MemoryItem(
                memory_id=r["memory_id"],
                session_id=r["session_id"],
                user_id=r.get("user_id"),
                content=r["content"],
                memory_type=r.get("memory_type", "short_term"),
                role=r.get("role"),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at", _now_iso()),
                score=r.get("score"),
            )
            for r in records
        ]
        return MemoryListResponse(
            items=items,
            total=len(items),
            page=1,
            page_size=payload.top_k,
            has_more=False,
        )
    except Exception as exc:
        logger.exception("search_memories failed")
        raise HTTPException(status_code=500, detail=f"Memory search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /memory/{memory_id}
# ---------------------------------------------------------------------------
@router.delete("/memory/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> MemoryDeleteResponse:
    """Delete a single memory entry by id (legacy)."""
    try:
        deleted = await memory_capability.delete(memory_id=memory_id, user_id=user.get("id"))
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        return MemoryDeleteResponse(deleted=True, memory_id=memory_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("delete_memory failed for memory_id=%s", memory_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /memory/bulk-delete
# ---------------------------------------------------------------------------
@router.post("/memory/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_memories(
    payload: BulkDeleteRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    memory_capability: Any = Depends(get_memory_capability),
) -> BulkDeleteResponse:
    """Delete multiple memories by id, or all memories for a session (legacy)."""
    if not payload.memory_ids and not payload.delete_all_for_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide memory_ids or set delete_all_for_session=true",
        )
    try:
        if payload.delete_all_for_session:
            if not payload.session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            count = await memory_capability.delete_session(
                session_id=payload.session_id, user_id=user.get("id")
            )
        else:
            count = await memory_capability.delete_many(
                memory_ids=payload.memory_ids or [], user_id=user.get("id")
            )
        return BulkDeleteResponse(deleted_count=count)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("bulk_delete_memories failed")
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {exc}") from exc
