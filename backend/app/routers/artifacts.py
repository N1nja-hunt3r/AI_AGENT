"""
artifacts.py

FastAPI router for artifact storage: upload, download, deletion, and
version history management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["artifacts"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ArtifactVersion(BaseModel):
    version: int
    size_bytes: int
    checksum: str
    created_at: str
    created_by: Optional[str] = None


class ArtifactMetadata(BaseModel):
    artifact_id: str
    name: str
    content_type: str
    current_version: int
    versions: List[ArtifactVersion] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ArtifactListResponse(BaseModel):
    artifacts: List[ArtifactMetadata]
    total: int
    page: int
    page_size: int


class UploadResponse(BaseModel):
    artifact: ArtifactMetadata


class DeleteResponse(BaseModel):
    artifact_id: str
    deleted: bool
    version: Optional[int] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_artifact_service(request: Request) -> Any:
    service = getattr(request.app.state, "artifact_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Artifact service unavailable")
    return service


def _to_metadata(raw: Dict[str, Any]) -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_id=raw["artifact_id"],
        name=raw.get("name", ""),
        content_type=raw.get("content_type", "application/octet-stream"),
        current_version=raw.get("current_version", 1),
        versions=[
            ArtifactVersion(
                version=v["version"],
                size_bytes=v.get("size_bytes", 0),
                checksum=v.get("checksum", ""),
                created_at=v.get("created_at", _now_iso()),
                created_by=v.get("created_by"),
            )
            for v in raw.get("versions", [])
        ],
        tags=raw.get("tags", []),
        created_at=raw.get("created_at", _now_iso()),
        updated_at=raw.get("updated_at", _now_iso()),
    )


# ---------------------------------------------------------------------------
# POST /artifacts/upload
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_artifact(
    name: Optional[str] = Query(default=None),
    artifact_id: Optional[str] = Query(default=None, description="Provide to create a new version of an existing artifact"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
    artifact_service: Any = Depends(get_artifact_service),
) -> UploadResponse:
    """Upload a new artifact, or a new version of an existing artifact."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    try:
        if artifact_id:
            raw = await artifact_service.add_version(
                artifact_id=artifact_id,
                user_id=user.get("id"),
                content=content,
                content_type=file.content_type or "application/octet-stream",
            )
            if raw is None:
                raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
        else:
            raw = await artifact_service.create_artifact(
                user_id=user.get("id"),
                name=name or file.filename or "artifact",
                content=content,
                content_type=file.content_type or "application/octet-stream",
                tags=tag_list,
            )
        return UploadResponse(artifact=_to_metadata(raw))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload_artifact failed")
        raise HTTPException(status_code=500, detail=f"Failed to upload artifact: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /artifacts
# ---------------------------------------------------------------------------
@router.get("", response_model=ArtifactListResponse)
async def list_artifacts(
    tag: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user),
    artifact_service: Any = Depends(get_artifact_service),
) -> ArtifactListResponse:
    """List artifacts owned by the current user, optionally filtered by tag."""
    try:
        offset = (page - 1) * page_size
        raw_artifacts, total = await artifact_service.list_artifacts(
            user_id=user.get("id"), tag=tag, offset=offset, limit=page_size
        )
        artifacts = [_to_metadata(a) for a in raw_artifacts]
        return ArtifactListResponse(artifacts=artifacts, total=total, page=page, page_size=page_size)
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_artifacts failed")
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /artifacts/{artifact_id}
# ---------------------------------------------------------------------------
@router.get("/{artifact_id}", response_model=ArtifactMetadata)
async def get_artifact_metadata(
    artifact_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    artifact_service: Any = Depends(get_artifact_service),
) -> ArtifactMetadata:
    """Retrieve metadata and version history for an artifact."""
    try:
        raw = await artifact_service.get_metadata(artifact_id=artifact_id, user_id=user.get("id"))
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
        return _to_metadata(raw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_artifact_metadata failed for artifact_id=%s", artifact_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve artifact metadata: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /artifacts/{artifact_id}/download
# ---------------------------------------------------------------------------
@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    version: Optional[int] = Query(default=None, description="Specific version; defaults to latest"),
    user: Dict[str, Any] = Depends(get_current_user),
    artifact_service: Any = Depends(get_artifact_service),
) -> StreamingResponse:
    """Download an artifact's content, optionally at a specific version."""
    try:
        result = await artifact_service.get_content(
            artifact_id=artifact_id, user_id=user.get("id"), version=version
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        async def _stream() -> Any:
            yield result["content"]

        filename = result.get("filename", artifact_id)
        return StreamingResponse(
            _stream(),
            media_type=result.get("content_type", "application/octet-stream"),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("download_artifact failed for artifact_id=%s", artifact_id)
        raise HTTPException(status_code=500, detail=f"Failed to download artifact: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /artifacts/{artifact_id}
# ---------------------------------------------------------------------------
@router.delete("/{artifact_id}", response_model=DeleteResponse)
async def delete_artifact(
    artifact_id: str,
    version: Optional[int] = Query(default=None, description="Delete a single version; omit to delete all"),
    user: Dict[str, Any] = Depends(get_current_user),
    artifact_service: Any = Depends(get_artifact_service),
) -> DeleteResponse:
    """Delete an artifact entirely, or a specific version of it."""
    try:
        deleted = await artifact_service.delete_artifact(
            artifact_id=artifact_id, user_id=user.get("id"), version=version
        )
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
        return DeleteResponse(artifact_id=artifact_id, deleted=True, version=version)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("delete_artifact failed for artifact_id=%s", artifact_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete artifact: {exc}") from exc
