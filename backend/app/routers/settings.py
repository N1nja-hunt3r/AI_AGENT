"""
settings.py

FastAPI router for user/application-level runtime settings: model
preferences, memory configuration, RAG configuration, and feature
flag toggles.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ModelSettings(BaseModel):
    provider: str = Field(default="nvidia_nim")
    model: str = Field(default="deepseek-ai/deepseek-v4-pro")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200_000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class MemorySettings(BaseModel):
    enabled: bool = True
    max_items: int = Field(default=1000, ge=0, le=100_000)
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)


class RAGSettings(BaseModel):
    enabled: bool = True
    default_namespace: str = Field(default="default")
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=2048)
    top_k: int = Field(default=5, ge=1, le=50)


class FeatureFlagSettings(BaseModel):
    enable_streaming: bool = True
    enable_function_calling: bool = True
    enable_multi_agent: bool = False
    enable_vector_search: bool = True
    enable_experimental: bool = False


class UserSettings(BaseModel):
    user_id: str
    model: ModelSettings
    memory: MemorySettings
    rag: RAGSettings
    features: FeatureFlagSettings
    updated_at: str


class UpdateSettingsRequest(BaseModel):
    model: Optional[ModelSettings] = None
    memory: Optional[MemorySettings] = None
    rag: Optional[RAGSettings] = None
    features: Optional[FeatureFlagSettings] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_settings_service(request: Request) -> Any:
    service = getattr(request.app.state, "settings_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Settings service unavailable")
    return service


def _defaults(user_id: str) -> UserSettings:
    return UserSettings(
        user_id=user_id,
        model=ModelSettings(),
        memory=MemorySettings(),
        rag=RAGSettings(),
        features=FeatureFlagSettings(),
        updated_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------
@router.get("", response_model=UserSettings)
async def get_settings(
    user: Dict[str, Any] = Depends(get_current_user),
    settings_service: Any = Depends(get_settings_service),
) -> UserSettings:
    """Retrieve the current user's runtime settings, falling back to defaults."""
    try:
        raw = await settings_service.get_settings(user_id=user.get("id"))
        if raw is None:
            return _defaults(user.get("id"))
        return UserSettings(
            user_id=user.get("id"),
            model=ModelSettings(**raw.get("model", {})),
            memory=MemorySettings(**raw.get("memory", {})),
            rag=RAGSettings(**raw.get("rag", {})),
            features=FeatureFlagSettings(**raw.get("features", {})),
            updated_at=raw.get("updated_at", _now_iso()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_settings failed")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve settings: {exc}") from exc


# ---------------------------------------------------------------------------
# PUT /settings
# ---------------------------------------------------------------------------
@router.put("", response_model=UserSettings)
async def update_settings(
    payload: UpdateSettingsRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    settings_service: Any = Depends(get_settings_service),
) -> UserSettings:
    """Update one or more settings sections for the current user."""
    update_data = {
        key: value.model_dump()
        for key, value in payload.model_dump(exclude_none=True).items()
    }
    if not update_data:
        raise HTTPException(status_code=400, detail="No settings fields provided")
    try:
        raw = await settings_service.update_settings(user_id=user.get("id"), updates=update_data)
        return UserSettings(
            user_id=user.get("id"),
            model=ModelSettings(**raw.get("model", {})),
            memory=MemorySettings(**raw.get("memory", {})),
            rag=RAGSettings(**raw.get("rag", {})),
            features=FeatureFlagSettings(**raw.get("features", {})),
            updated_at=raw.get("updated_at", _now_iso()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_settings failed")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /settings/reset
# ---------------------------------------------------------------------------
@router.post("/reset", response_model=UserSettings)
async def reset_settings(
    user: Dict[str, Any] = Depends(get_current_user),
    settings_service: Any = Depends(get_settings_service),
) -> UserSettings:
    """Reset the current user's settings to application defaults."""
    try:
        defaults = _defaults(user.get("id"))
        await settings_service.update_settings(
            user_id=user.get("id"),
            updates={
                "model": defaults.model.model_dump(),
                "memory": defaults.memory.model_dump(),
                "rag": defaults.rag.model_dump(),
                "features": defaults.features.model_dump(),
            },
        )
        return defaults
    except Exception as exc:  # noqa: BLE001
        logger.exception("reset_settings failed")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /settings/features
# ---------------------------------------------------------------------------
@router.get("/features", response_model=FeatureFlagSettings)
async def get_feature_flags(
    user: Dict[str, Any] = Depends(get_current_user),
    settings_service: Any = Depends(get_settings_service),
) -> FeatureFlagSettings:
    """Retrieve only the feature-flag section of the user's settings."""
    try:
        raw = await settings_service.get_settings(user_id=user.get("id"))
        features = (raw or {}).get("features", {})
        return FeatureFlagSettings(**features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_feature_flags failed")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feature flags: {exc}") from exc
