from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PREFIX = "/api/v1/api-keys"

router = APIRouter(tags=["api-keys"])

_api_keys: Dict[str, Dict[str, Any]] = {}

class ApiKeyOut(BaseModel):
    id: str
    name: str
    prefix: str
    createdAt: str
    lastUsedAt: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    expiresAt: Optional[str] = None

class ApiKeyWithSecret(ApiKeyOut):
    secret: str

class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    scopes: List[str] = Field(default_factory=list)

class CreateApiKeyResponse(BaseModel):
    key: ApiKeyOut
    secret: str

class ListApiKeysResponse(BaseModel):
    keys: List[ApiKeyOut]

async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user

@router.get("", response_model=ListApiKeysResponse)
async def list_api_keys(user: Dict[str, Any] = Depends(get_current_user)) -> ListApiKeysResponse:
    user_keys = [v for v in _api_keys.values() if v.get("user_id") == user.get("id")]
    return ListApiKeysResponse(
        keys=[
            ApiKeyOut(
                id=k["id"],
                name=k["name"],
                prefix=k["prefix"],
                createdAt=k["created_at"],
                lastUsedAt=k.get("last_used_at"),
                scopes=k.get("scopes", []),
                expiresAt=k.get("expires_at"),
            )
            for k in user_keys
        ]
    )

@router.post("", response_model=CreateApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: CreateApiKeyRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> CreateApiKeyResponse:
    key_id = str(uuid.uuid4())
    secret = str(uuid.uuid4()).replace("-", "") + str(uuid.uuid4()).replace("-", "")
    now = datetime.now(timezone.utc).isoformat()
    _api_keys[key_id] = {
        "id": key_id,
        "name": payload.name,
        "prefix": secret[:8],
        "scopes": payload.scopes,
        "user_id": user.get("id"),
        "created_at": now,
        "last_used_at": None,
        "expires_at": None,
    }
    return CreateApiKeyResponse(
        key=ApiKeyOut(
            id=key_id,
            name=payload.name,
            prefix=secret[:8],
            createdAt=now,
            scopes=payload.scopes,
        ),
        secret=secret,
    )

@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> None:
    if key_id not in _api_keys or _api_keys[key_id].get("user_id") != user.get("id"):
        raise HTTPException(status_code=404, detail="API key not found")
    del _api_keys[key_id]
