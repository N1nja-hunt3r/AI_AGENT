"""
auth.py

FastAPI router for authentication: JWT-based login/signup, token
refresh, logout, and OAuth provider redirect/callback handling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

SUPPORTED_OAUTH_PROVIDERS = {"google", "github", "microsoft"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=256)
    name: Optional[str] = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class LogoutResponse(BaseModel):
    logged_out: bool


class UserProfile(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    role: str
    created_at: str


class OAuthRedirectResponse(BaseModel):
    authorization_url: str
    provider: str
    state: str


class OAuthCallbackRequest(BaseModel):
    provider: str
    code: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_auth_service(request: Request) -> Any:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Auth service unavailable")
    return service


async def get_current_user(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Resolve and verify the current user from a bearer JWT."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    auth_service = get_auth_service(request)
    try:
        user = await auth_service.verify_access_token(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.user = user
    return user


# ---------------------------------------------------------------------------
# POST /auth/signup
# ---------------------------------------------------------------------------
@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    auth_service: Any = Depends(get_auth_service),
) -> TokenPair:
    """Register a new user account and issue an initial token pair."""
    try:
        existing = await auth_service.find_user_by_email(payload.email)
        if existing is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        user = await auth_service.create_user(
            email=payload.email, password=payload.password, name=payload.name
        )
        tokens = await auth_service.issue_token_pair(user_id=user["id"])
        return TokenPair(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens.get("expires_in", 3600),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("signup failed for email=%s", payload.email)
        raise HTTPException(status_code=500, detail=f"Signup failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------
@router.post("/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    auth_service: Any = Depends(get_auth_service),
) -> TokenPair:
    """Authenticate a user with email/password and issue a token pair."""
    try:
        user = await auth_service.authenticate(email=payload.email, password=payload.password)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
            )
        tokens = await auth_service.issue_token_pair(user_id=user["id"])
        return TokenPair(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens.get("expires_in", 3600),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("login failed for email=%s", payload.email)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------
@router.post("/refresh", response_model=TokenPair)
async def refresh_token(
    payload: RefreshRequest,
    auth_service: Any = Depends(get_auth_service),
) -> TokenPair:
    """Exchange a valid refresh token for a new access/refresh token pair."""
    try:
        tokens = await auth_service.refresh_access_token(payload.refresh_token)
        if tokens is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
            )
        return TokenPair(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens.get("expires_in", 3600),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("refresh_token failed")
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------
@router.post("/logout", response_model=LogoutResponse)
async def logout(
    payload: LogoutRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    auth_service: Any = Depends(get_auth_service),
) -> LogoutResponse:
    """Invalidate the current session's refresh token(s)."""
    try:
        await auth_service.revoke_session(
            user_id=user.get("id"), refresh_token=payload.refresh_token
        )
        return LogoutResponse(logged_out=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("logout failed for user_id=%s", user.get("id"))
        raise HTTPException(status_code=500, detail=f"Logout failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------
@router.get("/me", response_model=UserProfile)
async def get_profile(user: Dict[str, Any] = Depends(get_current_user)) -> UserProfile:
    """Return the authenticated user's profile."""
    return UserProfile(
        id=user["id"],
        email=user.get("email", ""),
        name=user.get("name"),
        role=user.get("role", "member"),
        created_at=user.get("created_at", _now_iso()),
    )


# ---------------------------------------------------------------------------
# GET /auth/oauth/{provider}
# ---------------------------------------------------------------------------
@router.get("/oauth/{provider}", response_model=OAuthRedirectResponse)
async def oauth_authorize(
    provider: str,
    auth_service: Any = Depends(get_auth_service),
) -> OAuthRedirectResponse:
    """Generate an OAuth authorization URL for the given provider."""
    if provider not in SUPPORTED_OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported OAuth provider: {provider}")
    try:
        result = await auth_service.build_oauth_redirect(provider=provider)
        return OAuthRedirectResponse(
            authorization_url=result["authorization_url"],
            provider=provider,
            state=result["state"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("oauth_authorize failed for provider=%s", provider)
        raise HTTPException(status_code=500, detail=f"OAuth redirect failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /auth/oauth/callback
# ---------------------------------------------------------------------------
@router.post("/oauth/callback", response_model=TokenPair)
async def oauth_callback(
    payload: OAuthCallbackRequest,
    auth_service: Any = Depends(get_auth_service),
) -> TokenPair:
    """Handle the OAuth provider callback, exchanging the code for tokens."""
    if payload.provider not in SUPPORTED_OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported OAuth provider: {payload.provider}")
    try:
        user = await auth_service.complete_oauth_flow(
            provider=payload.provider, code=payload.code, state=payload.state
        )
        if user is None:
            raise HTTPException(status_code=401, detail="OAuth authentication failed")
        tokens = await auth_service.issue_token_pair(user_id=user["id"])
        return TokenPair(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens.get("expires_in", 3600),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("oauth_callback failed for provider=%s", payload.provider)
        raise HTTPException(status_code=500, detail=f"OAuth callback failed: {exc}") from exc
