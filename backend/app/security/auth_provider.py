from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AuthError(Exception):
    pass


class InvalidCredentialsError(AuthError):
    pass


class TokenExpiredError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass


class SessionNotFoundError(AuthError):
    pass


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    password_salt: str
    roles: frozenset[str] = field(default_factory=frozenset)
    oauth_provider: Optional[str] = None
    oauth_subject: Optional[str] = None
    disabled: bool = False


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float
    last_seen_at: float
    expires_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= self.expires_at


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 0


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


class _MinimalJWT:
    """Minimal dependency-free JWT (HS256) implementation."""

    @staticmethod
    def encode(payload: dict[str, Any], secret: str) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), default=str).encode())
        signing_input = f"{header_b64}.{payload_b64}".encode()
        signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        sig_b64 = _b64url_encode(signature)
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    @staticmethod
    def decode(token: str, secret: str, *, verify_exp: bool = True) -> dict[str, Any]:
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
        except ValueError as exc:
            raise InvalidTokenError("malformed token") from exc

        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        try:
            actual_sig = _b64url_decode(sig_b64)
        except Exception as exc:
            raise InvalidTokenError("malformed token signature") from exc

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise InvalidTokenError("signature verification failed")

        try:
            payload = json.loads(_b64url_decode(payload_b64))
        except Exception as exc:
            raise InvalidTokenError("malformed token payload") from exc

        if verify_exp:
            exp = payload.get("exp")
            if exp is not None and time.time() >= exp:
                raise TokenExpiredError("token has expired")

        return payload


class AuthProvider:
    """JWT-based authentication with sessions, refresh tokens, and OAuth subject linking."""

    def __init__(
        self,
        *,
        jwt_secret: Optional[str] = None,
        access_token_ttl_seconds: int = 900,
        refresh_token_ttl_seconds: int = 60 * 60 * 24 * 14,
        session_ttl_seconds: int = 60 * 60 * 24,
        password_hash_iterations: int = 600_000,
    ) -> None:
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(64)
        self.access_token_ttl_seconds = access_token_ttl_seconds
        self.refresh_token_ttl_seconds = refresh_token_ttl_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self.password_hash_iterations = password_hash_iterations

        self._users: dict[str, User] = {}
        self._users_by_username: dict[str, str] = {}
        self._sessions: dict[str, Session] = {}
        self._revoked_refresh_jti: set[str] = set()
        self._oauth_index: dict[tuple[str, str], str] = {}

        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._login_count = 0
        self._failed_login_count = 0
        self._token_refresh_count = 0

    @staticmethod
    def _hash_password(password: str, *, salt: Optional[bytes] = None, iterations: int = 600_000) -> tuple[str, str]:
        salt = salt or secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return _b64url_encode(derived), _b64url_encode(salt)

    def register_user(self, username: str, password: str, *, roles: Optional[frozenset[str]] = None) -> User:
        if username in self._users_by_username:
            raise AuthError(f"username '{username}' already registered")
        pw_hash, salt = self._hash_password(password, iterations=self.password_hash_iterations)
        user = User(
            user_id=str(uuid.uuid4()), username=username, password_hash=pw_hash,
            password_salt=salt, roles=roles or frozenset(),
        )
        self._users[user.user_id] = user
        self._users_by_username[username] = user.user_id
        return user

    def register_oauth_user(self, *, provider: str, subject: str, username: str, roles: Optional[frozenset[str]] = None) -> User:
        key = (provider, subject)
        if key in self._oauth_index:
            return self._users[self._oauth_index[key]]
        user = User(
            user_id=str(uuid.uuid4()), username=username, password_hash="", password_salt="",
            roles=roles or frozenset(), oauth_provider=provider, oauth_subject=subject,
        )
        self._users[user.user_id] = user
        self._users_by_username[username] = user.user_id
        self._oauth_index[key] = user.user_id
        return user

    def _verify_password(self, user: User, password: str) -> bool:
        derived, _ = self._hash_password(password, salt=_b64url_decode(user.password_salt), iterations=self.password_hash_iterations)
        return hmac.compare_digest(derived, user.password_hash)

    def authenticate(self, username: str, password: str) -> tuple[User, Session, TokenPair]:
        user_id = self._users_by_username.get(username)
        user = self._users.get(user_id) if user_id else None
        if user is None or user.disabled or not self._verify_password(user, password):
            self._failed_login_count += 1
            raise InvalidCredentialsError("invalid username or password")
        self._login_count += 1
        session = self._create_session(user.user_id)
        tokens = self._issue_tokens(user, session)
        return user, session, tokens

    def authenticate_oauth(self, *, provider: str, subject: str) -> tuple[User, Session, TokenPair]:
        key = (provider, subject)
        user_id = self._oauth_index.get(key)
        if user_id is None or user_id not in self._users:
            raise InvalidCredentialsError(f"no linked OAuth account for provider '{provider}'")
        user = self._users[user_id]
        if user.disabled:
            raise InvalidCredentialsError("user account is disabled")
        self._login_count += 1
        session = self._create_session(user.user_id)
        tokens = self._issue_tokens(user, session)
        return user, session, tokens

    def _create_session(self, user_id: str) -> Session:
        now = time.time()
        session = Session(
            session_id=str(uuid.uuid4()), user_id=user_id, created_at=now,
            last_seen_at=now, expires_at=now + self.session_ttl_seconds,
        )
        self._sessions[session.session_id] = session
        return session

    def _issue_tokens(self, user: User, session: Session) -> TokenPair:
        now = time.time()
        access_payload = {
            "sub": user.user_id, "username": user.username, "roles": sorted(user.roles),
            "sid": session.session_id, "type": TokenType.ACCESS.value,
            "iat": int(now), "exp": int(now + self.access_token_ttl_seconds),
            "jti": str(uuid.uuid4()),
        }
        refresh_jti = str(uuid.uuid4())
        refresh_payload = {
            "sub": user.user_id, "sid": session.session_id, "type": TokenType.REFRESH.value,
            "iat": int(now), "exp": int(now + self.refresh_token_ttl_seconds), "jti": refresh_jti,
        }
        access_token = _MinimalJWT.encode(access_payload, self.jwt_secret)
        refresh_token = _MinimalJWT.encode(refresh_payload, self.jwt_secret)
        return TokenPair(access_token=access_token, refresh_token=refresh_token, expires_in=self.access_token_ttl_seconds)

    def verify_access_token(self, token: str) -> dict[str, Any]:
        payload = _MinimalJWT.decode(token, self.jwt_secret)
        if payload.get("type") != TokenType.ACCESS.value:
            raise InvalidTokenError("token is not an access token")
        session = self._sessions.get(payload.get("sid", ""))
        if session is None or session.is_expired():
            raise SessionNotFoundError("session not found or expired")
        return payload

    def refresh_access_token(self, refresh_token: str) -> TokenPair:
        payload = _MinimalJWT.decode(refresh_token, self.jwt_secret)
        if payload.get("type") != TokenType.REFRESH.value:
            raise InvalidTokenError("token is not a refresh token")
        jti = payload.get("jti", "")
        if jti in self._revoked_refresh_jti:
            raise InvalidTokenError("refresh token has been revoked")

        user = self._users.get(payload.get("sub", ""))
        session = self._sessions.get(payload.get("sid", ""))
        if user is None or session is None or session.is_expired():
            raise SessionNotFoundError("session not found or expired")

        self._revoked_refresh_jti.add(jti)
        session.last_seen_at = time.time()
        session.expires_at = max(session.expires_at, time.time() + self.session_ttl_seconds)
        self._token_refresh_count += 1
        return self._issue_tokens(user, session)

    def revoke_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is not None and session.is_expired():
            self._sessions.pop(session_id, None)
            return None
        return session

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    async def authenticate_async(self, username: str, password: str) -> tuple[User, Session, TokenPair]:
        async with self._lock:
            return await asyncio.to_thread(self.authenticate, username, password)

    async def authenticate_oauth_async(self, *, provider: str, subject: str) -> tuple[User, Session, TokenPair]:
        async with self._lock:
            return await asyncio.to_thread(self.authenticate_oauth, provider=provider, subject=subject)

    async def verify_access_token_async(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.verify_access_token, token)

    async def refresh_access_token_async(self, refresh_token: str) -> TokenPair:
        async with self._lock:
            return await asyncio.to_thread(self.refresh_access_token, refresh_token)

    async def revoke_session_async(self, session_id: str) -> bool:
        async with self._lock:
            return self.revoke_session(session_id)

    def health_check(self) -> HealthStatus:
        try:
            probe_payload = {"probe": True, "exp": int(time.time()) + 5}
            token = _MinimalJWT.encode(probe_payload, self.jwt_secret)
            decoded = _MinimalJWT.decode(token, self.jwt_secret)
            healthy = decoded.get("probe") is True
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "user_count": len(self._users),
                "active_session_count": len(self._sessions),
                "revoked_refresh_token_count": len(self._revoked_refresh_jti),
                "login_count": self._login_count,
                "failed_login_count": self._failed_login_count,
                "token_refresh_count": self._token_refresh_count,
            }
            return HealthStatus(healthy=healthy, component="auth_provider", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="auth_provider", details={"error": str(exc)})

    # ── Async wrapper methods matching auth router interface ──────────────

    async def find_user_by_email(self, email: str) -> Optional[dict]:
        async with self._lock:
            user_id = self._users_by_username.get(email)
            user = self._users.get(user_id) if user_id else None
            if user is None:
                return None
            return {"id": user.user_id, "email": email, "name": user.username}

    async def create_user(self, email: str, password: str, name: Optional[str] = None) -> dict:
        async with self._lock:
            user = self.register_user(username=email, password=password)
            # register_user may raise AuthError if duplicate
        return {"id": user.user_id, "email": email, "name": name or email}

    async def issue_token_pair(self, user_id: str) -> dict:
        async with self._lock:
            user = self._users.get(user_id)
            if not user:
                raise AuthError(f"user not found: {user_id}")
            session = await asyncio.to_thread(self._create_session, user_id)
            tokens = await asyncio.to_thread(self._issue_tokens, user, session)
        return {"access_token": tokens.access_token, "refresh_token": tokens.refresh_token, "expires_in": tokens.expires_in}

    async def authenticate(self, email: str, password: str) -> Optional[dict]:
        try:
            user, _session, _tokens = await self.authenticate_async(username=email, password=password)
            return {"id": user.user_id, "email": email, "name": user.username}
        except (InvalidCredentialsError, AuthError):
            return None

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "AuthProvider",
    "User",
    "Session",
    "TokenPair",
    "TokenType",
    "AuthError",
    "InvalidCredentialsError",
    "TokenExpiredError",
    "InvalidTokenError",
    "SessionNotFoundError",
    "HealthStatus",
]
