"""
security_connector.py - Production-grade security connector.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import jwt as pyjwt  # type: ignore[import]
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    pyjwt = None  # type: ignore

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None  # type: ignore

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    bcrypt = None  # type: ignore

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    aioredis = None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuthStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    FORBIDDEN = "forbidden"
    LOCKED = "locked"


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"
    CREATE = "create"
    LIST = "list"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditAction(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    DENIED = "denied"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    APPROVE = "approve"
    REJECT = "reject"
    SECRET_ACCESS = "secret_access"
    VALIDATION = "validation"
    RATE_LIMITED = "rate_limited"
    SANDBOX_EXEC = "sandbox_exec"


class SandboxStatus(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    SANDBOXED = "sandboxed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Identity:
    user_id: str
    username: str
    roles: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def has_role(self, role: str) -> bool:
        return role in self.roles or "admin" in self.roles

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN in self.permissions


@dataclass
class AuthToken:
    token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: str = ""
    issued_at: float = field(default_factory=time.time)


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    sanitized: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalGateResult:
    approved: bool
    approver: Optional[str] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int = 0
    reset_at: float = 0.0
    retry_after: float = 0.0


@dataclass
class AuditEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: AuditAction = AuditAction.ACCESS
    user_id: Optional[str] = None
    resource: Optional[str] = None
    success: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SandboxResult:
    status: SandboxStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    blocked_reason: Optional[str] = None


@dataclass
class SecurityConnectorConfig:
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600
    jwt_refresh_expiry_seconds: int = 86400
    encryption_key: Optional[str] = None
    bcrypt_rounds: int = 12
    rate_limit_default_rpm: int = 60
    rate_limit_burst: int = 10
    enable_sandbox: bool = True
    sandbox_allowed_modules: List[str] = field(default_factory=lambda: ["json", "math", "re"])
    sandbox_blocked_builtins: List[str] = field(default_factory=lambda: ["exec", "eval", "open", "__import__"])
    enable_audit: bool = True
    audit_max_entries: int = 10000
    enable_redis_rate_limit: bool = False
    redis_url: str = "redis://localhost:6379"
    approval_timeout: float = 300.0
    max_login_attempts: int = 5
    lockout_duration: float = 900.0
    secrets_encryption: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    password_min_length: int = 8
    password_require_special: bool = True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecurityConnectorError(Exception):
    pass

class AuthenticationError(SecurityConnectorError):
    pass

class AuthorizationError(SecurityConnectorError):
    pass

class ValidationError(SecurityConnectorError):
    pass

class EncryptionError(SecurityConnectorError):
    pass

class RateLimitExceededError(SecurityConnectorError):
    pass

class SandboxError(SecurityConnectorError):
    pass

class SecretNotFoundError(SecurityConnectorError):
    pass


# ---------------------------------------------------------------------------
# JWT Provider
# ---------------------------------------------------------------------------

class _JWTProvider:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._secret = config.jwt_secret or os.environ.get("JWT_SECRET", secrets.token_urlsafe(32))
        self._revoked: Set[str] = set()
        self._lock = asyncio.Lock()

    def _get_secret(self) -> str:
        return self._secret

    async def create_token(self, identity: Identity) -> AuthToken:
        if not _JWT_AVAILABLE:
            return self._create_simple_token(identity)

        now = time.time()
        exp = int(now + self._config.jwt_expiry_seconds)
        payload = {
            "sub": identity.user_id,
            "username": identity.username,
            "roles": identity.roles,
            "permissions": [p.value for p in identity.permissions],
            "session_id": identity.session_id,
            "iat": int(now),
            "exp": exp,
            "jti": str(uuid.uuid4()),
        }
        token = pyjwt.encode(payload, self._get_secret(), algorithm=self._config.jwt_algorithm)
        refresh_payload = {
            "sub": identity.user_id,
            "type": "refresh",
            "session_id": identity.session_id,
            "exp": int(now + self._config.jwt_refresh_expiry_seconds),
            "jti": str(uuid.uuid4()),
        }
        refresh_token = pyjwt.encode(refresh_payload, self._get_secret(), algorithm=self._config.jwt_algorithm)
        return AuthToken(
            token=token if isinstance(token, str) else token.decode(),
            refresh_token=refresh_token if isinstance(refresh_token, str) else refresh_token.decode(),
            expires_in=self._config.jwt_expiry_seconds,
        )

    def _create_simple_token(self, identity: Identity) -> AuthToken:
        payload = {
            "sub": identity.user_id,
            "username": identity.username,
            "roles": identity.roles,
            "session_id": identity.session_id,
            "exp": time.time() + self._config.jwt_expiry_seconds,
        }
        raw = json.dumps(payload, default=str)
        sig = hmac.new(self._secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{raw}.{sig}".encode()).decode()
        return AuthToken(token=token, expires_in=self._config.jwt_expiry_seconds)

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if token in self._revoked:
                return None

        if not _JWT_AVAILABLE:
            return self._verify_simple_token(token)

        try:
            payload = pyjwt.decode(
                token,
                self._get_secret(),
                algorithms=[self._config.jwt_algorithm],
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except pyjwt.InvalidTokenError as exc:
            logger.warning("JWT validation failed: %s", exc)
            return None

    def _verify_simple_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            raw, sig = decoded.rsplit(".", 1)
            expected_sig = hmac.new(self._secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return None
            payload = json.loads(raw)
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None

    async def revoke_token(self, token: str) -> None:
        async with self._lock:
            self._revoked.add(token)

    async def refresh(self, refresh_token: str) -> Optional[AuthToken]:
        payload = await self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        identity = Identity(
            user_id=payload["sub"],
            username=payload.get("username", ""),
            session_id=payload.get("session_id", str(uuid.uuid4())),
        )
        return await self.create_token(identity)


# ---------------------------------------------------------------------------
# Auth Provider
# ---------------------------------------------------------------------------

class _AuthProvider:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._users: Dict[str, Dict[str, Any]] = {}
        self._api_keys: Dict[str, Identity] = {}
        self._login_attempts: Dict[str, List[float]] = {}
        self._locked: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._external_auth: Optional[Callable[..., Any]] = None

    def set_external_auth(self, fn: Callable[..., Any]) -> None:
        self._external_auth = fn

    async def register_user(
        self,
        username: str,
        password: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_id = str(uuid.uuid4())
        hashed = self._hash_password(password)
        async with self._lock:
            self._users[username] = {
                "user_id": user_id,
                "username": username,
                "password_hash": hashed,
                "roles": roles or ["user"],
                "permissions": [p.value for p in (permissions or [Permission.READ])],
                "metadata": metadata or {},
                "created_at": time.time(),
            }
        return user_id

    def _hash_password(self, password: str) -> str:
        if _BCRYPT_AVAILABLE:
            return bcrypt.hashpw(
                password.encode(), bcrypt.gensalt(rounds=self._config.bcrypt_rounds)
            ).decode()
        salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}:{h.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        if _BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        try:
            salt, h = password_hash.split(":", 1)
            expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(h, expected.hex())
        except Exception:
            return False

    async def authenticate_password(self, username: str, password: str) -> Optional[Identity]:
        async with self._lock:
            locked_until = self._locked.get(username, 0)
            if time.time() < locked_until:
                raise AuthenticationError(f"Account '{username}' is locked")

        user = self._users.get(username)
        if not user:
            if self._external_auth:
                try:
                    if asyncio.iscoroutinefunction(self._external_auth):
                        return await self._external_auth(username, password)
                    return await asyncio.to_thread(self._external_auth, username, password)
                except Exception:
                    return None
            return None

        if not self._verify_password(password, user["password_hash"]):
            async with self._lock:
                attempts = self._login_attempts.setdefault(username, [])
                attempts.append(time.time())
                recent = [a for a in attempts if time.time() - a < 900]
                self._login_attempts[username] = recent
                if len(recent) >= self._config.max_login_attempts:
                    self._locked[username] = time.time() + self._config.lockout_duration
                    raise AuthenticationError(f"Account '{username}' locked due to too many failed attempts")
            return None

        async with self._lock:
            self._login_attempts.pop(username, None)

        return Identity(
            user_id=user["user_id"],
            username=username,
            roles=user["roles"],
            permissions=[Permission(p) for p in user["permissions"] if p in Permission.__members__.values()],
            metadata=user["metadata"],
            expires_at=time.time() + self._config.jwt_expiry_seconds,
        )

    async def register_api_key(self, identity: Identity) -> str:
        api_key = secrets.token_urlsafe(32)
        async with self._lock:
            self._api_keys[api_key] = identity
        return api_key

    async def authenticate_api_key(self, api_key: str) -> Optional[Identity]:
        return self._api_keys.get(api_key)

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        for u in self._users.values():
            if u["user_id"] == user_id:
                return u
        return None


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._windows: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        self._redis_client: Any = None

    async def _get_redis(self) -> Any:
        if not _REDIS_AVAILABLE:
            return None
        if self._redis_client is None:
            self._redis_client = aioredis.from_url(self._config.redis_url, decode_responses=True)
        return self._redis_client

    async def check(
        self,
        key: str,
        rpm: Optional[int] = None,
        burst: Optional[int] = None,
    ) -> RateLimitResult:
        limit = rpm or self._config.rate_limit_default_rpm
        max_burst = burst or self._config.rate_limit_burst

        if self._config.enable_redis_rate_limit:
            return await self._check_redis(key, limit)

        now = time.time()
        window_start = now - 60.0

        async with self._lock:
            times = self._windows.setdefault(key, [])
            times[:] = [t for t in times if t > window_start]
            count = len(times)
            if count >= limit + max_burst:
                oldest = min(times) if times else now
                reset_at = oldest + 60.0
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=max(0.0, reset_at - now),
                )
            times.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=max(0, limit - len(times)),
                reset_at=now + 60.0,
            )

    async def _check_redis(self, key: str, limit: int) -> RateLimitResult:
        try:
            client = await self._get_redis()
            if not client:
                return RateLimitResult(allowed=True, remaining=limit)
            redis_key = f"ratelimit:{key}"
            now = time.time()
            pipe = client.pipeline()
            pipe.zadd(redis_key, {str(uuid.uuid4()): now})
            pipe.zremrangebyscore(redis_key, 0, now - 60)
            pipe.zcard(redis_key)
            pipe.expire(redis_key, 60)
            results = await pipe.execute()
            count = results[2]
            if count > limit:
                return RateLimitResult(allowed=False, remaining=0, reset_at=now + 60, retry_after=60.0)
            return RateLimitResult(allowed=True, remaining=max(0, limit - count), reset_at=now + 60)
        except Exception as exc:
            logger.warning("Redis rate limit error: %s", exc)
            return RateLimitResult(allowed=True, remaining=limit)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._windows.pop(key, None)

    async def close(self) -> None:
        if self._redis_client:
            await self._redis_client.aclose()


# ---------------------------------------------------------------------------
# Encryption Service
# ---------------------------------------------------------------------------

class _EncryptionService:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._fernet: Any = None
        self._key: Optional[bytes] = None

    def _get_fernet(self) -> Any:
        if self._fernet:
            return self._fernet
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("cryptography package not installed. pip install cryptography")
        enc_key = self._config.encryption_key or os.environ.get("ENCRYPTION_KEY")
        if enc_key:
            key_bytes = enc_key.encode() if isinstance(enc_key, str) else enc_key
            if len(key_bytes) != 44:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"security_connector_salt",
                    iterations=100000,
                )
                key_bytes = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            self._key = key_bytes
        else:
            self._key = Fernet.generate_key()
        self._fernet = Fernet(self._key)
        return self._fernet

    async def encrypt(self, data: Any) -> str:
        try:
            fernet = self._get_fernet()
            if not isinstance(data, bytes):
                data = json.dumps(data, default=str).encode()
            encrypted = await asyncio.to_thread(fernet.encrypt, data)
            return encrypted.decode()
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    async def decrypt(self, encrypted: str) -> Any:
        try:
            fernet = self._get_fernet()
            decrypted = await asyncio.to_thread(fernet.decrypt, encrypted.encode())
            try:
                return json.loads(decrypted.decode())
            except json.JSONDecodeError:
                return decrypted.decode()
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc

    def hash(self, data: str, salt: Optional[str] = None) -> str:
        salt = salt or secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac("sha256", data.encode(), salt.encode(), 100000)
        return f"{salt}:{h.hex()}"

    def verify_hash(self, data: str, hashed: str) -> bool:
        try:
            salt, h = hashed.split(":", 1)
            expected = hashlib.pbkdf2_hmac("sha256", data.encode(), salt.encode(), 100000)
            return hmac.compare_digest(h, expected.hex())
        except Exception:
            return False

    def generate_key(self) -> str:
        if _CRYPTOGRAPHY_AVAILABLE:
            return Fernet.generate_key().decode()
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


# ---------------------------------------------------------------------------
# Permissions Manager
# ---------------------------------------------------------------------------

class _PermissionsManager:
    def __init__(self) -> None:
        self._role_permissions: Dict[str, Set[Permission]] = {
            "admin": set(Permission),
            "user": {Permission.READ, Permission.LIST},
            "operator": {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.LIST},
            "viewer": {Permission.READ, Permission.LIST},
        }
        self._resource_acl: Dict[str, Dict[str, Set[Permission]]] = {}

    def define_role(self, role: str, permissions: List[Permission]) -> None:
        self._role_permissions[role] = set(permissions)

    def grant(self, resource: str, role_or_user: str, permissions: List[Permission]) -> None:
        acl = self._resource_acl.setdefault(resource, {})
        acl.setdefault(role_or_user, set()).update(permissions)

    def revoke(self, resource: str, role_or_user: str, permissions: List[Permission]) -> None:
        acl = self._resource_acl.get(resource, {})
        if role_or_user in acl:
            acl[role_or_user] -= set(permissions)

    def check(self, identity: Identity, resource: str, permission: Permission) -> bool:
        # Check direct permission
        if identity.has_permission(permission):
            return True
        # Check role permissions
        for role in identity.roles:
            role_perms = self._role_permissions.get(role, set())
            if permission in role_perms:
                return True
        # Check resource ACL
        resource_acl = self._resource_acl.get(resource, {})
        for role in identity.roles:
            if permission in resource_acl.get(role, set()):
                return True
        if permission in resource_acl.get(identity.user_id, set()):
            return True
        return False

    def get_role_permissions(self, role: str) -> Set[Permission]:
        return self._role_permissions.get(role, set())


# ---------------------------------------------------------------------------
# Secrets Manager
# ---------------------------------------------------------------------------

class _SecretsManager:
    def __init__(self, encryption_service: _EncryptionService, encrypt: bool = True) -> None:
        self._enc = encryption_service
        self._encrypt = encrypt
        self._secrets: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._audit_callbacks: List[Callable[..., Any]] = []

    def add_audit_callback(self, fn: Callable[..., Any]) -> None:
        self._audit_callbacks.append(fn)

    async def set(self, key: str, value: str) -> None:
        stored = await self._enc.encrypt(value) if self._encrypt else value
        async with self._lock:
            self._secrets[key] = stored

    async def get(self, key: str, requester: str = "") -> str:
        async with self._lock:
            stored = self._secrets.get(key)
        if stored is None:
            env_val = os.environ.get(key)
            if env_val:
                return env_val
            raise SecretNotFoundError(f"Secret '{key}' not found")
        for cb in self._audit_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb("secret_access", {"key": key, "requester": requester}))
            except Exception:
                pass
        if self._encrypt:
            return await self._enc.decrypt(stored)
        return stored

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return bool(self._secrets.pop(key, None))

    async def list_keys(self) -> List[str]:
        async with self._lock:
            return list(self._secrets.keys())

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._secrets or bool(os.environ.get(key))


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class _Validator:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
            r"(--|;|\/\*|\*\/|xp_|0x[0-9a-f]+)",
        ]
        self._xss_patterns = [
            r"<script[\s\S]*?>[\s\S]*?<\/script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        self._path_traversal_patterns = [r"\.\./", r"\.\.\\", r"%2e%2e"]

    async def validate_input(
        self,
        data: Any,
        data_type: str = "generic",
        max_length: int = 10000,
    ) -> ValidationResult:
        errors: List[str] = []
        risk = RiskLevel.LOW

        if data is None:
            return ValidationResult(valid=True, sanitized=data)

        text = str(data) if not isinstance(data, str) else data

        if len(text) > max_length:
            errors.append(f"Input exceeds max length {max_length}")
            risk = RiskLevel.MEDIUM

        for pattern in self._sql_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append("Potential SQL injection detected")
                risk = RiskLevel.HIGH
                break

        for pattern in self._xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append("Potential XSS detected")
                risk = RiskLevel.HIGH
                break

        for pattern in self._path_traversal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append("Potential path traversal detected")
                risk = RiskLevel.HIGH
                break

        if data_type == "email":
            if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", text):
                errors.append("Invalid email format")
                risk = RiskLevel.MEDIUM

        elif data_type == "url":
            if not re.match(r"^https?://", text):
                errors.append("Invalid URL format")
                risk = RiskLevel.MEDIUM

        elif data_type == "password":
            if len(text) < self._config.password_min_length:
                errors.append(f"Password too short (min {self._config.password_min_length})")
            if self._config.password_require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", text):
                errors.append("Password must contain special characters")

        sanitized = self._sanitize(text) if not errors else None
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            risk_level=risk,
            sanitized=sanitized,
        )

    def _sanitize(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("'", "\\'").replace('"', '\\"')
        return text.strip()

    async def validate_json(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        try:
            if isinstance(data, str):
                json.loads(data)
            return ValidationResult(valid=True, sanitized=data)
        except Exception as exc:
            return ValidationResult(valid=False, errors=[f"Invalid JSON: {exc}"])


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class _Sandbox:
    def __init__(self, config: SecurityConnectorConfig) -> None:
        self._config = config
        self._blocked_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"__import__",
            r"open\(",
            r"exec\(",
            r"eval\(",
        ]

    async def evaluate(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> SandboxResult:
        if not self._config.enable_sandbox:
            return SandboxResult(status=SandboxStatus.ALLOWED)

        for pattern in self._blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return SandboxResult(
                    status=SandboxStatus.BLOCKED,
                    blocked_reason=f"Blocked pattern: {pattern}",
                )

        for builtin in self._config.sandbox_blocked_builtins:
            if builtin in code:
                return SandboxResult(
                    status=SandboxStatus.BLOCKED,
                    blocked_reason=f"Blocked builtin: {builtin}",
                )

        t0 = time.perf_counter()
        try:
            safe_globals = {
                "__builtins__": {
                    k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
                    if k not in self._config.sandbox_blocked_builtins
                } if isinstance(__builtins__, dict) else {},
                **(context or {}),
            }
            result = await asyncio.wait_for(
                asyncio.to_thread(eval, code, safe_globals),
                timeout=timeout,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return SandboxResult(
                status=SandboxStatus.SANDBOXED,
                output=result,
                execution_time_ms=elapsed,
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                status=SandboxStatus.BLOCKED,
                blocked_reason=f"Execution timed out after {timeout}s",
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return SandboxResult(
                status=SandboxStatus.SANDBOXED,
                error=str(exc),
                execution_time_ms=elapsed,
            )

    def is_safe_code(self, code: str) -> bool:
        for pattern in self._blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        return True


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class _AuditLogger:
    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: List[AuditEntry] = []
        self._max = max_entries
        self._lock = asyncio.Lock()
        self._external_handlers: List[Callable[..., Any]] = []

    def add_handler(self, fn: Callable[..., Any]) -> None:
        self._external_handlers.append(fn)

    async def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        success: bool = True,
        risk_level: RiskLevel = RiskLevel.LOW,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            action=action,
            user_id=user_id,
            resource=resource,
            success=success,
            risk_level=risk_level,
            details=details or {},
            ip_address=ip_address,
        )
        async with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

        logger.info(
            "[AUDIT] action=%s user=%s resource=%s success=%s risk=%s",
            action.value, user_id, resource, success, risk_level.value,
        )
        for handler in self._external_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(entry))
                else:
                    asyncio.get_event_loop().run_in_executor(None, handler, entry)
            except Exception as exc:
                logger.warning("Audit handler error: %s", exc)

        return entry

    async def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        risk_level: Optional[RiskLevel] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        async with self._lock:
            entries = list(reversed(self._entries))
        result = []
        for e in entries:
            if user_id and e.user_id != user_id:
                continue
            if action and e.action != action:
                continue
            if risk_level and e.risk_level != risk_level:
                continue
            result.append(e)
            if len(result) >= limit:
                break
        return result

    async def count(self) -> int:
        async with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Approval Gate
# ---------------------------------------------------------------------------

class _ApprovalGate:
    def __init__(self, timeout: float = 300.0) -> None:
        self._timeout = timeout
        self._pending: Dict[str, asyncio.Future] = {}  # type: ignore[type-arg]
        self._history: List[ApprovalGateResult] = []
        self._lock = asyncio.Lock()

    async def request(
        self,
        resource: str,
        requester: str,
        reason: str = "",
        notify_fn: Optional[Callable[..., Any]] = None,
    ) -> ApprovalGateResult:
        gate_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()  # type: ignore[type-arg]
        async with self._lock:
            self._pending[gate_id] = future

        if notify_fn:
            try:
                if asyncio.iscoroutinefunction(notify_fn):
                    asyncio.create_task(notify_fn(gate_id, resource, requester, reason))
                else:
                    asyncio.get_event_loop().run_in_executor(None, notify_fn, gate_id, resource, requester, reason)
            except Exception as exc:
                logger.warning("Approval notification error: %s", exc)

        logger.info("Approval gate [%s] waiting for resource '%s'", gate_id, resource)
        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
            gate_result = ApprovalGateResult(
                approved=result.get("approved", False),
                approver=result.get("approver"),
                reason=result.get("reason", ""),
            )
        except asyncio.TimeoutError:
            gate_result = ApprovalGateResult(approved=False, reason="Timeout")
        finally:
            async with self._lock:
                self._pending.pop(gate_id, None)

        self._history.append(gate_result)
        return gate_result

    async def respond(
        self,
        gate_id: str,
        approved: bool,
        approver: str = "",
        reason: str = "",
    ) -> bool:
        async with self._lock:
            future = self._pending.get(gate_id)
        if future and not future.done():
            future.set_result({"approved": approved, "approver": approver, "reason": reason})
            return True
        return False

    async def get_pending_ids(self) -> List[str]:
        async with self._lock:
            return list(self._pending.keys())


# ---------------------------------------------------------------------------
# SecurityConnector – Singleton
# ---------------------------------------------------------------------------

class SecurityConnector:
    _instance: Optional["SecurityConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = SecurityConnectorConfig()
        self._jwt_provider: Optional[_JWTProvider] = None
        self._auth_provider: Optional[_AuthProvider] = None
        self._rate_limiter: Optional[_RateLimiter] = None
        self._encryption: Optional[_EncryptionService] = None
        self._permissions: Optional[_PermissionsManager] = None
        self._secrets: Optional[_SecretsManager] = None
        self._validator: Optional[_Validator] = None
        self._sandbox: Optional[_Sandbox] = None
        self._audit_logger: Optional[_AuditLogger] = None
        self._approval_gate: Optional[_ApprovalGate] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SecurityConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "SecurityConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: Optional[SecurityConnectorConfig] = None,
    ) -> "SecurityConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        self._jwt_provider = _JWTProvider(self._config)
        self._auth_provider = _AuthProvider(self._config)
        self._rate_limiter = _RateLimiter(self._config)
        self._encryption = _EncryptionService(self._config)
        self._permissions = _PermissionsManager()
        self._secrets = _SecretsManager(self._encryption, self._config.secrets_encryption)
        self._validator = _Validator(self._config)
        self._sandbox = _Sandbox(self._config)
        self._audit_logger = _AuditLogger(self._config.audit_max_entries)
        self._approval_gate = _ApprovalGate(self._config.approval_timeout)

        # Wire audit callback to secrets
        self._secrets.add_audit_callback(self._on_secret_access)

        self._initialized = True
        logger.info(
            "SecurityConnector initialized: jwt=%s encryption=%s sandbox=%s",
            _JWT_AVAILABLE,
            _CRYPTOGRAPHY_AVAILABLE,
            self._config.enable_sandbox,
        )
        return self

    def configure_from_env(self) -> "SecurityConnector":
        self._config = SecurityConnectorConfig(
            jwt_secret=os.environ.get("JWT_SECRET", ""),
            jwt_algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
            jwt_expiry_seconds=int(os.environ.get("JWT_EXPIRY_SECONDS", "3600")),
            encryption_key=os.environ.get("ENCRYPTION_KEY"),
            rate_limit_default_rpm=int(os.environ.get("RATE_LIMIT_RPM", "60")),
            enable_redis_rate_limit=os.environ.get("REDIS_RATE_LIMIT", "").lower() == "true",
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        )
        return self

    # ------------------------------------------------------------------
    # Authenticate
    # ------------------------------------------------------------------

    async def authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[AuthStatus, Optional[Identity], Optional[AuthToken]]:
        await self._ensure_initialized()

        # JWT token auth
        if token:
            payload = await self._jwt_provider.verify_token(token)  # type: ignore[union-attr]
            if not payload:
                await self._audit(AuditAction.LOGIN, success=False, details={"method": "token"}, ip_address=ip_address)
                return AuthStatus.INVALID, None, None
            identity = Identity(
                user_id=payload["sub"],
                username=payload.get("username", ""),
                roles=payload.get("roles", []),
                permissions=[Permission(p) for p in payload.get("permissions", []) if p in Permission.__members__.values()],
                session_id=payload.get("session_id", str(uuid.uuid4())),
                expires_at=payload.get("exp"),
            )
            if identity.is_expired():
                await self._audit(AuditAction.LOGIN, user_id=identity.user_id, success=False, details={"reason": "expired"})
                return AuthStatus.EXPIRED, None, None
            await self._audit(AuditAction.LOGIN, user_id=identity.user_id, success=True, details={"method": "token"})
            return AuthStatus.AUTHENTICATED, identity, None

        # API key auth
        if api_key:
            identity = await self._auth_provider.authenticate_api_key(api_key)  # type: ignore[union-attr]
            if not identity:
                await self._audit(AuditAction.LOGIN, success=False, details={"method": "api_key"}, ip_address=ip_address)
                return AuthStatus.INVALID, None, None
            auth_token = await self._jwt_provider.create_token(identity)  # type: ignore[union-attr]
            await self._audit(AuditAction.LOGIN, user_id=identity.user_id, success=True, details={"method": "api_key"})
            return AuthStatus.AUTHENTICATED, identity, auth_token

        # Password auth
        if username and password:
            # Rate limit
            rate_result = await self._rate_limiter.check(f"login:{username}", rpm=10)  # type: ignore[union-attr]
            if not rate_result.allowed:
                await self._audit(AuditAction.RATE_LIMITED, details={"username": username})
                return AuthStatus.LOCKED, None, None

            try:
                identity = await self._auth_provider.authenticate_password(username, password)  # type: ignore[union-attr]
            except AuthenticationError as exc:
                await self._audit(AuditAction.LOGIN, success=False, details={"username": username, "error": str(exc)}, ip_address=ip_address)
                return AuthStatus.LOCKED, None, None

            if not identity:
                await self._audit(AuditAction.LOGIN, success=False, details={"username": username}, ip_address=ip_address)
                return AuthStatus.UNAUTHENTICATED, None, None

            auth_token = await self._jwt_provider.create_token(identity)  # type: ignore[union-attr]
            await self._audit(AuditAction.LOGIN, user_id=identity.user_id, success=True, details={"method": "password"}, ip_address=ip_address)
            return AuthStatus.AUTHENTICATED, identity, auth_token

        return AuthStatus.UNAUTHENTICATED, None, None

    async def logout(self, token: str, user_id: Optional[str] = None) -> bool:
        await self._ensure_initialized()
        await self._jwt_provider.revoke_token(token)  # type: ignore[union-attr]
        await self._audit(AuditAction.LOGOUT, user_id=user_id)
        return True

    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        await self._ensure_initialized()
        return await self._jwt_provider.refresh(refresh_token)  # type: ignore[union-attr]

    async def create_token(self, identity: Identity) -> AuthToken:
        await self._ensure_initialized()
        return await self._jwt_provider.create_token(identity)  # type: ignore[union-attr]

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        return await self._jwt_provider.verify_token(token)  # type: ignore[union-attr]

    async def register_user(
        self,
        username: str,
        password: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        await self._ensure_initialized()
        val = await self._validator.validate_input(password, "password")  # type: ignore[union-attr]
        if not val.valid:
            raise ValidationError(f"Password validation failed: {val.errors}")
        return await self._auth_provider.register_user(username, password, roles, permissions, metadata)  # type: ignore[union-attr]

    async def register_api_key(self, identity: Identity) -> str:
        await self._ensure_initialized()
        return await self._auth_provider.register_api_key(identity)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Authorize
    # ------------------------------------------------------------------

    async def authorize(
        self,
        identity: Identity,
        resource: str,
        permission: Permission,
        ip_address: Optional[str] = None,
    ) -> bool:
        await self._ensure_initialized()
        if identity.is_expired():
            await self._audit(AuditAction.DENIED, user_id=identity.user_id, resource=resource, success=False, details={"reason": "expired"})
            return False

        allowed = self._permissions.check(identity, resource, permission)  # type: ignore[union-attr]
        await self._audit(
            AuditAction.ACCESS if allowed else AuditAction.DENIED,
            user_id=identity.user_id,
            resource=resource,
            success=allowed,
            ip_address=ip_address,
        )
        return allowed

    async def require_role(self, identity: Identity, role: str) -> bool:
        return identity.has_role(role)

    async def require_permission(self, identity: Identity, permission: Permission) -> bool:
        return identity.has_permission(permission)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    async def validate(
        self,
        data: Any,
        data_type: str = "generic",
        max_length: int = 10000,
    ) -> ValidationResult:
        await self._ensure_initialized()
        result = await self._validator.validate_input(data, data_type, max_length)  # type: ignore[union-attr]
        await self._audit(
            AuditAction.VALIDATION,
            success=result.valid,
            risk_level=result.risk_level,
            details={"type": data_type, "errors": result.errors},
        )
        return result

    async def validate_json(self, data: Any) -> ValidationResult:
        await self._ensure_initialized()
        return await self._validator.validate_json(data)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Approve
    # ------------------------------------------------------------------

    async def approve(
        self,
        resource: str,
        requester: str,
        reason: str = "",
        notify_fn: Optional[Callable[..., Any]] = None,
    ) -> ApprovalGateResult:
        await self._ensure_initialized()
        result = await self._approval_gate.request(resource, requester, reason, notify_fn)  # type: ignore[union-attr]
        await self._audit(
            AuditAction.APPROVE if result.approved else AuditAction.REJECT,
            resource=resource,
            success=result.approved,
            details={"requester": requester, "reason": reason},
        )
        return result

    async def respond_to_approval(
        self,
        gate_id: str,
        approved: bool,
        approver: str = "",
        reason: str = "",
    ) -> bool:
        await self._ensure_initialized()
        return await self._approval_gate.respond(gate_id, approved, approver, reason)  # type: ignore[union-attr]

    async def get_pending_approvals(self) -> List[str]:
        await self._ensure_initialized()
        return await self._approval_gate.get_pending_ids()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Encrypt / Decrypt
    # ------------------------------------------------------------------

    async def encrypt(self, data: Any) -> str:
        await self._ensure_initialized()
        result = await self._encryption.encrypt(data)  # type: ignore[union-attr]
        await self._audit(AuditAction.ENCRYPT, success=True)
        return result

    async def decrypt(self, encrypted: str) -> Any:
        await self._ensure_initialized()
        result = await self._encryption.decrypt(encrypted)  # type: ignore[union-attr]
        await self._audit(AuditAction.DECRYPT, success=True)
        return result

    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        if not self._encryption:
            raise SecurityConnectorError("Not initialized")
        return self._encryption.hash(data, salt)

    def verify_hash(self, data: str, hashed: str) -> bool:
        if not self._encryption:
            raise SecurityConnectorError("Not initialized")
        return self._encryption.verify_hash(data, hashed)

    def generate_key(self) -> str:
        if not self._encryption:
            raise SecurityConnectorError("Not initialized")
        return self._encryption.generate_key()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def check_rate_limit(
        self,
        key: str,
        rpm: Optional[int] = None,
        burst: Optional[int] = None,
    ) -> RateLimitResult:
        await self._ensure_initialized()
        result = await self._rate_limiter.check(key, rpm, burst)  # type: ignore[union-attr]
        if not result.allowed:
            await self._audit(AuditAction.RATE_LIMITED, details={"key": key})
        return result

    async def reset_rate_limit(self, key: str) -> None:
        await self._ensure_initialized()
        await self._rate_limiter.reset(key)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Secrets
    # ------------------------------------------------------------------

    async def set_secret(self, key: str, value: str) -> None:
        await self._ensure_initialized()
        await self._secrets.set(key, value)  # type: ignore[union-attr]

    async def get_secret(self, key: str, requester: str = "") -> str:
        await self._ensure_initialized()
        return await self._secrets.get(key, requester)  # type: ignore[union-attr]

    async def delete_secret(self, key: str) -> bool:
        await self._ensure_initialized()
        return await self._secrets.delete(key)  # type: ignore[union-attr]

    async def secret_exists(self, key: str) -> bool:
        await self._ensure_initialized()
        return await self._secrets.exists(key)  # type: ignore[union-attr]

    async def list_secrets(self) -> List[str]:
        await self._ensure_initialized()
        return await self._secrets.list_keys()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------

    async def sandbox_execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> SandboxResult:
        await self._ensure_initialized()
        result = await self._sandbox.evaluate(code, context, timeout)  # type: ignore[union-attr]
        await self._audit(
            AuditAction.SANDBOX_EXEC,
            success=result.status != SandboxStatus.BLOCKED,
            risk_level=RiskLevel.HIGH if result.status == SandboxStatus.BLOCKED else RiskLevel.MEDIUM,
            details={"status": result.status.value, "blocked_reason": result.blocked_reason},
        )
        return result

    def is_safe_code(self, code: str) -> bool:
        if not self._sandbox:
            return True
        return self._sandbox.is_safe_code(code)

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def define_role(self, role: str, permissions: List[Permission]) -> None:
        if self._permissions:
            self._permissions.define_role(role, permissions)

    def grant_permission(self, resource: str, role_or_user: str, permissions: List[Permission]) -> None:
        if self._permissions:
            self._permissions.grant(resource, role_or_user, permissions)

    def revoke_permission(self, resource: str, role_or_user: str, permissions: List[Permission]) -> None:
        if self._permissions:
            self._permissions.revoke(resource, role_or_user, permissions)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    async def _audit(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        success: bool = True,
        risk_level: RiskLevel = RiskLevel.LOW,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        if self._audit_logger and self._config.enable_audit:
            await self._audit_logger.log(action, user_id, resource, success, risk_level, details, ip_address)

    async def query_audit_log(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        risk_level: Optional[RiskLevel] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        await self._ensure_initialized()
        return await self._audit_logger.query(user_id, action, risk_level, limit)  # type: ignore[union-attr]

    def add_audit_handler(self, fn: Callable[..., Any]) -> None:
        if self._audit_logger:
            self._audit_logger.add_handler(fn)

    async def _on_secret_access(self, event: str, details: Dict[str, Any]) -> None:
        await self._audit(
            AuditAction.SECRET_ACCESS,
            details=details,
            risk_level=RiskLevel.MEDIUM,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        audit_count = await self._audit_logger.count() if self._audit_logger else 0
        pending_approvals = await self._approval_gate.get_pending_ids() if self._approval_gate else []
        secrets_count = len(await self._secrets.list_keys()) if self._secrets else 0
        rate_ok = await self._rate_limiter.health_check() if self._rate_limiter else True  # type: ignore[union-attr, attr-defined]

        return {
            "healthy": True,
            "initialized": self._initialized,
            "components": {
                "jwt_provider": self._jwt_provider is not None,
                "jwt_available": _JWT_AVAILABLE,
                "auth_provider": self._auth_provider is not None,
                "rate_limiter": self._rate_limiter is not None,
                "rate_limiter_healthy": rate_ok,
                "encryption": self._encryption is not None,
                "cryptography_available": _CRYPTOGRAPHY_AVAILABLE,
                "bcrypt_available": _BCRYPT_AVAILABLE,
                "permissions": self._permissions is not None,
                "secrets_manager": self._secrets is not None,
                "validator": self._validator is not None,
                "sandbox": self._sandbox is not None,
                "audit_logger": self._audit_logger is not None,
                "approval_gate": self._approval_gate is not None,
            },
            "stats": {
                "audit_entries": audit_count,
                "pending_approvals": len(pending_approvals),
                "secrets_stored": secrets_count,
            },
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._rate_limiter:
            await self._rate_limiter.close()  # type: ignore[union-attr]
        self._initialized = False
        logger.info("SecurityConnector shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def __aenter__(self) -> "SecurityConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return f"SecurityConnector(initialized={self._initialized})"


# ---------------------------------------------------------------------------
# Type alias fix for missing import
# ---------------------------------------------------------------------------
from typing import Tuple  # noqa: E402


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def get_security_connector() -> SecurityConnector:
    return SecurityConnector.get_instance()


async def authenticate(username: str, password: str, **kwargs: Any):
    return await get_security_connector().authenticate(username=username, password=password, **kwargs)


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    return await get_security_connector().verify_token(token)


async def security_health_check() -> Dict[str, Any]:
    return await get_security_connector().health_check()
