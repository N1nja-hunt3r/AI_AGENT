from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SecretSource(str, Enum):
    ENVIRONMENT = "environment"
    MEMORY = "memory"
    VAULT = "vault"


class SecretsManagerError(Exception):
    pass


class SecretNotFoundError(SecretsManagerError):
    pass


class VaultBackendError(SecretsManagerError):
    pass


class VaultBackend:
    """Abstract interface for an external secrets vault (e.g. HashiCorp Vault, AWS Secrets Manager)."""

    async def read(self, path: str) -> Optional[str]:
        raise NotImplementedError

    async def write(self, path: str, value: str) -> None:
        raise NotImplementedError

    async def delete(self, path: str) -> None:
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError


class InMemoryVaultBackend(VaultBackend):
    """Local fallback vault implementation. Replace with a real backend in production."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def read(self, path: str) -> Optional[str]:
        return self._store.get(path)

    async def write(self, path: str, value: str) -> None:
        self._store[path] = value

    async def delete(self, path: str) -> None:
        self._store.pop(path, None)

    async def health_check(self) -> bool:
        return True


@dataclass
class SecretMetadata:
    name: str
    source: SecretSource
    created_at: float
    rotated_at: Optional[float] = None
    rotation_count: int = 0
    version: int = 1
    expires_at: Optional[float] = None
    tags: dict[str, str] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time.time()) >= self.expires_at


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


_SECRET_NAME_RE = re.compile(r"^[A-Za-z0-9_\-./]{1,256}$")


class SecretsManager:
    """API key and secret management with environment, in-memory, and vault sources."""

    def __init__(
        self,
        *,
        vault_backend: Optional[VaultBackend] = None,
        env_prefix: str = "",
        default_rotation_seconds: Optional[float] = None,
    ) -> None:
        self.vault_backend = vault_backend or InMemoryVaultBackend()
        self.env_prefix = env_prefix
        self.default_rotation_seconds = default_rotation_seconds
        self._memory_store: dict[str, str] = {}
        self._metadata: dict[str, SecretMetadata] = {}
        self._rotators: dict[str, Callable[[], str]] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._read_count = 0
        self._write_count = 0
        self._rotation_count = 0

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _SECRET_NAME_RE.match(name):
            raise SecretsManagerError(f"invalid secret name '{name}'")

    def get_from_env(self, name: str, *, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        self._validate_name(name)
        env_key = f"{self.env_prefix}{name}"
        value = os.environ.get(env_key)
        if value is None:
            if default is not None:
                return default
            if required:
                raise SecretNotFoundError(f"environment variable '{env_key}' not set")
            return None
        return value

    def set_in_memory(
        self,
        name: str,
        value: str,
        *,
        expires_in_seconds: Optional[float] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> SecretMetadata:
        self._validate_name(name)
        self._write_count += 1
        now = time.time()
        existing = self._metadata.get(name)
        meta = SecretMetadata(
            name=name,
            source=SecretSource.MEMORY,
            created_at=existing.created_at if existing else now,
            version=(existing.version + 1) if existing else 1,
            expires_at=(now + expires_in_seconds) if expires_in_seconds else None,
            tags=tags or {},
        )
        self._memory_store[name] = value
        self._metadata[name] = meta
        return meta

    def get_from_memory(self, name: str, *, required: bool = True) -> Optional[str]:
        self._validate_name(name)
        self._read_count += 1
        meta = self._metadata.get(name)
        if meta is not None and meta.is_expired():
            self._memory_store.pop(name, None)
            self._metadata.pop(name, None)
            if required:
                raise SecretNotFoundError(f"secret '{name}' has expired")
            return None
        value = self._memory_store.get(name)
        if value is None and required:
            raise SecretNotFoundError(f"secret '{name}' not found in memory store")
        return value

    def delete_from_memory(self, name: str) -> bool:
        self._metadata.pop(name, None)
        return self._memory_store.pop(name, None) is not None

    def register_rotator(self, name: str, rotator: Callable[[], str]) -> None:
        """Register a callable that produces a fresh secret value when rotation is triggered."""
        self._rotators[name] = rotator

    def rotate_memory_secret(self, name: str) -> SecretMetadata:
        self._validate_name(name)
        rotator = self._rotators.get(name)
        if rotator is None:
            raise SecretsManagerError(f"no rotator registered for secret '{name}'")
        new_value = rotator()
        meta = self.set_in_memory(name, new_value)
        meta.rotated_at = time.time()
        meta.rotation_count = (self._metadata[name].rotation_count if name in self._metadata else 0) + 1
        self._metadata[name] = meta
        self._rotation_count += 1
        return meta

    def get_secret(self, name: str, *, prefer: tuple[SecretSource, ...] = (SecretSource.MEMORY, SecretSource.ENVIRONMENT), required: bool = True) -> Optional[str]:
        for source in prefer:
            if source == SecretSource.MEMORY:
                value = self.get_from_memory(name, required=False)
                if value is not None:
                    return value
            elif source == SecretSource.ENVIRONMENT:
                value = self.get_from_env(name, required=False)
                if value is not None:
                    return value
        if required:
            raise SecretNotFoundError(f"secret '{name}' not found in any source {prefer}")
        return None

    def get_metadata(self, name: str) -> Optional[SecretMetadata]:
        return self._metadata.get(name)

    def list_secret_names(self) -> list[str]:
        return sorted(self._metadata.keys())

    async def get_from_vault(self, name: str, *, required: bool = True) -> Optional[str]:
        self._validate_name(name)
        self._read_count += 1
        try:
            value = await self.vault_backend.read(name)
        except Exception as exc:
            raise VaultBackendError(f"vault read failed for '{name}': {exc}") from exc
        if value is None and required:
            raise SecretNotFoundError(f"secret '{name}' not found in vault")
        return value

    async def set_in_vault(self, name: str, value: str, *, tags: Optional[dict[str, str]] = None) -> SecretMetadata:
        self._validate_name(name)
        self._write_count += 1
        async with self._lock:
            try:
                await self.vault_backend.write(name, value)
            except Exception as exc:
                raise VaultBackendError(f"vault write failed for '{name}': {exc}") from exc
            now = time.time()
            existing = self._metadata.get(name)
            meta = SecretMetadata(
                name=name, source=SecretSource.VAULT,
                created_at=existing.created_at if existing else now,
                version=(existing.version + 1) if existing else 1,
                tags=tags or {},
            )
            self._metadata[name] = meta
            return meta

    async def delete_from_vault(self, name: str) -> None:
        async with self._lock:
            try:
                await self.vault_backend.delete(name)
            except Exception as exc:
                raise VaultBackendError(f"vault delete failed for '{name}': {exc}") from exc
            self._metadata.pop(name, None)

    async def rotate_vault_secret(self, name: str) -> SecretMetadata:
        rotator = self._rotators.get(name)
        if rotator is None:
            raise SecretsManagerError(f"no rotator registered for secret '{name}'")
        new_value = rotator()
        meta = await self.set_in_vault(name, new_value)
        meta.rotated_at = time.time()
        async with self._lock:
            meta.rotation_count = (self._metadata[name].rotation_count if name in self._metadata else 0) + 1
            self._metadata[name] = meta
            self._rotation_count += 1
        return meta

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "memory_secret_count": len(self._memory_store),
                "tracked_metadata_count": len(self._metadata),
                "registered_rotators": list(self._rotators.keys()),
                "read_count": self._read_count,
                "write_count": self._write_count,
                "rotation_count": self._rotation_count,
            }
            return HealthStatus(healthy=True, component="secrets_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="secrets_manager", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        try:
            vault_healthy = await self.vault_backend.health_check()
        except Exception:
            vault_healthy = False
        base = self.health_check()
        base.details["vault_backend_healthy"] = vault_healthy
        return HealthStatus(healthy=base.healthy and vault_healthy, component="secrets_manager", details=base.details)


__all__ = [
    "SecretsManager",
    "VaultBackend",
    "InMemoryVaultBackend",
    "SecretMetadata",
    "SecretSource",
    "SecretsManagerError",
    "SecretNotFoundError",
    "VaultBackendError",
    "HealthStatus",
]
