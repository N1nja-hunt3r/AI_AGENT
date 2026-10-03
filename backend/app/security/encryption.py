from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes as _crypto_hashes
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


class EncryptionError(Exception):
    pass


class DecryptionError(EncryptionError):
    pass


class KeyNotFoundError(EncryptionError):
    pass


@dataclass(frozen=True)
class EncryptedPayload:
    ciphertext: str
    nonce: str
    key_id: str
    algorithm: str = "AES-256-GCM"
    salt: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        d = {
            "ciphertext": self.ciphertext,
            "nonce": self.nonce,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
        }
        if self.salt:
            d["salt"] = self.salt
        return d

    @staticmethod
    def from_dict(data: dict[str, str]) -> "EncryptedPayload":
        return EncryptedPayload(
            ciphertext=data["ciphertext"],
            nonce=data["nonce"],
            key_id=data["key_id"],
            algorithm=data.get("algorithm", "AES-256-GCM"),
            salt=data.get("salt"),
        )


@dataclass(frozen=True)
class KeyMetadata:
    key_id: str
    created_at: float
    rotated_from: Optional[str] = None
    active: bool = True


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode("ascii"))


class EncryptionManager:
    """AES-256-GCM encryption, hashing, and key management."""

    KEY_SIZE_BYTES = 32
    NONCE_SIZE_BYTES = 12

    def __init__(self, *, master_key: Optional[bytes] = None) -> None:
        if not _HAS_CRYPTOGRAPHY:
            raise EncryptionError(
                "the 'cryptography' package is required for EncryptionManager; install with: pip install cryptography"
            )
        self._keys: dict[str, bytes] = {}
        self._key_metadata: dict[str, KeyMetadata] = {}
        self._active_key_id: Optional[str] = None
        self._created_at = time.time()
        self._encrypt_count = 0
        self._decrypt_count = 0

        if master_key is not None:
            self.import_key("master", master_key)
            self._active_key_id = "master"
        else:
            self.generate_key("master", activate=True)

    def generate_key(self, key_id: str, *, activate: bool = False) -> KeyMetadata:
        key_bytes = secrets.token_bytes(self.KEY_SIZE_BYTES)
        self._keys[key_id] = key_bytes
        meta = KeyMetadata(key_id=key_id, created_at=time.time())
        self._key_metadata[key_id] = meta
        if activate or self._active_key_id is None:
            self._active_key_id = key_id
        return meta

    def import_key(self, key_id: str, key_bytes: bytes) -> KeyMetadata:
        if len(key_bytes) != self.KEY_SIZE_BYTES:
            raise EncryptionError(f"key must be exactly {self.KEY_SIZE_BYTES} bytes, got {len(key_bytes)}")
        self._keys[key_id] = key_bytes
        meta = KeyMetadata(key_id=key_id, created_at=time.time())
        self._key_metadata[key_id] = meta
        return meta

    def derive_key(self, key_id: str, *, passphrase: str, salt: Optional[bytes] = None, iterations: int = 600_000) -> tuple[KeyMetadata, bytes]:
        salt = salt or secrets.token_bytes(16)
        kdf = PBKDF2HMAC(algorithm=_crypto_hashes.SHA256(), length=self.KEY_SIZE_BYTES, salt=salt, iterations=iterations)
        key_bytes = kdf.derive(passphrase.encode("utf-8"))
        meta = self.import_key(key_id, key_bytes)
        return meta, salt

    def rotate_key(self, *, new_key_id: Optional[str] = None) -> KeyMetadata:
        old_id = self._active_key_id
        new_id = new_key_id or f"key-{secrets.token_hex(8)}"
        meta = self.generate_key(new_id, activate=True)
        if old_id is not None and old_id in self._key_metadata:
            old_meta = self._key_metadata[old_id]
            self._key_metadata[old_id] = KeyMetadata(
                key_id=old_meta.key_id, created_at=old_meta.created_at, rotated_from=old_meta.rotated_from, active=False
            )
        self._key_metadata[new_id] = KeyMetadata(key_id=new_id, created_at=meta.created_at, rotated_from=old_id, active=True)
        return self._key_metadata[new_id]

    def deactivate_key(self, key_id: str) -> None:
        if key_id not in self._key_metadata:
            raise KeyNotFoundError(f"key '{key_id}' not found")
        old = self._key_metadata[key_id]
        self._key_metadata[key_id] = KeyMetadata(
            key_id=old.key_id, created_at=old.created_at, rotated_from=old.rotated_from, active=False
        )

    def list_keys(self) -> list[KeyMetadata]:
        return list(self._key_metadata.values())

    def encrypt(self, plaintext: str | bytes, *, key_id: Optional[str] = None, associated_data: Optional[bytes] = None) -> EncryptedPayload:
        self._encrypt_count += 1
        active_id = key_id or self._active_key_id
        if active_id is None or active_id not in self._keys:
            raise KeyNotFoundError(f"key '{active_id}' not found")
        key_bytes = self._keys[active_id]
        data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        nonce = secrets.token_bytes(self.NONCE_SIZE_BYTES)
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        return EncryptedPayload(ciphertext=_b64e(ciphertext), nonce=_b64e(nonce), key_id=active_id)

    def decrypt(self, payload: EncryptedPayload, *, associated_data: Optional[bytes] = None) -> bytes:
        self._decrypt_count += 1
        if payload.key_id not in self._keys:
            raise KeyNotFoundError(f"key '{payload.key_id}' not found")
        key_bytes = self._keys[payload.key_id]
        aesgcm = AESGCM(key_bytes)
        try:
            return aesgcm.decrypt(_b64d(payload.nonce), _b64d(payload.ciphertext), associated_data)
        except Exception as exc:
            raise DecryptionError(f"decryption failed: {exc}") from exc

    def decrypt_str(self, payload: EncryptedPayload, *, associated_data: Optional[bytes] = None) -> str:
        return self.decrypt(payload, associated_data=associated_data).decode("utf-8")

    def encrypt_secret(self, secret_value: str, *, secret_name: str) -> EncryptedPayload:
        return self.encrypt(secret_value, associated_data=secret_name.encode("utf-8"))

    def decrypt_secret(self, payload: EncryptedPayload, *, secret_name: str) -> str:
        return self.decrypt_str(payload, associated_data=secret_name.encode("utf-8"))

    @staticmethod
    def hash_sha256(data: str | bytes) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def hash_sha512(data: str | bytes) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hashlib.sha512(raw).hexdigest()

    @staticmethod
    def hmac_sign(data: str | bytes, *, key: bytes) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hmac.new(key, raw, hashlib.sha256).hexdigest()

    @staticmethod
    def hmac_verify(data: str | bytes, *, key: bytes, signature: str) -> bool:
        expected = EncryptionManager.hmac_sign(data, key=key)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def hash_password(password: str, *, salt: Optional[bytes] = None, iterations: int = 600_000) -> tuple[str, str]:
        salt = salt or os.urandom(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return _b64e(derived), _b64e(salt)

    @staticmethod
    def verify_password(password: str, *, hash_b64: str, salt_b64: str, iterations: int = 600_000) -> bool:
        salt = _b64d(salt_b64)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(_b64e(derived), hash_b64)

    @staticmethod
    def generate_token(num_bytes: int = 32) -> str:
        return secrets.token_urlsafe(num_bytes)

    def health_check(self) -> HealthStatus:
        try:
            probe = self.encrypt("health-check-probe")
            recovered = self.decrypt_str(probe)
            healthy = recovered == "health-check-probe"
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "active_key_id": self._active_key_id,
                "key_count": len(self._keys),
                "active_keys": [m.key_id for m in self._key_metadata.values() if m.active],
                "encrypt_count": self._encrypt_count,
                "decrypt_count": self._decrypt_count,
                "cryptography_available": _HAS_CRYPTOGRAPHY,
            }
            return HealthStatus(healthy=healthy, component="encryption", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="encryption", details={"error": str(exc)})


__all__ = [
    "EncryptionManager",
    "EncryptedPayload",
    "KeyMetadata",
    "EncryptionError",
    "DecryptionError",
    "KeyNotFoundError",
    "HealthStatus",
]
