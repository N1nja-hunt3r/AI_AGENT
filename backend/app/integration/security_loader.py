"""
security_loader.py

Initializes and registers all security subsystem components:
validator, sandbox, approval_gate, rate_limit, audit_logger,
permissions, encryption, secrets_manager and auth_provider.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus, Registry, get_registry

logger = logging.getLogger("security_loader")

_COMPONENT_NAMES: List[str] = [
    "validator",
    "sandbox",
    "approval_gate",
    "rate_limit",
    "audit_logger",
    "permissions",
    "encryption",
    "secrets_manager",
    "auth_provider",
]

_CLASS_MAP: Dict[str, str] = {
    "validator": "Validator",
    "sandbox": "Sandbox",
    "approval_gate": "ApprovalGate",
    "rate_limit": "RateLimiter",
    "audit_logger": "AuditLogger",
    "permissions": "PermissionsManager",
    "encryption": "EncryptionService",
    "secrets_manager": "SecretsManager",
    "auth_provider": "AuthProvider",
}


class _StubSecurityComponent:
    """Fallback no-op implementation used when a real module is unavailable."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._initialized = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return f"<StubSecurityComponent name={self.name!r}>"


def _import_component(module_name: str, class_name: str) -> Any:
    """
    Attempt to import a real component class from security.<module_name>.
    Falls back to a stub if the module is not present, so the loader
    degrades gracefully in environments where the security package has
    not been fully provisioned.
    """
    try:
        module = __import__(f"app.security.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        logger.warning(
            "security.%s.%s not found; using stub fallback.", module_name, class_name
        )
        return None


@dataclass
class SecurityLoaderConfig:
    enabled: bool = True
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    component_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fail_fast: bool = False


class SecurityLoader:
    """Loads, wires and supervises all security components."""

    def __init__(
        self,
        container: Optional[DependencyContainer] = None,
        config: Optional[SecurityLoaderConfig] = None,
    ) -> None:
        self._container = container or get_container()
        self._registry: Registry = get_registry("security")
        self._config = config or SecurityLoaderConfig()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._metrics: Dict[str, int] = {"init_success": 0, "init_failure": 0}

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            if not self._config.enabled:
                logger.info("Security subsystem disabled via configuration.")
                return

            await self.register_components()

            for name in self._registry.list():
                if self._config.feature_flags.get(name, True) is False:
                    logger.info("Skipping disabled security component '%s'.", name)
                    continue
                component = self._registry.get(name)
                try:
                    cfg = self._config.component_config.get(name, {})
                    if hasattr(component, "initialize"):
                        await component.initialize(cfg)
                    self._metrics["init_success"] += 1
                    logger.info("Initialized security component '%s'.", name)
                except Exception:
                    self._metrics["init_failure"] += 1
                    logger.exception("Failed to initialize security component '%s'.", name)
                    if self._config.fail_fast:
                        raise

            self._initialized = True

    async def register_components(self) -> None:
        for name in _COMPONENT_NAMES:
            if self._registry.has(name):
                continue
            cls = _import_component(name, _CLASS_MAP[name])
            instance = cls() if cls is not None else _StubSecurityComponent(name)
            self._registry.register(
                name,
                instance,
                version="1.0.0",
                metadata={"kind": "security"},
                health_check=getattr(instance, "health_check", None),
            )
            self._container.register_singleton(f"security:{name}", instance)

    def get_component(self, name: str) -> Any:
        return self._registry.get(name)

    def list_components(self) -> List[str]:
        return self._registry.list()

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    async def shutdown(self) -> None:
        async with self._lock:
            for name in self._registry.list():
                component = self._registry.get(name)
                try:
                    if hasattr(component, "shutdown"):
                        await component.shutdown()
                except Exception:
                    logger.exception("Error shutting down security component '%s'.", name)
            self._initialized = False

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_loader_lock = asyncio.Lock()
_security_loader: Optional[SecurityLoader] = None


async def get_security_loader() -> SecurityLoader:
    global _security_loader
    async with _loader_lock:
        if _security_loader is None:
            _security_loader = SecurityLoader()
        return _security_loader
