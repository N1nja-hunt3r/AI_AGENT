"""
dependency_container.py

Thread-safe dependency injection container supporting singleton,
transient and factory-based service registration, lazy resolution,
and first-class helpers for capability, agent and router injection.
"""

from __future__ import annotations

import inspect
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

T = TypeVar("T")

Factory = Callable[..., Any]
Key = Union[str, Type[Any]]


class Lifetime(str, Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    FACTORY = "factory"
    LAZY_SINGLETON = "lazy_singleton"


class ServiceNotRegisteredError(KeyError):
    pass


class ServiceDescriptor:
    __slots__ = ("key", "lifetime", "provider", "instance", "tags")

    def __init__(
        self,
        key: Key,
        lifetime: Lifetime,
        provider: Factory,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.key = key
        self.lifetime = lifetime
        self.provider = provider
        self.instance: Any = None
        self.tags: Dict[str, Any] = tags or {}


def _key_name(key: Key) -> str:
    return key if isinstance(key, str) else key.__name__


class DependencyContainer:
    """A thread-safe service container for dependency injection."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._descriptors: Dict[str, ServiceDescriptor] = {}

    # ------------------------------------------------------------------ #
    # Core registration API
    # ------------------------------------------------------------------ #
    def register_singleton(
        self,
        key: Key,
        instance_or_factory: Union[Any, Factory],
        *,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        name = _key_name(key)
        with self._lock:
            if inspect.isclass(instance_or_factory):
                descriptor = ServiceDescriptor(key, Lifetime.SINGLETON, instance_or_factory, tags)
            elif callable(instance_or_factory) and not isinstance(instance_or_factory, (str, bytes, dict, list)):
                descriptor = ServiceDescriptor(key, Lifetime.SINGLETON, instance_or_factory, tags)
            else:
                descriptor = ServiceDescriptor(key, Lifetime.SINGLETON, lambda: instance_or_factory, tags)
                descriptor.instance = instance_or_factory
            self._descriptors[name] = descriptor

    def register_transient(
        self,
        key: Key,
        factory: Factory,
        *,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        name = _key_name(key)
        with self._lock:
            self._descriptors[name] = ServiceDescriptor(key, Lifetime.TRANSIENT, factory, tags)

    def register_factory(
        self,
        key: Key,
        factory: Factory,
        *,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        name = _key_name(key)
        with self._lock:
            self._descriptors[name] = ServiceDescriptor(key, Lifetime.FACTORY, factory, tags)

    def register_lazy_singleton(
        self,
        key: Key,
        factory: Factory,
        *,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a singleton whose factory is only invoked on first resolve()."""
        name = _key_name(key)
        with self._lock:
            self._descriptors[name] = ServiceDescriptor(key, Lifetime.LAZY_SINGLETON, factory, tags)

    def register_instance(self, key: Key, instance: Any, *, tags: Optional[Dict[str, Any]] = None) -> None:
        name = _key_name(key)
        with self._lock:
            descriptor = ServiceDescriptor(key, Lifetime.SINGLETON, lambda: instance, tags)
            descriptor.instance = instance
            self._descriptors[name] = descriptor

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #
    def resolve(self, key: Key, *args: Any, **kwargs: Any) -> Any:
        name = _key_name(key)
        with self._lock:
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                raise ServiceNotRegisteredError(f"Service '{name}' is not registered.")

            if descriptor.lifetime in (Lifetime.SINGLETON, Lifetime.LAZY_SINGLETON):
                if descriptor.instance is None:
                    descriptor.instance = descriptor.provider(*args, **kwargs)
                return descriptor.instance

            # TRANSIENT / FACTORY: always create a new instance.
            return descriptor.provider(*args, **kwargs)

    def try_resolve(self, key: Key, *args: Any, **kwargs: Any) -> Optional[Any]:
        try:
            return self.resolve(key, *args, **kwargs)
        except ServiceNotRegisteredError:
            return None

    def is_registered(self, key: Key) -> bool:
        with self._lock:
            return _key_name(key) in self._descriptors

    def unregister(self, key: Key) -> None:
        with self._lock:
            self._descriptors.pop(_key_name(key), None)

    def list_services(self) -> Dict[str, Lifetime]:
        with self._lock:
            return {k: v.lifetime for k, v in self._descriptors.items()}

    # ------------------------------------------------------------------ #
    # Domain-specific convenience helpers
    # ------------------------------------------------------------------ #
    def register_capability(self, name: str, capability: Any) -> None:
        self.register_singleton(f"capability:{name}", capability, tags={"kind": "capability"})

    def get_capability(self, name: str) -> Any:
        return self.resolve(f"capability:{name}")

    def register_agent(self, name: str, agent: Any) -> None:
        self.register_singleton(f"agent:{name}", agent, tags={"kind": "agent"})

    def get_agent(self, name: str) -> Any:
        return self.resolve(f"agent:{name}")

    def register_router(self, name: str, router: Any) -> None:
        self.register_singleton(f"router:{name}", router, tags={"kind": "router"})

    def get_router(self, name: str) -> Any:
        return self.resolve(f"router:{name}")

    def register_service(self, name: str, service: Any) -> None:
        self.register_singleton(f"service:{name}", service, tags={"kind": "service"})

    def get_service(self, name: str) -> Any:
        return self.resolve(f"service:{name}")

    def by_tag(self, key: str, value: Any) -> Dict[str, Any]:
        with self._lock:
            return {
                name: (d.instance if d.instance is not None else d.provider)
                for name, d in self._descriptors.items()
                if d.tags.get(key) == value
            }

    def clear(self) -> None:
        with self._lock:
            self._descriptors.clear()


_container_lock = threading.RLock()
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Return the process-wide singleton DependencyContainer instance."""
    global _container
    with _container_lock:
        if _container is None:
            _container = DependencyContainer()
        return _container
