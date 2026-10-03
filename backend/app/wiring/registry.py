"""
Connector Registry - Production-grade registry for managing connectors.

Provides discovery, registration, versioning, health monitoring,
singleton pattern, thread safety, and full type annotations.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

__all__ = [
    "ConnectorRegistry",
    "BaseConnector",
    "ConnectorHealth",
    "ConnectorStatus",
    "ConnectorMetadata",
    "ConnectorVersion",
    "RegistryError",
    "ConnectorNotFoundError",
    "ConnectorAlreadyRegisteredError",
    "VersionConflictError",
    "connector",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConnector")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    """Base exception for all registry errors."""


class ConnectorNotFoundError(RegistryError):
    """Raised when a requested connector cannot be found."""


class ConnectorAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a duplicate connector."""


class VersionConflictError(RegistryError):
    """Raised on incompatible version operations."""


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ConnectorVersion:
    """Semantic version representation for connectors."""

    major: int
    minor: int
    patch: int
    pre_release: str = ""

    # ---------- construction helpers ----------

    @classmethod
    def parse(cls, version_string: str) -> "ConnectorVersion":
        """Parse a semantic version string, e.g. ``'1.2.3'`` or ``'2.0.0-beta'``."""
        if not version_string:
            raise ValueError("Version string must not be empty.")
        pre_release = ""
        if "-" in version_string:
            version_string, pre_release = version_string.split("-", 1)
        parts = version_string.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid version format '{version_string}'. Expected 'MAJOR.MINOR.PATCH'."
            )
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Non-integer version components: {exc}") from exc
        return cls(major=major, minor=minor, patch=patch, pre_release=pre_release)

    # ---------- utilities ----------

    def is_compatible_with(self, other: "ConnectorVersion") -> bool:
        """Two versions are compatible when they share the same major version."""
        return self.major == other.major

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.pre_release}" if self.pre_release else base

    def __repr__(self) -> str:
        return f"ConnectorVersion('{self!s}')"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ConnectorStatus(Enum):
    """Lifecycle / health status of a connector."""

    UNKNOWN = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    DISABLED = auto()


@dataclass
class ConnectorHealth:
    """Snapshot of a connector's health at a point in time."""

    status: ConnectorStatus = ConnectorStatus.UNKNOWN
    message: str = ""
    last_checked: float = field(default_factory=time.monotonic)
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        return self.status == ConnectorStatus.HEALTHY

    def __str__(self) -> str:
        latency = f", latency={self.latency_ms:.2f}ms" if self.latency_ms is not None else ""
        return f"ConnectorHealth(status={self.status.name}{latency}, msg='{self.message}')"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ConnectorMetadata:
    """Descriptive metadata attached to a registered connector."""

    name: str
    version: ConnectorVersion
    description: str = ""
    author: str = ""
    tags: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    deprecated: bool = False
    deprecation_message: str = ""
    registered_at: float = field(default_factory=time.monotonic)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Connector name must not be empty.")

    # convenience

    def qualified_name(self) -> str:
        """Return ``name@version`` string."""
        return f"{self.name}@{self.version}"

    def __str__(self) -> str:
        return self.qualified_name()


# ---------------------------------------------------------------------------
# Base Connector
# ---------------------------------------------------------------------------


class BaseConnector(ABC):
    """
    Abstract base class every connector must inherit from.

    Subclasses **must** implement:
    - :meth:`connect`
    - :meth:`disconnect`
    - :meth:`health_check`
    """

    #: Class-level metadata; populated by :func:`connector` decorator or manual assignment.
    metadata: ClassVar[ConnectorMetadata]

    def __init__(self) -> None:
        self._connected: bool = False
        self._health: ConnectorHealth = ConnectorHealth()
        self._lock: threading.Lock = threading.Lock()

    # ---------- abstract interface ----------

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection."""

    @abstractmethod
    def health_check(self) -> ConnectorHealth:
        """Probe the connector and return its current health."""

    # ---------- concrete helpers ----------

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_health(self) -> ConnectorHealth:
        with self._lock:
            return self._health

    def _set_health(self, health: ConnectorHealth) -> None:
        with self._lock:
            self._health = health

    # ---------- context manager ----------

    def __enter__(self) -> "BaseConnector":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        meta = getattr(self.__class__, "metadata", None)
        label = str(meta) if meta else self.__class__.__name__
        return f"<{label} connected={self._connected}>"


# ---------------------------------------------------------------------------
# Registry entry (internal)
# ---------------------------------------------------------------------------


@dataclass
class _RegistryEntry(Generic[T]):
    """Internal record stored in the registry."""

    connector_class: Type[T]
    metadata: ConnectorMetadata
    health: ConnectorHealth = field(default_factory=ConnectorHealth)
    enabled: bool = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ConnectorRegistry:
    """
    Thread-safe singleton registry for :class:`BaseConnector` implementations.

    Features
    --------
    - **Singleton**: Only one instance exists per process.
    - **Thread-safe**: All mutations are guarded by a reentrant lock.
    - **Discovery**: Iterate / filter registered connectors by name, tag, capability.
    - **Versioning**: Multiple versions of the same connector name can coexist.
    - **Health monitoring**: Per-connector health snapshots with background refresh.
    - **Typed**: Full generic and type-annotated public API.
    """

    _instance: ClassVar[Optional["ConnectorRegistry"]] = None
    _singleton_lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> "ConnectorRegistry":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._lock: threading.RLock = threading.RLock()
        # Primary store: (name, version_str) -> _RegistryEntry
        self._entries: Dict[Tuple[str, str], _RegistryEntry[Any]] = {}
        # Background health-check thread management
        self._health_thread: Optional[threading.Thread] = None
        self._health_interval: float = 30.0  # seconds
        self._health_stop_event: threading.Event = threading.Event()
        self._initialized = True
        logger.info("ConnectorRegistry initialised (singleton).")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        connector_class: Type[T],
        *,
        allow_override: bool = False,
    ) -> None:
        """
        Register a connector class.

        Parameters
        ----------
        connector_class:
            The :class:`BaseConnector` subclass to register.
        allow_override:
            When ``True`` an existing entry for the same ``(name, version)``
            key will be silently replaced; otherwise raises
            :class:`ConnectorAlreadyRegisteredError`.

        Raises
        ------
        TypeError
            If *connector_class* is not a subclass of :class:`BaseConnector`.
        AttributeError
            If *connector_class* is missing a ``metadata`` class attribute.
        ConnectorAlreadyRegisteredError
            If the connector is already registered and *allow_override* is ``False``.
        """
        if not (isinstance(connector_class, type) and issubclass(connector_class, BaseConnector)):
            raise TypeError(
                f"Expected a subclass of BaseConnector, got {connector_class!r}."
            )
        if not hasattr(connector_class, "metadata"):
            raise AttributeError(
                f"{connector_class.__name__} is missing a 'metadata' class attribute."
            )

        meta: ConnectorMetadata = connector_class.metadata
        key = self._make_key(meta.name, meta.version)

        with self._lock:
            if key in self._entries and not allow_override:
                raise ConnectorAlreadyRegisteredError(
                    f"Connector '{meta.qualified_name()}' is already registered. "
                    "Use allow_override=True to replace it."
                )
            entry: _RegistryEntry[T] = _RegistryEntry(
                connector_class=connector_class,
                metadata=meta,
            )
            self._entries[key] = entry
            logger.info("Registered connector: %s", meta.qualified_name())

        if meta.deprecated:
            logger.warning(
                "Connector '%s' is deprecated. %s",
                meta.qualified_name(),
                meta.deprecation_message,
            )

    def unregister(self, name: str, version: ConnectorVersion) -> None:
        """
        Remove a connector from the registry.

        Raises
        ------
        ConnectorNotFoundError
            If no matching connector is found.
        """
        key = self._make_key(name, version)
        with self._lock:
            if key not in self._entries:
                raise ConnectorNotFoundError(
                    f"No connector '{name}@{version}' found in registry."
                )
            del self._entries[key]
            logger.info("Unregistered connector: %s@%s", name, version)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get(
        self,
        name: str,
        version: Optional[ConnectorVersion] = None,
    ) -> Type[BaseConnector]:
        """
        Retrieve a connector class by name (and optionally version).

        When *version* is omitted the latest registered version is returned.

        Raises
        ------
        ConnectorNotFoundError
            If no matching connector is found.
        """
        with self._lock:
            if version is not None:
                key = self._make_key(name, version)
                entry = self._entries.get(key)
                if entry is None:
                    raise ConnectorNotFoundError(
                        f"Connector '{name}@{version}' not found."
                    )
                return entry.connector_class

            # Latest version
            candidates = [
                e for e in self._entries.values() if e.metadata.name == name and e.enabled
            ]
            if not candidates:
                raise ConnectorNotFoundError(
                    f"No enabled connector with name '{name}' found."
                )
            best = max(candidates, key=lambda e: e.metadata.version)
            return best.connector_class

    def get_all_versions(self, name: str) -> List[ConnectorMetadata]:
        """Return metadata for every registered version of *name*, sorted ascending."""
        with self._lock:
            versions = [
                e.metadata
                for e in self._entries.values()
                if e.metadata.name == name
            ]
        return sorted(versions, key=lambda m: m.version)

    def discover(
        self,
        *,
        tag: Optional[str] = None,
        capability: Optional[str] = None,
        include_disabled: bool = False,
        include_deprecated: bool = True,
    ) -> List[ConnectorMetadata]:
        """
        Discover registered connectors with optional filtering.

        Parameters
        ----------
        tag:
            If given, only connectors whose ``tags`` set contains this value.
        capability:
            If given, only connectors whose ``capabilities`` set contains this value.
        include_disabled:
            Include connectors that have been disabled.
        include_deprecated:
            Include connectors marked as deprecated.

        Returns
        -------
        List[ConnectorMetadata]
            Sorted list of matching :class:`ConnectorMetadata` objects.
        """
        with self._lock:
            entries = list(self._entries.values())

        results: List[ConnectorMetadata] = []
        for entry in entries:
            if not include_disabled and not entry.enabled:
                continue
            if not include_deprecated and entry.metadata.deprecated:
                continue
            if tag and tag not in entry.metadata.tags:
                continue
            if capability and capability not in entry.metadata.capabilities:
                continue
            results.append(entry.metadata)

        return sorted(results, key=lambda m: (m.name, m.version))

    def list_names(self) -> List[str]:
        """Return a sorted list of unique connector names."""
        with self._lock:
            return sorted({e.metadata.name for e in self._entries.values()})

    def __iter__(self) -> Iterator[ConnectorMetadata]:
        """Iterate over all registered connector metadata records."""
        with self._lock:
            snapshot = list(self._entries.values())
        for entry in snapshot:
            yield entry.metadata

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, item: object) -> bool:
        """Support ``(name, version)`` or ``ConnectorMetadata`` membership test."""
        if isinstance(item, tuple) and len(item) == 2:
            name, version = item
            if isinstance(version, ConnectorVersion):
                key = self._make_key(name, version)
                with self._lock:
                    return key in self._entries
        if isinstance(item, ConnectorMetadata):
            key = self._make_key(item.name, item.version)
            with self._lock:
                return key in self._entries
        return False

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    def compatible_versions(
        self, name: str, version: ConnectorVersion
    ) -> List[ConnectorMetadata]:
        """
        Return all registered versions of *name* that are compatible with *version*
        (i.e. share the same major version number), sorted ascending.
        """
        all_versions = self.get_all_versions(name)
        return [m for m in all_versions if m.version.is_compatible_with(version)]

    def latest_version(self, name: str) -> Optional[ConnectorVersion]:
        """Return the latest registered :class:`ConnectorVersion` for *name*."""
        candidates = self.get_all_versions(name)
        if not candidates:
            return None
        return max(m.version for m in candidates)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def update_health(
        self,
        name: str,
        version: ConnectorVersion,
        health: ConnectorHealth,
    ) -> None:
        """Manually push a health snapshot for a registered connector."""
        key = self._make_key(name, version)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise ConnectorNotFoundError(
                    f"Cannot update health; connector '{name}@{version}' not found."
                )
            entry.health = health
        logger.debug(
            "Health updated for %s@%s → %s", name, version, health.status.name
        )

    def get_health(
        self, name: str, version: ConnectorVersion
    ) -> ConnectorHealth:
        """Return the last known health for a connector."""
        key = self._make_key(name, version)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise ConnectorNotFoundError(
                    f"Connector '{name}@{version}' not found."
                )
            return entry.health

    def health_report(self) -> Dict[str, ConnectorHealth]:
        """Return a snapshot ``{qualified_name: health}`` for every connector."""
        with self._lock:
            return {
                entry.metadata.qualified_name(): entry.health
                for entry in self._entries.values()
            }

    def probe_all(self) -> Dict[str, ConnectorHealth]:
        """
        Instantiate every registered, enabled connector and call
        :meth:`~BaseConnector.health_check` on it.

        Results are persisted back into the registry and returned.

        .. warning::
            This creates *transient* connector instances purely for probing.
            It does **not** maintain persistent connections.
        """
        report: Dict[str, ConnectorHealth] = {}
        with self._lock:
            entries = list(self._entries.values())

        for entry in entries:
            if not entry.enabled:
                continue
            qname = entry.metadata.qualified_name()
            try:
                start = time.monotonic()
                instance = entry.connector_class()
                health = instance.health_check()
                health.latency_ms = (time.monotonic() - start) * 1_000
                health.last_checked = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                health = ConnectorHealth(
                    status=ConnectorStatus.UNHEALTHY,
                    message=str(exc),
                    last_checked=time.monotonic(),
                )
                logger.warning("Health probe failed for %s: %s", qname, exc)

            try:
                self.update_health(
                    entry.metadata.name,
                    entry.metadata.version,
                    health,
                )
            except ConnectorNotFoundError:
                pass  # Connector was unregistered concurrently – skip silently.

            report[qname] = health

        return report

    # ------------------------------------------------------------------
    # Background health monitoring
    # ------------------------------------------------------------------

    def start_health_monitoring(self, interval: float = 30.0) -> None:
        """
        Start a daemon thread that periodically calls :meth:`probe_all`.

        Parameters
        ----------
        interval:
            Seconds between successive health sweeps. Defaults to 30 s.
        """
        with self._lock:
            if self._health_thread and self._health_thread.is_alive():
                logger.debug("Health monitoring thread already running.")
                return
            self._health_interval = interval
            self._health_stop_event.clear()
            self._health_thread = threading.Thread(
                target=self._health_loop,
                name="ConnectorRegistry-HealthMonitor",
                daemon=True,
            )
            self._health_thread.start()
            logger.info(
                "Health monitoring started (interval=%.1fs).", interval
            )

    def stop_health_monitoring(self, timeout: float = 5.0) -> None:
        """Signal the background health thread to stop and wait for it."""
        self._health_stop_event.set()
        if self._health_thread:
            self._health_thread.join(timeout=timeout)
            logger.info("Health monitoring stopped.")

    def _health_loop(self) -> None:
        while not self._health_stop_event.wait(timeout=self._health_interval):
            try:
                self.probe_all()
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error in health loop: %s", exc)

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(self, name: str, version: ConnectorVersion) -> None:
        """Re-enable a previously disabled connector."""
        self._set_enabled(name, version, enabled=True)

    def disable(self, name: str, version: ConnectorVersion) -> None:
        """Disable a connector so it is excluded from discovery and probing."""
        self._set_enabled(name, version, enabled=False)

    def _set_enabled(
        self, name: str, version: ConnectorVersion, *, enabled: bool
    ) -> None:
        key = self._make_key(name, version)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise ConnectorNotFoundError(
                    f"Connector '{name}@{version}' not found."
                )
            entry.enabled = enabled
        state = "enabled" if enabled else "disabled"
        logger.info("Connector %s@%s %s.", name, version, state)

    # ------------------------------------------------------------------
    # Reset (testing / lifecycle)
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered connectors and stop health monitoring."""
        self.stop_health_monitoring()
        with self._lock:
            self._entries.clear()
        logger.warning("ConnectorRegistry cleared.")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Destroy the singleton instance.

        .. warning::
            Intended for **testing only**. Do not call in production code.
        """
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None
        logger.warning("ConnectorRegistry singleton reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(name: str, version: ConnectorVersion) -> Tuple[str, str]:
        return (name, str(version))

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return f"ConnectorRegistry(connectors={count})"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def connector(
    name: str,
    version: str,
    *,
    description: str = "",
    author: str = "",
    tags: Optional[Set[str]] = None,
    capabilities: Optional[Set[str]] = None,
    deprecated: bool = False,
    deprecation_message: str = "",
    auto_register: bool = True,
    allow_override: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator that attaches :class:`ConnectorMetadata` to a
    :class:`BaseConnector` subclass and optionally registers it.

    Example
    -------
    .. code-block:: python

        @connector(
            name="postgres",
            version="1.0.0",
            tags={"database", "sql"},
            capabilities={"read", "write", "transaction"},
        )
        class PostgresConnector(BaseConnector):
            ...
    """

    def decorator(cls: Type[T]) -> Type[T]:
        parsed_version = ConnectorVersion.parse(version)
        meta = ConnectorMetadata(
            name=name,
            version=parsed_version,
            description=description,
            author=author,
            tags=tags or set(),
            capabilities=capabilities or set(),
            deprecated=deprecated,
            deprecation_message=deprecation_message,
            extra=extra or {},
        )
        cls.metadata = meta  # type: ignore[attr-defined]
        if auto_register:
            ConnectorRegistry().register(cls, allow_override=allow_override)
        return cls

    return decorator
