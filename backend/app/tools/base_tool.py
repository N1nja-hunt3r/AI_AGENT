"""Abstract base class for tools with lifecycle, metadata, and health checks."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ToolPriority(IntEnum):
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class ToolStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class ToolError(Exception):
    """Base error for tool operations."""


class ToolInitializationError(ToolError):
    """Raised when a tool fails to initialize."""


class ToolShutdownError(ToolError):
    """Raised when a tool fails to shut down cleanly."""


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""


class ToolTimeoutError(ToolError):
    """Raised when a tool operation exceeds its configured timeout."""


class ToolStateError(ToolError):
    """Raised when an operation is invoked in an invalid tool state."""


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    version: str = "1.0.0"
    description: str = ""
    priority: ToolPriority = ToolPriority.NORMAL
    timeout: float = 30.0
    tags: tuple[str, ...] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolHealth:
    name: str
    healthy: bool
    status: ToolStatus
    latency_ms: float
    timestamp: float
    detail: Optional[str] = None


@dataclass(frozen=True)
class ToolResult(Generic[TOutput]):
    success: bool
    output: Optional[TOutput]
    error: Optional[str]
    latency_ms: float


class BaseTool(ABC, Generic[TInput, TOutput]):
    """Abstract base class for executable tools.

    Subclasses implement `_initialize`, `_shutdown`, `_execute`, and
    optionally `_health_check`. The public `initialize`, `shutdown`,
    `execute`, and `health_check` methods enforce state transitions,
    timeouts, and consistent error handling around those hooks.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        priority: ToolPriority = ToolPriority.NORMAL,
        timeout: float = 30.0,
        tags: Optional[tuple[str, ...]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self._metadata = ToolMetadata(
            name=name,
            version=version,
            description=description,
            priority=priority,
            timeout=timeout,
            tags=tags or (),
            extra=extra or {},
        )
        self._status = ToolStatus.UNINITIALIZED
        self._lock = asyncio.Lock()
        self._last_health: Optional[ToolHealth] = None

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def priority(self) -> ToolPriority:
        return self._metadata.priority

    @property
    def timeout(self) -> float:
        return self._metadata.timeout

    @property
    def status(self) -> ToolStatus:
        return self._status

    @property
    def last_health(self) -> Optional[ToolHealth]:
        return self._last_health

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the tool, transitioning it to READY on success."""
        async with self._lock:
            if self._status is ToolStatus.READY:
                return
            if self._status is ToolStatus.SHUTDOWN:
                raise ToolStateError(f"Tool '{self.name}' has been shut down and cannot be reinitialized.")

            self._status = ToolStatus.INITIALIZING
            try:
                await asyncio.wait_for(self._initialize(), timeout=self._metadata.timeout)
            except asyncio.TimeoutError as exc:
                self._status = ToolStatus.ERROR
                raise ToolTimeoutError(
                    f"Tool '{self.name}' initialization timed out after {self._metadata.timeout}s."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - normalize to ToolError
                self._status = ToolStatus.ERROR
                raise ToolInitializationError(
                    f"Tool '{self.name}' failed to initialize: {exc}"
                ) from exc
            else:
                self._status = ToolStatus.READY

    async def shutdown(self) -> None:
        """Shut down the tool, transitioning it to SHUTDOWN regardless of outcome."""
        async with self._lock:
            if self._status is ToolStatus.SHUTDOWN:
                return
            self._status = ToolStatus.SHUTTING_DOWN
            try:
                await asyncio.wait_for(self._shutdown(), timeout=self._metadata.timeout)
            except asyncio.TimeoutError as exc:
                logger.warning("Tool '%s' shutdown timed out: %s", self.name, exc)
                raise ToolTimeoutError(
                    f"Tool '{self.name}' shutdown timed out after {self._metadata.timeout}s."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - normalize to ToolError
                logger.warning("Tool '%s' shutdown raised an error: %s", self.name, exc)
                raise ToolShutdownError(f"Tool '{self.name}' failed to shut down cleanly: {exc}") from exc
            finally:
                self._status = ToolStatus.SHUTDOWN

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        input_data: TInput,
        *,
        timeout: Optional[float] = None,
    ) -> ToolResult[TOutput]:
        """Execute the tool against `input_data`, enforcing readiness and timeout."""
        if self._status is not ToolStatus.READY:
            raise ToolStateError(
                f"Tool '{self.name}' is not ready for execution (status={self._status.value})."
            )

        effective_timeout = timeout if timeout is not None else self._metadata.timeout
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive.")

        start = time.monotonic()
        try:
            output = await asyncio.wait_for(self._execute(input_data), timeout=effective_timeout)
            latency_ms = (time.monotonic() - start) * 1000.0
            return ToolResult(success=True, output=output, error=None, latency_ms=latency_ms)
        except asyncio.TimeoutError as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            raise ToolTimeoutError(
                f"Tool '{self.name}' execution timed out after {effective_timeout}s."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize to ToolError
            latency_ms = (time.monotonic() - start) * 1000.0
            raise ToolExecutionError(f"Tool '{self.name}' execution failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self) -> ToolHealth:
        """Run the tool's health probe and record the resulting status."""
        start = time.monotonic()
        try:
            probe_ok = await asyncio.wait_for(self._health_check(), timeout=self._metadata.timeout)
            latency_ms = (time.monotonic() - start) * 1000.0
            healthy = bool(probe_ok) and self._status is ToolStatus.READY
            health = ToolHealth(
                name=self.name,
                healthy=healthy,
                status=self._status,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully into a health result
            latency_ms = (time.monotonic() - start) * 1000.0
            health = ToolHealth(
                name=self.name,
                healthy=False,
                status=self._status,
                latency_ms=latency_ms,
                timestamp=time.time(),
                detail=str(exc),
            )

        self._last_health = health
        return health

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _initialize(self) -> None:
        """Perform tool-specific setup (connections, resource allocation, etc.)."""
        ...

    @abstractmethod
    async def _shutdown(self) -> None:
        """Perform tool-specific teardown (closing connections, releasing resources)."""
        ...

    @abstractmethod
    async def _execute(self, input_data: TInput) -> TOutput:
        """Perform the tool's core operation against `input_data`."""
        ...

    async def _health_check(self) -> bool:
        """Default health probe; subclasses may override with deeper checks."""
        return self._status is ToolStatus.READY
