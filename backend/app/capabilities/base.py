"""
Capability Base - Abstract foundation for all capabilities in the AI Agent Platform.

This module defines the core abstractions that all capabilities must implement:
- Capability: Abstract base class with lifecycle and execution methods
- CapabilityMetadata: Immutable capability description and versioning
- ExecutionContext: Request context for capability execution
- CapabilityResult: Standardized execution output wrapper

Capabilities are pluggable components that provide specific functionality
to the agent system (memory, RAG, tools, web search, custom integrations).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
)

from app.agents.models import CapabilityType, SessionState

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")  # Generic output type
ConfigT = TypeVar("ConfigT", bound="CapabilityConfig")


# =============================================================================
# ENUMS
# =============================================================================


class CapabilityState(Enum):
    """
    Lifecycle state of a capability.

    Tracks the capability through its lifecycle from
    creation to shutdown.
    """

    UNINITIALIZED = auto()  # Created but not initialized
    INITIALIZING = auto()  # Currently initializing
    READY = auto()  # Initialized and ready for execution
    DEGRADED = auto()  # Operational but with reduced functionality
    EXECUTING = auto()  # Currently executing an action
    SHUTTING_DOWN = auto()  # Currently shutting down
    SHUTDOWN = auto()  # Fully shut down
    ERROR = auto()  # In error state

    @property
    def is_operational(self) -> bool:
        """Check if capability can accept executions."""
        return self in {self.READY, self.DEGRADED, self.EXECUTING}

    @property
    def is_terminal(self) -> bool:
        """Check if capability is in a terminal state."""
        return self in {self.SHUTDOWN, self.ERROR}


class CapabilityStatus(Enum):
    """Status of a capability execution result."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"


class CapabilityPriority(Enum):
    """
    Execution priority levels for capabilities.

    Higher priority capabilities are initialized first
    and may receive preferential resource allocation.
    """

    CRITICAL = 100  # Must succeed, highest priority
    HIGH = 75  # Important, prefer over normal
    NORMAL = 50  # Default priority
    LOW = 25  # Background, can be delayed
    BACKGROUND = 0  # Lowest priority, opportunistic

    def __lt__(self, other: CapabilityPriority) -> bool:
        return self.value < other.value

    def __gt__(self, other: CapabilityPriority) -> bool:
        return self.value > other.value


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CapabilityError(Exception):
    """Base exception for capability errors."""

    def __init__(
        self,
        message: str,
        capability_name: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.capability_name = capability_name
        self.cause = cause


class CapabilityNotReadyError(CapabilityError):
    """Capability is not in a ready state."""

    def __init__(self, capability_name: str, current_state: CapabilityState) -> None:
        super().__init__(
            f"Capability '{capability_name}' is not ready (state: {current_state.name})",
            capability_name=capability_name,
        )
        self.current_state = current_state


class CapabilityInitializationError(CapabilityError):
    """Capability failed to initialize."""

    pass


class CapabilityExecutionError(CapabilityError):
    """Capability execution failed."""

    def __init__(
        self,
        message: str,
        capability_name: str | None = None,
        action: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, capability_name, cause)
        self.action = action


class CapabilityTimeoutError(CapabilityError):
    """Capability execution timed out."""

    def __init__(
        self,
        capability_name: str,
        action: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            f"Capability '{capability_name}' action '{action}' timed out after {timeout_seconds}s",
            capability_name=capability_name,
        )
        self.action = action
        self.timeout_seconds = timeout_seconds


class UnsupportedActionError(CapabilityError):
    """Requested action is not supported by the capability."""

    def __init__(
        self,
        capability_name: str,
        action: str,
        supported_actions: list[str],
    ) -> None:
        super().__init__(
            f"Capability '{capability_name}' does not support action '{action}'. "
            f"Supported: {supported_actions}",
            capability_name=capability_name,
        )
        self.action = action
        self.supported_actions = supported_actions


# =============================================================================
# DEPENDENCY DECLARATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class CapabilityDependency:
    """
    Declares a dependency on another capability.

    Used to express that one capability requires another
    to be available and initialized before it can operate.

    Attributes:
        capability_type: Type of required capability
        required: If True, initialization fails without this dependency
        min_version: Minimum required version (semver string)
        max_version: Maximum compatible version (semver string)
    """

    capability_type: CapabilityType
    required: bool = True
    min_version: str | None = None
    max_version: str | None = None

    def is_compatible(self, version: str) -> bool:
        """Check if a version satisfies this dependency."""
        if self.min_version and not self._version_gte(version, self.min_version):
            return False
        if self.max_version and not self._version_lte(version, self.max_version):
            return False
        return True

    @staticmethod
    def _version_gte(version: str, minimum: str) -> bool:
        """Check if version >= minimum (simple semver comparison)."""
        return tuple(map(int, version.split("."))) >= tuple(map(int, minimum.split(".")))

    @staticmethod
    def _version_lte(version: str, maximum: str) -> bool:
        """Check if version <= maximum (simple semver comparison)."""
        return tuple(map(int, version.split("."))) <= tuple(map(int, maximum.split(".")))


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CapabilityConfig:
    """
    Base configuration for capabilities.

    Subclass this for capability-specific configuration.

    Attributes:
        enabled: Whether the capability is enabled
        timeout_seconds: Default execution timeout
        max_retries: Maximum retry attempts
        priority: Execution priority level
        rate_limit_per_minute: Max executions per minute (0 = unlimited)
        custom_settings: Additional capability-specific settings
    """

    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 0
    priority: CapabilityPriority = CapabilityPriority.NORMAL
    rate_limit_per_minute: int = 0
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def get_setting(self, key: str, default: T = None) -> T:
        """Get a custom setting with optional default."""
        return self.custom_settings.get(key, default)


# =============================================================================
# METADATA
# =============================================================================


@dataclass(frozen=True, slots=True)
class CapabilityMetadata:
    """
    Immutable metadata describing a capability.

    Provides identity, versioning, and dependency information
    for capability discovery and compatibility checking.

    Attributes:
        name: Unique capability identifier
        version: Semantic version string (e.g., "1.2.3")
        capability_type: Type classification
        description: Human-readable description
        author: Capability author/maintainer
        dependencies: Required/optional capability dependencies
        supported_actions: List of supported action names
        tags: Searchable tags for discovery
        deprecated: Whether this capability is deprecated
        deprecation_message: Message explaining deprecation
    """

    name: str
    version: str
    capability_type: CapabilityType
    description: str = ""
    author: str = ""
    dependencies: tuple[CapabilityDependency, ...] = ()
    supported_actions: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    config_schema: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    deprecated: bool = False
    deprecation_message: str = ""

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if not self.name:
            raise ValueError("Capability name cannot be empty")
        if not self.version:
            raise ValueError("Capability version cannot be empty")

    @property
    def version_tuple(self) -> tuple[int, ...]:
        """Parse version string into tuple for comparison."""
        try:
            return tuple(int(x) for x in self.version.split("."))
        except ValueError:
            return (0, 0, 0)

    @property
    def required_dependencies(self) -> list[CapabilityDependency]:
        """Get list of required dependencies."""
        return [d for d in self.dependencies if d.required]

    @property
    def optional_dependencies(self) -> list[CapabilityDependency]:
        """Get list of optional dependencies."""
        return [d for d in self.dependencies if not d.required]

    def supports_action(self, action: str) -> bool:
        """Check if an action is supported."""
        return action in self.supported_actions or not self.supported_actions


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass(slots=True)
class ExecutionContext:
    """
    Context passed to capability execute() method.

    Provides all contextual information needed for execution
    including session state, agent information, and request metadata.

    Attributes:
        session: Current session state
        action: Action being executed
        parameters: Action parameters
        request_id: Unique request identifier
        agent_id: ID of the executing agent (multi-agent support)
        agent_name: Name of the executing agent
        timeout_seconds: Execution timeout
        metadata: Additional context metadata
        previous_results: Results from dependency steps
        created_at: Context creation timestamp
    """

    session: SessionState
    action: str
    parameters: dict[str, Any]
    request_id: str = ""
    agent_id: str | None = None
    agent_name: str | None = None
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_results: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def session_id(self) -> str:
        """Get session ID from session state."""
        return self.session.session_id

    @property
    def has_agent_context(self) -> bool:
        """Check if executing within an agent context."""
        return self.agent_id is not None

    def get_parameter(self, key: str, default: T = None) -> T:
        """Get a parameter with optional default."""
        return self.parameters.get(key, default)

    def get_previous_result(self, step_id: str, default: T = None) -> T:
        """Get a previous step result with optional default."""
        return self.previous_results.get(step_id, default)

    def with_timeout(self, timeout_seconds: float) -> ExecutionContext:
        """Create a copy with different timeout."""
        return ExecutionContext(
            session=self.session,
            action=self.action,
            parameters=self.parameters,
            request_id=self.request_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            timeout_seconds=timeout_seconds,
            metadata=self.metadata,
            previous_results=self.previous_results,
            created_at=self.created_at,
        )


# =============================================================================
# EXECUTION RESULT
# =============================================================================


@dataclass(slots=True)
class CapabilityResult(Generic[T]):
    """
    Standardized result wrapper from capability execution.

    Provides a consistent structure for all capability outputs
    including success/failure status, timing, and metadata.

    Attributes:
        success: Whether execution succeeded
        output: The result data (generic type)
        error: Error message if failed
        execution_time_ms: Time taken to execute
        metadata: Additional result metadata
        cached: Whether result was served from cache
        partial: Whether result is partial/incomplete
    """

    success: bool
    output: T | None = None
    error: str | None = None
    status: CapabilityStatus | None = None
    data: dict[str, Any] | None = None
    execution_time: float = 0.0
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    partial: bool = False

    @classmethod
    def success_result(
        cls,
        output: T,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        cached: bool = False,
    ) -> CapabilityResult[T]:
        """Factory for successful result."""
        return cls(
            success=True,
            output=output,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            cached=cached,
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        partial_output: T | None = None,
    ) -> CapabilityResult[T]:
        """Factory for failed result."""
        return cls(
            success=False,
            output=partial_output,
            error=error,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            partial=partial_output is not None,
        )

    def map(self, func: Callable[[T], Any]) -> CapabilityResult[Any]:
        """Transform the output if successful."""
        if self.success and self.output is not None:
            return CapabilityResult.success_result(
                output=func(self.output),
                execution_time_ms=self.execution_time_ms,
                metadata=self.metadata,
                cached=self.cached,
            )
        return self  # type: ignore


# =============================================================================
# CAPABILITY ABSTRACT BASE
# =============================================================================


class Capability(ABC, Generic[T]):
    """
    Abstract base class for all capabilities.

    Capabilities are modular components that provide specific
    functionality to the agent system. All capabilities must
    implement this interface.

    Lifecycle:
        1. __init__: Create instance with config
        2. initialize(): Async setup (connections, resources)
        3. execute(): Perform actions (can be called many times)
        4. shutdown(): Async cleanup

    Multi-Agent Support:
        - on_agent_attach(): Called when agent starts using capability
        - on_agent_detach(): Called when agent stops using capability

    Type Parameters:
        T: The output type returned by execute()
    """

    # Class-level metadata (override in subclasses)
    METADATA: ClassVar[CapabilityMetadata]

    def __init__(
        self,
        config: CapabilityConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize capability instance.

        Args:
            config: Capability configuration
            logger: Optional logger instance
        """
        self._config = config or CapabilityConfig()
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._state = CapabilityState.UNINITIALIZED
        self._state_lock = asyncio.Lock()

        # Execution tracking
        self._execution_count = 0
        self._error_count = 0
        self._total_execution_time_ms = 0.0
        self._last_execution_time: datetime | None = None
        self._last_error: str | None = None

        # Multi-agent tracking
        self._attached_agents: set[str] = set()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return self.__class__.METADATA

    @property
    def name(self) -> str:
        """Get capability name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Get capability version."""
        return self.metadata.version

    @property
    def capability_type(self) -> CapabilityType:
        """Get capability type."""
        return self.metadata.capability_type

    @property
    def state(self) -> CapabilityState:
        """Get current capability state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if capability is ready for execution."""
        return self._state.is_operational

    @property
    def is_enabled(self) -> bool:
        """Check if capability is enabled in config."""
        return self._config.enabled

    @property
    def priority(self) -> CapabilityPriority:
        """Get capability priority."""
        return self._config.priority

    @property
    def config(self) -> CapabilityConfig:
        """Get capability configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the capability.

        Called once before first execution. Use for:
        - Establishing connections
        - Loading resources
        - Warming up caches
        - Validating configuration

        Raises:
            CapabilityInitializationError: If initialization fails
        """
        async with self._state_lock:
            if self._state != CapabilityState.UNINITIALIZED:
                self._logger.warning(
                    f"Capability {self.name} already initialized (state: {self._state.name})"
                )
                return

            self._state = CapabilityState.INITIALIZING

        try:
            self._logger.info(f"Initializing capability: {self.name} v{self.version}")

            # Call subclass initialization
            await self._do_initialize()

            async with self._state_lock:
                self._state = CapabilityState.READY

            self._logger.info(f"Capability {self.name} initialized successfully")

        except Exception as e:
            async with self._state_lock:
                self._state = CapabilityState.ERROR
                self._last_error = str(e)

            self._logger.error(f"Failed to initialize capability {self.name}: {e}")
            raise CapabilityInitializationError(
                f"Failed to initialize {self.name}: {e}",
                capability_name=self.name,
                cause=e,
            )

    async def shutdown(self) -> None:
        """
        Shutdown the capability.

        Called when capability is being removed or system is shutting down.
        Use for:
        - Closing connections
        - Flushing buffers
        - Releasing resources
        - Cleanup operations
        """
        async with self._state_lock:
            if self._state.is_terminal:
                return

            self._state = CapabilityState.SHUTTING_DOWN

        try:
            self._logger.info(f"Shutting down capability: {self.name}")

            # Detach all agents
            for agent_id in list(self._attached_agents):
                await self.on_agent_detach(agent_id)

            # Call subclass shutdown
            await self._do_shutdown()

            async with self._state_lock:
                self._state = CapabilityState.SHUTDOWN

            self._logger.info(f"Capability {self.name} shut down successfully")

        except Exception as e:
            self._logger.error(f"Error during shutdown of {self.name}: {e}")
            async with self._state_lock:
                self._state = CapabilityState.ERROR

    async def health_check(self) -> bool:
        """
        Check if capability is healthy and operational.

        Returns:
            True if capability is healthy

        Override _do_health_check() for custom health logic.
        """
        if not self._state.is_operational:
            return False

        try:
            return await self._do_health_check()
        except Exception as e:
            self._logger.warning(f"Health check failed for {self.name}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
    ) -> T:
        """
        Execute an action with the given parameters.

        This is the main entry point for capability execution.
        Handles state validation, timeout, and error tracking.

        Args:
            action: The action to perform
            parameters: Action parameters
            context: Session state or full execution context

        Returns:
            Action result (type T)

        Raises:
            CapabilityNotReadyError: If capability is not ready
            UnsupportedActionError: If action is not supported
            CapabilityTimeoutError: If execution times out
            CapabilityExecutionError: If execution fails
        """
        # Validate state
        if not self._state.is_operational:
            raise CapabilityNotReadyError(self.name, self._state)

        # Validate action
        if not self.validate_action(action):
            raise UnsupportedActionError(
                self.name,
                action,
                list(self.metadata.supported_actions),
            )

        # Build execution context
        if isinstance(context, SessionState):
            exec_context = ExecutionContext(
                session=context,
                action=action,
                parameters=parameters,
                timeout_seconds=self._config.timeout_seconds,
            )
        else:
            exec_context = context

        # Execute with tracking
        start_time = time.monotonic()

        try:
            async with self._state_lock:
                self._state = CapabilityState.EXECUTING

            # Execute with timeout
            timeout = exec_context.timeout_seconds or self._config.timeout_seconds

            result = await asyncio.wait_for(
                self._do_execute(exec_context),
                timeout=timeout,
            )

            # Update tracking
            execution_time = (time.monotonic() - start_time) * 1000
            self._execution_count += 1
            self._total_execution_time_ms += execution_time
            self._last_execution_time = datetime.now(timezone.utc)

            return result

        except asyncio.TimeoutError:
            self._error_count += 1
            raise CapabilityTimeoutError(
                self.name,
                action,
                exec_context.timeout_seconds,
            )

        except CapabilityError:
            self._error_count += 1
            raise

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            raise CapabilityExecutionError(
                f"Execution failed: {e}",
                capability_name=self.name,
                action=action,
                cause=e,
            )

        finally:
            async with self._state_lock:
                if self._state == CapabilityState.EXECUTING:
                    self._state = CapabilityState.READY

    async def execute_safe(
        self,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
    ) -> CapabilityResult[T]:
        """
        Execute action and return result wrapper (never raises).

        Convenience method that wraps execute() and catches
        all exceptions, returning a CapabilityResult.

        Args:
            action: The action to perform
            parameters: Action parameters
            context: Session state or execution context

        Returns:
            CapabilityResult with success/failure status
        """
        start_time = time.monotonic()

        try:
            output = await self.execute(action, parameters, context)
            execution_time = (time.monotonic() - start_time) * 1000

            return CapabilityResult.success_result(
                output=output,
                execution_time_ms=execution_time,
                metadata={"capability": self.name, "action": action},
            )

        except Exception as e:
            execution_time = (time.monotonic() - start_time) * 1000

            return CapabilityResult.failure_result(
                error=str(e),
                execution_time_ms=execution_time,
                metadata={
                    "capability": self.name,
                    "action": action,
                    "error_type": type(e).__name__,
                },
            )

    def validate_action(self, action: str) -> bool:
        """
        Check if an action is supported.

        Args:
            action: Action name to validate

        Returns:
            True if action is supported
        """
        # If no actions specified, all are allowed
        if not self.metadata.supported_actions:
            return True

        return action in self.metadata.supported_actions

    def get_supported_actions(self) -> list[str]:
        """Get list of supported actions."""
        return list(self.metadata.supported_actions)

    # -------------------------------------------------------------------------
    # Multi-Agent Support
    # -------------------------------------------------------------------------

    async def on_agent_attach(self, agent_id: str, agent_name: str | None = None) -> None:
        """
        Called when an agent starts using this capability.

        Override for agent-specific initialization (e.g., per-agent caches).

        Args:
            agent_id: Unique agent identifier
            agent_name: Optional agent name
        """
        self._attached_agents.add(agent_id)
        self._logger.debug(f"Agent {agent_id} attached to capability {self.name}")

    async def on_agent_detach(self, agent_id: str) -> None:
        """
        Called when an agent stops using this capability.

        Override for agent-specific cleanup.

        Args:
            agent_id: Unique agent identifier
        """
        self._attached_agents.discard(agent_id)
        self._logger.debug(f"Agent {agent_id} detached from capability {self.name}")

    @property
    def attached_agent_count(self) -> int:
        """Get number of attached agents."""
        return len(self._attached_agents)

    def is_agent_attached(self, agent_id: str) -> bool:
        """Check if an agent is attached."""
        return agent_id in self._attached_agents

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get capability statistics."""
        avg_execution_time = (
            self._total_execution_time_ms / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        return {
            "name": self.name,
            "version": self.version,
            "type": self.capability_type.value,
            "state": self._state.name,
            "enabled": self.is_enabled,
            "priority": self._config.priority.name,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "total_execution_time_ms": self._total_execution_time_ms,
            "avg_execution_time_ms": avg_execution_time,
            "last_execution": (
                self._last_execution_time.isoformat()
                if self._last_execution_time
                else None
            ),
            "last_error": self._last_error,
            "attached_agents": len(self._attached_agents),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._error_count = 0
        self._total_execution_time_ms = 0.0
        self._last_execution_time = None
        self._last_error = None

    # -------------------------------------------------------------------------
    # Abstract Methods (Implement in Subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        Subclass initialization logic.

        Override this method to implement capability-specific
        initialization (connections, resources, etc.).
        """
        ...

    @abstractmethod
    async def _do_shutdown(self) -> None:
        """
        Subclass shutdown logic.

        Override this method to implement capability-specific
        cleanup (close connections, flush buffers, etc.).
        """
        ...

    @abstractmethod
    async def _do_execute(self, context: ExecutionContext) -> T:
        """
        Subclass execution logic.

        Override this method to implement the capability's
        core functionality.

        Args:
            context: Execution context with action and parameters

        Returns:
            Execution result (type T)
        """
        ...

    async def _do_health_check(self) -> bool:
        """
        Subclass health check logic.

        Override for custom health checking. Default returns True.

        Returns:
            True if healthy
        """
        return True


# =============================================================================
# BASE IMPLEMENTATIONS
# =============================================================================


class NoOpCapability(Capability[None]):
    """
    No-operation capability for testing and placeholders.

    Always succeeds and returns None.
    """

    METADATA = CapabilityMetadata(
        name="noop",
        version="1.0.0",
        capability_type=CapabilityType.TOOLS,
        description="No-operation capability for testing",
        supported_actions=("noop", "echo", "sleep"),
        tags=("testing", "debug"),
    )

    async def _do_initialize(self) -> None:
        pass

    async def _do_shutdown(self) -> None:
        pass

    async def _do_execute(self, context: ExecutionContext) -> None:
        if context.action == "echo":
            return context.get_parameter("message")  # type: ignore[return-value]
        elif context.action == "sleep":
            duration = context.get_parameter("seconds", 1.0)
            await asyncio.sleep(duration)
        return None


# =============================================================================
# CAPABILITY WRAPPER
# =============================================================================


class CapabilityWrapper(Generic[T]):
    """
    Wrapper that adds cross-cutting concerns to a capability.

    Provides:
    - Logging
    - Metrics collection
    - Rate limiting
    - Circuit breaking (future)

    Use this to enhance capabilities without modifying them.
    """

    def __init__(
        self,
        capability: Capability[T],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Wrap a capability.

        Args:
            capability: The capability to wrap
            logger: Optional logger
        """
        self._capability = capability
        self._logger = logger or logging.getLogger(__name__)

    @property
    def capability(self) -> Capability[T]:
        """Get the wrapped capability."""
        return self._capability

    async def execute(
        self,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
    ) -> T:
        """Execute with logging and metrics."""
        self._logger.debug(
            f"Executing {self._capability.name}.{action}",
            extra={"parameters": parameters},
        )

        start_time = time.monotonic()

        try:
            result = await self._capability.execute(action, parameters, context)

            execution_time = (time.monotonic() - start_time) * 1000
            self._logger.debug(
                f"Completed {self._capability.name}.{action} in {execution_time:.2f}ms"
            )

            return result

        except Exception as e:
            execution_time = (time.monotonic() - start_time) * 1000
            self._logger.error(
                f"Failed {self._capability.name}.{action} after {execution_time:.2f}ms: {e}"
            )
            raise


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_capability_metadata(
    name: str,
    version: str,
    capability_type: CapabilityType,
    description: str = "",
    supported_actions: list[str] | None = None,
    dependencies: list[CapabilityDependency] | None = None,
    tags: list[str] | None = None,
) -> CapabilityMetadata:
    """
    Factory function to create CapabilityMetadata.

    Args:
        name: Unique capability name
        version: Semantic version string
        capability_type: Type classification
        description: Human-readable description
        supported_actions: List of supported actions
        dependencies: Capability dependencies
        tags: Searchable tags

    Returns:
        CapabilityMetadata instance
    """
    return CapabilityMetadata(
        name=name,
        version=version,
        capability_type=capability_type,
        description=description,
        supported_actions=tuple(supported_actions or []),
        dependencies=tuple(dependencies or []),
        tags=tuple(tags or []),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CapabilityState",
    "CapabilityStatus",
    "CapabilityPriority",
    "CapabilityType",
    # Exceptions
    "CapabilityError",
    "CapabilityNotReadyError",
    "CapabilityInitializationError",
    "CapabilityExecutionError",
    "CapabilityTimeoutError",
    "UnsupportedActionError",
    # Data Classes
    "CapabilityDependency",
    "CapabilityConfig",
    "CapabilityMetadata",
    "ExecutionContext",
    "CapabilityResult",
    # Abstract Base
    "Capability",
    # Implementations
    "NoOpCapability",
    "CapabilityWrapper",
    # Factory
    "create_capability_metadata",
]
