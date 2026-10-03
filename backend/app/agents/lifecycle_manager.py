"""
Lifecycle Manager - Component lifecycle orchestration for the AI Agent Platform.

This module provides comprehensive lifecycle management including:
- Startup and shutdown orchestration
- Warmup and initialization sequences
- Capability initialization with dependency ordering
- Health checks and readiness probes
- Graceful cleanup and resource release
- Multi-agent lifecycle coordination
- Hooks and event emission
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class LifecycleState(Enum):
    """States in the component lifecycle."""

    CREATED = auto()  # Initial state
    INITIALIZING = auto()  # Running initialization
    INITIALIZED = auto()  # Initialization complete
    WARMING_UP = auto()  # Running warmup
    READY = auto()  # Fully operational
    DEGRADED = auto()  # Partially operational
    STOPPING = auto()  # Shutdown in progress
    STOPPED = auto()  # Fully stopped
    FAILED = auto()  # Failed state
    RESTARTING = auto()  # Restart in progress


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of managed components."""

    CORE = "core"  # Core system components
    CAPABILITY = "capability"  # Agent capabilities
    SERVICE = "service"  # Background services
    INTEGRATION = "integration"  # External integrations
    AGENT = "agent"  # Agent instances
    EXTENSION = "extension"  # Extensions/plugins


class ShutdownMode(Enum):
    """Shutdown behavior modes."""

    GRACEFUL = auto()  # Wait for tasks to complete
    IMMEDIATE = auto()  # Stop immediately
    FORCED = auto()  # Force stop, may lose data


class InitializationPriority(Enum):
    """Initialization priority levels."""

    CRITICAL = 0  # Must initialize first
    HIGH = 1  # Initialize early
    NORMAL = 2  # Standard priority
    LOW = 3  # Initialize late
    DEFERRED = 4  # Initialize on demand


# =============================================================================
# PROTOCOLS
# =============================================================================


@runtime_checkable
class Initializable(Protocol):
    """Protocol for components that can be initialized."""

    async def initialize(self) -> None:
        """Initialize the component."""
        ...


@runtime_checkable
class Startable(Protocol):
    """Protocol for components that can be started."""

    async def start(self) -> None:
        """Start the component."""
        ...


@runtime_checkable
class Stoppable(Protocol):
    """Protocol for components that can be stopped."""

    async def stop(self) -> None:
        """Stop the component."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components with health checks."""

    async def health_check(self) -> HealthStatus:
        """Check component health."""
        ...


@runtime_checkable
class Warmable(Protocol):
    """Protocol for components that support warmup."""

    async def warmup(self) -> None:
        """Warm up the component."""
        ...


@runtime_checkable
class Cleanable(Protocol):
    """Protocol for components that need cleanup."""

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        component_id: Component identifier
        status: Health status
        message: Status message
        latency_ms: Check latency in milliseconds
        timestamp: When check was performed
        details: Additional health details
    """

    component_id: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class ComponentInfo:
    """
    Information about a managed component.

    Attributes:
        component_id: Unique component identifier
        name: Component name
        component_type: Type of component
        priority: Initialization priority
        state: Current lifecycle state
        dependencies: Component dependencies
        instance: Component instance
        initialized_at: When initialized
        started_at: When started
        metadata: Additional metadata
    """

    component_id: str
    name: str
    component_type: ComponentType
    priority: InitializationPriority = InitializationPriority.NORMAL
    state: LifecycleState = LifecycleState.CREATED
    dependencies: list[str] = field(default_factory=list)
    instance: Any = None
    initialized_at: datetime | None = None
    started_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "name": self.name,
            "component_type": self.component_type.value,
            "priority": self.priority.name,
            "state": self.state.name,
            "dependencies": self.dependencies,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "metadata": self.metadata,
        }


@dataclass
class LifecycleEvent:
    """
    Event emitted during lifecycle transitions.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of lifecycle event
        component_id: Component identifier
        old_state: Previous state
        new_state: New state
        timestamp: When event occurred
        duration_ms: Transition duration
        error: Error if transition failed
        metadata: Additional event data
    """

    event_id: str
    event_type: str
    component_id: str | None
    old_state: LifecycleState | None
    new_state: LifecycleState
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "component_id": self.component_id,
            "old_state": self.old_state.name if self.old_state else None,
            "new_state": self.new_state.name,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class LifecycleConfig:
    """
    Configuration for lifecycle manager.

    Attributes:
        startup_timeout_seconds: Timeout for startup
        shutdown_timeout_seconds: Timeout for shutdown
        warmup_timeout_seconds: Timeout for warmup
        health_check_interval_seconds: Health check interval
        health_check_timeout_seconds: Health check timeout
        max_initialization_retries: Max init retries
        initialization_retry_delay_seconds: Delay between retries
        enable_warmup: Enable warmup phase
        enable_health_checks: Enable health checks
        shutdown_mode: Default shutdown mode
        parallel_initialization: Initialize in parallel
        fail_fast: Stop on first failure
    """

    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0
    warmup_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 5.0
    max_initialization_retries: int = 3
    initialization_retry_delay_seconds: float = 1.0
    enable_warmup: bool = True
    enable_health_checks: bool = True
    shutdown_mode: ShutdownMode = ShutdownMode.GRACEFUL
    parallel_initialization: bool = True
    fail_fast: bool = False


# =============================================================================
# LIFECYCLE HOOKS
# =============================================================================


LifecycleHook = Callable[["LifecycleManager", LifecycleEvent], Awaitable[None]]


@dataclass
class HookRegistration:
    """Registration for a lifecycle hook."""

    hook_id: str
    event_type: str
    callback: LifecycleHook
    priority: int = 0
    once: bool = False


# =============================================================================
# EXCEPTIONS
# =============================================================================


class LifecycleError(Exception):
    """Base exception for lifecycle errors."""

    def __init__(
        self,
        message: str,
        component_id: str | None = None,
        state: LifecycleState | None = None,
    ) -> None:
        super().__init__(message)
        self.component_id = component_id
        self.state = state


class InitializationError(LifecycleError):
    """Raised when initialization fails."""

    pass


class StartupError(LifecycleError):
    """Raised when startup fails."""

    pass


class ShutdownError(LifecycleError):
    """Raised when shutdown fails."""

    pass


class WarmupError(LifecycleError):
    """Raised when warmup fails."""

    pass


class HealthCheckError(LifecycleError):
    """Raised when health check fails."""

    pass


class DependencyError(LifecycleError):
    """Raised when dependency resolution fails."""

    pass


# =============================================================================
# DEPENDENCY RESOLVER
# =============================================================================


class DependencyResolver:
    """
    Resolves component dependencies for initialization ordering.

    Uses topological sort to determine initialization order.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def resolve(
        self,
        components: dict[str, ComponentInfo],
    ) -> list[list[str]]:
        """
        Resolve dependencies and return initialization order.

        Args:
            components: Dictionary of component info

        Returns:
            List of component ID batches (can be initialized in parallel)

        Raises:
            DependencyError: If circular dependency detected
        """
        # Build dependency graph
        graph: dict[str, set[str]] = {}
        in_degree: dict[str, int] = {}

        for comp_id, info in components.items():
            if comp_id not in graph:
                graph[comp_id] = set()
                in_degree[comp_id] = 0

            for dep_id in info.dependencies:
                if dep_id not in components:
                    self._logger.warning(
                        f"Component {comp_id} depends on unknown component {dep_id}"
                    )
                    continue

                if dep_id not in graph:
                    graph[dep_id] = set()
                    in_degree[dep_id] = 0

                graph[dep_id].add(comp_id)
                in_degree[comp_id] = in_degree.get(comp_id, 0) + 1

        # Topological sort with batching
        batches: list[list[str]] = []
        remaining = set(components.keys())

        while remaining:
            # Find all components with no unresolved dependencies
            batch = [
                comp_id for comp_id in remaining
                if in_degree.get(comp_id, 0) == 0
            ]

            if not batch:
                # Circular dependency detected
                cycle = self._find_cycle(graph, remaining)
                raise DependencyError(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )

            # Sort batch by priority
            batch.sort(
                key=lambda c: (
                    components[c].priority.value,
                    components[c].name,
                )
            )

            batches.append(batch)

            # Remove batch from remaining and update in-degrees
            for comp_id in batch:
                remaining.remove(comp_id)
                for dependent in graph.get(comp_id, set()):
                    in_degree[dependent] -= 1

        return batches

    def _find_cycle(
        self,
        graph: dict[str, set[str]],
        nodes: set[str],
    ) -> list[str]:
        """Find a cycle in the graph."""
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor in nodes:
                    result = dfs(neighbor)
                    if result:
                        return result

            path.pop()
            return None

        for node in nodes:
            result = dfs(node)
            if result:
                return result

        return list(nodes)[:3] + ["..."]  # Fallback


# =============================================================================
# HEALTH CHECKER
# =============================================================================


class HealthChecker:
    """
    Performs health checks on components.

    Supports periodic checks and aggregated health status.
    """

    def __init__(
        self,
        config: LifecycleConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._results: dict[str, HealthCheckResult] = {}
        self._check_task: asyncio.Task | None = None
        self._running = False

    async def check_component(
        self,
        component_id: str,
        instance: Any,
    ) -> HealthCheckResult:
        """
        Check health of a single component.

        Args:
            component_id: Component identifier
            instance: Component instance

        Returns:
            Health check result
        """
        start_time = time.monotonic()

        try:
            if isinstance(instance, HealthCheckable):
                status = await asyncio.wait_for(
                    instance.health_check(),
                    timeout=self._config.health_check_timeout_seconds,
                )
                message = "Health check passed"
            else:
                # Component doesn't support health checks
                status = HealthStatus.UNKNOWN
                message = "Component does not support health checks"

        except asyncio.TimeoutError:
            status = HealthStatus.UNHEALTHY
            message = "Health check timed out"

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Health check failed: {e}"
            self._logger.error(f"Health check failed for {component_id}: {e}")

        latency_ms = (time.monotonic() - start_time) * 1000

        result = HealthCheckResult(
            component_id=component_id,
            status=status,
            message=message,
            latency_ms=latency_ms,
        )

        self._results[component_id] = result
        return result

    async def check_all(
        self,
        components: dict[str, ComponentInfo],
    ) -> dict[str, HealthCheckResult]:
        """
        Check health of all components.

        Args:
            components: Dictionary of component info

        Returns:
            Dictionary of health check results
        """
        tasks = []

        for comp_id, info in components.items():
            if info.instance and info.state == LifecycleState.READY:
                tasks.append(self.check_component(comp_id, info.instance))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return dict(self._results)

    def get_aggregate_status(self) -> HealthStatus:
        """Get aggregate health status."""
        if not self._results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self._results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_results(self) -> dict[str, HealthCheckResult]:
        """Get all health check results."""
        return dict(self._results)

    async def start_periodic_checks(
        self,
        components: dict[str, ComponentInfo],
    ) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(
            self._periodic_check_loop(components)
        )

    async def stop_periodic_checks(self) -> None:
        """Stop periodic health checks."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    async def _periodic_check_loop(
        self,
        components: dict[str, ComponentInfo],
    ) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.check_all(components)
                await asyncio.sleep(self._config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Periodic health check error: {e}")
                await asyncio.sleep(self._config.health_check_interval_seconds)


# =============================================================================
# AGENT LIFECYCLE COORDINATOR
# =============================================================================


@dataclass
class AgentInfo:
    """
    Information about a managed agent.

    Attributes:
        agent_id: Unique agent identifier
        name: Agent name
        state: Current lifecycle state
        capabilities: Agent capabilities
        instance: Agent instance
        created_at: When created
        metadata: Additional metadata
    """

    agent_id: str
    name: str
    state: LifecycleState = LifecycleState.CREATED
    capabilities: list[str] = field(default_factory=list)
    instance: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentLifecycleCoordinator:
    """
    Coordinates lifecycle across multiple agents.

    Supports agent registration, coordination, and orchestration.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._agents: dict[str, AgentInfo] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        instance: Any,
        capabilities: list[str] | None = None,
    ) -> AgentInfo:
        """
        Register an agent.

        Args:
            agent_id: Agent identifier
            name: Agent name
            instance: Agent instance
            capabilities: Agent capabilities

        Returns:
            Agent info
        """
        async with self._lock:
            info = AgentInfo(
                agent_id=agent_id,
                name=name,
                instance=instance,
                capabilities=capabilities or [],
            )
            self._agents[agent_id] = info
            self._logger.info(f"Registered agent: {name} ({agent_id})")
            return info

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        async with self._lock:
            if agent_id in self._agents:
                info = self._agents.pop(agent_id)
                self._logger.info(f"Unregistered agent: {info.name} ({agent_id})")

    async def get_agent(self, agent_id: str) -> AgentInfo | None:
        """Get agent info."""
        return self._agents.get(agent_id)

    async def list_agents(
        self,
        state: LifecycleState | None = None,
    ) -> list[AgentInfo]:
        """List all agents, optionally filtered by state."""
        agents = list(self._agents.values())
        if state:
            agents = [a for a in agents if a.state == state]
        return agents

    async def update_agent_state(
        self,
        agent_id: str,
        state: LifecycleState,
    ) -> None:
        """Update agent state."""
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].state = state

    async def initialize_all_agents(self) -> list[str]:
        """
        Initialize all registered agents.

        Returns:
            List of failed agent IDs
        """
        failed: list[str] = []

        for agent_id, info in self._agents.items():
            try:
                if isinstance(info.instance, Initializable):
                    await info.instance.initialize()
                info.state = LifecycleState.INITIALIZED
                self._logger.info(f"Initialized agent: {info.name}")
            except Exception as e:
                self._logger.error(f"Failed to initialize agent {info.name}: {e}")
                info.state = LifecycleState.FAILED
                failed.append(agent_id)

        return failed

    async def start_all_agents(self) -> list[str]:
        """
        Start all initialized agents.

        Returns:
            List of failed agent IDs
        """
        failed: list[str] = []

        for agent_id, info in self._agents.items():
            if info.state != LifecycleState.INITIALIZED:
                continue

            try:
                if isinstance(info.instance, Startable):
                    await info.instance.start()
                info.state = LifecycleState.READY
                self._logger.info(f"Started agent: {info.name}")
            except Exception as e:
                self._logger.error(f"Failed to start agent {info.name}: {e}")
                info.state = LifecycleState.FAILED
                failed.append(agent_id)

        return failed

    async def stop_all_agents(self) -> list[str]:
        """
        Stop all running agents.

        Returns:
            List of failed agent IDs
        """
        failed: list[str] = []

        for agent_id, info in self._agents.items():
            if info.state not in {LifecycleState.READY, LifecycleState.DEGRADED}:
                continue

            try:
                info.state = LifecycleState.STOPPING
                if isinstance(info.instance, Stoppable):
                    await info.instance.stop()
                info.state = LifecycleState.STOPPED
                self._logger.info(f"Stopped agent: {info.name}")
            except Exception as e:
                self._logger.error(f"Failed to stop agent {info.name}: {e}")
                info.state = LifecycleState.FAILED
                failed.append(agent_id)

        return failed

    async def broadcast_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast event to all agents."""
        for info in self._agents.values():
            if info.state == LifecycleState.READY:
                if hasattr(info.instance, "on_event"):
                    try:
                        await info.instance.on_event(event_type, data)
                    except Exception as e:
                        self._logger.error(
                            f"Error broadcasting to agent {info.name}: {e}"
                        )


# =============================================================================
# LIFECYCLE MANAGER
# =============================================================================


class LifecycleManager:
    """
    Central lifecycle manager for the AI Agent Platform.

    Orchestrates startup, shutdown, warmup, health checks,
    and cleanup for all components.
    """

    def __init__(
        self,
        config: LifecycleConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize lifecycle manager.

        Args:
            config: Lifecycle configuration
            logger: Optional logger
        """
        self._config = config or LifecycleConfig()
        self._logger = logger or logging.getLogger(__name__)

        # State
        self._state = LifecycleState.CREATED
        self._components: dict[str, ComponentInfo] = {}
        self._initialization_order: list[list[str]] = []

        # Helpers
        self._resolver = DependencyResolver(logger)
        self._health_checker = HealthChecker(self._config, logger)
        self._agent_coordinator = AgentLifecycleCoordinator(logger)

        # Hooks
        self._hooks: dict[str, list[HookRegistration]] = {}

        # Synchronization
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._startup_time: float | None = None
        self._shutdown_time: float | None = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return self._state == LifecycleState.READY

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._state in {
            LifecycleState.READY,
            LifecycleState.DEGRADED,
        }

    @property
    def agent_coordinator(self) -> AgentLifecycleCoordinator:
        """Get agent coordinator."""
        return self._agent_coordinator

    # -------------------------------------------------------------------------
    # Component Registration
    # -------------------------------------------------------------------------

    async def register_component(
        self,
        component_id: str,
        name: str,
        instance: Any,
        component_type: ComponentType = ComponentType.SERVICE,
        priority: InitializationPriority = InitializationPriority.NORMAL,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComponentInfo:
        """
        Register a component for lifecycle management.

        Args:
            component_id: Unique component identifier
            name: Component name
            instance: Component instance
            component_type: Type of component
            priority: Initialization priority
            dependencies: Component dependencies
            metadata: Additional metadata

        Returns:
            Component info
        """
        async with self._lock:
            info = ComponentInfo(
                component_id=component_id,
                name=name,
                component_type=component_type,
                priority=priority,
                dependencies=dependencies or [],
                instance=instance,
                metadata=metadata or {},
            )

            self._components[component_id] = info
            self._logger.info(f"Registered component: {name} ({component_id})")

            return info

    async def unregister_component(self, component_id: str) -> None:
        """Unregister a component."""
        async with self._lock:
            if component_id in self._components:
                info = self._components.pop(component_id)
                self._logger.info(
                    f"Unregistered component: {info.name} ({component_id})"
                )

    def get_component(self, component_id: str) -> ComponentInfo | None:
        """Get component info."""
        return self._components.get(component_id)

    def list_components(
        self,
        component_type: ComponentType | None = None,
        state: LifecycleState | None = None,
    ) -> list[ComponentInfo]:
        """List components with optional filtering."""
        components = list(self._components.values())

        if component_type:
            components = [c for c in components if c.component_type == component_type]

        if state:
            components = [c for c in components if c.state == state]

        return components

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    def on(
        self,
        event_type: str,
        callback: LifecycleHook,
        priority: int = 0,
        once: bool = False,
    ) -> str:
        """
        Register a lifecycle hook.

        Args:
            event_type: Event type to hook
            callback: Callback function
            priority: Hook priority (higher = earlier)
            once: Remove after first invocation

        Returns:
            Hook ID
        """
        hook_id = str(uuid4())

        registration = HookRegistration(
            hook_id=hook_id,
            event_type=event_type,
            callback=callback,
            priority=priority,
            once=once,
        )

        if event_type not in self._hooks:
            self._hooks[event_type] = []

        self._hooks[event_type].append(registration)
        self._hooks[event_type].sort(key=lambda h: h.priority, reverse=True)

        return hook_id

    def off(self, hook_id: str) -> bool:
        """
        Remove a lifecycle hook.

        Args:
            hook_id: Hook ID to remove

        Returns:
            True if removed
        """
        for event_type, hooks in self._hooks.items():
            for hook in hooks:
                if hook.hook_id == hook_id:
                    hooks.remove(hook)
                    return True
        return False

    async def _emit_event(
        self,
        event_type: str,
        component_id: str | None = None,
        old_state: LifecycleState | None = None,
        new_state: LifecycleState | None = None,
        duration_ms: float = 0.0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a lifecycle event."""
        event = LifecycleEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            component_id=component_id,
            old_state=old_state,
            new_state=new_state or self._state,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )

        hooks = self._hooks.get(event_type, [])
        hooks_to_remove: list[HookRegistration] = []

        for hook in hooks:
            try:
                await hook.callback(self, event)
                if hook.once:
                    hooks_to_remove.append(hook)
            except Exception as e:
                self._logger.error(f"Hook error for {event_type}: {e}")

        for hook in hooks_to_remove:
            hooks.remove(hook)

    # -------------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Start the system.

        Initializes all components, runs warmup, and starts health checks.

        Raises:
            StartupError: If startup fails
        """
        if self._state not in {LifecycleState.CREATED, LifecycleState.STOPPED}:
            raise StartupError(
                f"Cannot start from state: {self._state.name}",
                state=self._state,
            )

        start_time = time.monotonic()
        self._logger.info("Starting system...")

        try:
            # Emit pre-startup event
            await self._emit_event("pre_startup")

            # Initialize components
            await self._initialize_components()

            # Warmup if enabled
            if self._config.enable_warmup:
                await self._warmup_components()

            # Start health checks if enabled
            if self._config.enable_health_checks:
                await self._health_checker.start_periodic_checks(self._components)

            # Update state
            self._state = LifecycleState.READY
            self._startup_time = time.monotonic() - start_time

            self._logger.info(
                f"System started successfully in {self._startup_time:.2f}s"
            )

            # Emit post-startup event
            await self._emit_event(
                "post_startup",
                new_state=LifecycleState.READY,
                duration_ms=self._startup_time * 1000,
            )

        except Exception as e:
            self._state = LifecycleState.FAILED
            self._logger.error(f"Startup failed: {e}")
            await self._emit_event(
                "startup_failed",
                new_state=LifecycleState.FAILED,
                error=str(e),
            )
            raise StartupError(f"Startup failed: {e}") from e

    async def _initialize_components(self) -> None:
        """Initialize all registered components."""
        self._state = LifecycleState.INITIALIZING
        await self._emit_event("initializing", new_state=LifecycleState.INITIALIZING)

        # Resolve dependencies
        self._initialization_order = self._resolver.resolve(self._components)

        self._logger.info(
            f"Initialization order: {len(self._initialization_order)} batches"
        )

        # Initialize in batches
        for batch_index, batch in enumerate(self._initialization_order):
            self._logger.debug(f"Initializing batch {batch_index + 1}: {batch}")

            if self._config.parallel_initialization:
                # Initialize batch in parallel
                tasks = [
                    self._initialize_component(comp_id)
                    for comp_id in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for failures
                for comp_id, result in zip(batch, results):
                    if isinstance(result, Exception):
                        if self._config.fail_fast:
                            raise InitializationError(
                                f"Failed to initialize {comp_id}: {result}",
                                component_id=comp_id,
                            )
                        self._logger.error(f"Failed to initialize {comp_id}: {result}")
                        self._components[comp_id].state = LifecycleState.FAILED
            else:
                # Initialize sequentially
                for comp_id in batch:
                    try:
                        await self._initialize_component(comp_id)
                    except Exception as e:
                        if self._config.fail_fast:
                            raise
                        self._logger.error(f"Failed to initialize {comp_id}: {e}")
                        self._components[comp_id].state = LifecycleState.FAILED

        self._state = LifecycleState.INITIALIZED
        await self._emit_event("initialized", new_state=LifecycleState.INITIALIZED)

    async def _initialize_component(self, component_id: str) -> None:
        """Initialize a single component."""
        info = self._components.get(component_id)
        if not info:
            return

        start_time = time.monotonic()
        retries = 0

        while retries <= self._config.max_initialization_retries:
            try:
                if isinstance(info.instance, Initializable):
                    await asyncio.wait_for(
                        info.instance.initialize(),
                        timeout=self._config.startup_timeout_seconds,
                    )

                info.state = LifecycleState.INITIALIZED
                info.initialized_at = datetime.now(timezone.utc)

                duration_ms = (time.monotonic() - start_time) * 1000
                self._logger.info(
                    f"Initialized {info.name} in {duration_ms:.1f}ms"
                )

                await self._emit_event(
                    "component_initialized",
                    component_id=component_id,
                    old_state=LifecycleState.CREATED,
                    new_state=LifecycleState.INITIALIZED,
                    duration_ms=duration_ms,
                )
                return

            except asyncio.TimeoutError:
                self._logger.warning(
                    f"Initialization timeout for {info.name}, retry {retries + 1}"
                )
            except Exception as e:
                self._logger.warning(
                    f"Initialization error for {info.name}: {e}, retry {retries + 1}"
                )

            retries += 1
            if retries <= self._config.max_initialization_retries:
                await asyncio.sleep(self._config.initialization_retry_delay_seconds)

        raise InitializationError(
            f"Failed to initialize {info.name} after {retries} attempts",
            component_id=component_id,
        )

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    async def warmup(self) -> None:
        """
        Warm up all components.

        Raises:
            WarmupError: If warmup fails
        """
        await self._warmup_components()

    async def _warmup_components(self) -> None:
        """Warm up all initialized components."""
        self._state = LifecycleState.WARMING_UP
        await self._emit_event("warming_up", new_state=LifecycleState.WARMING_UP)

        warmup_tasks = []

        for comp_id, info in self._components.items():
            if info.state != LifecycleState.INITIALIZED:
                continue

            if isinstance(info.instance, Warmable):
                warmup_tasks.append(self._warmup_component(comp_id))

        if warmup_tasks:
            results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self._logger.warning(f"Warmup error: {result}")

        await self._emit_event("warmed_up", new_state=LifecycleState.READY)

    async def _warmup_component(self, component_id: str) -> None:
        """Warm up a single component."""
        info = self._components.get(component_id)
        if not info or not isinstance(info.instance, Warmable):
            return

        start_time = time.monotonic()

        try:
            await asyncio.wait_for(
                info.instance.warmup(),
                timeout=self._config.warmup_timeout_seconds,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            self._logger.info(f"Warmed up {info.name} in {duration_ms:.1f}ms")

            await self._emit_event(
                "component_warmed_up",
                component_id=component_id,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            self._logger.warning(f"Warmup timeout for {info.name}")
        except Exception as e:
            self._logger.warning(f"Warmup error for {info.name}: {e}")

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(
        self,
        mode: ShutdownMode | None = None,
    ) -> None:
        """
        Shut down the system.

        Args:
            mode: Shutdown mode (defaults to config)

        Raises:
            ShutdownError: If shutdown fails
        """
        if self._state in {LifecycleState.STOPPED, LifecycleState.STOPPING}:
            return

        mode = mode or self._config.shutdown_mode
        start_time = time.monotonic()

        self._logger.info(f"Shutting down system ({mode.name})...")
        self._state = LifecycleState.STOPPING

        try:
            # Emit pre-shutdown event
            await self._emit_event("pre_shutdown", new_state=LifecycleState.STOPPING)

            # Stop health checks
            await self._health_checker.stop_periodic_checks()

            # Stop agents
            await self._agent_coordinator.stop_all_agents()

            # Stop components in reverse order
            if mode == ShutdownMode.GRACEFUL:
                await self._graceful_shutdown()
            elif mode == ShutdownMode.IMMEDIATE:
                await self._immediate_shutdown()
            else:
                await self._forced_shutdown()

            # Cleanup
            await self._cleanup_components()

            # Update state
            self._state = LifecycleState.STOPPED
            self._shutdown_time = time.monotonic() - start_time
            self._shutdown_event.set()

            self._logger.info(
                f"System shut down in {self._shutdown_time:.2f}s"
            )

            # Emit post-shutdown event
            await self._emit_event(
                "post_shutdown",
                new_state=LifecycleState.STOPPED,
                duration_ms=self._shutdown_time * 1000,
            )

        except Exception as e:
            self._state = LifecycleState.FAILED
            self._logger.error(f"Shutdown failed: {e}")
            await self._emit_event(
                "shutdown_failed",
                new_state=LifecycleState.FAILED,
                error=str(e),
            )
            raise ShutdownError(f"Shutdown failed: {e}") from e

    async def _graceful_shutdown(self) -> None:
        """Graceful shutdown - wait for tasks to complete."""
        # Reverse initialization order
        for batch in reversed(self._initialization_order):
            tasks = []

            for comp_id in batch:
                info = self._components.get(comp_id)
                if info and isinstance(info.instance, Stoppable):
                    tasks.append(self._stop_component(comp_id))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _immediate_shutdown(self) -> None:
        """Immediate shutdown - stop without waiting."""
        tasks = []

        for comp_id, info in self._components.items():
            if isinstance(info.instance, Stoppable):
                tasks.append(self._stop_component(comp_id, timeout=5.0))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _forced_shutdown(self) -> None:
        """Forced shutdown - cancel all tasks."""
        for comp_id, info in self._components.items():
            info.state = LifecycleState.STOPPED

    async def _stop_component(
        self,
        component_id: str,
        timeout: float | None = None,
    ) -> None:
        """Stop a single component."""
        info = self._components.get(component_id)
        if not info or not isinstance(info.instance, Stoppable):
            return

        timeout = timeout or self._config.shutdown_timeout_seconds
        start_time = time.monotonic()

        try:
            info.state = LifecycleState.STOPPING

            await asyncio.wait_for(
                info.instance.stop(),
                timeout=timeout,
            )

            info.state = LifecycleState.STOPPED
            duration_ms = (time.monotonic() - start_time) * 1000

            self._logger.info(f"Stopped {info.name} in {duration_ms:.1f}ms")

            await self._emit_event(
                "component_stopped",
                component_id=component_id,
                old_state=LifecycleState.STOPPING,
                new_state=LifecycleState.STOPPED,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            self._logger.warning(f"Stop timeout for {info.name}")
            info.state = LifecycleState.STOPPED
        except Exception as e:
            self._logger.error(f"Stop error for {info.name}: {e}")
            info.state = LifecycleState.FAILED

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Clean up all components."""
        await self._cleanup_components()

    async def _cleanup_components(self) -> None:
        """Clean up all components."""
        await self._emit_event("cleaning_up")

        tasks = []

        for comp_id, info in self._components.items():
            if isinstance(info.instance, Cleanable):
                tasks.append(self._cleanup_component(comp_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        await self._emit_event("cleaned_up")

    async def _cleanup_component(self, component_id: str) -> None:
        """Clean up a single component."""
        info = self._components.get(component_id)
        if not info or not isinstance(info.instance, Cleanable):
            return

        try:
            await info.instance.cleanup()
            self._logger.debug(f"Cleaned up {info.name}")
        except Exception as e:
            self._logger.error(f"Cleanup error for {info.name}: {e}")

    # -------------------------------------------------------------------------
    # Health Checks
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all components.

        Returns:
            Health check results
        """
        results = await self._health_checker.check_all(self._components)
        aggregate = self._health_checker.get_aggregate_status()

        # Update state based on health
        if self._state == LifecycleState.READY:
            if aggregate == HealthStatus.UNHEALTHY:
                self._state = LifecycleState.DEGRADED
        elif self._state == LifecycleState.DEGRADED:
            if aggregate == HealthStatus.HEALTHY:
                self._state = LifecycleState.READY

        return {
            "status": aggregate.value,
            "state": self._state.name,
            "components": {
                comp_id: result.to_dict()
                for comp_id, result in results.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def readiness_check(self) -> bool:
        """
        Check if system is ready to serve requests.

        Returns:
            True if ready
        """
        return self._state == LifecycleState.READY

    async def liveness_check(self) -> bool:
        """
        Check if system is alive.

        Returns:
            True if alive
        """
        return self._state not in {
            LifecycleState.STOPPED,
            LifecycleState.FAILED,
        }

    # -------------------------------------------------------------------------
    # Signal Handling
    # -------------------------------------------------------------------------

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),  # type: ignore[misc]
            )

        self._logger.info("Signal handlers configured")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        self._logger.info(f"Received signal {sig.name}, initiating shutdown...")
        await self.shutdown()

    # -------------------------------------------------------------------------
    # Wait Methods
    # -------------------------------------------------------------------------

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()

    async def wait_for_ready(
        self,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait for system to become ready.

        Args:
            timeout: Maximum wait time

        Returns:
            True if ready, False if timeout
        """
        start_time = time.monotonic()

        while True:
            if self._state == LifecycleState.READY:
                return True

            if self._state == LifecycleState.FAILED:
                return False

            if timeout:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            await asyncio.sleep(0.1)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get lifecycle statistics."""
        component_states: dict[str, int] = {}
        for info in self._components.values():
            state_name = info.state.name
            component_states[state_name] = component_states.get(state_name, 0) + 1

        return {
            "state": self._state.name,
            "startup_time_seconds": self._startup_time,
            "shutdown_time_seconds": self._shutdown_time,
            "total_components": len(self._components),
            "component_states": component_states,
            "health_status": self._health_checker.get_aggregate_status().value,
            "registered_agents": len(self._agent_coordinator._agents),
            "registered_hooks": sum(len(h) for h in self._hooks.values()),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_lifecycle_manager(
    startup_timeout: float = 60.0,
    shutdown_timeout: float = 30.0,
    enable_warmup: bool = True,
    enable_health_checks: bool = True,
    parallel_initialization: bool = True,
    fail_fast: bool = False,
    logger: logging.Logger | None = None,
) -> LifecycleManager:
    """
    Factory function to create configured LifecycleManager.

    Args:
        startup_timeout: Startup timeout in seconds
        shutdown_timeout: Shutdown timeout in seconds
        enable_warmup: Enable warmup phase
        enable_health_checks: Enable health checks
        parallel_initialization: Initialize in parallel
        fail_fast: Stop on first failure
        logger: Optional logger

    Returns:
        Configured LifecycleManager
    """
    config = LifecycleConfig(
        startup_timeout_seconds=startup_timeout,
        shutdown_timeout_seconds=shutdown_timeout,
        enable_warmup=enable_warmup,
        enable_health_checks=enable_health_checks,
        parallel_initialization=parallel_initialization,
        fail_fast=fail_fast,
    )

    return LifecycleManager(config=config, logger=logger)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LifecycleState",
    "HealthStatus",
    "ComponentType",
    "ShutdownMode",
    "InitializationPriority",
    # Protocols
    "Initializable",
    "Startable",
    "Stoppable",
    "HealthCheckable",
    "Warmable",
    "Cleanable",
    # Data Structures
    "HealthCheckResult",
    "ComponentInfo",
    "LifecycleEvent",
    "LifecycleConfig",
    "HookRegistration",
    "AgentInfo",
    # Exceptions
    "LifecycleError",
    "InitializationError",
    "StartupError",
    "ShutdownError",
    "WarmupError",
    "HealthCheckError",
    "DependencyError",
    # Components
    "DependencyResolver",
    "HealthChecker",
    "AgentLifecycleCoordinator",
    # Manager
    "LifecycleManager",
    # Factory
    "create_lifecycle_manager",
]
