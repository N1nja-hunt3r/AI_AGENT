"""
Capability Registry - Central capability management for the AI Agent Platform.

This module provides the registry system for managing capabilities:
- Registration and lifecycle management
- Dependency resolution and initialization ordering
- Version selection and compatibility checking
- Execution dispatch and routing
- Multi-agent capability isolation

The registry is the single source of truth for all available capabilities
and coordinates their initialization, execution, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)
from weakref import WeakValueDictionary

from app.capabilities.base import (
    Capability,
    CapabilityDependency,
    CapabilityMetadata,
    CapabilityPriority,
    CapabilityState,
    ExecutionContext,
)
from app.agents.models import CapabilityType, SessionState

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
CapabilityT = TypeVar("CapabilityT", bound=Capability[Any])


# =============================================================================
# ENUMS
# =============================================================================


class RegistryState(Enum):
    """State of the capability registry."""

    IDLE = auto()  # No operations in progress
    INITIALIZING = auto()  # Initializing capabilities
    READY = auto()  # All capabilities initialized
    SHUTTING_DOWN = auto()  # Shutting down capabilities
    SHUTDOWN = auto()  # Registry shut down


class DiscoveryStrategy(Enum):
    """Strategy for capability discovery."""

    FIRST = auto()  # Return first match
    ALL = auto()  # Return all matches
    HIGHEST_PRIORITY = auto()  # Return highest priority match
    LATEST_VERSION = auto()  # Return latest version match
    BEST_MATCH = auto()  # Return best overall match (priority + version)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class RegistryError(Exception):
    """Base exception for registry errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context


class CapabilityNotFoundError(RegistryError):
    """Requested capability not found in registry."""

    def __init__(self, identifier: str | CapabilityType) -> None:
        super().__init__(
            f"Capability not found: {identifier}",
            identifier=identifier,
        )
        self.identifier = identifier


class DuplicateCapabilityError(RegistryError):
    """Capability with same name already registered."""

    def __init__(self, name: str, existing_version: str) -> None:
        super().__init__(
            f"Capability '{name}' already registered (version {existing_version})",
            name=name,
            existing_version=existing_version,
        )
        self.name = name
        self.existing_version = existing_version


class DependencyResolutionError(RegistryError):
    """Failed to resolve capability dependencies."""

    def __init__(self, message: str, capability_name: str | None = None) -> None:
        super().__init__(message, capability_name=capability_name)
        self.capability_name = capability_name


class CircularDependencyError(DependencyResolutionError):
    """Circular dependency detected."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(
            f"Circular dependency detected: {' -> '.join(cycle)}",
        )
        self.cycle = cycle


class VersionConflictError(RegistryError):
    """Version requirement conflict."""

    def __init__(
        self,
        capability_name: str,
        required_version: str,
        available_version: str,
    ) -> None:
        super().__init__(
            f"Version conflict for '{capability_name}': "
            f"required {required_version}, available {available_version}",
            capability_name=capability_name,
        )
        self.required_version = required_version
        self.available_version = available_version


# =============================================================================
# CAPABILITY ENTRY
# =============================================================================


@dataclass
class CapabilityEntry:
    """
    Registry entry wrapping a capability instance.

    Holds the capability along with registration metadata
    and runtime state tracking.

    Attributes:
        capability: The capability instance
        registered_at: When capability was registered
        initialized_at: When capability was initialized
        last_used: Last execution timestamp
        use_count: Number of executions
        error_count: Number of errors
        tags: Additional searchable tags
    """

    capability: Capability[Any]
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    initialized_at: datetime | None = None
    last_used: datetime | None = None
    use_count: int = 0
    error_count: int = 0
    tags: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Get capability name."""
        return self.capability.name

    @property
    def version(self) -> str:
        """Get capability version."""
        return self.capability.version

    @property
    def capability_type(self) -> CapabilityType:
        """Get capability type."""
        return self.capability.capability_type

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return self.capability.metadata

    @property
    def state(self) -> CapabilityState:
        """Get capability state."""
        return self.capability.state

    @property
    def priority(self) -> CapabilityPriority:
        """Get capability priority."""
        return self.capability.priority

    @property
    def is_ready(self) -> bool:
        """Check if capability is ready."""
        return self.capability.is_ready

    @property
    def all_tags(self) -> set[str]:
        """Get all tags (metadata + entry tags)."""
        return set(self.metadata.tags) | self.tags

    def record_use(self, success: bool = True) -> None:
        """Record a capability use."""
        self.use_count += 1
        self.last_used = datetime.now(timezone.utc)
        if not success:
            self.error_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.capability_type.value,
            "state": self.state.name,
            "priority": self.priority.name,
            "registered_at": self.registered_at.isoformat(),
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "error_count": self.error_count,
            "tags": list(self.all_tags),
        }


# =============================================================================
# DISCOVERY CRITERIA
# =============================================================================


@dataclass
class DiscoveryCriteria:
    """
    Criteria for capability discovery.

    Attributes:
        capability_type: Filter by type
        name_pattern: Filter by name pattern (supports *)
        tags: Filter by tags (any match)
        min_version: Minimum version requirement
        max_version: Maximum version requirement
        state: Filter by capability state
        priority: Minimum priority level
        exclude_deprecated: Exclude deprecated capabilities
    """

    capability_type: CapabilityType | None = None
    name_pattern: str | None = None
    tags: set[str] | None = None
    min_version: str | None = None
    max_version: str | None = None
    state: CapabilityState | None = None
    priority: CapabilityPriority | None = None
    exclude_deprecated: bool = True

    def matches(self, entry: CapabilityEntry) -> bool:
        """Check if entry matches criteria."""
        # Type filter
        if self.capability_type and entry.capability_type != self.capability_type:
            return False

        # Name pattern filter
        if self.name_pattern:
            if not self._match_pattern(entry.name, self.name_pattern):
                return False

        # Tags filter (any match)
        if self.tags:
            if not self.tags & entry.all_tags:
                return False

        # Version filters
        if self.min_version:
            if not self._version_gte(entry.version, self.min_version):
                return False

        if self.max_version:
            if not self._version_lte(entry.version, self.max_version):
                return False

        # State filter
        if self.state and entry.state != self.state:
            return False

        # Priority filter
        if self.priority and entry.priority < self.priority:
            return False

        # Deprecated filter
        if self.exclude_deprecated and entry.metadata.deprecated:
            return False

        return True

    @staticmethod
    def _match_pattern(name: str, pattern: str) -> bool:
        """Simple pattern matching with * wildcard."""
        if "*" not in pattern:
            return name == pattern

        parts = pattern.split("*")
        if len(parts) == 2:
            prefix, suffix = parts
            return name.startswith(prefix) and name.endswith(suffix)

        return pattern.replace("*", "") in name

    @staticmethod
    def _version_gte(version: str, minimum: str) -> bool:
        """Check if version >= minimum."""
        try:
            v = tuple(int(x) for x in version.split("."))
            m = tuple(int(x) for x in minimum.split("."))
            return v >= m
        except ValueError:
            return False

    @staticmethod
    def _version_lte(version: str, maximum: str) -> bool:
        """Check if version <= maximum."""
        try:
            v = tuple(int(x) for x in version.split("."))
            m = tuple(int(x) for x in maximum.split("."))
            return v <= m
        except ValueError:
            return False


# =============================================================================
# DEPENDENCY RESOLVER
# =============================================================================


class DependencyResolver:
    """
    Resolves capability dependencies and determines initialization order.

    Uses topological sort to order capabilities by their dependencies,
    detecting circular dependencies.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def resolve(
        self,
        entries: dict[str, CapabilityEntry],
        type_index: dict[CapabilityType, list[str]],
    ) -> list[str]:
        """
        Resolve initialization order for capabilities.

        Args:
            entries: All registered capability entries
            type_index: Index of capabilities by type

        Returns:
            List of capability names in initialization order

        Raises:
            CircularDependencyError: If circular dependency detected
            DependencyResolutionError: If required dependency missing
        """
        # Build dependency graph
        graph: dict[str, set[str]] = {}
        in_degree: dict[str, int] = {}

        for name, entry in entries.items():
            graph[name] = set()
            in_degree[name] = 0

        # Add edges for dependencies
        for name, entry in entries.items():
            for dep in entry.metadata.dependencies:
                # Find capability that satisfies dependency
                dep_name = self._find_dependency(dep, entries, type_index)

                if dep_name is None:
                    if dep.required:
                        raise DependencyResolutionError(
                            f"Required dependency {dep.capability_type.value} "
                            f"not found for capability '{name}'",
                            capability_name=name,
                        )
                    continue

                # Add edge: dep_name -> name (dep must init before name)
                if dep_name not in graph[name]:
                    graph[name].add(dep_name)
                    in_degree[name] = in_degree.get(name, 0) + 1

        # Topological sort (Kahn's algorithm)
        queue: list[str] = [n for n, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            # Sort by priority (higher priority first)
            queue.sort(key=lambda n: entries[n].priority.value, reverse=True)
            node = queue.pop(0)
            result.append(node)

            for dependent in entries.keys():
                if node in graph[dependent]:
                    graph[dependent].remove(node)
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for cycles
        if len(result) != len(entries):
            remaining = set(entries.keys()) - set(result)
            cycle = self._find_cycle(remaining, graph)
            raise CircularDependencyError(cycle)

        return result

    def _find_dependency(
        self,
        dep: CapabilityDependency,
        entries: dict[str, CapabilityEntry],
        type_index: dict[CapabilityType, list[str]],
    ) -> str | None:
        """Find a capability that satisfies a dependency."""
        candidates = type_index.get(dep.capability_type, [])

        for name in candidates:
            entry = entries[name]
            if dep.is_compatible(entry.version):
                return name

        return None

    def _find_cycle(
        self,
        nodes: set[str],
        graph: dict[str, set[str]],
    ) -> list[str]:
        """Find a cycle in the remaining nodes."""
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

            for dep in graph.get(node, set()):
                if dep in nodes:
                    result = dfs(dep)
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
# VERSION SELECTOR
# =============================================================================


class VersionSelector:
    """
    Selects the best capability version based on requirements.

    Supports semantic versioning comparison and compatibility checking.
    """

    @staticmethod
    def select_best(
        entries: list[CapabilityEntry],
        min_version: str | None = None,
        max_version: str | None = None,
        prefer_latest: bool = True,
    ) -> CapabilityEntry | None:
        """
        Select the best capability from candidates.

        Args:
            entries: Candidate capability entries
            min_version: Minimum version requirement
            max_version: Maximum version requirement
            prefer_latest: If True, prefer latest version

        Returns:
            Best matching entry or None
        """
        if not entries:
            return None

        # Filter by version requirements
        candidates = []
        for entry in entries:
            if min_version and not VersionSelector._version_gte(entry.version, min_version):
                continue
            if max_version and not VersionSelector._version_lte(entry.version, max_version):
                continue
            candidates.append(entry)

        if not candidates:
            return None

        # Sort by version and priority
        def sort_key(e: CapabilityEntry) -> tuple[tuple[int, ...], int]:
            version_tuple = tuple(int(x) for x in e.version.split("."))
            return (version_tuple, e.priority.value)

        candidates.sort(key=sort_key, reverse=prefer_latest)
        return candidates[0]

    @staticmethod
    def _version_gte(version: str, minimum: str) -> bool:
        try:
            v = tuple(int(x) for x in version.split("."))
            m = tuple(int(x) for x in minimum.split("."))
            return v >= m
        except ValueError:
            return False

    @staticmethod
    def _version_lte(version: str, maximum: str) -> bool:
        try:
            v = tuple(int(x) for x in version.split("."))
            m = tuple(int(x) for x in maximum.split("."))
            return v <= m
        except ValueError:
            return False


# =============================================================================
# EXECUTION DISPATCHER
# =============================================================================


class ExecutionDispatcher:
    """
    Dispatches execution requests to appropriate capabilities.

    Handles capability selection, execution routing, and
    result collection.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._logger = logger or logging.getLogger(__name__)

    async def dispatch(
        self,
        capability_type: CapabilityType,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
        capability_name: str | None = None,
    ) -> Any:
        """
        Dispatch execution to a capability.

        Args:
            capability_type: Type of capability to use
            action: Action to execute
            parameters: Action parameters
            context: Execution context
            capability_name: Specific capability name (optional)

        Returns:
            Execution result

        Raises:
            CapabilityNotFoundError: If no suitable capability found
        """
        # Find capability
        if capability_name:
            entry = self._registry.get_entry(capability_name)
            if entry is None:
                raise CapabilityNotFoundError(capability_name)
        else:
            entry = self._registry.get_by_type(capability_type)
            if entry is None:
                raise CapabilityNotFoundError(capability_type)

        # Execute
        self._logger.debug(
            f"Dispatching {action} to {entry.name}",
            extra={"capability": entry.name, "action": action},
        )

        try:
            result = await entry.capability.execute(action, parameters, context)
            entry.record_use(success=True)
            return result

        except Exception:
            entry.record_use(success=False)
            raise

    async def dispatch_to_all(
        self,
        capability_type: CapabilityType,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
    ) -> list[tuple[str, Any]]:
        """
        Dispatch execution to all capabilities of a type.

        Args:
            capability_type: Type of capabilities to use
            action: Action to execute
            parameters: Action parameters
            context: Execution context

        Returns:
            List of (capability_name, result) tuples
        """
        entries = self._registry.get_all_by_type(capability_type)
        results: list[tuple[str, Any]] = []

        for entry in entries:
            if not entry.is_ready:
                continue

            try:
                result = await entry.capability.execute(action, parameters, context)
                entry.record_use(success=True)
                results.append((entry.name, result))
            except Exception as e:
                entry.record_use(success=False)
                self._logger.warning(f"Dispatch to {entry.name} failed: {e}")
                results.append((entry.name, e))

        return results


# =============================================================================
# AGENT CAPABILITY VIEW
# =============================================================================


class AgentCapabilityView:
    """
    Agent-scoped view of available capabilities.

    Provides isolation between agents, allowing each agent
    to have its own set of available capabilities.

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        agent_id: str,
        agent_name: str | None = None,
        allowed_types: set[CapabilityType] | None = None,
        allowed_names: set[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Create agent capability view.

        Args:
            registry: Parent registry
            agent_id: Unique agent identifier
            agent_name: Optional agent name
            allowed_types: Allowed capability types (None = all)
            allowed_names: Allowed capability names (None = all)
            logger: Optional logger
        """
        self._registry = registry
        self._agent_id = agent_id
        self._agent_name = agent_name or agent_id
        self._allowed_types = allowed_types
        self._allowed_names = allowed_names
        self._logger = logger or logging.getLogger(__name__)

        # Track attached capabilities
        self._attached: set[str] = set()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def is_allowed(self, entry: CapabilityEntry) -> bool:
        """Check if capability is allowed for this agent."""
        if self._allowed_types and entry.capability_type not in self._allowed_types:
            return False
        if self._allowed_names and entry.name not in self._allowed_names:
            return False
        return True

    def get(self, name: str) -> Capability[Any] | None:
        """Get capability by name if allowed."""
        entry = self._registry.get_entry(name)
        if entry and self.is_allowed(entry):
            return entry.capability
        return None

    def get_by_type(self, capability_type: CapabilityType) -> Capability[Any] | None:
        """Get capability by type if allowed."""
        if self._allowed_types and capability_type not in self._allowed_types:
            return None

        entry = self._registry.get_by_type(capability_type)
        if entry and self.is_allowed(entry):
            return entry.capability
        return None

    def list_available(self) -> list[CapabilityEntry]:
        """List all available capabilities for this agent."""
        return [
            entry
            for entry in self._registry.list_all()
            if self.is_allowed(entry)
        ]

    async def attach_all(self) -> None:
        """Attach agent to all allowed capabilities."""
        for entry in self.list_available():
            if entry.name not in self._attached:
                await entry.capability.on_agent_attach(self._agent_id, self._agent_name)
                self._attached.add(entry.name)

    async def detach_all(self) -> None:
        """Detach agent from all capabilities."""
        for name in list(self._attached):
            entry = self._registry.get_entry(name)
            if entry:
                await entry.capability.on_agent_detach(self._agent_id)
            self._attached.discard(name)

    async def execute(
        self,
        capability_type: CapabilityType,
        action: str,
        parameters: dict[str, Any],
        context: SessionState,
    ) -> Any:
        """Execute action through this agent view."""
        capability = self.get_by_type(capability_type)
        if capability is None:
            raise CapabilityNotFoundError(capability_type)

        # Create agent-scoped context
        exec_context = ExecutionContext(
            session=context,
            action=action,
            parameters=parameters,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
        )

        return await capability.execute(action, parameters, exec_context)


# =============================================================================
# REGISTRY SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class RegistrySnapshot:
    """
    Immutable snapshot of registry state.

    Useful for diagnostics, monitoring, and state comparison.
    """

    timestamp: datetime
    state: RegistryState
    capability_count: int
    capabilities: tuple[dict[str, Any], ...]
    type_counts: dict[str, int]
    health_status: dict[str, bool]

    @classmethod
    def create(
        cls,
        registry: CapabilityRegistry,
        health_status: dict[str, bool] | None = None,
    ) -> RegistrySnapshot:
        """Create snapshot from registry."""
        entries = registry.list_all()

        type_counts: dict[str, int] = {}
        for entry in entries:
            type_name = entry.capability_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return cls(
            timestamp=datetime.now(timezone.utc),
            state=registry.state,
            capability_count=len(entries),
            capabilities=tuple(e.to_dict() for e in entries),
            type_counts=type_counts,
            health_status=health_status or {},
        )


# =============================================================================
# MAIN REGISTRY
# =============================================================================


class CapabilityRegistry:
    """
    Central registry for managing capabilities.

    Provides registration, discovery, lifecycle management,
    and execution dispatch for all capabilities in the system.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the capability registry.

        Args:
            logger: Optional logger instance
        """
        self._logger = logger or logging.getLogger(__name__)

        # State
        self._state = RegistryState.IDLE
        self._state_lock = asyncio.Lock()

        # Thread safety
        self._lock = threading.RLock()

        # Primary storage
        self._entries: dict[str, CapabilityEntry] = {}

        # Indexes for fast lookup
        self._by_type: dict[CapabilityType, list[str]] = defaultdict(list)
        self._by_tag: dict[str, set[str]] = defaultdict(set)
        self._by_priority: dict[CapabilityPriority, list[str]] = defaultdict(list)

        # Components
        self._dependency_resolver = DependencyResolver(logger)
        self._version_selector = VersionSelector()
        self._dispatcher: ExecutionDispatcher | None = None

        # Agent views (weak references)
        self._agent_views: WeakValueDictionary[str, AgentCapabilityView] = WeakValueDictionary()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> RegistryState:
        """Get registry state."""
        return self._state

    @property
    def count(self) -> int:
        """Get number of registered capabilities."""
        with self._lock:
            return len(self._entries)

    @property
    def dispatcher(self) -> ExecutionDispatcher:
        """Get execution dispatcher."""
        if self._dispatcher is None:
            self._dispatcher = ExecutionDispatcher(self, self._logger)
        return self._dispatcher

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(
        self,
        capability: Capability[Any],
        tags: set[str] | None = None,
        replace: bool = False,
    ) -> CapabilityEntry:
        """
        Register a capability.

        Args:
            capability: Capability instance to register
            tags: Additional tags for discovery
            replace: If True, replace existing capability with same name

        Returns:
            CapabilityEntry for the registered capability

        Raises:
            DuplicateCapabilityError: If capability exists and replace=False
        """
        with self._lock:
            name = capability.name

            # Check for existing
            if name in self._entries:
                if not replace:
                    existing = self._entries[name]
                    raise DuplicateCapabilityError(name, existing.version)

                # Unregister existing
                self._remove_from_indexes(name)

            # Create entry
            entry = CapabilityEntry(
                capability=capability,
                tags=tags or set(),
            )

            # Store
            self._entries[name] = entry

            # Update indexes
            self._add_to_indexes(entry)

            self._logger.info(
                f"Registered capability: {name} v{capability.version} "
                f"(type: {capability.capability_type.value})"
            )

            return entry

    def unregister(self, name: str) -> bool:
        """
        Unregister a capability.

        Args:
            name: Capability name to unregister

        Returns:
            True if capability was unregistered
        """
        with self._lock:
            if name not in self._entries:
                return False

            self._remove_from_indexes(name)
            del self._entries[name]

            self._logger.info(f"Unregistered capability: {name}")
            return True

    def _add_to_indexes(self, entry: CapabilityEntry) -> None:
        """Add entry to all indexes."""
        name = entry.name

        # Type index
        self._by_type[entry.capability_type].append(name)

        # Tag index
        for tag in entry.all_tags:
            self._by_tag[tag].add(name)

        # Priority index
        self._by_priority[entry.priority].append(name)

    def _remove_from_indexes(self, name: str) -> None:
        """Remove entry from all indexes."""
        entry = self._entries.get(name)
        if not entry:
            return

        # Type index
        if name in self._by_type[entry.capability_type]:
            self._by_type[entry.capability_type].remove(name)

        # Tag index
        for tag in entry.all_tags:
            self._by_tag[tag].discard(name)

        # Priority index
        if name in self._by_priority[entry.priority]:
            self._by_priority[entry.priority].remove(name)

    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

    def get(self, name: str) -> Capability[Any] | None:
        """
        Get capability by name.

        Args:
            name: Capability name

        Returns:
            Capability instance or None
        """
        with self._lock:
            entry = self._entries.get(name)
            return entry.capability if entry else None

    def get_entry(self, name: str) -> CapabilityEntry | None:
        """
        Get capability entry by name.

        Args:
            name: Capability name

        Returns:
            CapabilityEntry or None
        """
        with self._lock:
            return self._entries.get(name)

    def get_by_type(
        self,
        capability_type: CapabilityType,
        prefer_ready: bool = True,
    ) -> CapabilityEntry | None:
        """
        Get best capability of a type.

        Args:
            capability_type: Type to look up
            prefer_ready: Prefer ready capabilities

        Returns:
            Best matching CapabilityEntry or None
        """
        with self._lock:
            names = self._by_type.get(capability_type, [])
            if not names:
                return None

            entries = [self._entries[n] for n in names]

            # Filter to ready if preferred
            if prefer_ready:
                ready = [e for e in entries if e.is_ready]
                if ready:
                    entries = ready

            # Select best by priority and version
            return self._version_selector.select_best(entries)

    def get_all_by_type(
        self,
        capability_type: CapabilityType,
    ) -> list[CapabilityEntry]:
        """
        Get all capabilities of a type.

        Args:
            capability_type: Type to look up

        Returns:
            List of matching entries
        """
        with self._lock:
            names = self._by_type.get(capability_type, [])
            return [self._entries[n] for n in names]

    def __getitem__(self, name: str) -> Capability[Any]:
        """Get capability by name (raises if not found)."""
        capability = self.get(name)
        if capability is None:
            raise CapabilityNotFoundError(name)
        return capability

    def __contains__(self, name: str) -> bool:
        """Check if capability is registered."""
        with self._lock:
            return name in self._entries

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover(
        self,
        criteria: DiscoveryCriteria | None = None,
        strategy: DiscoveryStrategy = DiscoveryStrategy.ALL,
    ) -> list[CapabilityEntry]:
        """
        Discover capabilities matching criteria.

        Args:
            criteria: Discovery criteria (None = all)
            strategy: Discovery strategy

        Returns:
            List of matching entries
        """
        with self._lock:
            if criteria is None:
                entries = list(self._entries.values())
            else:
                entries = [e for e in self._entries.values() if criteria.matches(e)]

            if not entries:
                return []

            # Apply strategy
            if strategy == DiscoveryStrategy.FIRST:
                return entries[:1]

            elif strategy == DiscoveryStrategy.HIGHEST_PRIORITY:
                entries.sort(key=lambda e: e.priority.value, reverse=True)
                return entries[:1]

            elif strategy == DiscoveryStrategy.LATEST_VERSION:
                entries.sort(
                    key=lambda e: tuple(int(x) for x in e.version.split(".")),
                    reverse=True,
                )
                return entries[:1]

            elif strategy == DiscoveryStrategy.BEST_MATCH:
                # Sort by priority then version
                entries.sort(
                    key=lambda e: (
                        e.priority.value,
                        tuple(int(x) for x in e.version.split(".")),
                    ),
                    reverse=True,
                )
                return entries[:1]

            # ALL strategy
            return entries

    def discover_by_tag(self, tag: str) -> list[CapabilityEntry]:
        """
        Discover capabilities by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching entries
        """
        with self._lock:
            names = self._by_tag.get(tag, set())
            return [self._entries[n] for n in names]

    def list_all(self) -> list[CapabilityEntry]:
        """List all registered capabilities."""
        with self._lock:
            return list(self._entries.values())

    def list_types(self) -> list[CapabilityType]:
        """List all registered capability types."""
        with self._lock:
            return [t for t, names in self._by_type.items() if names]

    def list_tags(self) -> list[str]:
        """List all registered tags."""
        with self._lock:
            return [t for t, names in self._by_tag.items() if names]

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def initialize_all(self) -> dict[str, bool]:
        """
        Initialize all capabilities in dependency order.

        Returns:
            Dictionary mapping capability names to success status
        """
        async with self._state_lock:
            if self._state != RegistryState.IDLE:
                self._logger.warning(f"Registry not idle (state: {self._state.name})")
                return {}

            self._state = RegistryState.INITIALIZING

        results: dict[str, bool] = {}

        try:
            # Resolve initialization order
            with self._lock:
                order = self._dependency_resolver.resolve(
                    self._entries,
                    dict(self._by_type),
                )

            self._logger.info(f"Initializing {len(order)} capabilities")

            # Initialize in order
            for name in order:
                entry = self._entries.get(name)
                if not entry:
                    continue

                try:
                    await entry.capability.initialize()
                    entry.initialized_at = datetime.now(timezone.utc)
                    results[name] = True
                    self._logger.debug(f"Initialized: {name}")

                except Exception as e:
                    results[name] = False
                    self._logger.error(f"Failed to initialize {name}: {e}")

            async with self._state_lock:
                self._state = RegistryState.READY

            success_count = sum(1 for v in results.values() if v)
            self._logger.info(
                f"Initialization complete: {success_count}/{len(results)} succeeded"
            )

        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            async with self._state_lock:
                self._state = RegistryState.IDLE
            raise

        return results

    async def shutdown_all(self) -> dict[str, bool]:
        """
        Shutdown all capabilities in reverse dependency order.

        Returns:
            Dictionary mapping capability names to success status
        """
        async with self._state_lock:
            self._state = RegistryState.SHUTTING_DOWN

        results: dict[str, bool] = {}

        try:
            # Get reverse initialization order
            with self._lock:
                try:
                    order = self._dependency_resolver.resolve(
                        self._entries,
                        dict(self._by_type),
                    )
                    order.reverse()
                except Exception:
                    # Fallback to any order
                    order = list(self._entries.keys())

            self._logger.info(f"Shutting down {len(order)} capabilities")

            # Shutdown in reverse order
            for name in order:
                entry = self._entries.get(name)
                if not entry:
                    continue

                try:
                    await entry.capability.shutdown()
                    results[name] = True
                    self._logger.debug(f"Shut down: {name}")

                except Exception as e:
                    results[name] = False
                    self._logger.error(f"Failed to shut down {name}: {e}")

            async with self._state_lock:
                self._state = RegistryState.SHUTDOWN

            self._logger.info("Shutdown complete")

        except Exception as e:
            self._logger.error(f"Shutdown failed: {e}")
            raise

        return results

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    async def health_status(self) -> dict[str, bool]:
        """
        Get health status of all capabilities.

        Returns:
            Dictionary mapping capability names to health status
        """
        results: dict[str, bool] = {}

        with self._lock:
            entries = list(self._entries.values())

        for entry in entries:
            try:
                results[entry.name] = await entry.capability.health_check()
            except Exception:
                results[entry.name] = False

        return results

    async def health_check(self) -> bool:
        """
        Check if registry is healthy (all capabilities healthy).

        Returns:
            True if all capabilities are healthy
        """
        status = await self.health_status()
        return all(status.values()) if status else True

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        capability_type: CapabilityType,
        action: str,
        parameters: dict[str, Any],
        context: SessionState | ExecutionContext,
        capability_name: str | None = None,
    ) -> Any:
        """
        Execute action on a capability.

        Args:
            capability_type: Type of capability
            action: Action to execute
            parameters: Action parameters
            context: Execution context
            capability_name: Specific capability name (optional)

        Returns:
            Execution result
        """
        return await self.dispatcher.dispatch(
            capability_type=capability_type,
            action=action,
            parameters=parameters,
            context=context,
            capability_name=capability_name,
        )

    # -------------------------------------------------------------------------
    # Multi-Agent Support
    # -------------------------------------------------------------------------

    def create_agent_view(
        self,
        agent_id: str,
        agent_name: str | None = None,
        allowed_types: set[CapabilityType] | None = None,
        allowed_names: set[str] | None = None,
    ) -> AgentCapabilityView:
        """
        Create an agent-scoped capability view.

        Args:
            agent_id: Unique agent identifier
            agent_name: Optional agent name
            allowed_types: Allowed capability types
            allowed_names: Allowed capability names

        Returns:
            AgentCapabilityView for the agent
        """
        view = AgentCapabilityView(
            registry=self,
            agent_id=agent_id,
            agent_name=agent_name,
            allowed_types=allowed_types,
            allowed_names=allowed_names,
            logger=self._logger,
        )

        self._agent_views[agent_id] = view
        self._logger.debug(f"Created agent view: {agent_id}")

        return view

    def get_agent_view(self, agent_id: str) -> AgentCapabilityView | None:
        """Get existing agent view."""
        return self._agent_views.get(agent_id)

    # -------------------------------------------------------------------------
    # Dependency Resolution
    # -------------------------------------------------------------------------

    def resolve_dependencies(self) -> list[str]:
        """
        Resolve capability initialization order.

        Returns:
            List of capability names in initialization order

        Raises:
            CircularDependencyError: If circular dependency detected
        """
        with self._lock:
            return self._dependency_resolver.resolve(
                self._entries,
                dict(self._by_type),
            )

    def get_dependencies(self, name: str) -> list[CapabilityDependency]:
        """Get dependencies for a capability."""
        with self._lock:
            entry = self._entries.get(name)
            if not entry:
                return []
            return list(entry.metadata.dependencies)

    def get_dependents(self, name: str) -> list[str]:
        """Get capabilities that depend on the given capability."""
        with self._lock:
            entry = self._entries.get(name)
            if not entry:
                return []

            dependents = []
            for other_name, other_entry in self._entries.items():
                if other_name == name:
                    continue

                for dep in other_entry.metadata.dependencies:
                    if dep.capability_type == entry.capability_type:
                        dependents.append(other_name)
                        break

            return dependents

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def snapshot(self) -> RegistrySnapshot:
        """Create immutable snapshot of registry state."""
        return RegistrySnapshot.create(self)

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            type_counts = {
                t.value: len(names)
                for t, names in self._by_type.items()
                if names
            }

            priority_counts = {
                p.name: len(names)
                for p, names in self._by_priority.items()
                if names
            }

            ready_count = sum(
                1 for e in self._entries.values()
                if e.is_ready
            )

            return {
                "state": self._state.name,
                "total_capabilities": len(self._entries),
                "ready_capabilities": ready_count,
                "type_counts": type_counts,
                "priority_counts": priority_counts,
                "tag_count": len([t for t, n in self._by_tag.items() if n]),
                "agent_views": len(self._agent_views),
            }

    def __repr__(self) -> str:
        return f"CapabilityRegistry(count={self.count}, state={self._state.name})"


# =============================================================================
# GLOBAL REGISTRY (Optional Singleton)
# =============================================================================

_global_registry: CapabilityRegistry | None = None
_global_lock = threading.Lock()


def get_global_registry() -> CapabilityRegistry:
    """
    Get the global capability registry singleton.

    Returns:
        Global CapabilityRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = CapabilityRegistry()

    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry

    with _global_lock:
        _global_registry = None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_registry(
    capabilities: list[Capability[Any]] | None = None,
    logger: logging.Logger | None = None,
) -> CapabilityRegistry:
    """
    Factory function to create a configured registry.

    Args:
        capabilities: Initial capabilities to register
        logger: Optional logger

    Returns:
        Configured CapabilityRegistry
    """
    registry = CapabilityRegistry(logger=logger)

    if capabilities:
        for capability in capabilities:
            registry.register(capability)

    return registry


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RegistryState",
    "DiscoveryStrategy",
    # Exceptions
    "RegistryError",
    "CapabilityNotFoundError",
    "DuplicateCapabilityError",
    "DependencyResolutionError",
    "CircularDependencyError",
    "VersionConflictError",
    # Data Classes
    "CapabilityEntry",
    "DiscoveryCriteria",
    "RegistrySnapshot",
    # Components
    "DependencyResolver",
    "VersionSelector",
    "ExecutionDispatcher",
    "AgentCapabilityView",
    # Main Registry
    "CapabilityRegistry",
    # Global
    "get_global_registry",
    "reset_global_registry",
    # Factory
    "create_registry",
]
