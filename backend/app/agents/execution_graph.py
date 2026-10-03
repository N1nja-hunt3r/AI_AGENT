"""
Execution Graph - DAG-based execution planning for the AI Agent Platform.

This module provides comprehensive graph-based execution including:
- Directed Acyclic Graph (DAG) construction and validation
- Dependency management and resolution
- Parallel execution with concurrency control
- Topological sorting with multiple strategies
- Visualization support (Mermaid, DOT, ASCII)
- Rollback and recovery support
- Execution state tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# ENUMS
# =============================================================================


class NodeState(Enum):
    """Execution state of a node."""

    PENDING = auto()  # Not yet executed
    READY = auto()  # Dependencies satisfied, ready to execute
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Execution failed
    SKIPPED = auto()  # Skipped due to dependency failure
    CANCELLED = auto()  # Cancelled by user or system
    ROLLED_BACK = auto()  # Rolled back after failure


class EdgeType(Enum):
    """Types of edges in the graph."""

    DEPENDENCY = "dependency"  # Standard dependency
    SOFT_DEPENDENCY = "soft_dependency"  # Optional dependency
    DATA_FLOW = "data_flow"  # Data passes between nodes
    CONDITIONAL = "conditional"  # Conditional execution


class ExecutionStrategy(Enum):
    """Strategies for graph execution."""

    PARALLEL = auto()  # Execute independent nodes in parallel
    SEQUENTIAL = auto()  # Execute nodes one at a time
    BREADTH_FIRST = auto()  # Level by level execution
    PRIORITY = auto()  # Priority-based execution


class SortStrategy(Enum):
    """Strategies for topological sorting."""

    KAHN = "kahn"  # Kahn's algorithm
    DFS = "dfs"  # Depth-first search
    PRIORITY = "priority"  # Priority-aware sorting
    LEVEL = "level"  # Level-based grouping


class RollbackStrategy(Enum):
    """Strategies for rollback."""

    NONE = auto()  # No rollback
    REVERSE_ORDER = auto()  # Rollback in reverse execution order
    PARALLEL = auto()  # Rollback in parallel
    SELECTIVE = auto()  # Only rollback affected nodes


class VisualizationFormat(Enum):
    """Output formats for visualization."""

    MERMAID = "mermaid"
    DOT = "dot"
    ASCII = "ascii"
    JSON = "json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class NodeResult:
    """
    Result of node execution.

    Attributes:
        node_id: Node identifier
        success: Whether execution succeeded
        output: Execution output
        error: Error message if failed
        start_time: Execution start time
        end_time: Execution end time
        duration_ms: Execution duration
        metadata: Additional result data
    """

    node_id: str
    success: bool
    output: Any = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "success": self.success,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """
    Edge connecting two nodes.

    Attributes:
        edge_id: Unique edge identifier
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of edge
        condition: Condition for conditional edges
        data_key: Key for data flow edges
        metadata: Additional edge data
    """

    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DEPENDENCY
    condition: Callable[[Any], bool] | None = None
    data_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "data_key": self.data_key,
            "metadata": self.metadata,
        }


@dataclass
class Node(Generic[T]):
    """
    Node in the execution graph.

    Attributes:
        node_id: Unique node identifier
        name: Human-readable name
        action: Action to execute
        parameters: Action parameters
        state: Current execution state
        priority: Execution priority (higher = earlier)
        timeout_seconds: Execution timeout
        retries: Maximum retry attempts
        rollback_action: Action to rollback this node
        result: Execution result
        metadata: Additional node data
    """

    node_id: str
    name: str
    action: Callable[..., Awaitable[T]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    state: NodeState = NodeState.PENDING
    priority: int = 0
    timeout_seconds: float = 60.0
    retries: int = 0
    rollback_action: Callable[..., Awaitable[None]] | None = None
    result: NodeResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Graph relationships (populated by graph)
    _incoming_edges: list[str] = field(default_factory=list)
    _outgoing_edges: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "state": self.state.name,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "parameters": self.parameters,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionLevel:
    """
    A level in the execution graph (nodes that can run in parallel).

    Attributes:
        level: Level number (0 = root)
        node_ids: Node IDs at this level
    """

    level: int
    node_ids: list[str]


@dataclass
class GraphStats:
    """
    Statistics about the execution graph.

    Attributes:
        total_nodes: Total number of nodes
        total_edges: Total number of edges
        depth: Maximum depth of the graph
        width: Maximum width (nodes at any level)
        completed_nodes: Number of completed nodes
        failed_nodes: Number of failed nodes
        pending_nodes: Number of pending nodes
    """

    total_nodes: int = 0
    total_edges: int = 0
    depth: int = 0
    width: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    pending_nodes: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "depth": self.depth,
            "width": self.width,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "pending_nodes": self.pending_nodes,
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================


class GraphError(Exception):
    """Base exception for graph errors."""

    pass


class CycleDetectedError(GraphError):
    """Raised when a cycle is detected in the graph."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Cycle detected: {' -> '.join(cycle)}")


class NodeNotFoundError(GraphError):
    """Raised when a node is not found."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class EdgeNotFoundError(GraphError):
    """Raised when an edge is not found."""

    def __init__(self, edge_id: str) -> None:
        self.edge_id = edge_id
        super().__init__(f"Edge not found: {edge_id}")


class ExecutionError(GraphError):
    """Raised when execution fails."""

    def __init__(
        self,
        message: str,
        node_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.node_id = node_id
        self.cause = cause
        super().__init__(message)


class RollbackError(GraphError):
    """Raised when rollback fails."""

    def __init__(
        self,
        message: str,
        failed_nodes: list[str] | None = None,
    ) -> None:
        self.failed_nodes = failed_nodes or []
        super().__init__(message)


# =============================================================================
# TOPOLOGICAL SORTER
# =============================================================================


class TopologicalSorter:
    """
    Performs topological sorting on the execution graph.

    Supports multiple sorting strategies.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def sort(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        strategy: SortStrategy = SortStrategy.KAHN,
    ) -> list[str]:
        """
        Sort nodes topologically.

        Args:
            nodes: Dictionary of nodes
            edges: Dictionary of edges
            strategy: Sorting strategy

        Returns:
            List of node IDs in topological order

        Raises:
            CycleDetectedError: If cycle is detected
        """
        if strategy == SortStrategy.KAHN:
            return self._kahn_sort(nodes, edges)
        elif strategy == SortStrategy.DFS:
            return self._dfs_sort(nodes, edges)
        elif strategy == SortStrategy.PRIORITY:
            return self._priority_sort(nodes, edges)
        elif strategy == SortStrategy.LEVEL:
            levels = self.get_levels(nodes, edges)
            return [nid for level in levels for nid in level.node_ids]
        else:
            return self._kahn_sort(nodes, edges)

    def _kahn_sort(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
    ) -> list[str]:
        """Kahn's algorithm for topological sort."""
        # Build adjacency and in-degree
        in_degree: dict[str, int] = {nid: 0 for nid in nodes}
        adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}

        for edge in edges.values():
            if edge.source_id in nodes and edge.target_id in nodes:
                adjacency[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        # Start with nodes having no dependencies
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result: list[str] = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(nodes):
            # Cycle detected - find it
            remaining = set(nodes.keys()) - set(result)
            cycle = self._find_cycle(remaining, adjacency)
            raise CycleDetectedError(cycle)

        return result

    def _dfs_sort(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
    ) -> list[str]:
        """DFS-based topological sort."""
        adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}

        for edge in edges.values():
            if edge.source_id in nodes and edge.target_id in nodes:
                adjacency[edge.source_id].append(edge.target_id)

        visited: set[str] = set()
        temp_visited: set[str] = set()
        result: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in temp_visited:
                # Find cycle
                cycle = self._find_cycle({node_id}, adjacency)
                raise CycleDetectedError(cycle)

            if node_id in visited:
                return

            temp_visited.add(node_id)

            for neighbor in adjacency[node_id]:
                visit(neighbor)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        for node_id in nodes:
            if node_id not in visited:
                visit(node_id)

        return list(reversed(result))

    def _priority_sort(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
    ) -> list[str]:
        """Priority-aware topological sort."""
        # Get basic topological order
        self._kahn_sort(nodes, edges)

        # Group by levels
        levels = self.get_levels(nodes, edges)

        # Sort each level by priority
        result: list[str] = []
        for level in levels:
            sorted_level = sorted(
                level.node_ids,
                key=lambda nid: (-nodes[nid].priority, nodes[nid].name),
            )
            result.extend(sorted_level)

        return result

    def get_levels(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
    ) -> list[ExecutionLevel]:
        """
        Get execution levels (nodes that can run in parallel).

        Args:
            nodes: Dictionary of nodes
            edges: Dictionary of edges

        Returns:
            List of execution levels
        """
        # Build in-degree
        in_degree: dict[str, int] = {nid: 0 for nid in nodes}
        adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}

        for edge in edges.values():
            if edge.source_id in nodes and edge.target_id in nodes:
                adjacency[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        levels: list[ExecutionLevel] = []
        remaining = set(nodes.keys())
        level_num = 0

        while remaining:
            # Find nodes with no remaining dependencies
            current_level = [
                nid for nid in remaining
                if in_degree[nid] == 0
            ]

            if not current_level:
                # Cycle detected
                cycle = self._find_cycle(remaining, adjacency)
                raise CycleDetectedError(cycle)

            levels.append(ExecutionLevel(level=level_num, node_ids=current_level))

            # Remove current level and update in-degrees
            for node_id in current_level:
                remaining.remove(node_id)
                for neighbor in adjacency[node_id]:
                    if neighbor in remaining:
                        in_degree[neighbor] -= 1

            level_num += 1

        return levels

    def _find_cycle(
        self,
        nodes: set[str],
        adjacency: dict[str, list[str]],
    ) -> list[str]:
        """Find a cycle in the graph."""
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited or node not in nodes:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                result = dfs(neighbor)
                if result:
                    return result

            path.pop()
            return None

        for node in nodes:
            result = dfs(node)
            if result:
                return result

        return list(nodes)[:3] + ["..."]


# =============================================================================
# GRAPH VISUALIZER
# =============================================================================


class GraphVisualizer:
    """
    Generates visualizations of the execution graph.

    Supports multiple output formats.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def visualize(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        format: VisualizationFormat = VisualizationFormat.MERMAID,
        title: str = "Execution Graph",
    ) -> str:
        """
        Generate visualization of the graph.

        Args:
            nodes: Dictionary of nodes
            edges: Dictionary of edges
            format: Output format
            title: Graph title

        Returns:
            Visualization string
        """
        if format == VisualizationFormat.MERMAID:
            return self._to_mermaid(nodes, edges, title)
        elif format == VisualizationFormat.DOT:
            return self._to_dot(nodes, edges, title)
        elif format == VisualizationFormat.ASCII:
            return self._to_ascii(nodes, edges, title)
        elif format == VisualizationFormat.JSON:
            return self._to_json(nodes, edges, title)
        else:
            return self._to_mermaid(nodes, edges, title)

    def _to_mermaid(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        title: str,
    ) -> str:
        """Generate Mermaid diagram."""
        lines = ["---", f"title: {title}", "---", "flowchart TD"]

        # State to style mapping
        state_styles = {
            NodeState.PENDING: ":::pending",
            NodeState.READY: ":::ready",
            NodeState.RUNNING: ":::running",
            NodeState.COMPLETED: ":::completed",
            NodeState.FAILED: ":::failed",
            NodeState.SKIPPED: ":::skipped",
            NodeState.CANCELLED: ":::cancelled",
            NodeState.ROLLED_BACK: ":::rolledback",
        }

        # Add nodes
        for node_id, node in nodes.items():
            safe_id = node_id.replace("-", "_")
            style = state_styles.get(node.state, "")
            lines.append(f"    {safe_id}[{node.name}]{style}")

        # Add edges
        for edge in edges.values():
            source = edge.source_id.replace("-", "_")
            target = edge.target_id.replace("-", "_")

            if edge.edge_type == EdgeType.SOFT_DEPENDENCY:
                lines.append(f"    {source} -.-> {target}")
            elif edge.edge_type == EdgeType.DATA_FLOW:
                label = edge.data_key or "data"
                lines.append(f"    {source} -->|{label}| {target}")
            elif edge.edge_type == EdgeType.CONDITIONAL:
                lines.append(f"    {source} -.->|condition| {target}")
            else:
                lines.append(f"    {source} --> {target}")

        # Add styles
        lines.extend([
            "",
            "    classDef pending fill:#gray,stroke:#333",
            "    classDef ready fill:#yellow,stroke:#333",
            "    classDef running fill:#blue,stroke:#333,color:#fff",
            "    classDef completed fill:#green,stroke:#333,color:#fff",
            "    classDef failed fill:#red,stroke:#333,color:#fff",
            "    classDef skipped fill:#orange,stroke:#333",
            "    classDef cancelled fill:#purple,stroke:#333,color:#fff",
            "    classDef rolledback fill:#pink,stroke:#333",
        ])

        return "\n".join(lines)

    def _to_dot(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        title: str,
    ) -> str:
        """Generate DOT (Graphviz) diagram."""
        lines = [
            f'digraph "{title}" {{',
            "    rankdir=TB;",
            "    node [shape=box, style=rounded];",
            "",
        ]

        # State to color mapping
        state_colors = {
            NodeState.PENDING: "gray",
            NodeState.READY: "yellow",
            NodeState.RUNNING: "blue",
            NodeState.COMPLETED: "green",
            NodeState.FAILED: "red",
            NodeState.SKIPPED: "orange",
            NodeState.CANCELLED: "purple",
            NodeState.ROLLED_BACK: "pink",
        }

        # Add nodes
        for node_id, node in nodes.items():
            safe_id = node_id.replace("-", "_")
            color = state_colors.get(node.state, "gray")
            lines.append(
                f'    {safe_id} [label="{node.name}", fillcolor={color}, style=filled];'
            )

        lines.append("")

        # Add edges
        for edge in edges.values():
            source = edge.source_id.replace("-", "_")
            target = edge.target_id.replace("-", "_")

            attrs = []
            if edge.edge_type == EdgeType.SOFT_DEPENDENCY:
                attrs.append("style=dashed")
            elif edge.edge_type == EdgeType.DATA_FLOW:
                attrs.append(f'label="{edge.data_key or "data"}"')
            elif edge.edge_type == EdgeType.CONDITIONAL:
                attrs.append("style=dotted")
                attrs.append('label="condition"')

            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(f"    {source} -> {target}{attr_str};")

        lines.append("}")

        return "\n".join(lines)

    def _to_ascii(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        title: str,
    ) -> str:
        """Generate ASCII representation."""
        lines = [
            "=" * 60,
            f" {title}",
            "=" * 60,
            "",
        ]

        # Get levels for layout
        sorter = TopologicalSorter()
        try:
            levels = sorter.get_levels(nodes, edges)
        except CycleDetectedError:
            levels = [ExecutionLevel(0, list(nodes.keys()))]

        # State symbols
        state_symbols = {
            NodeState.PENDING: "○",
            NodeState.READY: "◐",
            NodeState.RUNNING: "◑",
            NodeState.COMPLETED: "●",
            NodeState.FAILED: "✗",
            NodeState.SKIPPED: "⊘",
            NodeState.CANCELLED: "⊗",
            NodeState.ROLLED_BACK: "↺",
        }

        # Build adjacency for arrows
        adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
        for edge in edges.values():
            if edge.source_id in nodes and edge.target_id in nodes:
                adjacency[edge.source_id].append(edge.target_id)

        # Render levels
        for level in levels:
            lines.append(f"Level {level.level}:")

            for node_id in level.node_ids:
                node = nodes[node_id]
                symbol = state_symbols.get(node.state, "?")
                deps = adjacency.get(node_id, [])

                line = f"  {symbol} {node.name} [{node.state.name}]"
                if deps:
                    line += f" -> {', '.join(nodes[d].name for d in deps if d in nodes)}"

                lines.append(line)

            lines.append("")

        # Legend
        lines.extend([
            "-" * 60,
            "Legend:",
            "  ○ Pending  ◐ Ready  ◑ Running  ● Completed",
            "  ✗ Failed   ⊘ Skipped  ⊗ Cancelled  ↺ Rolled Back",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _to_json(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        title: str,
    ) -> str:
        """Generate JSON representation."""
        import json

        data = {
            "title": title,
            "nodes": [node.to_dict() for node in nodes.values()],
            "edges": [edge.to_dict() for edge in edges.values()],
        }

        return json.dumps(data, indent=2)


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass
class ExecutionContext:
    """
    Context passed during graph execution.

    Attributes:
        graph_id: Graph identifier
        execution_id: Execution run identifier
        data: Shared data between nodes
        results: Results from completed nodes
        start_time: Execution start time
        metadata: Additional context data
    """

    graph_id: str
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    data: dict[str, Any] = field(default_factory=dict)
    results: dict[str, NodeResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_result(self, node_id: str) -> NodeResult | None:
        """Get result for a node."""
        return self.results.get(node_id)

    def get_output(self, node_id: str) -> Any:
        """Get output from a node."""
        result = self.results.get(node_id)
        return result.output if result else None

    def set_data(self, key: str, value: Any) -> None:
        """Set shared data."""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get shared data."""
        return self.data.get(key, default)


# =============================================================================
# EXECUTION GRAPH
# =============================================================================


class ExecutionGraph:
    """
    Directed Acyclic Graph for execution planning.

    Manages nodes, edges, dependencies, and execution.
    """

    def __init__(
        self,
        graph_id: str | None = None,
        name: str = "Execution Graph",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize execution graph.

        Args:
            graph_id: Unique graph identifier
            name: Graph name
            logger: Optional logger
        """
        self.graph_id = graph_id or str(uuid4())
        self.name = name
        self._logger = logger or logging.getLogger(__name__)

        # Graph structure
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

        # Helpers
        self._sorter = TopologicalSorter(logger)
        self._visualizer = GraphVisualizer(logger)

        # Execution state
        self._execution_order: list[str] | None = None
        self._levels: list[ExecutionLevel] | None = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        name: str,
        action: Callable[..., Awaitable[Any]] | None = None,
        parameters: dict[str, Any] | None = None,
        priority: int = 0,
        timeout_seconds: float = 60.0,
        retries: int = 0,
        rollback_action: Callable[..., Awaitable[None]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        """
        Add a node to the graph.

        Args:
            node_id: Unique node identifier
            name: Human-readable name
            action: Async action to execute
            parameters: Action parameters
            priority: Execution priority
            timeout_seconds: Execution timeout
            retries: Maximum retries
            rollback_action: Rollback action
            metadata: Additional metadata

        Returns:
            Created node
        """
        node = Node(
            node_id=node_id,
            name=name,
            action=action,
            parameters=parameters or {},
            priority=priority,
            timeout_seconds=timeout_seconds,
            retries=retries,
            rollback_action=rollback_action,
            metadata=metadata or {},
        )

        self._nodes[node_id] = node
        self._invalidate_cache()

        self._logger.debug(f"Added node: {name} ({node_id})")
        return node

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and its edges.

        Args:
            node_id: Node to remove

        Raises:
            NodeNotFoundError: If node not found
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)

        # Remove associated edges
        edges_to_remove = [
            eid for eid, edge in self._edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]

        for edge_id in edges_to_remove:
            del self._edges[edge_id]

        del self._nodes[node_id]
        self._invalidate_cache()

        self._logger.debug(f"Removed node: {node_id}")

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes

    def list_nodes(
        self,
        state: NodeState | None = None,
    ) -> list[Node]:
        """List all nodes, optionally filtered by state."""
        nodes = list(self._nodes.values())
        if state:
            nodes = [n for n in nodes if n.state == state]
        return nodes

    # -------------------------------------------------------------------------
    # Edge Management
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEPENDENCY,
        condition: Callable[[Any], bool] | None = None,
        data_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """
        Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            condition: Condition for conditional edges
            data_key: Key for data flow
            metadata: Additional metadata

        Returns:
            Created edge

        Raises:
            NodeNotFoundError: If source or target not found
        """
        if source_id not in self._nodes:
            raise NodeNotFoundError(source_id)
        if target_id not in self._nodes:
            raise NodeNotFoundError(target_id)

        edge_id = f"{source_id}->{target_id}"

        edge = Edge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
            data_key=data_key,
            metadata=metadata or {},
        )

        self._edges[edge_id] = edge

        # Update node relationships
        self._nodes[source_id]._outgoing_edges.append(edge_id)
        self._nodes[target_id]._incoming_edges.append(edge_id)

        self._invalidate_cache()

        self._logger.debug(f"Added edge: {source_id} -> {target_id}")
        return edge

    def add_dependency(
        self,
        node_id: str,
        depends_on: str,
        soft: bool = False,
    ) -> Edge:
        """
        Add a dependency (convenience method).

        Args:
            node_id: Node that has the dependency
            depends_on: Node that must complete first
            soft: Whether dependency is optional

        Returns:
            Created edge
        """
        edge_type = EdgeType.SOFT_DEPENDENCY if soft else EdgeType.DEPENDENCY
        return self.add_edge(depends_on, node_id, edge_type=edge_type)

    def remove_edge(self, edge_id: str) -> None:
        """
        Remove an edge.

        Args:
            edge_id: Edge to remove

        Raises:
            EdgeNotFoundError: If edge not found
        """
        if edge_id not in self._edges:
            raise EdgeNotFoundError(edge_id)

        edge = self._edges[edge_id]

        # Update node relationships
        if edge.source_id in self._nodes:
            self._nodes[edge.source_id]._outgoing_edges.remove(edge_id)
        if edge.target_id in self._nodes:
            self._nodes[edge.target_id]._incoming_edges.remove(edge_id)

        del self._edges[edge_id]
        self._invalidate_cache()

    def get_edge(self, edge_id: str) -> Edge | None:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get IDs of nodes that this node depends on."""
        node = self._nodes.get(node_id)
        if not node:
            return []

        dependencies = []
        for edge_id in node._incoming_edges:
            edge = self._edges.get(edge_id)
            if edge:
                dependencies.append(edge.source_id)

        return dependencies

    def get_dependents(self, node_id: str) -> list[str]:
        """Get IDs of nodes that depend on this node."""
        node = self._nodes.get(node_id)
        if not node:
            return []

        dependents = []
        for edge_id in node._outgoing_edges:
            edge = self._edges.get(edge_id)
            if edge:
                dependents.append(edge.target_id)

        return dependents

    # -------------------------------------------------------------------------
    # Graph Analysis
    # -------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the graph.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        # Check for cycles
        try:
            self._sorter.sort(self._nodes, self._edges)
        except CycleDetectedError as e:
            errors.append(f"Cycle detected: {' -> '.join(e.cycle)}")

        # Check for orphan nodes (no action and no edges)
        for node_id, node in self._nodes.items():
            if (
                node.action is None
                and not node._incoming_edges
                and not node._outgoing_edges
            ):
                errors.append(f"Orphan node with no action: {node_id}")

        # Check edge references
        for edge_id, edge in self._edges.items():
            if edge.source_id not in self._nodes:
                errors.append(f"Edge {edge_id} references missing source: {edge.source_id}")
            if edge.target_id not in self._nodes:
                errors.append(f"Edge {edge_id} references missing target: {edge.target_id}")

        return errors

    def is_valid(self) -> bool:
        """Check if graph is valid."""
        return len(self.validate()) == 0

    def get_execution_order(
        self,
        strategy: SortStrategy = SortStrategy.KAHN,
    ) -> list[str]:
        """
        Get topological execution order.

        Args:
            strategy: Sorting strategy

        Returns:
            List of node IDs in execution order
        """
        if self._execution_order is None:
            self._execution_order = self._sorter.sort(
                self._nodes, self._edges, strategy
            )
        return list(self._execution_order)

    def get_levels(self) -> list[ExecutionLevel]:
        """
        Get execution levels for parallel execution.

        Returns:
            List of execution levels
        """
        if self._levels is None:
            self._levels = self._sorter.get_levels(self._nodes, self._edges)
        return list(self._levels)

    def get_stats(self) -> GraphStats:
        """Get graph statistics."""
        levels = self.get_levels()

        state_counts = {state: 0 for state in NodeState}
        for node in self._nodes.values():
            state_counts[node.state] += 1

        return GraphStats(
            total_nodes=len(self._nodes),
            total_edges=len(self._edges),
            depth=len(levels),
            width=max((len(level.node_ids) for level in levels), default=0),
            completed_nodes=state_counts[NodeState.COMPLETED],
            failed_nodes=state_counts[NodeState.FAILED],
            pending_nodes=state_counts[NodeState.PENDING],
        )

    def get_ready_nodes(self) -> list[str]:
        """Get nodes that are ready to execute."""
        ready = []

        for node_id, node in self._nodes.items():
            if node.state != NodeState.PENDING:
                continue

            # Check all dependencies are satisfied
            dependencies = self.get_dependencies(node_id)
            all_satisfied = all(
                self._nodes[dep_id].state == NodeState.COMPLETED
                for dep_id in dependencies
                if dep_id in self._nodes
            )

            if all_satisfied:
                ready.append(node_id)

        return ready

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._execution_order = None
        self._levels = None

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
        max_concurrency: int = 10,
        fail_fast: bool = False,
        context: ExecutionContext | None = None,
    ) -> ExecutionContext:
        """
        Execute the graph.

        Args:
            strategy: Execution strategy
            max_concurrency: Maximum concurrent executions
            fail_fast: Stop on first failure
            context: Execution context

        Returns:
            Execution context with results

        Raises:
            ExecutionError: If execution fails
        """
        # Validate graph
        errors = self.validate()
        if errors:
            raise ExecutionError(f"Invalid graph: {errors[0]}")

        # Create context
        context = context or ExecutionContext(graph_id=self.graph_id)

        self._logger.info(f"Executing graph: {self.name} ({strategy.name})")

        try:
            if strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(context, max_concurrency, fail_fast)
            elif strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(context, fail_fast)
            elif strategy == ExecutionStrategy.BREADTH_FIRST:
                await self._execute_breadth_first(context, max_concurrency, fail_fast)
            elif strategy == ExecutionStrategy.PRIORITY:
                await self._execute_priority(context, max_concurrency, fail_fast)

        except Exception as e:
            self._logger.error(f"Graph execution failed: {e}")
            raise

        return context

    async def _execute_parallel(
        self,
        context: ExecutionContext,
        max_concurrency: int,
        fail_fast: bool,
    ) -> None:
        """Execute with parallel strategy."""
        semaphore = asyncio.Semaphore(max_concurrency)
        failed = False

        while True:
            if failed and fail_fast:
                break

            ready_nodes = self.get_ready_nodes()
            if not ready_nodes:
                # Check if all done or stuck
                pending = [n for n in self._nodes.values() if n.state == NodeState.PENDING]
                if not pending:
                    break
                # Nodes are stuck (dependencies failed)
                for node in pending:
                    node.state = NodeState.SKIPPED
                break

            # Execute ready nodes in parallel
            tasks = []
            for node_id in ready_nodes:
                task = asyncio.create_task(
                    self._execute_node_with_semaphore(node_id, context, semaphore)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    failed = True
                    if fail_fast:
                        break

    async def _execute_sequential(
        self,
        context: ExecutionContext,
        fail_fast: bool,
    ) -> None:
        """Execute sequentially."""
        order = self.get_execution_order()

        for node_id in order:
            node = self._nodes[node_id]

            # Check dependencies
            deps = self.get_dependencies(node_id)
            deps_satisfied = all(
                self._nodes[d].state == NodeState.COMPLETED
                for d in deps if d in self._nodes
            )

            if not deps_satisfied:
                node.state = NodeState.SKIPPED
                continue

            try:
                await self._execute_node(node_id, context)
            except Exception:
                if fail_fast:
                    raise

    async def _execute_breadth_first(
        self,
        context: ExecutionContext,
        max_concurrency: int,
        fail_fast: bool,
    ) -> None:
        """Execute level by level."""
        levels = self.get_levels()
        semaphore = asyncio.Semaphore(max_concurrency)

        for level in levels:
            tasks = []

            for node_id in level.node_ids:
                node = self._nodes[node_id]

                # Check dependencies
                deps = self.get_dependencies(node_id)
                deps_satisfied = all(
                    self._nodes[d].state == NodeState.COMPLETED
                    for d in deps if d in self._nodes
                )

                if not deps_satisfied:
                    node.state = NodeState.SKIPPED
                    continue

                task = asyncio.create_task(
                    self._execute_node_with_semaphore(node_id, context, semaphore)
                )
                tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                if fail_fast:
                    for result in results:
                        if isinstance(result, Exception):
                            raise result

    async def _execute_priority(
        self,
        context: ExecutionContext,
        max_concurrency: int,
        fail_fast: bool,
    ) -> None:
        """Execute with priority ordering."""
        order = self._sorter.sort(self._nodes, self._edges, SortStrategy.PRIORITY)
        semaphore = asyncio.Semaphore(max_concurrency)

        # Group by priority for parallel execution within priority
        priority_groups: dict[int, list[str]] = {}
        for node_id in order:
            priority = self._nodes[node_id].priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(node_id)

        for priority in sorted(priority_groups.keys(), reverse=True):
            node_ids = priority_groups[priority]
            tasks = []

            for node_id in node_ids:
                node = self._nodes[node_id]

                deps = self.get_dependencies(node_id)
                deps_satisfied = all(
                    self._nodes[d].state == NodeState.COMPLETED
                    for d in deps if d in self._nodes
                )

                if not deps_satisfied:
                    node.state = NodeState.SKIPPED
                    continue

                task = asyncio.create_task(
                    self._execute_node_with_semaphore(node_id, context, semaphore)
                )
                tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                if fail_fast:
                    for result in results:
                        if isinstance(result, Exception):
                            raise result

    async def _execute_node_with_semaphore(
        self,
        node_id: str,
        context: ExecutionContext,
        semaphore: asyncio.Semaphore,
    ) -> NodeResult:
        """Execute node with semaphore for concurrency control."""
        async with semaphore:
            return await self._execute_node(node_id, context)

    async def _execute_node(
        self,
        node_id: str,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute a single node."""
        node = self._nodes[node_id]
        node.state = NodeState.RUNNING

        start_time = datetime.now(timezone.utc)
        start_mono = time.monotonic()

        self._logger.debug(f"Executing node: {node.name}")

        retries = 0
        last_error: Exception | None = None

        while retries <= node.retries:
            try:
                if node.action:
                    # Prepare parameters with context data
                    params = dict(node.parameters)
                    params["_context"] = context

                    # Execute with timeout
                    output = await asyncio.wait_for(
                        node.action(**params),
                        timeout=node.timeout_seconds,
                    )
                else:
                    output = None

                # Success
                end_time = datetime.now(timezone.utc)
                duration_ms = (time.monotonic() - start_mono) * 1000

                result = NodeResult(
                    node_id=node_id,
                    success=True,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                )

                node.state = NodeState.COMPLETED
                node.result = result
                context.results[node_id] = result

                self._logger.debug(
                    f"Node completed: {node.name} ({duration_ms:.1f}ms)"
                )

                return result

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Node {node_id} timed out after {node.timeout_seconds}s"
                )
                retries += 1

            except Exception as e:
                last_error = e
                retries += 1

            if retries <= node.retries:
                self._logger.warning(
                    f"Node {node.name} failed, retry {retries}/{node.retries}"
                )
                await asyncio.sleep(0.5 * retries)  # Backoff

        # All retries exhausted
        end_time = datetime.now(timezone.utc)
        duration_ms = (time.monotonic() - start_mono) * 1000

        result = NodeResult(
            node_id=node_id,
            success=False,
            error=str(last_error),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
        )

        node.state = NodeState.FAILED
        node.result = result
        context.results[node_id] = result

        self._logger.error(f"Node failed: {node.name} - {last_error}")

        return result

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    async def rollback(
        self,
        context: ExecutionContext,
        strategy: RollbackStrategy = RollbackStrategy.REVERSE_ORDER,
        node_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Rollback executed nodes.

        Args:
            context: Execution context
            strategy: Rollback strategy
            node_ids: Specific nodes to rollback (None = all completed)

        Returns:
            List of failed rollback node IDs

        Raises:
            RollbackError: If rollback fails critically
        """
        self._logger.info(f"Rolling back graph: {self.name} ({strategy.name})")

        # Determine nodes to rollback
        if node_ids:
            to_rollback = node_ids
        else:
            to_rollback = [
                nid for nid, node in self._nodes.items()
                if node.state == NodeState.COMPLETED and node.rollback_action
            ]

        if not to_rollback:
            return []

        failed: list[str] = []

        if strategy == RollbackStrategy.REVERSE_ORDER:
            # Get execution order and reverse it
            order = self.get_execution_order()
            rollback_order = [nid for nid in reversed(order) if nid in to_rollback]

            for node_id in rollback_order:
                success = await self._rollback_node(node_id, context)
                if not success:
                    failed.append(node_id)

        elif strategy == RollbackStrategy.PARALLEL:
            tasks = [
                self._rollback_node(node_id, context)
                for node_id in to_rollback
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node_id, result in zip(to_rollback, results):
                if isinstance(result, Exception) or result is False:
                    failed.append(node_id)

        elif strategy == RollbackStrategy.SELECTIVE:
            # Only rollback nodes affected by failures
            failed_nodes = [
                nid for nid, node in self._nodes.items()
                if node.state == NodeState.FAILED
            ]

            affected = set()
            for failed_id in failed_nodes:
                affected.update(self._get_affected_nodes(failed_id))

            selective_rollback = [nid for nid in to_rollback if nid in affected]

            for node_id in selective_rollback:
                success = await self._rollback_node(node_id, context)
                if not success:
                    failed.append(node_id)

        if failed:
            self._logger.warning(f"Rollback failed for nodes: {failed}")

        return failed

    async def _rollback_node(
        self,
        node_id: str,
        context: ExecutionContext,
    ) -> bool:
        """Rollback a single node."""
        node = self._nodes.get(node_id)
        if not node or not node.rollback_action:
            return True

        self._logger.debug(f"Rolling back node: {node.name}")

        try:
            await asyncio.wait_for(
                node.rollback_action(_context=context),
                timeout=node.timeout_seconds,
            )
            node.state = NodeState.ROLLED_BACK
            self._logger.debug(f"Rolled back node: {node.name}")
            return True

        except Exception as e:
            self._logger.error(f"Rollback failed for {node.name}: {e}")
            return False

    def _get_affected_nodes(self, node_id: str) -> set[str]:
        """Get all nodes affected by a node (dependents recursively)."""
        affected: set[str] = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in affected:
                continue

            affected.add(current)
            dependents = self.get_dependents(current)
            queue.extend(dependents)

        return affected

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all nodes to pending state."""
        for node in self._nodes.values():
            node.state = NodeState.PENDING
            node.result = None

        self._logger.debug("Graph reset to pending state")

    def reset_failed(self) -> None:
        """Reset only failed and skipped nodes."""
        for node in self._nodes.values():
            if node.state in {NodeState.FAILED, NodeState.SKIPPED}:
                node.state = NodeState.PENDING
                node.result = None

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def visualize(
        self,
        format: VisualizationFormat = VisualizationFormat.MERMAID,
    ) -> str:
        """
        Generate visualization of the graph.

        Args:
            format: Output format

        Returns:
            Visualization string
        """
        return self._visualizer.visualize(
            self._nodes, self._edges, format, self.name
        )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "stats": self.get_stats().to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        actions: dict[str, Callable[..., Awaitable[Any]]] | None = None,
    ) -> ExecutionGraph:
        """
        Create graph from dictionary.

        Args:
            data: Graph data
            actions: Mapping of node IDs to action functions

        Returns:
            ExecutionGraph instance
        """
        actions = actions or {}

        graph = cls(
            graph_id=data.get("graph_id"),
            name=data.get("name", "Execution Graph"),
        )

        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data["node_id"]
            graph.add_node(
                node_id=node_id,
                name=node_data["name"],
                action=actions.get(node_id),
                parameters=node_data.get("parameters", {}),
                priority=node_data.get("priority", 0),
                timeout_seconds=node_data.get("timeout_seconds", 60.0),
                retries=node_data.get("retries", 0),
                metadata=node_data.get("metadata", {}),
            )

        # Add edges
        for edge_data in data.get("edges", []):
            graph.add_edge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=EdgeType(edge_data.get("edge_type", "dependency")),
                data_key=edge_data.get("data_key"),
                metadata=edge_data.get("metadata", {}),
            )

        return graph


# =============================================================================
# GRAPH BUILDER
# =============================================================================


class GraphBuilder:
    """
    Fluent builder for constructing execution graphs.

    Provides a convenient API for building complex graphs.
    """

    def __init__(
        self,
        name: str = "Execution Graph",
        graph_id: str | None = None,
    ) -> None:
        """
        Initialize builder.

        Args:
            name: Graph name
            graph_id: Optional graph ID
        """
        self._graph = ExecutionGraph(graph_id=graph_id, name=name)
        self._current_node: str | None = None

    def node(
        self,
        node_id: str,
        name: str | None = None,
        action: Callable[..., Awaitable[Any]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """
        Add a node.

        Args:
            node_id: Node identifier
            name: Node name (defaults to node_id)
            action: Node action
            **kwargs: Additional node parameters

        Returns:
            Self for chaining
        """
        self._graph.add_node(
            node_id=node_id,
            name=name or node_id,
            action=action,
            **kwargs,
        )
        self._current_node = node_id
        return self

    def depends_on(self, *node_ids: str, soft: bool = False) -> GraphBuilder:
        """
        Add dependencies for current node.

        Args:
            *node_ids: Nodes to depend on
            soft: Whether dependencies are soft

        Returns:
            Self for chaining
        """
        if not self._current_node:
            raise GraphError("No current node - call node() first")

        for dep_id in node_ids:
            self._graph.add_dependency(self._current_node, dep_id, soft=soft)

        return self

    def then(
        self,
        node_id: str,
        name: str | None = None,
        action: Callable[..., Awaitable[Any]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """
        Add a node that depends on the current node.

        Args:
            node_id: Node identifier
            name: Node name
            action: Node action
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        previous = self._current_node

        self.node(node_id, name, action, **kwargs)

        if previous:
            self.depends_on(previous)

        return self

    def parallel(
        self,
        *nodes: tuple[str, str | None, Callable[..., Awaitable[Any]] | None],
    ) -> GraphBuilder:
        """
        Add multiple nodes that can run in parallel.

        Args:
            *nodes: Tuples of (node_id, name, action)

        Returns:
            Self for chaining
        """
        previous = self._current_node

        for node_spec in nodes:
            node_id = node_spec[0]
            name = node_spec[1] if len(node_spec) > 1 else None
            action = node_spec[2] if len(node_spec) > 2 else None

            self._graph.add_node(node_id, name or node_id, action)

            if previous:
                self._graph.add_dependency(node_id, previous)

        # Set current to None since we have multiple parallel nodes
        self._current_node = None

        return self

    def join(
        self,
        node_id: str,
        name: str | None = None,
        action: Callable[..., Awaitable[Any]] | None = None,
        from_nodes: list[str] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """
        Add a node that joins multiple parallel nodes.

        Args:
            node_id: Node identifier
            name: Node name
            action: Node action
            from_nodes: Nodes to join (None = all leaf nodes)
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        self._graph.add_node(node_id, name or node_id, action, **kwargs)

        if from_nodes:
            for dep_id in from_nodes:
                self._graph.add_dependency(node_id, dep_id)
        else:
            # Find all leaf nodes (no outgoing edges)
            for nid, node in self._graph._nodes.items():
                if nid != node_id and not node._outgoing_edges:
                    self._graph.add_dependency(node_id, nid)

        self._current_node = node_id
        return self

    def edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEPENDENCY,
        **kwargs: Any,
    ) -> GraphBuilder:
        """
        Add an edge directly.

        Args:
            source_id: Source node
            target_id: Target node
            edge_type: Edge type
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        self._graph.add_edge(source_id, target_id, edge_type, **kwargs)
        return self

    def with_rollback(
        self,
        rollback_action: Callable[..., Awaitable[None]],
    ) -> GraphBuilder:
        """
        Add rollback action to current node.

        Args:
            rollback_action: Rollback function

        Returns:
            Self for chaining
        """
        if not self._current_node:
            raise GraphError("No current node - call node() first")

        node = self._graph.get_node(self._current_node)
        if node:
            node.rollback_action = rollback_action

        return self

    def build(self) -> ExecutionGraph:
        """
        Build and return the graph.

        Returns:
            Constructed ExecutionGraph

        Raises:
            GraphError: If graph is invalid
        """
        errors = self._graph.validate()
        if errors:
            raise GraphError(f"Invalid graph: {'; '.join(errors)}")

        return self._graph


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_graph(
    name: str = "Execution Graph",
    graph_id: str | None = None,
    logger: logging.Logger | None = None,
) -> ExecutionGraph:
    """
    Create a new execution graph.

    Args:
        name: Graph name
        graph_id: Optional graph ID
        logger: Optional logger

    Returns:
        New ExecutionGraph
    """
    return ExecutionGraph(graph_id=graph_id, name=name, logger=logger)


def create_builder(
    name: str = "Execution Graph",
    graph_id: str | None = None,
) -> GraphBuilder:
    """
    Create a new graph builder.

    Args:
        name: Graph name
        graph_id: Optional graph ID

    Returns:
        New GraphBuilder
    """
    return GraphBuilder(name=name, graph_id=graph_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "NodeState",
    "EdgeType",
    "ExecutionStrategy",
    "SortStrategy",
    "RollbackStrategy",
    "VisualizationFormat",
    # Data Structures
    "NodeResult",
    "Edge",
    "Node",
    "ExecutionLevel",
    "GraphStats",
    "ExecutionContext",
    # Exceptions
    "GraphError",
    "CycleDetectedError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "ExecutionError",
    "RollbackError",
    # Components
    "TopologicalSorter",
    "GraphVisualizer",
    # Main Classes
    "ExecutionGraph",
    "GraphBuilder",
    # Factory Functions
    "create_graph",
    "create_builder",
]
