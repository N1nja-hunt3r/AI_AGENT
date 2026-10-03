"""
Models - Core data structures for the AI Agent Platform.

This module contains all enums, data models, and schemas used throughout
the agent controller system. Models are organized into logical sections:
- Enums: Type classifications and status indicators
- Request/Response: API boundary models
- Analysis: Intent and routing models
- Planning: Execution plan models
- State: Session and persistence models
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# =============================================================================
# ENUMS - Type Classifications
# =============================================================================


class IntentType(Enum):
    """
    Classification of user intent types.

    Used by IntentAnalyzer to categorize the primary purpose
    of a user's request.

    Attributes:
        QUERY: Information retrieval requests
        ACTION: Requests to execute operations or tasks
        CONVERSATION: General dialogue and chat
        ANALYSIS: Deep analysis or reasoning tasks
        CREATION: Content or artifact generation
        UNKNOWN: Unclassifiable intent
    """

    QUERY = "query"
    ACTION = "action"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    CREATION = "creation"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> IntentType:
        """
        Safely convert string to IntentType.

        Args:
            value: String representation of intent type

        Returns:
            Matching IntentType or UNKNOWN if not found
        """
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class CapabilityType(Enum):
    """
    Available capability types in the agent system.

    Each capability represents a distinct functional module
    that can be invoked during request processing.

    Attributes:
        MEMORY: Short and long-term memory operations
        TOOLS: External tool and function execution
        RAG: Retrieval-augmented generation for knowledge
        WEB_SEARCH: Real-time web search capability
        CHAT: Conversational chat model
        CODING: Code generation and analysis
        VISION: Image understanding and processing
        FILES: File system operations
        COMPUTER: Computer interaction (screen, mouse, keyboard)
    """

    MEMORY = "memory"
    TOOLS = "tools"
    RAG = "rag"
    WEB_SEARCH = "web_search"
    CHAT = "chat"
    CODING = "coding"
    VISION = "vision"
    FILES = "files"
    COMPUTER = "computer"

    @property
    def display_name(self) -> str:
        """Human-readable name for the capability."""
        names = {
            self.MEMORY: "Memory",
            self.TOOLS: "Tools",
            self.RAG: "Knowledge Base",
            self.WEB_SEARCH: "Web Search",
            self.CHAT: "Chat",
            self.CODING: "Coding",
            self.VISION: "Vision",
            self.FILES: "Files",
            self.COMPUTER: "Computer",
        }
        return names.get(self, self.value.replace("_", " ").title())


# =============================================================================
# ENUMS - Status Indicators
# =============================================================================


class PlanStepStatus(Enum):
    """
    Execution status of an individual plan step.

    Tracks the lifecycle of a step from creation through completion.

    Attributes:
        PENDING: Step has not started execution
        RUNNING: Step is currently executing
        COMPLETED: Step finished successfully
        FAILED: Step encountered an error
        SKIPPED: Step was skipped (fallback or dependency failure)
    """

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

    @property
    def is_terminal(self) -> bool:
        """Check if this status represents a final state."""
        return self in {self.COMPLETED, self.FAILED, self.SKIPPED}

    @property
    def is_success(self) -> bool:
        """Check if this status represents successful completion."""
        return self in {self.COMPLETED, self.SKIPPED}


class PlanStatus(Enum):
    """
    Overall execution status of an execution plan.

    Tracks the lifecycle of the entire plan.

    Attributes:
        CREATED: Plan has been generated but not started
        EXECUTING: Plan is currently being executed
        COMPLETED: All steps finished successfully
        FAILED: Plan execution failed
        REPLANNING: Plan is being regenerated after failure
    """

    CREATED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REPLANNING = auto()

    @property
    def is_terminal(self) -> bool:
        """Check if this status represents a final state."""
        return self in {self.COMPLETED, self.FAILED}

    @property
    def allows_execution(self) -> bool:
        """Check if plan can be executed in this state."""
        return self in {self.CREATED, self.REPLANNING}


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """
    Immutable incoming request to the agent controller.

    This is the primary input model for all agent interactions.
    Immutability ensures request integrity throughout processing.

    Attributes:
        request_id: Unique identifier for this request (UUID)
        session_id: Session identifier for state continuity (UUID)
        content: The user's input content/message
        metadata: Additional request metadata (user_id, timestamp, etc.)

    Example:
        >>> request = AgentRequest.create(
        ...     content="What is the weather today?",
        ...     metadata={"user_id": "user_123"}
        ... )
    """

    request_id: str
    session_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request fields after initialization."""
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")

    @classmethod
    def create(
        cls,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRequest:
        """
        Factory method to create a new request with generated IDs.

        Args:
            content: The user's input content
            session_id: Optional session ID (generated if not provided)
            metadata: Optional request metadata

        Returns:
            New AgentRequest instance
        """
        return cls(
            request_id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
        )

    @property
    def user_id(self) -> str | None:
        """Extract user_id from metadata if present."""
        return self.metadata.get("user_id")

    @property
    def timestamp(self) -> datetime | None:
        """Extract timestamp from metadata if present."""
        ts = self.metadata.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return None
        return None


@dataclass(slots=True)
class AgentResponse:
    """
    Response from the agent controller.

    This is the primary output model for all agent interactions.

    Attributes:
        request_id: Reference to the original request
        session_id: Session identifier for continuity
        content: The response content/message
        success: Whether the request was handled successfully
        metadata: Response metadata (timing, capabilities used, etc.)
        error: Error information if unsuccessful

    Example:
        >>> response = AgentResponse(
        ...     request_id="abc-123",
        ...     session_id="session-456",
        ...     content="The weather today is sunny.",
        ...     success=True,
        ...     metadata={"execution_time_ms": 150}
        ... )
    """

    request_id: str
    session_id: str
    content: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate response consistency."""
        if not self.success and not self.error:
            self.error = "Unknown error"

    @classmethod
    def error_response(
        cls,
        request: AgentRequest,
        error_message: str,
        content: str | None = None,
    ) -> AgentResponse:
        """
        Factory method to create an error response.

        Args:
            request: The original request
            error_message: Description of the error
            content: Optional user-facing content

        Returns:
            AgentResponse indicating failure
        """
        return cls(
            request_id=request.request_id,
            session_id=request.session_id,
            content=content or "An error occurred while processing your request.",
            success=False,
            error=error_message,
        )

    @property
    def execution_time_ms(self) -> float:
        """Extract execution time from metadata."""
        return float(self.metadata.get("total_execution_time_ms", 0))

    @property
    def capabilities_used(self) -> list[str]:
        """Extract list of capabilities used from metadata."""
        return list(self.metadata.get("capabilities_used", []))


# =============================================================================
# ANALYSIS MODELS
# =============================================================================


@dataclass(slots=True)
class IntentAnalysis:
    """
    Result of analyzing user intent.

    Produced by IntentAnalyzer to guide capability routing
    and execution planning.

    Attributes:
        intent_type: Classified type of intent
        confidence: Confidence score (0.0 to 1.0)
        entities: Extracted entities from the request
        complexity_score: Estimated task complexity (1-10)
        requires_context: Whether historical context is needed
        requires_current_info: Whether real-time information is needed

    Example:
        >>> analysis = IntentAnalysis(
        ...     intent_type=IntentType.QUERY,
        ...     confidence=0.95,
        ...     entities={"topic": "weather", "location": "New York"},
        ...     complexity_score=3,
        ...     requires_current_info=True
        ... )
    """

    intent_type: IntentType
    confidence: float
    entities: dict[str, Any]
    complexity_score: int
    requires_context: bool = False
    requires_current_info: bool = False

    def __post_init__(self) -> None:
        """Validate analysis values."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.complexity_score = max(1, min(10, self.complexity_score))

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence exceeds threshold (0.8)."""
        return self.confidence >= 0.8

    @property
    def is_complex(self) -> bool:
        """Check if task is considered complex (score >= 7)."""
        return self.complexity_score >= 7

    @property
    def is_simple(self) -> bool:
        """Check if task is considered simple (score <= 3)."""
        return self.complexity_score <= 3

    def has_entity(self, key: str) -> bool:
        """Check if a specific entity was extracted."""
        return key in self.entities

    def get_entity(self, key: str, default: Any = None) -> Any:
        """Get an entity value with optional default."""
        return self.entities.get(key, default)


@dataclass(slots=True)
class CapabilityDecision:
    """
    Routing decision for which capabilities to invoke.

    Produced by CapabilityRouter based on intent analysis
    and session context.

    Attributes:
        use_memory: Whether to query/update memory
        use_rag: Whether to use RAG for knowledge retrieval
        use_web_search: Whether to search the web
        use_chat: Whether to use conversational chat model
        use_coding: Whether to use code generation model
        use_vision: Whether to use image understanding
        use_files: Whether to use file operations
        use_computer: Whether to use computer interaction
        tools_to_use: List of specific tools to invoke
        reasoning: Explanation of routing decisions
    """

    use_memory: bool = False
    use_rag: bool = False
    use_web_search: bool = False
    use_chat: bool = True
    use_coding: bool = False
    use_vision: bool = False
    use_files: bool = False
    use_computer: bool = False
    tools_to_use: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def active_capabilities(self) -> list[CapabilityType]:
        """Return list of capabilities that will be used."""
        capabilities: list[CapabilityType] = []
        if self.use_memory:
            capabilities.append(CapabilityType.MEMORY)
        if self.use_rag:
            capabilities.append(CapabilityType.RAG)
        if self.use_web_search:
            capabilities.append(CapabilityType.WEB_SEARCH)
        if self.use_chat:
            capabilities.append(CapabilityType.CHAT)
        if self.use_coding:
            capabilities.append(CapabilityType.CODING)
        if self.use_vision:
            capabilities.append(CapabilityType.VISION)
        if self.use_files:
            capabilities.append(CapabilityType.FILES)
        if self.use_computer:
            capabilities.append(CapabilityType.COMPUTER)
        if self.tools_to_use:
            capabilities.append(CapabilityType.TOOLS)
        return capabilities

    @property
    def capability_count(self) -> int:
        """Return number of active capabilities."""
        return len(self.active_capabilities)

    @property
    def requires_external_calls(self) -> bool:
        """Check if decision requires external API calls."""
        return self.use_web_search or bool(self.tools_to_use)

    @property
    def is_direct_response(self) -> bool:
        """Check if no capabilities are needed (direct LLM response)."""
        return self.capability_count == 0


# =============================================================================
# PLANNING MODELS
# =============================================================================


@dataclass(slots=True)
class StepResult:
    """
    Result from executing a plan step.

    Captures the outcome of a single step execution including
    output data, timing, and error information.

    Attributes:
        step_id: Reference to the executed step
        success: Whether execution succeeded
        output: Output data from the step
        error: Error information if failed
        execution_time_ms: Time taken to execute
        metadata: Additional execution metadata

    Example:
        >>> result = StepResult(
        ...     step_id="step_1",
        ...     success=True,
        ...     output={"documents": [...]},
        ...     execution_time_ms=45.2
        ... )
    """

    step_id: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        step_id: str,
        output: Any,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        """Factory for successful step result."""
        return cls(
            step_id=step_id,
            success=True,
            output=output,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        step_id: str,
        error: str,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        """Factory for failed step result."""
        return cls(
            step_id=step_id,
            success=False,
            output=None,
            error=error,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def skipped_result(
        cls,
        step_id: str,
        reason: str,
    ) -> StepResult:
        """Factory for skipped step result."""
        return cls(
            step_id=step_id,
            success=True,
            output=None,
            metadata={"skipped": True, "reason": reason},
        )

    @property
    def was_skipped(self) -> bool:
        """Check if this step was skipped."""
        return self.metadata.get("skipped", False)


@dataclass(slots=True)
class PlanStep:
    """
    Single step in an execution plan.

    Represents an atomic unit of work to be executed by a capability.

    Attributes:
        step_id: Unique identifier for this step
        capability: Which capability this step uses
        action: Specific action to perform
        parameters: Parameters for the action
        dependencies: Step IDs that must complete before this step
        fallback_action: Alternative action if primary fails
        timeout_seconds: Maximum execution time
        status: Current execution status
        result: Execution result (populated after completion)

    Example:
        >>> step = PlanStep(
        ...     step_id="step_1",
        ...     capability=CapabilityType.RAG,
        ...     action="search",
        ...     parameters={"query": "machine learning basics"},
        ...     timeout_seconds=30
        ... )
    """

    step_id: str
    capability: CapabilityType
    action: str
    parameters: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    fallback_action: str | None = None
    timeout_seconds: int = 30
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: StepResult | None = None

    def __post_init__(self) -> None:
        """Validate step configuration."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def has_dependencies(self) -> bool:
        """Check if this step has dependencies."""
        return bool(self.dependencies)

    @property
    def has_fallback(self) -> bool:
        """Check if this step has a fallback action."""
        return self.fallback_action is not None

    @property
    def is_complete(self) -> bool:
        """Check if step has finished execution."""
        return self.status.is_terminal

    @property
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.status.is_success

    def depends_on(self, step_id: str) -> bool:
        """Check if this step depends on another step."""
        return step_id in self.dependencies

    def mark_running(self) -> None:
        """Mark step as currently executing."""
        self.status = PlanStepStatus.RUNNING

    def mark_completed(self, result: StepResult) -> None:
        """Mark step as completed with result."""
        self.status = PlanStepStatus.COMPLETED
        self.result = result

    def mark_failed(self, result: StepResult) -> None:
        """Mark step as failed with result."""
        self.status = PlanStepStatus.FAILED
        self.result = result

    def mark_skipped(self, reason: str) -> None:
        """Mark step as skipped."""
        self.status = PlanStepStatus.SKIPPED
        self.result = StepResult.skipped_result(self.step_id, reason)


@dataclass(slots=True)
class ExecutionPlan:
    """
    Complete execution plan for a request.

    Contains an ordered sequence of steps with dependencies,
    retry policies, and execution state.

    Attributes:
        plan_id: Unique identifier for this plan
        steps: Ordered list of execution steps
        status: Current plan status
        allows_replan: Whether replanning is permitted on failure
        max_retries: Maximum retry attempts per step
        created_at: Plan creation timestamp

    Example:
        >>> plan = ExecutionPlan(
        ...     plan_id="plan-abc-123",
        ...     steps=[step1, step2, step3],
        ...     allows_replan=True
        ... )
    """

    plan_id: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.CREATED
    allows_replan: bool = True
    max_retries: int = 2
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate plan configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    @classmethod
    def create(
        cls,
        steps: list[PlanStep],
        allows_replan: bool = True,
        max_retries: int = 2,
    ) -> ExecutionPlan:
        """Factory method to create a new plan with generated ID."""
        return cls(
            plan_id=str(uuid.uuid4()),
            steps=steps,
            allows_replan=allows_replan,
            max_retries=max_retries,
        )

    @property
    def step_count(self) -> int:
        """Return total number of steps."""
        return len(self.steps)

    @property
    def completed_steps(self) -> list[PlanStep]:
        """Return list of completed steps."""
        return [s for s in self.steps if s.status == PlanStepStatus.COMPLETED]

    @property
    def failed_steps(self) -> list[PlanStep]:
        """Return list of failed steps."""
        return [s for s in self.steps if s.status == PlanStepStatus.FAILED]

    @property
    def pending_steps(self) -> list[PlanStep]:
        """Return list of pending steps."""
        return [s for s in self.steps if s.status == PlanStepStatus.PENDING]

    @property
    def progress_percentage(self) -> float:
        """Calculate execution progress as percentage."""
        if not self.steps:
            return 100.0
        completed = sum(1 for s in self.steps if s.is_complete)
        return (completed / len(self.steps)) * 100

    @property
    def is_executable(self) -> bool:
        """Check if plan can be executed."""
        return self.status.allows_execution

    def ordered_steps(self) -> list[PlanStep]:
        """
        Return steps in dependency-respecting execution order.

        Uses topological sort to ensure dependencies are satisfied.

        Returns:
            List of steps in execution order

        Raises:
            ValueError: If circular dependency is detected
        """
        executed: set[str] = set()
        ordered: list[PlanStep] = []
        remaining = list(self.steps)

        while remaining:
            ready = [
                step
                for step in remaining
                if all(dep in executed for dep in step.dependencies)
            ]

            if not ready and remaining:
                cycle_steps = [s.step_id for s in remaining]
                raise ValueError(
                    f"Circular dependency detected in plan. "
                    f"Affected steps: {cycle_steps}"
                )

            for step in ready:
                ordered.append(step)
                executed.add(step.step_id)
                remaining.remove(step)

        return ordered

    def get_step(self, step_id: str) -> PlanStep | None:
        """Retrieve a step by its ID."""
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_step_results(self) -> list[StepResult]:
        """Get all available step results."""
        return [s.result for s in self.steps if s.result is not None]

    def mark_executing(self) -> None:
        """Mark plan as currently executing."""
        self.status = PlanStatus.EXECUTING

    def mark_completed(self) -> None:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED

    def mark_failed(self) -> None:
        """Mark plan as failed."""
        self.status = PlanStatus.FAILED

    def mark_replanning(self) -> None:
        """Mark plan as undergoing replanning."""
        self.status = PlanStatus.REPLANNING


# =============================================================================
# STATE MODELS
# =============================================================================


@dataclass(slots=True)
class SessionState:
    """
    Persistent state for an agent session.

    Maintains conversation history, memory references, and
    execution state across multiple requests.

    Attributes:
        session_id: Unique session identifier
        history: Conversation history (request/response pairs)
        memory_refs: References to stored memories
        active_plan: Currently executing plan
        metadata: Session metadata (user preferences, constraints)
        created_at: Session creation time
        last_active: Last activity timestamp

    Example:
        >>> state = SessionState(session_id="session-123")
        >>> state.add_to_history(request, response)
    """

    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    active_plan: ExecutionPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    MAX_HISTORY_SIZE: int = 100  # Class constant for history limit

    @classmethod
    def create(
        cls,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        """Factory method to create a new session."""
        return cls(
            session_id=session_id or str(uuid.uuid4()),
            metadata=metadata or {},
        )

    @property
    def history_size(self) -> int:
        """Return number of history entries."""
        return len(self.history)

    @property
    def has_history(self) -> bool:
        """Check if session has conversation history."""
        return bool(self.history)

    @property
    def has_active_plan(self) -> bool:
        """Check if session has an active execution plan."""
        return self.active_plan is not None

    @property
    def session_duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        return (self.last_active - self.created_at).total_seconds()

    def add_to_history(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> None:
        """
        Append a request-response pair to history.

        Automatically manages history size and updates last_active.

        Args:
            request: The user's request
            response: The agent's response
        """
        entry = {
            "request_id": request.request_id,
            "content": request.content,
            "response": response.content,
            "success": response.success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)
        self._trim_history()
        self.touch()

    def get_recent_history(self, count: int = 5) -> list[dict[str, Any]]:
        """
        Get the most recent history entries.

        Args:
            count: Number of entries to return

        Returns:
            List of recent history entries
        """
        return self.history[-count:] if self.history else []

    def add_memory_ref(self, ref: str) -> None:
        """Add a memory reference to the session."""
        if ref not in self.memory_refs:
            self.memory_refs.append(ref)

    def remove_memory_ref(self, ref: str) -> None:
        """Remove a memory reference from the session."""
        if ref in self.memory_refs:
            self.memory_refs.remove(ref)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.history.clear()

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now(timezone.utc)

    def set_active_plan(self, plan: ExecutionPlan) -> None:
        """Set the active execution plan."""
        self.active_plan = plan
        self.touch()

    def clear_active_plan(self) -> None:
        """Clear the active execution plan."""
        self.active_plan = None

    def _trim_history(self) -> None:
        """Trim history to maximum size."""
        if len(self.history) > self.MAX_HISTORY_SIZE:
            self.history = self.history[-self.MAX_HISTORY_SIZE :]

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state to dictionary."""
        return {
            "session_id": self.session_id,
            "history": self.history,
            "memory_refs": self.memory_refs,
            "active_plan_id": self.active_plan.plan_id if self.active_plan else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Deserialize session state from dictionary."""
        return cls(
            session_id=data["session_id"],
            history=data.get("history", []),
            memory_refs=data.get("memory_refs", []),
            active_plan=None,  # Plan must be loaded separately
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
        )


# =============================================================================
# TYPE ALIASES (for external use)
# =============================================================================

# Collection types for common patterns
StepResults = list[StepResult]
HistoryEntry = dict[str, Any]
EntityMap = dict[str, Any]
Metadata = dict[str, Any]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "IntentType",
    "CapabilityType",
    "PlanStepStatus",
    "PlanStatus",
    # Request/Response
    "AgentRequest",
    "AgentResponse",
    # Analysis
    "IntentAnalysis",
    "CapabilityDecision",
    # Planning
    "PlanStep",
    "StepResult",
    "ExecutionPlan",
    # State
    "SessionState",
    # Type Aliases
    "StepResults",
    "HistoryEntry",
    "EntityMap",
    "Metadata",
]
