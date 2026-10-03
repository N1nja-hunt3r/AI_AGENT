"""
Agent Controller - Core orchestration engine for the AI Agent Platform.

This module contains the central AgentController class that orchestrates
the complete request lifecycle: validation, intent analysis, capability
routing, planning, execution, and response assembly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Optional, Protocol, TypeVar

from app.providers.provider_registry import ProviderRegistry
from app.providers.provider_router import ProviderRouter, TaskType

# =============================================================================
# ENUMS
# =============================================================================


class IntentType(Enum):
    """Classification of user intent types."""

    QUERY = "query"  # Information retrieval
    ACTION = "action"  # Execute an operation
    CONVERSATION = "conversation"  # General dialogue
    ANALYSIS = "analysis"  # Deep analysis task
    CREATION = "creation"  # Generate content/artifacts
    UNKNOWN = "unknown"


class PlanStepStatus(Enum):
    """Execution status of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(Enum):
    """Overall execution plan status."""

    CREATED = "created"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


class CapabilityType(Enum):
    """Available capability types in the system."""

    MEMORY = "memory"
    TOOLS = "tools"
    RAG = "rag"
    WEB_SEARCH = "web_search"
    CHAT = "chat"
    CODING = "coding"
    VISION = "vision"
    FILES = "files"
    COMPUTER = "computer"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(frozen=True)
class AgentRequest:
    """
    Immutable incoming request to the agent controller.

    Attributes:
        request_id: Unique identifier for this request
        session_id: Session identifier for state continuity
        content: The user's input content
        metadata: Additional request metadata (user_id, timestamp, etc.)
    """

    request_id: str
    session_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRequest:
        """Factory method to create a new request with generated IDs."""
        return cls(
            request_id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
        )


@dataclass
class IntentAnalysis:
    """
    Result of analyzing user intent.

    Attributes:
        intent_type: Classified type of intent
        confidence: Confidence score (0.0 to 1.0)
        entities: Extracted entities from the request
        complexity_score: Estimated task complexity (1-10)
        requires_context: Whether historical context is needed
        requires_current_info: Whether real-time information is needed
    """

    intent_type: IntentType
    confidence: float
    entities: dict[str, Any]
    complexity_score: int
    requires_context: bool = False
    requires_current_info: bool = False


@dataclass
class CapabilityDecision:
    """
    Routing decision for which capabilities to invoke.

    Attributes:
        use_memory: Whether to query/update memory
        use_rag: Whether to use RAG for knowledge retrieval
        use_web_search: Whether to search the web
        tools_to_use: List of specific tools to invoke
        reasoning: Explanation of routing decisions
    """

    use_memory: bool = False
    use_rag: bool = False
    use_web_search: bool = False
    tools_to_use: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def active_capabilities(self) -> list[CapabilityType]:
        """Return list of capabilities that will be used."""
        capabilities = []
        if self.use_memory:
            capabilities.append(CapabilityType.MEMORY)
        if self.use_rag:
            capabilities.append(CapabilityType.RAG)
        if self.use_web_search:
            capabilities.append(CapabilityType.WEB_SEARCH)
        if self.tools_to_use:
            capabilities.append(CapabilityType.TOOLS)
        return capabilities


@dataclass
class PlanStep:
    """
    Single step in an execution plan.

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


@dataclass
class StepResult:
    """
    Result from executing a plan step.

    Attributes:
        step_id: Reference to the executed step
        success: Whether execution succeeded
        output: Output data from the step
        error: Error information if failed
        execution_time_ms: Time taken to execute
        metadata: Additional execution metadata
    """

    step_id: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a request.

    Attributes:
        plan_id: Unique identifier for this plan
        steps: Ordered list of execution steps
        status: Current plan status
        allows_replan: Whether replanning is permitted on failure
        max_retries: Maximum retry attempts per step
        created_at: Plan creation timestamp
    """

    plan_id: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.CREATED
    allows_replan: bool = True
    max_retries: int = 2
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def ordered_steps(self) -> list[PlanStep]:
        """Return steps in dependency-respecting execution order."""
        # Topological sort based on dependencies
        executed: set[str] = set()
        ordered: list[PlanStep] = []
        remaining = list(self.steps)

        while remaining:
            # Find steps with all dependencies satisfied
            ready = [
                step
                for step in remaining
                if all(dep in executed for dep in step.dependencies)
            ]

            if not ready and remaining:
                # Circular dependency detected
                raise ExecutionError(
                    "Circular dependency detected in execution plan",
                    step_id=remaining[0].step_id,
                )

            for step in ready:
                ordered.append(step)
                executed.add(step.step_id)
                remaining.remove(step)

        return ordered

    def get_step(self, step_id: str) -> PlanStep | None:
        """Retrieve a step by its ID."""
        return next((s for s in self.steps if s.step_id == step_id), None)


@dataclass
class SessionState:
    """
    Persistent state for an agent session.

    Attributes:
        session_id: Unique session identifier
        history: Conversation history
        memory_refs: References to stored memories
        active_plan: Currently executing plan
        metadata: Session metadata
        created_at: Session creation time
        last_active: Last activity timestamp
    """

    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    active_plan: ExecutionPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_to_history(
        self, request: AgentRequest, response: AgentResponse
    ) -> None:
        """Append a request-response pair to history."""
        self.history.append(
            {
                "request_id": request.request_id,
                "content": request.content,
                "response": response.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.last_active = datetime.now(timezone.utc)


@dataclass
class AgentResponse:
    """
    Response from the agent controller.

    Attributes:
        request_id: Reference to the original request
        session_id: Session identifier
        content: The response content
        success: Whether the request was handled successfully
        metadata: Response metadata (timing, capabilities used, etc.)
        error: Error information if unsuccessful
    """

    request_id: str
    session_id: str
    content: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ControllerError(Exception):
    """Base exception for controller errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context


class ValidationError(ControllerError):
    """Request validation failed."""

    pass


class IntentAnalysisError(ControllerError):
    """Intent analysis failed."""

    pass


class PlanningError(ControllerError):
    """Plan generation failed."""

    pass


class ExecutionError(ControllerError):
    """Plan execution failed."""

    def __init__(
        self, message: str, step_id: str | None = None, **context: Any
    ) -> None:
        super().__init__(message, step_id=step_id, **context)
        self.step_id = step_id


# =============================================================================
# PROTOCOLS & ABSTRACT BASES
# =============================================================================

T = TypeVar("T")


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a completion from the LLM."""
        ...

    async def complete_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured completion matching the schema."""
        ...


class Capability(ABC, Generic[T]):
    """
    Abstract base class for all capabilities.

    Capabilities are modular components that provide specific
    functionality (memory, tools, RAG, web search).
    """

    @property
    @abstractmethod
    def capability_type(self) -> CapabilityType:
        """Return the type of this capability."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this capability."""
        ...

    @abstractmethod
    async def execute(
        self, action: str, parameters: dict[str, Any], context: SessionState
    ) -> T:
        """
        Execute an action with the given parameters.

        Args:
            action: The specific action to perform
            parameters: Action parameters
            context: Current session state

        Returns:
            Capability-specific result type
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the capability is operational."""
        ...


class StateBackend(Protocol):
    """Protocol for state persistence backends."""

    async def load(self, session_id: str) -> SessionState | None:
        """Load session state from storage."""
        ...

    async def save(self, state: SessionState) -> None:
        """Save session state to storage."""
        ...

    async def delete(self, session_id: str) -> None:
        """Delete session state from storage."""
        ...


# =============================================================================
# CONTROLLER COMPONENTS
# =============================================================================


class RequestGateway:
    """
    Handles request ingestion, validation, and normalization.

    Responsibilities:
        - Validate request structure and content
        - Normalize input format
        - Apply rate limiting (placeholder)
        - Authenticate requests (placeholder)
    """

    MAX_CONTENT_LENGTH = 32000
    MIN_CONTENT_LENGTH = 1

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def validate(self, request: AgentRequest) -> AgentRequest:
        """
        Validate and normalize an incoming request.

        Args:
            request: The incoming agent request

        Returns:
            Validated request (same instance if valid)

        Raises:
            ValidationError: If request is invalid
        """
        self._validate_content(request.content)
        self._validate_session_id(request.session_id)
        self._logger.debug(
            "Request validated",
            extra={"request_id": request.request_id},
        )
        return request

    def _validate_content(self, content: str) -> None:
        """Validate request content."""
        if not content or not content.strip():
            raise ValidationError("Request content cannot be empty")

        if len(content) < self.MIN_CONTENT_LENGTH:
            raise ValidationError(
                f"Content too short (min {self.MIN_CONTENT_LENGTH} chars)"
            )

        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Content too long (max {self.MAX_CONTENT_LENGTH} chars)"
            )

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID format."""
        if not session_id:
            raise ValidationError("Session ID is required")

        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValidationError("Invalid session ID format (expected UUID)")


class IntentAnalyzer:
    """
    Analyzes user requests to determine intent and extract entities.

    Uses LLM to classify intent type, extract relevant entities,
    and assess task complexity.
    """

    ANALYSIS_SYSTEM_PROMPT = """You are an intent analysis system. Analyze the user's request and extract:
1. Intent type (query, action, conversation, analysis, creation)
2. Key entities mentioned
3. Complexity score (1-10)
4. Whether historical context is needed
5. Whether current/real-time information is needed

Respond in JSON format."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)

    async def analyze(
        self,
        request: AgentRequest,
        history: list[dict[str, Any]] | None = None,
    ) -> IntentAnalysis:
        """
        Analyze the intent of a user request.

        Args:
            request: The user's request
            history: Optional conversation history for context

        Returns:
            IntentAnalysis with classified intent and extracted entities

        Raises:
            IntentAnalysisError: If analysis fails
        """
        try:
            prompt = self._build_analysis_prompt(request, history)

            result = await self._llm.complete_structured(
                prompt=prompt,
                response_schema={
                    "intent_type": "string",
                    "confidence": "float",
                    "entities": "object",
                    "complexity_score": "integer",
                    "requires_context": "boolean",
                    "requires_current_info": "boolean",
                },
                system_prompt=self.ANALYSIS_SYSTEM_PROMPT,
            )

            return IntentAnalysis(
                intent_type=IntentType(result.get("intent_type", "unknown")),
                confidence=float(result.get("confidence", 0.5)),
                entities=result.get("entities", {}),
                complexity_score=int(result.get("complexity_score", 5)),
                requires_context=bool(result.get("requires_context", False)),
                requires_current_info=bool(result.get("requires_current_info", False)),
            )

        except Exception as e:
            self._logger.error(f"Intent analysis failed: {e}")
            raise IntentAnalysisError(
                f"Failed to analyze intent: {e}",
                request_id=request.request_id,
            )

    def _build_analysis_prompt(
        self,
        request: AgentRequest,
        history: list[dict[str, Any]] | None,
    ) -> str:
        """Build the prompt for intent analysis."""
        prompt_parts = [f"User request: {request.content}"]

        if history:
            recent_history = history[-5:]  # Last 5 turns
            history_text = "\n".join(
                f"- {h.get('content', '')}" for h in recent_history
            )
            prompt_parts.append(f"\nRecent conversation:\n{history_text}")

        return "\n".join(prompt_parts)


class CapabilityRouter:
    """
    Determines which capabilities should be invoked for a request.

    Makes routing decisions based on intent analysis, available
    capabilities, and session context.
    """

    def __init__(
        self,
        available_capabilities: list[CapabilityType],
        available_tools: list[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._available = set(available_capabilities)
        self._available_tools = available_tools or []
        self._logger = logger or logging.getLogger(__name__)

    def decide(
        self,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> CapabilityDecision:
        """
        Decide which capabilities to use for the given intent.

        Args:
            intent: Analyzed intent from the request
            session: Current session state

        Returns:
            CapabilityDecision indicating which capabilities to invoke
        """
        decision = CapabilityDecision()
        reasoning_parts: list[str] = []

        # Memory decision
        if self._should_use_memory(intent, session):
            decision.use_memory = True
            reasoning_parts.append("Memory: context reference detected")

        # RAG decision
        if self._should_use_rag(intent):
            decision.use_rag = True
            reasoning_parts.append("RAG: domain knowledge required")

        # Web search decision
        if self._should_use_web_search(intent):
            decision.use_web_search = True
            reasoning_parts.append("Web: current information needed")

        # Tools decision
        tools = self._select_tools(intent)
        if tools:
            decision.tools_to_use = tools
            reasoning_parts.append(f"Tools: {', '.join(tools)}")

        decision.reasoning = "; ".join(reasoning_parts) or "Direct response"

        self._logger.debug(
            "Capability routing decision",
            extra={
                "capabilities": [c.value for c in decision.active_capabilities],
                "reasoning": decision.reasoning,
            },
        )

        return decision

    def _should_use_memory(
        self, intent: IntentAnalysis, session: SessionState
    ) -> bool:
        """Determine if memory capability should be used."""
        if CapabilityType.MEMORY not in self._available:
            return False

        # Use memory if context is required or session has memory refs
        return intent.requires_context or bool(session.memory_refs)

    def _should_use_rag(self, intent: IntentAnalysis) -> bool:
        """Determine if RAG capability should be used."""
        if CapabilityType.RAG not in self._available:
            return False

        # Use RAG for queries and analysis that need domain knowledge
        knowledge_intents = {IntentType.QUERY, IntentType.ANALYSIS}
        return intent.intent_type in knowledge_intents and intent.complexity_score >= 3

    def _should_use_web_search(self, intent: IntentAnalysis) -> bool:
        """Determine if web search capability should be used."""
        if CapabilityType.WEB_SEARCH not in self._available:
            return False

        return intent.requires_current_info

    def _select_tools(self, intent: IntentAnalysis) -> list[str]:
        """Select which tools to use based on intent."""
        if CapabilityType.TOOLS not in self._available:
            return []

        if intent.intent_type != IntentType.ACTION:
            return []

        # Match tools based on entities (simplified matching)
        selected: list[str] = []
        requested_tools = intent.entities.get("tools", [])

        for tool in requested_tools:
            if tool in self._available_tools:
                selected.append(tool)

        return selected


class Planner:
    """
    Generates execution plans based on intent and capability decisions.

    Creates ordered sequences of steps with dependencies,
    fallback strategies, and retry policies.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)

    async def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
    ) -> ExecutionPlan:
        """
        Create an execution plan for the request.

        Args:
            request: The original user request
            intent: Analyzed intent
            capabilities: Routing decision

        Returns:
            ExecutionPlan with ordered steps

        Raises:
            PlanningError: If plan generation fails
        """
        try:
            steps: list[PlanStep] = []
            step_counter = 0

            def next_step_id() -> str:
                nonlocal step_counter
                step_counter += 1
                return f"step_{step_counter}"

            # Build steps based on capability decisions
            memory_step_id: str | None = None
            rag_step_id: str | None = None
            web_step_id: str | None = None

            # Memory retrieval (no dependencies)
            if capabilities.use_memory:
                memory_step_id = next_step_id()
                steps.append(
                    PlanStep(
                        step_id=memory_step_id,
                        capability=CapabilityType.MEMORY,
                        action="retrieve",
                        parameters={"query": request.content},
                    )
                )

            # RAG retrieval (no dependencies)
            if capabilities.use_rag:
                rag_step_id = next_step_id()
                steps.append(
                    PlanStep(
                        step_id=rag_step_id,
                        capability=CapabilityType.RAG,
                        action="search",
                        parameters={
                            "query": request.content,
                            "entities": intent.entities,
                        },
                    )
                )

            # Web search (no dependencies)
            if capabilities.use_web_search:
                web_step_id = next_step_id()
                steps.append(
                    PlanStep(
                        step_id=web_step_id,
                        capability=CapabilityType.WEB_SEARCH,
                        action="search",
                        parameters={"query": request.content},
                        fallback_action="skip",
                    )
                )

            # Tool executions (depend on context gathering)
            context_deps = [
                sid
                for sid in [memory_step_id, rag_step_id, web_step_id]
                if sid is not None
            ]

            for tool_name in capabilities.tools_to_use:
                steps.append(
                    PlanStep(
                        step_id=next_step_id(),
                        capability=CapabilityType.TOOLS,
                        action=tool_name,
                        parameters=intent.entities.get("tool_params", {}).get(
                            tool_name, {}
                        ),
                        dependencies=context_deps,
                    )
                )

            # Final synthesis step (depends on all previous steps)
            all_step_ids = [s.step_id for s in steps]
            steps.append(
                PlanStep(
                    step_id=next_step_id(),
                    capability=CapabilityType.MEMORY,  # Uses LLM via memory capability
                    action="synthesize",
                    parameters={
                        "request": request.content,
                        "intent": intent.intent_type.value,
                    },
                    dependencies=all_step_ids,
                )
            )

            plan = ExecutionPlan(
                plan_id=str(uuid.uuid4()),
                steps=steps,
                allows_replan=intent.complexity_score <= 7,
            )

            self._logger.info(
                "Execution plan created",
                extra={
                    "plan_id": plan.plan_id,
                    "step_count": len(steps),
                },
            )

            return plan

        except Exception as e:
            self._logger.error(f"Planning failed: {e}")
            raise PlanningError(
                f"Failed to create execution plan: {e}",
                request_id=request.request_id,
            )

    async def replan(
        self,
        original_plan: ExecutionPlan,
        failed_step: PlanStep,
        error: Exception,
        completed_results: list[StepResult],
    ) -> ExecutionPlan:
        """
        Create a new plan after a step failure.

        Args:
            original_plan: The plan that failed
            failed_step: The step that failed
            error: The error that occurred
            completed_results: Results from completed steps

        Returns:
            New ExecutionPlan with adjusted steps
        """
        self._logger.info(
            "Replanning after failure",
            extra={
                "plan_id": original_plan.plan_id,
                "failed_step": failed_step.step_id,
            },
        )

        # Remove failed step and its dependents
        failed_id = failed_step.step_id
        remaining_steps = [
            step
            for step in original_plan.steps
            if step.step_id != failed_id
            and failed_id not in step.dependencies
            and step.status == PlanStepStatus.PENDING
        ]

        # Update dependencies to remove reference to failed step
        for step in remaining_steps:
            step.dependencies = [d for d in step.dependencies if d != failed_id]

        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            steps=remaining_steps,
            allows_replan=False,  # Only one replan attempt
            max_retries=1,
        )


class Executor:
    """
    Executes plan steps using registered capabilities.

    Handles step execution, error recovery, timeouts,
    and result collection.
    """

    def __init__(
        self,
        capabilities: dict[CapabilityType, Capability[Any]],
        logger: logging.Logger | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._logger = logger or logging.getLogger(__name__)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        session: SessionState,
    ) -> list[StepResult]:
        """
        Execute all steps in a plan.

        Args:
            plan: The execution plan
            session: Current session state

        Returns:
            List of results from all executed steps
        """
        plan.status = PlanStatus.EXECUTING
        results: list[StepResult] = []
        results_by_id: dict[str, StepResult] = {}

        for step in plan.ordered_steps():
            result = await self._execute_step(step, session, results_by_id)
            results.append(result)
            results_by_id[step.step_id] = result

            if not result.success and not step.fallback_action:
                plan.status = PlanStatus.FAILED
                break

        if plan.status != PlanStatus.FAILED:
            plan.status = PlanStatus.COMPLETED

        return results

    async def _execute_step(
        self,
        step: PlanStep,
        session: SessionState,
        previous_results: dict[str, StepResult],
    ) -> StepResult:
        """Execute a single plan step."""
        step.status = PlanStepStatus.RUNNING
        start_time = datetime.now(timezone.utc)

        try:
            capability = self._capabilities.get(step.capability)
            if not capability:
                raise ExecutionError(
                    f"Capability not available: {step.capability.value}",
                    step_id=step.step_id,
                )

            # Inject previous results into parameters
            enriched_params = {
                **step.parameters,
                "_previous_results": {
                    dep: previous_results[dep].output
                    for dep in step.dependencies
                    if dep in previous_results and previous_results[dep].success
                },
            }

            # Execute with timeout
            output = await asyncio.wait_for(
                capability.execute(step.action, enriched_params, session),
                timeout=step.timeout_seconds,
            )

            step.status = PlanStepStatus.COMPLETED
            execution_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            result = StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            step.status = PlanStepStatus.FAILED
            result = StepResult(
                step_id=step.step_id,
                success=False,
                output=None,
                error=f"Step timed out after {step.timeout_seconds}s",
            )

        except Exception as e:
            step.status = PlanStepStatus.FAILED
            self._logger.error(
                f"Step execution failed: {e}",
                extra={"step_id": step.step_id},
            )

            # Try fallback if available
            if step.fallback_action == "skip":
                step.status = PlanStepStatus.SKIPPED
                result = StepResult(
                    step_id=step.step_id,
                    success=True,
                    output=None,
                    metadata={"skipped": True, "reason": str(e)},
                )
            else:
                result = StepResult(
                    step_id=step.step_id,
                    success=False,
                    output=None,
                    error=str(e),
                )

        step.result = result
        return result


class StateManager:
    """
    Manages session state lifecycle.

    Handles loading, saving, checkpointing, and cleanup
    of session state.
    """

    def __init__(
        self,
        backend: StateBackend,
        logger: logging.Logger | None = None,
    ) -> None:
        self._backend = backend
        self._logger = logger or logging.getLogger(__name__)
        self._cache: dict[str, SessionState] = {}

    async def load_or_create(self, session_id: str) -> SessionState:
        """
        Load existing session or create a new one.

        Args:
            session_id: The session identifier

        Returns:
            SessionState (existing or newly created)
        """
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]

        # Try loading from backend
        state = await self._backend.load(session_id)

        if state is None:
            state = SessionState(session_id=session_id)
            self._logger.debug(f"Created new session: {session_id}")
        else:
            self._logger.debug(f"Loaded existing session: {session_id}")

        self._cache[session_id] = state
        return state

    async def checkpoint(
        self,
        session: SessionState,
        step: PlanStep,
        result: StepResult,
    ) -> None:
        """
        Save a checkpoint after step completion.

        Args:
            session: Current session state
            step: Completed step
            result: Step result
        """
        session.last_active = datetime.now(timezone.utc)

        # Async save (fire and forget for checkpoints)
        asyncio.create_task(self._backend.save(session))

        self._logger.debug(
            "Checkpoint saved",
            extra={"session_id": session.session_id, "step_id": step.step_id},
        )

    async def persist(self, session: SessionState) -> None:
        """
        Persist session state to backend.

        Args:
            session: Session state to persist
        """
        session.last_active = datetime.now(timezone.utc)
        await self._backend.save(session)
        self._cache[session.session_id] = session

    async def cleanup(self, session_id: str) -> None:
        """Remove session from cache and backend."""
        self._cache.pop(session_id, None)
        await self._backend.delete(session_id)


class ResponseAssembler:
    """
    Assembles final responses from execution results.

    Combines outputs from multiple steps into a coherent
    response with appropriate metadata.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)

    async def build(
        self,
        request: AgentRequest,
        results: list[StepResult],
        session: SessionState,
    ) -> AgentResponse:
        """
        Build the final response from execution results.

        Args:
            request: Original request
            results: Results from plan execution
            session: Session state

        Returns:
            Assembled AgentResponse
        """
        # Check for failures
        failures = [r for r in results if not r.success]
        if failures and all(not r.success for r in results):
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                content="I encountered an error processing your request.",
                success=False,
                error=failures[0].error,
            )

        # Find synthesis result (last step)
        synthesis_result = next(
            (r for r in reversed(results) if r.success and r.output),
            None,
        )

        if synthesis_result and isinstance(synthesis_result.output, str):
            content = synthesis_result.output
        else:
            # Fallback: combine successful outputs
            outputs = [
                str(r.output)
                for r in results
                if r.success and r.output is not None
            ]
            content = await self._synthesize_outputs(request.content, outputs)

        # Build metadata
        metadata = {
            "steps_executed": len(results),
            "steps_successful": sum(1 for r in results if r.success),
            "total_execution_time_ms": sum(r.execution_time_ms for r in results),
            "capabilities_used": list(
                {r.metadata.get("capability") for r in results if r.metadata}
            ),
        }

        return AgentResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            content=content,
            success=True,
            metadata=metadata,
        )

    async def _synthesize_outputs(
        self, original_request: str, outputs: list[str]
    ) -> str:
        """Use LLM to synthesize multiple outputs into coherent response."""
        if not outputs:
            return "I was unable to gather the information needed to respond."

        prompt = f"""Original request: {original_request}

Gathered information:
{chr(10).join(f'- {output}' for output in outputs)}

Respond directly to the user's original request in a natural, helpful way.
Do not mention the gather information steps or internal processing."""

        return await self._llm.complete(prompt)


# =============================================================================
# MAIN CONTROLLER
# =============================================================================


class AgentController:
    """
    Central orchestration engine for the AI Agent Platform.

    Coordinates the complete request lifecycle through validation,
    intent analysis, capability routing, planning, execution,
    and response assembly.

    This is the primary entry point for processing agent requests.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        state_backend: StateBackend,
        capabilities: dict[CapabilityType, Capability[Any]],
        available_tools: list[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the Agent Controller.

        Args:
            llm_client: LLM client for AI operations
            state_backend: Backend for state persistence
            capabilities: Registered capability implementations
            available_tools: List of available tool names
            logger: Logger instance
        """
        self._logger = logger or logging.getLogger(__name__)

        # Initialize components
        self._gateway = RequestGateway(logger=self._logger)
        self._intent_analyzer = IntentAnalyzer(llm_client, logger=self._logger)
        self._capability_router = CapabilityRouter(
            available_capabilities=list(capabilities.keys()),
            available_tools=available_tools,
            logger=self._logger,
        )
        self._planner = Planner(llm_client, logger=self._logger)
        self._executor = Executor(capabilities, logger=self._logger)
        self._state_manager = StateManager(state_backend, logger=self._logger)
        self._response_assembler = ResponseAssembler(llm_client, logger=self._logger)

    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a user request through the complete agent lifecycle.

        This is the main entry point for all agent interactions.

        Args:
            request: The incoming agent request

        Returns:
            AgentResponse with the result

        Lifecycle:
            1. Validate request
            2. Load/create session state
            3. Analyze intent
            4. Route to capabilities
            5. Create execution plan
            6. Execute plan
            7. Update state
            8. Assemble response
        """
        try:
            # Phase 1: Validation
            validated_request = self._gateway.validate(request)

            # Phase 2: State Loading
            session = await self._state_manager.load_or_create(
                validated_request.session_id
            )

            # Phase 3: Intent Analysis
            intent = await self._intent_analyzer.analyze(
                request=validated_request,
                history=session.history,
            )

            # Phase 4: Capability Routing
            capabilities_needed = self._capability_router.decide(
                intent=intent,
                session=session,
            )

            # Phase 5: Planning
            plan = await self._planner.create_plan(
                request=validated_request,
                intent=intent,
                capabilities=capabilities_needed,
            )
            session.active_plan = plan

            # Phase 6: Execution
            results = await self._execute_with_recovery(plan, session)

            # Phase 7: State Update
            response = await self._response_assembler.build(
                request=validated_request,
                results=results,
                session=session,
            )
            session.add_to_history(validated_request, response)
            await self._state_manager.persist(session)

            # Phase 8: Return Response
            self._logger.info(
                "Request completed",
                extra={
                    "request_id": request.request_id,
                    "success": response.success,
                },
            )

            return response

        except ValidationError as e:
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                content=f"Invalid request: {e.message}",
                success=False,
                error=e.message,
            )

        except ControllerError as e:
            self._logger.error(f"Controller error: {e}", extra=e.context)
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                content="An error occurred while processing your request.",
                success=False,
                error=e.message,
            )

        except Exception as e:
            self._logger.exception("Unexpected error in request handling")
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                content="An unexpected error occurred.",
                success=False,
                error=str(e),
            )

    async def _execute_with_recovery(
        self,
        plan: ExecutionPlan,
        session: SessionState,
    ) -> list[StepResult]:
        """
        Execute plan with automatic recovery on failure.

        Args:
            plan: Execution plan
            session: Session state

        Returns:
            List of step results
        """
        results = await self._executor.execute_plan(plan, session)

        # Check for failures that need replanning
        if plan.status == PlanStatus.FAILED and plan.allows_replan:
            failed_step = next(
                (s for s in plan.steps if s.status == PlanStepStatus.FAILED),
                None,
            )

            if failed_step:
                self._logger.info("Attempting replan after failure")
                plan.status = PlanStatus.REPLANNING

                new_plan = await self._planner.replan(
                    original_plan=plan,
                    failed_step=failed_step,
                    error=Exception(failed_step.result.error if failed_step.result else "Unknown"),
                    completed_results=results,
                )

                additional_results = await self._executor.execute_plan(
                    new_plan, session
                )
                results.extend(additional_results)

        return results

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all controller components.

        Returns:
            Dictionary mapping component names to health status
        """
        health: dict[str, bool] = {"controller": True}

        for cap_type, capability in self._executor._capabilities.items():
            try:
                health[cap_type.value] = await capability.health_check()
            except Exception:
                health[cap_type.value] = False

        return health


# =============================================================================
# DEFAULT / MOCK IMPLEMENTATIONS FOR AGENT LOADER
# =============================================================================


class ProviderLLMClient:
    """LLM client backed by the provider registry / router."""

    def __init__(
        self,
        router: ProviderRouter,
        registry: Optional[ProviderRegistry] = None,
    ) -> None:
        self._router = router
        self._registry = registry

    @classmethod
    def from_registry(cls, registry: ProviderRegistry) -> ProviderLLMClient:
        return cls(router=ProviderRouter(registry), registry=registry)

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        provider = self._router.select(TaskType.PLANNING)
        content, _ = await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content

    async def complete_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        provider = self._router.select(TaskType.PLANNING)
        schema_str = "\n".join(f"  {k}: {v}" for k, v in response_schema.items())
        structured_prompt = (
            f"{prompt}\n\nRespond in JSON matching this schema:\n{schema_str}"
        )
        content, _ = await provider.generate(
            prompt=structured_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2000,
        )
        import json

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"intent_type": "conversation", "confidence": 0.5}


class _MockLLMClient:
    """Minimal LLM client mock for default agent creation."""

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        return "Mock response"

    async def complete_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        return {
            "intent_type": "conversation",
            "confidence": 0.5,
            "entities": {},
            "complexity_score": 1,
            "requires_context": False,
            "requires_current_info": False,
        }


class _InMemoryStateBackend:
    """In-memory state backend for default agent creation."""

    def __init__(self) -> None:
        self._storage: dict[str, SessionState] = {}

    async def load(self, session_id: str) -> SessionState | None:
        return self._storage.get(session_id)

    async def save(self, state: SessionState) -> None:
        self._storage[state.session_id] = state

    async def delete(self, session_id: str) -> None:
        self._storage.pop(session_id, None)


def create_agent(
    registry: Optional[ProviderRegistry] = None,
) -> AgentController:
    """Factory function for agent_loader compatibility."""
    if registry is not None:
        router = ProviderRouter(registry)
        llm_client = ProviderLLMClient(router=router, registry=registry)
    else:
        llm_client = _MockLLMClient()
    return AgentController(
        llm_client=llm_client,
        state_backend=_InMemoryStateBackend(),
        capabilities={},
        available_tools=[],
    )
