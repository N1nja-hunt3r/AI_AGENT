"""
Planner - Execution plan generation for the AI Agent Platform.

This module generates execution plans that orchestrate capability
invocations to fulfill user requests. It handles:
- Plan generation from intent and capability decisions
- Step dependency management
- Parallel execution optimization
- Replanning on failure
- Multi-agent workflow support (future)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from app.agents.models import (
    AgentRequest,
    CapabilityDecision,
    CapabilityType,
    ExecutionPlan,
    IntentAnalysis,
    PlanStep,
    SessionState,
    StepResult,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class StepType(Enum):
    """
    Classification of plan step types.

    Used to categorize steps for optimization and visualization.
    """

    CONTEXT_GATHERING = auto()  # Memory, RAG retrieval
    INFORMATION_RETRIEVAL = auto()  # Web search, API calls
    COMPUTATION = auto()  # Tool execution, calculations
    SYNTHESIS = auto()  # Final response generation
    SIDE_EFFECT = auto()  # State changes, notifications
    CHECKPOINT = auto()  # State persistence
    AGENT_HANDOFF = auto()  # Multi-agent delegation


class PlanningStrategy(Enum):
    """
    Strategy for plan generation.

    Different strategies optimize for different goals.
    """

    SEQUENTIAL = auto()  # Execute steps one by one
    PARALLEL_OPTIMIZED = auto()  # Maximize parallelism
    FAIL_FAST = auto()  # Stop on first failure
    RESILIENT = auto()  # Continue despite failures


# Default timeouts by capability type
DEFAULT_TIMEOUTS: dict[CapabilityType, int] = {
    CapabilityType.MEMORY: 10,
    CapabilityType.RAG: 30,
    CapabilityType.WEB_SEARCH: 45,
    CapabilityType.TOOLS: 60,
    CapabilityType.COMPUTER: 60,
}


# =============================================================================
# STEP TEMPLATES
# =============================================================================


@dataclass
class StepTemplate:
    """
    Template for generating plan steps.

    Templates define the structure of steps for each capability,
    allowing consistent step generation with customizable parameters.

    Attributes:
        capability: Target capability type
        action: Default action name
        default_params: Default parameters
        timeout_seconds: Default timeout
        fallback_action: Default fallback
        step_type: Classification of step
        parallelizable: Whether step can run in parallel
    """

    capability: CapabilityType
    action: str
    default_params: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    fallback_action: str | None = None
    step_type: StepType = StepType.COMPUTATION
    parallelizable: bool = True

    def create_step(
        self,
        step_id: str,
        parameters: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        timeout_override: int | None = None,
    ) -> PlanStep:
        """
        Create a PlanStep from this template.

        Args:
            step_id: Unique step identifier
            parameters: Override/extend default parameters
            dependencies: Step dependencies
            timeout_override: Override default timeout

        Returns:
            Configured PlanStep
        """
        merged_params = {**self.default_params}
        if parameters:
            merged_params.update(parameters)

        return PlanStep(
            step_id=step_id,
            capability=self.capability,
            action=self.action,
            parameters=merged_params,
            dependencies=dependencies or [],
            fallback_action=self.fallback_action,
            timeout_seconds=timeout_override or self.timeout_seconds,
        )


# Pre-defined step templates
STEP_TEMPLATES: dict[str, StepTemplate] = {
    # Memory templates
    "memory_retrieve": StepTemplate(
        capability=CapabilityType.MEMORY,
        action="retrieve",
        timeout_seconds=10,
        step_type=StepType.CONTEXT_GATHERING,
        parallelizable=True,
    ),
    "memory_store": StepTemplate(
        capability=CapabilityType.MEMORY,
        action="store",
        timeout_seconds=10,
        step_type=StepType.SIDE_EFFECT,
        parallelizable=False,
    ),
    # RAG templates
    "rag_search": StepTemplate(
        capability=CapabilityType.RAG,
        action="search",
        timeout_seconds=30,
        step_type=StepType.CONTEXT_GATHERING,
        parallelizable=True,
    ),
    "rag_retrieve": StepTemplate(
        capability=CapabilityType.RAG,
        action="retrieve",
        timeout_seconds=30,
        step_type=StepType.CONTEXT_GATHERING,
        parallelizable=True,
    ),
    # Web search templates
    "web_search": StepTemplate(
        capability=CapabilityType.WEB_SEARCH,
        action="search",
        timeout_seconds=45,
        fallback_action="skip",
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    "web_fetch": StepTemplate(
        capability=CapabilityType.WEB_SEARCH,
        action="fetch",
        timeout_seconds=30,
        fallback_action="skip",
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    # Tool templates
    "tool_execute": StepTemplate(
        capability=CapabilityType.TOOLS,
        action="execute",
        timeout_seconds=60,
        step_type=StepType.COMPUTATION,
        parallelizable=False,
    ),
    # File templates
    "file_read": StepTemplate(
        capability=CapabilityType.FILES,
        action="read",
        timeout_seconds=30,
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    "file_search": StepTemplate(
        capability=CapabilityType.FILES,
        action="search",
        timeout_seconds=30,
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    "file_write": StepTemplate(
        capability=CapabilityType.FILES,
        action="write",
        timeout_seconds=30,
        step_type=StepType.SIDE_EFFECT,
        parallelizable=False,
    ),
    "file_mkdir": StepTemplate(
        capability=CapabilityType.FILES,
        action="mkdir",
        timeout_seconds=15,
        step_type=StepType.SIDE_EFFECT,
        parallelizable=False,
    ),
    # Coding templates
    "coding_read": StepTemplate(
        capability=CapabilityType.CODING,
        action="read",
        timeout_seconds=30,
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    "coding_analyze": StepTemplate(
        capability=CapabilityType.CODING,
        action="analyze",
        timeout_seconds=60,
        step_type=StepType.COMPUTATION,
        parallelizable=False,
    ),
    "coding_write": StepTemplate(
        capability=CapabilityType.CODING,
        action="write",
        timeout_seconds=30,
        step_type=StepType.SIDE_EFFECT,
        parallelizable=False,
    ),
    # Vision templates
    "vision_analyze": StepTemplate(
        capability=CapabilityType.VISION,
        action="analyze",
        timeout_seconds=60,
        step_type=StepType.COMPUTATION,
        parallelizable=False,
    ),
    "vision_describe": StepTemplate(
        capability=CapabilityType.VISION,
        action="describe",
        timeout_seconds=45,
        step_type=StepType.INFORMATION_RETRIEVAL,
        parallelizable=True,
    ),
    # Computer templates
    "computer_execute": StepTemplate(
        capability=CapabilityType.COMPUTER,
        action="terminal_execute",
        timeout_seconds=60,
        step_type=StepType.SIDE_EFFECT,
        parallelizable=False,
    ),
    # Synthesis templates
    "synthesize": StepTemplate(
        capability=CapabilityType.MEMORY,  # Uses LLM via memory capability
        action="synthesize",
        timeout_seconds=30,
        step_type=StepType.SYNTHESIS,
        parallelizable=False,
    ),
}


# =============================================================================
# STEP BUILDER
# =============================================================================


class StepBuilder:
    """
    Fluent builder for creating plan steps.

    Provides a chainable API for step construction with validation.
    """

    def __init__(self, step_id: str | None = None) -> None:
        """
        Initialize step builder.

        Args:
            step_id: Optional step ID (generated if not provided)
        """
        self._step_id = step_id or f"step_{uuid.uuid4().hex[:8]}"
        self._capability: CapabilityType | None = None
        self._action: str = ""
        self._parameters: dict[str, Any] = {}
        self._dependencies: list[str] = []
        self._fallback_action: str | None = None
        self._timeout_seconds: int = 30
        self._metadata: dict[str, Any] = {}

    def with_capability(self, capability: CapabilityType) -> StepBuilder:
        """Set the capability type."""
        self._capability = capability
        self._timeout_seconds = DEFAULT_TIMEOUTS.get(capability, 30)
        return self

    def with_action(self, action: str) -> StepBuilder:
        """Set the action name."""
        self._action = action
        return self

    def with_parameters(self, **params: Any) -> StepBuilder:
        """Add parameters."""
        self._parameters.update(params)
        return self

    def with_parameter(self, key: str, value: Any) -> StepBuilder:
        """Add a single parameter."""
        self._parameters[key] = value
        return self

    def depends_on(self, *step_ids: str) -> StepBuilder:
        """Add dependencies."""
        self._dependencies.extend(step_ids)
        return self

    def with_fallback(self, fallback: str) -> StepBuilder:
        """Set fallback action."""
        self._fallback_action = fallback
        return self

    def with_timeout(self, seconds: int) -> StepBuilder:
        """Set timeout."""
        self._timeout_seconds = seconds
        return self

    def with_metadata(self, **metadata: Any) -> StepBuilder:
        """Add metadata."""
        self._metadata.update(metadata)
        return self

    def from_template(
        self,
        template_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> StepBuilder:
        """
        Initialize from a template.

        Args:
            template_name: Name of template in STEP_TEMPLATES
            parameters: Override parameters

        Returns:
            Self for chaining

        Raises:
            ValueError: If template not found
        """
        template = STEP_TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        self._capability = template.capability
        self._action = template.action
        self._parameters = {**template.default_params}
        if parameters:
            self._parameters.update(parameters)
        self._fallback_action = template.fallback_action
        self._timeout_seconds = template.timeout_seconds
        self._metadata["step_type"] = template.step_type.name
        self._metadata["parallelizable"] = template.parallelizable

        return self

    def build(self) -> PlanStep:
        """
        Build the PlanStep.

        Returns:
            Configured PlanStep

        Raises:
            ValueError: If required fields are missing
        """
        if not self._capability:
            raise ValueError("Capability is required")
        if not self._action:
            raise ValueError("Action is required")

        # Store metadata in parameters
        if self._metadata:
            self._parameters["_metadata"] = self._metadata

        return PlanStep(
            step_id=self._step_id,
            capability=self._capability,
            action=self._action,
            parameters=self._parameters,
            dependencies=list(set(self._dependencies)),  # Dedupe
            fallback_action=self._fallback_action,
            timeout_seconds=self._timeout_seconds,
        )


# =============================================================================
# PLAN BUILDER
# =============================================================================


class PlanBuilder:
    """
    Fluent builder for creating execution plans.

    Provides a high-level API for plan construction with
    automatic dependency management.
    """

    def __init__(self, plan_id: str | None = None) -> None:
        """
        Initialize plan builder.

        Args:
            plan_id: Optional plan ID (generated if not provided)
        """
        self._plan_id = plan_id or str(uuid.uuid4())
        self._steps: list[PlanStep] = []
        self._step_counter = 0
        self._allows_replan = True
        self._max_retries = 2
        self._metadata: dict[str, Any] = {}

        # Track step groups for dependency management
        self._current_group: list[str] = []
        self._previous_group: list[str] = []

    def _next_step_id(self, prefix: str = "step") -> str:
        """Generate next step ID."""
        self._step_counter += 1
        return f"{prefix}_{self._step_counter}"

    # -------------------------------------------------------------------------
    # Step Addition
    # -------------------------------------------------------------------------

    def add_step(self, step: PlanStep) -> PlanBuilder:
        """Add a pre-built step."""
        self._steps.append(step)
        self._current_group.append(step.step_id)
        return self

    def add_memory_retrieval(
        self,
        query: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add memory retrieval step."""
        step = (
            StepBuilder(step_id or self._next_step_id("memory"))
            .from_template("memory_retrieve", {"query": query})
            .build()
        )
        return self.add_step(step)

    def add_rag_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add RAG search step."""
        params: dict[str, Any] = {"query": query}
        if filters:
            params["filters"] = filters

        step = (
            StepBuilder(step_id or self._next_step_id("rag"))
            .from_template("rag_search", params)
            .build()
        )
        return self.add_step(step)

    def add_web_search(
        self,
        query: str,
        max_results: int = 5,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add web search step."""
        step = (
            StepBuilder(step_id or self._next_step_id("web"))
            .from_template("web_search", {"query": query, "max_results": max_results})
            .build()
        )
        return self.add_step(step)

    def add_tool_execution(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        step_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanBuilder:
        """Add tool execution step."""
        builder = (
            StepBuilder(step_id or self._next_step_id("tool"))
            .from_template("tool_execute", {"tool_name": tool_name, **(parameters or {})})
        )

        if depends_on:
            builder.depends_on(*depends_on)
        elif self._previous_group:
            builder.depends_on(*self._previous_group)

        return self.add_step(builder.build())

    def add_synthesis(
        self,
        request_content: str,
        intent_type: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add synthesis step (depends on all previous steps)."""
        all_step_ids = [s.step_id for s in self._steps]

        step = (
            StepBuilder(step_id or self._next_step_id("synthesis"))
            .from_template(
                "synthesize",
                {"request": request_content, "intent": intent_type},
            )
            .depends_on(*all_step_ids)
            .build()
        )
        return self.add_step(step)

    def add_computer_execution(
        self,
        command: str,
        step_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanBuilder:
        """Add a computer execution step (runs an OS command)."""
        builder = (
            StepBuilder(step_id or self._next_step_id("computer"))
            .from_template("computer_execute", {"command": command})
        )

        if depends_on:
            builder.depends_on(*depends_on)
        elif self._previous_group:
            builder.depends_on(*self._previous_group)

        return self.add_step(builder.build())

    def add_file_search(
        self,
        query: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add a file search step."""
        step = (
            StepBuilder(step_id or self._next_step_id("file"))
            .from_template("file_search", {"query": query})
            .build()
        )
        return self.add_step(step)

    def add_file_read(
        self,
        path: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add a file read step."""
        step = (
            StepBuilder(step_id or self._next_step_id("file"))
            .from_template("file_read", {"path": path})
            .build()
        )
        return self.add_step(step)

    def add_file_write(
        self,
        path: str,
        content: str,
        step_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanBuilder:
        """Add a file write step."""
        builder = (
            StepBuilder(step_id or self._next_step_id("file"))
            .from_template("file_write", {"path": path, "content": content})
        )
        if depends_on:
            builder.depends_on(*depends_on)
        elif self._previous_group:
            builder.depends_on(*self._previous_group)
        return self.add_step(builder.build())

    def add_file_mkdir(
        self,
        path: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add a directory creation step."""
        step = (
            StepBuilder(step_id or self._next_step_id("file"))
            .from_template("file_mkdir", {"path": path})
            .build()
        )
        return self.add_step(step)

    def add_coding_read(
        self,
        path: str,
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add a code file read step."""
        step = (
            StepBuilder(step_id or self._next_step_id("code"))
            .from_template("coding_read", {"path": path})
            .build()
        )
        return self.add_step(step)

    def add_coding_analyze(
        self,
        path: str,
        task: str,
        step_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanBuilder:
        """Add a code analysis/editing step."""
        builder = (
            StepBuilder(step_id or self._next_step_id("code"))
            .from_template("coding_analyze", {"path": path, "task": task})
        )
        if depends_on:
            builder.depends_on(*depends_on)
        elif self._previous_group:
            builder.depends_on(*self._previous_group)
        return self.add_step(builder.build())

    def add_vision_analyze(
        self,
        image_path: str | None = None,
        prompt: str = "",
        step_id: str | None = None,
    ) -> PlanBuilder:
        """Add a vision analysis step (screenshot or image)."""
        params: dict[str, Any] = {"prompt": prompt}
        if image_path:
            params["image_path"] = image_path
        step = (
            StepBuilder(step_id or self._next_step_id("vision"))
            .from_template("vision_analyze", params)
            .build()
        )
        return self.add_step(step)

    def add_custom_step(
        self,
        capability: CapabilityType,
        action: str,
        parameters: dict[str, Any] | None = None,
        step_id: str | None = None,
        depends_on: list[str] | None = None,
        timeout: int = 30,
        fallback: str | None = None,
    ) -> PlanBuilder:
        """Add a custom step with full control."""
        builder = (
            StepBuilder(step_id or self._next_step_id("custom"))
            .with_capability(capability)
            .with_action(action)
            .with_timeout(timeout)
        )

        if parameters:
            builder.with_parameters(**parameters)

        if depends_on:
            builder.depends_on(*depends_on)

        if fallback:
            builder.with_fallback(fallback)

        return self.add_step(builder.build())

    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------

    def start_parallel_group(self) -> PlanBuilder:
        """
        Start a new parallel execution group.

        Steps added after this call will be in the same group
        and can execute in parallel.
        """
        if self._current_group:
            self._previous_group = self._current_group.copy()
        self._current_group = []
        return self

    def end_parallel_group(self) -> PlanBuilder:
        """
        End current parallel group.

        Subsequent steps will depend on all steps in this group.
        """
        self._previous_group = self._current_group.copy()
        self._current_group = []
        return self

    def add_barrier(self) -> PlanBuilder:
        """
        Add a synchronization barrier.

        All subsequent steps will depend on all previous steps.
        """
        self._previous_group = [s.step_id for s in self._steps]
        self._current_group = []
        return self

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def with_replan(self, allowed: bool = True) -> PlanBuilder:
        """Set whether replanning is allowed."""
        self._allows_replan = allowed
        return self

    def with_max_retries(self, retries: int) -> PlanBuilder:
        """Set maximum retries per step."""
        self._max_retries = retries
        return self

    def with_metadata(self, **metadata: Any) -> PlanBuilder:
        """Add plan metadata."""
        self._metadata.update(metadata)
        return self

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self) -> ExecutionPlan:
        """
        Build the execution plan.

        Returns:
            Configured ExecutionPlan

        Raises:
            ValueError: If plan is invalid
        """
        if not self._steps:
            raise ValueError("Plan must have at least one step")

        # Validate dependencies
        step_ids = {s.step_id for s in self._steps}
        for step in self._steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step {step.step_id} depends on unknown step: {dep}"
                    )

        plan = ExecutionPlan(
            plan_id=self._plan_id,
            steps=self._steps,
            allows_replan=self._allows_replan,
            max_retries=self._max_retries,
        )

        # Store metadata
        if self._metadata:
            # Metadata stored in first step's parameters for now
            # Could be extended to plan-level metadata
            pass

        return plan


# =============================================================================
# PLANNING STRATEGIES
# =============================================================================


class PlanningStrategyHandler(ABC):
    """Abstract base for planning strategy implementations."""

    @abstractmethod
    def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState,
    ) -> ExecutionPlan:
        """
        Create an execution plan.

        Args:
            request: User request
            intent: Intent analysis
            capabilities: Routing decision
            session: Session state

        Returns:
            ExecutionPlan
        """
        ...


class StandardPlanningStrategy(PlanningStrategyHandler):
    """
    Standard planning strategy with parallel optimization.

    Creates plans that:
    - Execute context gathering in parallel
    - Execute tools after context is available
    - Synthesize results at the end
    """

    def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState,
    ) -> ExecutionPlan:
        """Create a standard execution plan."""
        builder = PlanBuilder()

        # Phase 1: Context gathering (parallel)
        builder.start_parallel_group()

        if capabilities.use_memory:
            builder.add_memory_retrieval(query=request.content)

        if capabilities.use_rag:
            builder.add_rag_search(
                query=request.content,
                filters={"entities": intent.entities} if intent.entities else None,
            )

        if capabilities.use_web_search:
            builder.add_web_search(query=request.content)

        builder.end_parallel_group()

        # Phase 2: File operations
        if capabilities.use_files:
            builder.add_file_search(query=request.content)

        # Phase 3: Coding operations
        if capabilities.use_coding:
            builder.add_coding_analyze(
                path=intent.entities.get("file_path", ""),
                task=request.content,
            )

        # Phase 4: Vision operations
        if capabilities.use_vision:
            builder.add_vision_analyze(
                image_path=intent.entities.get("image_path"),
                prompt=request.content,
            )

        # Phase 5: Computer execution (OS commands / browser / desktop)
        if capabilities.use_computer:
            builder.add_computer_execution(command=request.content)

        # Phase 6: Tool execution (sequential, depends on context)
        for tool_name in capabilities.tools_to_use:
            tool_params = intent.entities.get("tool_params", {}).get(tool_name, {})
            builder.add_tool_execution(
                tool_name=tool_name,
                parameters=tool_params,
            )

        # Phase 7: Synthesis
        builder.add_synthesis(
            request_content=request.content,
            intent_type=intent.intent_type.value,
        )

        # Configure based on complexity
        builder.with_replan(intent.complexity_score <= 7)
        builder.with_max_retries(2 if intent.complexity_score <= 5 else 1)

        return builder.build()


class MinimalPlanningStrategy(PlanningStrategyHandler):
    """
    Minimal planning strategy for simple requests.

    Creates lightweight plans with fewer steps for
    low-complexity requests.
    """

    def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState,
    ) -> ExecutionPlan:
        """Create a minimal execution plan."""
        builder = PlanBuilder()

        # Only add essential steps
        if capabilities.use_memory and session.has_history:
            builder.add_memory_retrieval(query=request.content)

        if capabilities.use_rag:
            builder.add_rag_search(query=request.content)

        # Skip web search for minimal plans unless explicitly needed
        if capabilities.use_web_search and intent.requires_current_info:
            builder.add_web_search(query=request.content, max_results=3)

        # File / coding / vision (minimal — single step per type)
        if capabilities.use_files:
            builder.add_file_search(query=request.content)

        if capabilities.use_coding:
            builder.add_coding_analyze(
                path=intent.entities.get("file_path", ""),
                task=request.content,
            )

        if capabilities.use_vision:
            builder.add_vision_analyze(prompt=request.content)

        # Computer execution (if needed)
        if capabilities.use_computer:
            builder.add_computer_execution(command=request.content)

        # Single tool only
        if capabilities.tools_to_use:
            builder.add_tool_execution(
                tool_name=capabilities.tools_to_use[0],
            )

        # Always synthesize
        builder.add_synthesis(
            request_content=request.content,
            intent_type=intent.intent_type.value,
        )

        builder.with_replan(False)
        builder.with_max_retries(1)

        return builder.build()


class ResilientPlanningStrategy(PlanningStrategyHandler):
    """
    Resilient planning strategy for critical requests.

    Creates plans with fallbacks and redundancy for
    high-reliability requirements.
    """

    def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState,
    ) -> ExecutionPlan:
        """Create a resilient execution plan."""
        builder = PlanBuilder()

        # Context gathering with fallbacks
        builder.start_parallel_group()

        if capabilities.use_memory:
            builder.add_custom_step(
                capability=CapabilityType.MEMORY,
                action="retrieve",
                parameters={"query": request.content},
                fallback="skip",
                timeout=15,
            )

        if capabilities.use_rag:
            builder.add_custom_step(
                capability=CapabilityType.RAG,
                action="search",
                parameters={"query": request.content},
                fallback="skip",
                timeout=45,
            )

        if capabilities.use_web_search:
            builder.add_custom_step(
                capability=CapabilityType.WEB_SEARCH,
                action="search",
                parameters={"query": request.content, "max_results": 10},
                fallback="skip",
                timeout=60,
            )

        builder.end_parallel_group()

        # File operations
        if capabilities.use_files:
            builder.add_custom_step(
                capability=CapabilityType.FILES,
                action="search",
                parameters={"query": request.content},
                fallback="skip",
                timeout=30,
            )

        # Coding operations
        if capabilities.use_coding:
            builder.add_custom_step(
                capability=CapabilityType.CODING,
                action="analyze",
                parameters={
                    "path": intent.entities.get("file_path", ""),
                    "task": request.content,
                },
                fallback="skip",
                timeout=60,
            )

        # Vision operations
        if capabilities.use_vision:
            builder.add_custom_step(
                capability=CapabilityType.VISION,
                action="analyze",
                parameters={"prompt": request.content},
                fallback="skip",
                timeout=60,
            )

        # Computer execution
        if capabilities.use_computer:
            builder.add_custom_step(
                capability=CapabilityType.COMPUTER,
                action="terminal_execute",
                parameters={"command": request.content},
                fallback="skip",
                timeout=90,
            )

        # Tools with individual fallbacks
        for tool_name in capabilities.tools_to_use:
            builder.add_custom_step(
                capability=CapabilityType.TOOLS,
                action="execute",
                parameters={"tool_name": tool_name},
                fallback="skip",
                timeout=90,
            )

        # Synthesis
        builder.add_synthesis(
            request_content=request.content,
            intent_type=intent.intent_type.value,
        )

        builder.with_replan(True)
        builder.with_max_retries(3)

        return builder.build()


# =============================================================================
# MULTI-AGENT SUPPORT (FUTURE)
# =============================================================================


@dataclass
class AgentDefinition:
    """
    Definition of an agent for multi-agent workflows.

    Attributes:
        agent_id: Unique agent identifier
        name: Human-readable name
        capabilities: Capabilities this agent provides
        specialization: What this agent specializes in
        priority: Selection priority
    """

    agent_id: str
    name: str
    capabilities: set[CapabilityType]
    specialization: str = ""
    priority: int = 0


@dataclass
class AgentHandoffStep:
    """
    Step representing handoff to another agent.

    Attributes:
        step_id: Step identifier
        target_agent_id: Agent to hand off to
        task_description: What the agent should do
        context: Context to pass to agent
        await_result: Whether to wait for result
    """

    step_id: str
    target_agent_id: str
    task_description: str
    context: dict[str, Any] = field(default_factory=dict)
    await_result: bool = True

    def to_plan_step(self) -> PlanStep:
        """Convert to standard PlanStep."""
        return PlanStep(
            step_id=self.step_id,
            capability=CapabilityType.TOOLS,  # Agents are invoked via tools
            action="agent_handoff",
            parameters={
                "target_agent_id": self.target_agent_id,
                "task_description": self.task_description,
                "context": self.context,
                "await_result": self.await_result,
            },
            timeout_seconds=300,  # Longer timeout for agent tasks
        )


class MultiAgentPlanner:
    """
    Planner extension for multi-agent workflows.

    Handles agent selection, task decomposition, and
    coordination between multiple agents.

    Note: This is a foundation for future multi-agent support.
    """

    def __init__(
        self,
        agents: list[AgentDefinition] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize multi-agent planner.

        Args:
            agents: Available agent definitions
            logger: Optional logger
        """
        self._agents = {a.agent_id: a for a in (agents or [])}
        self._logger = logger or logging.getLogger(__name__)

    def register_agent(self, agent: AgentDefinition) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent

    def select_agent(
        self,
        required_capabilities: set[CapabilityType],
        specialization_hint: str = "",
    ) -> AgentDefinition | None:
        """
        Select best agent for given requirements.

        Args:
            required_capabilities: Capabilities needed
            specialization_hint: Preferred specialization

        Returns:
            Best matching agent or None
        """
        candidates: list[tuple[AgentDefinition, int]] = []

        for agent in self._agents.values():
            # Check capability coverage
            if not required_capabilities.issubset(agent.capabilities):
                continue

            # Score based on specialization match and priority
            score = agent.priority
            if specialization_hint and specialization_hint in agent.specialization:
                score += 10

            candidates.append((agent, score))

        if not candidates:
            return None

        # Return highest scoring agent
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def create_handoff_step(
        self,
        target_agent_id: str,
        task: str,
        context: dict[str, Any] | None = None,
        step_id: str | None = None,
    ) -> PlanStep:
        """
        Create a step that hands off to another agent.

        Args:
            target_agent_id: Agent to hand off to
            task: Task description
            context: Context to pass
            step_id: Optional step ID

        Returns:
            PlanStep for agent handoff
        """
        handoff = AgentHandoffStep(
            step_id=step_id or f"handoff_{uuid.uuid4().hex[:8]}",
            target_agent_id=target_agent_id,
            task_description=task,
            context=context or {},
        )
        return handoff.to_plan_step()

    def decompose_for_agents(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
    ) -> list[tuple[AgentDefinition, str]]:
        """
        Decompose request into agent tasks.

        Args:
            request: User request
            intent: Intent analysis

        Returns:
            List of (agent, task_description) tuples
        """
        # Placeholder for future implementation
        # Would use LLM to decompose complex tasks
        return []


# =============================================================================
# MAIN PLANNER
# =============================================================================


@dataclass
class PlannerConfig:
    """
    Configuration for the Planner.

    Attributes:
        default_strategy: Default planning strategy
        complexity_threshold_minimal: Max complexity for minimal strategy
        complexity_threshold_resilient: Min complexity for resilient strategy
        enable_multi_agent: Enable multi-agent planning
        max_steps: Maximum steps in a plan
    """

    default_strategy: PlanningStrategy = PlanningStrategy.PARALLEL_OPTIMIZED
    complexity_threshold_minimal: int = 3
    complexity_threshold_resilient: int = 8
    enable_multi_agent: bool = False
    max_steps: int = 20


class Planner:
    """
    Generates execution plans for agent requests.

    The planner analyzes requests and creates optimized execution
    plans that orchestrate capability invocations.

    This is the main entry point for plan generation.
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        multi_agent_planner: MultiAgentPlanner | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            config: Planner configuration
            multi_agent_planner: Optional multi-agent planner
            logger: Optional logger
        """
        self._config = config or PlannerConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._multi_agent = multi_agent_planner

        # Initialize strategies
        self._strategies: dict[PlanningStrategy, PlanningStrategyHandler] = {
            PlanningStrategy.PARALLEL_OPTIMIZED: StandardPlanningStrategy(),
            PlanningStrategy.SEQUENTIAL: StandardPlanningStrategy(),  # Same for now
            PlanningStrategy.FAIL_FAST: MinimalPlanningStrategy(),
            PlanningStrategy.RESILIENT: ResilientPlanningStrategy(),
        }

    # -------------------------------------------------------------------------
    # Plan Creation
    # -------------------------------------------------------------------------

    async def create_plan(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState | None = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for the request.

        Args:
            request: User request
            intent: Intent analysis result
            capabilities: Capability routing decision
            session: Optional session state

        Returns:
            ExecutionPlan ready for execution

        Raises:
            PlanningError: If plan generation fails
        """
        session = session or SessionState.create()

        # Select strategy based on complexity
        strategy = self._select_strategy(intent)

        self._logger.debug(
            f"Creating plan with strategy: {strategy.name}",
            extra={
                "intent_type": intent.intent_type.value,
                "complexity": intent.complexity_score,
            },
        )

        # Generate plan
        handler = self._strategies[strategy]
        plan = handler.create_plan(request, intent, capabilities, session)

        # Validate plan
        self._validate_plan(plan)

        self._logger.info(
            f"Plan created: {plan.plan_id}",
            extra={
                "step_count": plan.step_count,
                "strategy": strategy.name,
            },
        )

        return plan

    def create_plan_sync(
        self,
        request: AgentRequest,
        intent: IntentAnalysis,
        capabilities: CapabilityDecision,
        session: SessionState | None = None,
    ) -> ExecutionPlan:
        """
        Synchronous version of create_plan.

        For use in non-async contexts.
        """
        session = session or SessionState.create()
        strategy = self._select_strategy(intent)
        handler = self._strategies[strategy]
        plan = handler.create_plan(request, intent, capabilities, session)
        self._validate_plan(plan)
        return plan

    # -------------------------------------------------------------------------
    # Replanning
    # -------------------------------------------------------------------------

    async def replan(
        self,
        original_plan: ExecutionPlan,
        failed_step: PlanStep,
        error: Exception,
        completed_results: list[StepResult],
    ) -> ExecutionPlan:
        """
        Create a new plan after step failure.

        Args:
            original_plan: The plan that failed
            failed_step: The step that failed
            error: The error that occurred
            completed_results: Results from completed steps

        Returns:
            New ExecutionPlan with adjusted steps
        """
        self._logger.info(
            f"Replanning after failure in step {failed_step.step_id}",
            extra={"error": str(error)},
        )

        # Get completed step IDs
        completed_ids = {r.step_id for r in completed_results if r.success}

        # Filter remaining steps
        remaining_steps: list[PlanStep] = []
        for step in original_plan.steps:
            # Skip completed and failed steps
            if step.step_id in completed_ids or step.step_id == failed_step.step_id:
                continue

            # Skip steps that depend on failed step
            if failed_step.step_id in step.dependencies:
                self._logger.debug(f"Skipping step {step.step_id} (depends on failed)")
                continue

            # Update dependencies to remove failed step
            new_deps = [d for d in step.dependencies if d != failed_step.step_id]

            # Create new step with updated dependencies
            new_step = PlanStep(
                step_id=step.step_id,
                capability=step.capability,
                action=step.action,
                parameters=step.parameters,
                dependencies=new_deps,
                fallback_action=step.fallback_action,
                timeout_seconds=step.timeout_seconds,
            )
            remaining_steps.append(new_step)

        if not remaining_steps:
            # No steps remaining, create minimal synthesis plan
            remaining_steps = [
                StepBuilder(f"synthesis_recovery_{uuid.uuid4().hex[:8]}")
                .from_template("synthesize", {
                    "request": "Generate response from available context",
                    "intent": "recovery",
                    "_recovery_mode": True,
                    "_completed_results": [r.step_id for r in completed_results],
                })
                .build()
            ]

        new_plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            steps=remaining_steps,
            allows_replan=False,  # Only one replan attempt
            max_retries=1,
        )

        self._logger.info(
            f"Replan created: {new_plan.plan_id}",
            extra={"remaining_steps": len(remaining_steps)},
        )

        return new_plan

    # -------------------------------------------------------------------------
    # Plan Modification
    # -------------------------------------------------------------------------

    def add_step_to_plan(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        after_step_id: str | None = None,
    ) -> ExecutionPlan:
        """
        Add a step to an existing plan.

        Args:
            plan: Plan to modify
            step: Step to add
            after_step_id: Insert after this step (appends if None)

        Returns:
            Modified plan
        """
        new_steps = list(plan.steps)

        if after_step_id:
            # Find insertion point
            idx = next(
                (i for i, s in enumerate(new_steps) if s.step_id == after_step_id),
                None,
            )
            if idx is not None:
                new_steps.insert(idx + 1, step)
            else:
                new_steps.append(step)
        else:
            new_steps.append(step)

        return ExecutionPlan(
            plan_id=plan.plan_id,
            steps=new_steps,
            status=plan.status,
            allows_replan=plan.allows_replan,
            max_retries=plan.max_retries,
            created_at=plan.created_at,
        )

    def remove_step_from_plan(
        self,
        plan: ExecutionPlan,
        step_id: str,
    ) -> ExecutionPlan:
        """
        Remove a step from a plan.

        Args:
            plan: Plan to modify
            step_id: Step to remove

        Returns:
            Modified plan
        """
        new_steps = [s for s in plan.steps if s.step_id != step_id]

        # Update dependencies
        for step in new_steps:
            step.dependencies = [d for d in step.dependencies if d != step_id]

        return ExecutionPlan(
            plan_id=plan.plan_id,
            steps=new_steps,
            status=plan.status,
            allows_replan=plan.allows_replan,
            max_retries=plan.max_retries,
            created_at=plan.created_at,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _select_strategy(self, intent: IntentAnalysis) -> PlanningStrategy:
        """Select planning strategy based on intent."""
        if intent.complexity_score <= self._config.complexity_threshold_minimal:
            return PlanningStrategy.FAIL_FAST

        if intent.complexity_score >= self._config.complexity_threshold_resilient:
            return PlanningStrategy.RESILIENT

        return self._config.default_strategy

    def _validate_plan(self, plan: ExecutionPlan) -> None:
        """
        Validate plan structure.

        Raises:
            ValueError: If plan is invalid
        """
        if plan.step_count > self._config.max_steps:
            raise ValueError(
                f"Plan exceeds maximum steps: {plan.step_count} > {self._config.max_steps}"
            )

        # Check for cycles (will raise if found)
        try:
            plan.ordered_steps()
        except ValueError as e:
            raise ValueError(f"Invalid plan structure: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get planner statistics."""
        return {
            "default_strategy": self._config.default_strategy.name,
            "complexity_threshold_minimal": self._config.complexity_threshold_minimal,
            "complexity_threshold_resilient": self._config.complexity_threshold_resilient,
            "max_steps": self._config.max_steps,
            "multi_agent_enabled": self._config.enable_multi_agent,
            "available_strategies": [s.name for s in self._strategies.keys()],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_planner(
    config: PlannerConfig | None = None,
    agents: list[AgentDefinition] | None = None,
    logger: logging.Logger | None = None,
) -> Planner:
    """
    Factory function to create a configured Planner.

    Args:
        config: Planner configuration
        agents: Agent definitions for multi-agent support
        logger: Optional logger

    Returns:
        Configured Planner instance
    """
    multi_agent = None
    if agents:
        multi_agent = MultiAgentPlanner(agents=agents, logger=logger)

    return Planner(
        config=config,
        multi_agent_planner=multi_agent,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StepType",
    "PlanningStrategy",
    # Templates
    "StepTemplate",
    "STEP_TEMPLATES",
    # Builders
    "StepBuilder",
    "PlanBuilder",
    # Strategies
    "PlanningStrategyHandler",
    "StandardPlanningStrategy",
    "MinimalPlanningStrategy",
    "ResilientPlanningStrategy",
    # Multi-agent
    "AgentDefinition",
    "AgentHandoffStep",
    "MultiAgentPlanner",
    # Main
    "PlannerConfig",
    "Planner",
    # Factory
    "create_planner",
]
