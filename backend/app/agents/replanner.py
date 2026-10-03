"""
Replanner - Adaptive plan modification for the AI Agent Platform.

This module provides comprehensive replanning capabilities including:
- Failure detection and classification
- Plan modification and adaptation
- Fallback plan generation
- Recovery strategies
- Multi-agent workflow coordination
- Learning from failures
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)
from uuid import uuid4

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class FailureType(Enum):
    """Types of failures that can trigger replanning."""

    # Execution failures
    STEP_FAILED = "step_failed"
    STEP_TIMEOUT = "step_timeout"
    STEP_REJECTED = "step_rejected"
    DEPENDENCY_FAILED = "dependency_failed"

    # Resource failures
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"

    # Tool failures
    TOOL_ERROR = "tool_error"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_TIMEOUT = "tool_timeout"

    # Agent failures
    AGENT_ERROR = "agent_error"
    AGENT_UNAVAILABLE = "agent_unavailable"
    AGENT_OVERLOADED = "agent_overloaded"

    # External failures
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Validation failures
    VALIDATION_ERROR = "validation_error"
    PRECONDITION_FAILED = "precondition_failed"
    POSTCONDITION_FAILED = "postcondition_failed"

    # Other
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


class FailureSeverity(Enum):
    """Severity levels for failures."""

    LOW = auto()  # Minor issue, can continue
    MEDIUM = auto()  # Significant issue, may need adjustment
    HIGH = auto()  # Major issue, likely needs replanning
    CRITICAL = auto()  # Severe issue, plan cannot continue
    FATAL = auto()  # Unrecoverable, must abort


class ReplanStrategy(Enum):
    """Strategies for replanning."""

    RETRY = "retry"  # Retry the failed step
    SKIP = "skip"  # Skip the failed step
    SUBSTITUTE = "substitute"  # Use alternative step
    ROLLBACK = "rollback"  # Rollback and try different path
    DECOMPOSE = "decompose"  # Break step into smaller steps
    DELEGATE = "delegate"  # Delegate to another agent
    ESCALATE = "escalate"  # Escalate to human
    ABORT = "abort"  # Abort the plan
    FULL_REPLAN = "full_replan"  # Generate entirely new plan


class ReplanStatus(Enum):
    """Status of a replan operation."""

    PENDING = auto()
    ANALYZING = auto()
    PLANNING = auto()
    READY = auto()
    APPLIED = auto()
    FAILED = auto()
    REJECTED = auto()


class RecoveryMode(Enum):
    """Modes for recovery behavior."""

    AGGRESSIVE = auto()  # Try multiple strategies quickly
    CONSERVATIVE = auto()  # Careful, minimal changes
    BALANCED = auto()  # Balance between speed and safety
    INTERACTIVE = auto()  # Require human approval


# =============================================================================
# FAILURE DETECTION
# =============================================================================


@dataclass
class FailureContext:
    """
    Context information about a failure.

    Attributes:
        failure_id: Unique failure identifier
        failure_type: Type of failure
        severity: Failure severity
        step_id: Failed step identifier
        step_index: Index in plan
        error_message: Error description
        error_code: Error code if available
        timestamp: When failure occurred
        retry_count: Number of retries attempted
        stack_trace: Error stack trace
        affected_steps: Steps affected by this failure
        metadata: Additional failure data
    """

    failure_id: str
    failure_type: FailureType
    severity: FailureSeverity
    step_id: str
    step_index: int
    error_message: str
    error_code: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    stack_trace: str | None = None
    affected_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_retryable(self) -> bool:
        """Check if failure is potentially retryable."""
        non_retryable = {
            FailureType.VALIDATION_ERROR,
            FailureType.PRECONDITION_FAILED,
            FailureType.CANCELLED,
            FailureType.BUDGET_EXCEEDED,
            FailureType.QUOTA_EXHAUSTED,
        }
        return self.failure_type not in non_retryable

    @property
    def is_transient(self) -> bool:
        """Check if failure is likely transient."""
        transient = {
            FailureType.STEP_TIMEOUT,
            FailureType.RATE_LIMITED,
            FailureType.NETWORK_ERROR,
            FailureType.SERVICE_UNAVAILABLE,
            FailureType.AGENT_OVERLOADED,
        }
        return self.failure_type in transient

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_id": self.failure_id,
            "failure_type": self.failure_type.value,
            "severity": self.severity.name,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "is_retryable": self.is_retryable,
            "is_transient": self.is_transient,
            "affected_steps": self.affected_steps,
            "metadata": self.metadata,
        }


class FailureDetector:
    """
    Detects and classifies failures from execution results.

    Analyzes errors, exceptions, and results to determine
    failure type and severity.
    """

    # Error patterns for classification
    ERROR_PATTERNS: dict[str, FailureType] = {
        "timeout": FailureType.STEP_TIMEOUT,
        "timed out": FailureType.STEP_TIMEOUT,
        "rate limit": FailureType.RATE_LIMITED,
        "too many requests": FailureType.RATE_LIMITED,
        "429": FailureType.RATE_LIMITED,
        "quota": FailureType.QUOTA_EXHAUSTED,
        "budget": FailureType.BUDGET_EXCEEDED,
        "network": FailureType.NETWORK_ERROR,
        "connection": FailureType.NETWORK_ERROR,
        "dns": FailureType.NETWORK_ERROR,
        "unavailable": FailureType.SERVICE_UNAVAILABLE,
        "503": FailureType.SERVICE_UNAVAILABLE,
        "502": FailureType.SERVICE_UNAVAILABLE,
        "not found": FailureType.RESOURCE_UNAVAILABLE,
        "404": FailureType.RESOURCE_UNAVAILABLE,
        "validation": FailureType.VALIDATION_ERROR,
        "invalid": FailureType.VALIDATION_ERROR,
        "permission": FailureType.STEP_REJECTED,
        "denied": FailureType.STEP_REJECTED,
        "unauthorized": FailureType.STEP_REJECTED,
        "forbidden": FailureType.STEP_REJECTED,
        "cancelled": FailureType.CANCELLED,
        "aborted": FailureType.CANCELLED,
    }

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def detect(
        self,
        step_id: str,
        step_index: int,
        error: Exception | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        retry_count: int = 0,
    ) -> FailureContext:
        """
        Detect and classify a failure.

        Args:
            step_id: Step identifier
            step_index: Step index in plan
            error: Exception if available
            error_message: Error message
            result: Step result if available
            retry_count: Number of retries

        Returns:
            FailureContext with classification
        """
        # Extract error message
        if error_message is None and error:
            error_message = str(error)
        error_message = error_message or "Unknown error"

        # Classify failure type
        failure_type = self._classify_type(error_message, error, result)

        # Determine severity
        severity = self._classify_severity(failure_type, retry_count, result)

        # Extract error code
        error_code = self._extract_error_code(error, result)

        # Get stack trace
        stack_trace = None
        if error:
            import traceback
            stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))

        return FailureContext(
            failure_id=str(uuid4()),
            failure_type=failure_type,
            severity=severity,
            step_id=step_id,
            step_index=step_index,
            error_message=error_message,
            error_code=error_code,
            retry_count=retry_count,
            stack_trace=stack_trace,
            metadata={
                "error_type": type(error).__name__ if error else None,
                "result": result,
            },
        )

    def _classify_type(
        self,
        error_message: str,
        error: Exception | None,
        result: dict[str, Any] | None,
    ) -> FailureType:
        """Classify failure type from error information."""
        error_lower = error_message.lower()

        # Check patterns
        for pattern, failure_type in self.ERROR_PATTERNS.items():
            if pattern in error_lower:
                return failure_type

        # Check exception types
        if error:
            error_type = type(error).__name__.lower()

            if "timeout" in error_type:
                return FailureType.STEP_TIMEOUT
            if "connection" in error_type:
                return FailureType.NETWORK_ERROR
            if "validation" in error_type:
                return FailureType.VALIDATION_ERROR
            if "permission" in error_type or "auth" in error_type:
                return FailureType.STEP_REJECTED

        # Check result
        if result:
            status = result.get("status", "").lower()
            if status in {"timeout", "timed_out"}:
                return FailureType.STEP_TIMEOUT
            if status in {"rejected", "denied"}:
                return FailureType.STEP_REJECTED

        return FailureType.STEP_FAILED

    def _classify_severity(
        self,
        failure_type: FailureType,
        retry_count: int,
        result: dict[str, Any] | None,
    ) -> FailureSeverity:
        """Classify failure severity."""
        # Critical failures
        critical_types = {
            FailureType.BUDGET_EXCEEDED,
            FailureType.QUOTA_EXHAUSTED,
            FailureType.CANCELLED,
        }
        if failure_type in critical_types:
            return FailureSeverity.CRITICAL

        # High severity after multiple retries
        if retry_count >= 3:
            return FailureSeverity.HIGH

        # Medium severity failures
        medium_types = {
            FailureType.VALIDATION_ERROR,
            FailureType.PRECONDITION_FAILED,
            FailureType.TOOL_UNAVAILABLE,
            FailureType.AGENT_UNAVAILABLE,
        }
        if failure_type in medium_types:
            return FailureSeverity.MEDIUM

        # Low severity for transient failures
        transient_types = {
            FailureType.STEP_TIMEOUT,
            FailureType.RATE_LIMITED,
            FailureType.NETWORK_ERROR,
            FailureType.SERVICE_UNAVAILABLE,
        }
        if failure_type in transient_types and retry_count < 2:
            return FailureSeverity.LOW

        return FailureSeverity.MEDIUM

    def _extract_error_code(
        self,
        error: Exception | None,
        result: dict[str, Any] | None,
    ) -> str | None:
        """Extract error code from error or result."""
        if result:
            for key in ["error_code", "code", "status_code"]:
                if key in result:
                    return str(result[key])

        if error and hasattr(error, "code"):
            return str(error.code)

        return None


# =============================================================================
# PLAN STRUCTURES
# =============================================================================


@dataclass
class PlanStep:
    """
    Represents a step in an execution plan.

    Attributes:
        step_id: Unique step identifier
        action: Action to perform
        parameters: Action parameters
        dependencies: IDs of dependent steps
        agent_id: Assigned agent
        timeout_seconds: Step timeout
        retries: Maximum retries
        fallback_step_id: Fallback step if this fails
        metadata: Additional step data
    """

    step_id: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    agent_id: str | None = None
    timeout_seconds: float = 60.0
    retries: int = 3
    fallback_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, new_id: str | None = None) -> PlanStep:
        """Create a copy of this step."""
        return PlanStep(
            step_id=new_id or str(uuid4()),
            action=self.action,
            parameters=dict(self.parameters),
            dependencies=list(self.dependencies),
            agent_id=self.agent_id,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            fallback_step_id=self.fallback_step_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "agent_id": self.agent_id,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "fallback_step_id": self.fallback_step_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            action=data["action"],
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            agent_id=data.get("agent_id"),
            timeout_seconds=data.get("timeout_seconds", 60.0),
            retries=data.get("retries", 3),
            fallback_step_id=data.get("fallback_step_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionPlan:
    """
    Represents an execution plan.

    Attributes:
        plan_id: Unique plan identifier
        goal: Plan goal description
        steps: List of plan steps
        version: Plan version
        created_at: Creation timestamp
        parent_plan_id: Parent plan if this is a replan
        metadata: Additional plan data
    """

    plan_id: str
    goal: str
    steps: list[PlanStep]
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_plan_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> PlanStep | None:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> int:
        """Get step index by ID."""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                return i
        return -1

    def get_dependent_steps(self, step_id: str) -> list[PlanStep]:
        """Get steps that depend on the given step."""
        return [s for s in self.steps if step_id in s.dependencies]

    def get_remaining_steps(self, from_index: int) -> list[PlanStep]:
        """Get steps from index onwards."""
        return self.steps[from_index:]

    def clone(self, new_id: str | None = None) -> ExecutionPlan:
        """Create a copy of this plan."""
        return ExecutionPlan(
            plan_id=new_id or str(uuid4()),
            goal=self.goal,
            steps=[s.clone() for s in self.steps],
            version=self.version + 1,
            parent_plan_id=self.plan_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "parent_plan_id": self.parent_plan_id,
            "metadata": self.metadata,
        }


# =============================================================================
# REPLAN RESULT
# =============================================================================


@dataclass
class ReplanResult:
    """
    Result of a replanning operation.

    Attributes:
        replan_id: Unique replan identifier
        status: Replan status
        strategy: Strategy used
        original_plan: Original plan
        modified_plan: Modified plan
        failure_context: Failure that triggered replan
        changes: Description of changes made
        confidence: Confidence in the replan (0-1)
        reasoning: Explanation of replan decisions
        alternatives: Alternative strategies considered
        created_at: When replan was created
        metadata: Additional data
    """

    replan_id: str
    status: ReplanStatus
    strategy: ReplanStrategy
    original_plan: ExecutionPlan
    modified_plan: ExecutionPlan | None
    failure_context: FailureContext
    changes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: list[ReplanStrategy] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if replan was successful."""
        return self.status == ReplanStatus.READY and self.modified_plan is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "replan_id": self.replan_id,
            "status": self.status.name,
            "strategy": self.strategy.value,
            "changes": self.changes,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [s.value for s in self.alternatives],
            "failure": self.failure_context.to_dict(),
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================


@dataclass
class StrategyRule:
    """Rule for selecting a replan strategy."""

    failure_types: set[FailureType]
    max_severity: FailureSeverity
    max_retries: int
    strategy: ReplanStrategy
    priority: int = 0
    conditions: dict[str, Any] = field(default_factory=dict)


class StrategySelector:
    """
    Selects appropriate replan strategy based on failure context.

    Uses rules and heuristics to determine the best strategy.
    """

    DEFAULT_RULES: list[StrategyRule] = [
        # Retry for transient failures
        StrategyRule(
            failure_types={
                FailureType.STEP_TIMEOUT,
                FailureType.RATE_LIMITED,
                FailureType.NETWORK_ERROR,
                FailureType.SERVICE_UNAVAILABLE,
            },
            max_severity=FailureSeverity.MEDIUM,
            max_retries=3,
            strategy=ReplanStrategy.RETRY,
            priority=100,
        ),
        # Skip for non-critical failures
        StrategyRule(
            failure_types={
                FailureType.RESOURCE_UNAVAILABLE,
                FailureType.TOOL_UNAVAILABLE,
            },
            max_severity=FailureSeverity.LOW,
            max_retries=5,
            strategy=ReplanStrategy.SKIP,
            priority=80,
            conditions={"step_optional": True},
        ),
        # Substitute for tool failures
        StrategyRule(
            failure_types={
                FailureType.TOOL_ERROR,
                FailureType.TOOL_UNAVAILABLE,
                FailureType.TOOL_TIMEOUT,
            },
            max_severity=FailureSeverity.HIGH,
            max_retries=5,
            strategy=ReplanStrategy.SUBSTITUTE,
            priority=70,
        ),
        # Delegate for agent failures
        StrategyRule(
            failure_types={
                FailureType.AGENT_ERROR,
                FailureType.AGENT_UNAVAILABLE,
                FailureType.AGENT_OVERLOADED,
            },
            max_severity=FailureSeverity.HIGH,
            max_retries=5,
            strategy=ReplanStrategy.DELEGATE,
            priority=60,
        ),
        # Decompose for complex failures
        StrategyRule(
            failure_types={
                FailureType.STEP_FAILED,
                FailureType.VALIDATION_ERROR,
            },
            max_severity=FailureSeverity.MEDIUM,
            max_retries=2,
            strategy=ReplanStrategy.DECOMPOSE,
            priority=50,
        ),
        # Rollback for dependency failures
        StrategyRule(
            failure_types={
                FailureType.DEPENDENCY_FAILED,
                FailureType.PRECONDITION_FAILED,
            },
            max_severity=FailureSeverity.HIGH,
            max_retries=5,
            strategy=ReplanStrategy.ROLLBACK,
            priority=40,
        ),
        # Escalate for critical failures
        StrategyRule(
            failure_types={
                FailureType.STEP_REJECTED,
                FailureType.BUDGET_EXCEEDED,
            },
            max_severity=FailureSeverity.CRITICAL,
            max_retries=10,
            strategy=ReplanStrategy.ESCALATE,
            priority=30,
        ),
        # Full replan for severe failures
        StrategyRule(
            failure_types={
                FailureType.STEP_FAILED,
                FailureType.POSTCONDITION_FAILED,
            },
            max_severity=FailureSeverity.HIGH,
            max_retries=1,
            strategy=ReplanStrategy.FULL_REPLAN,
            priority=20,
        ),
        # Abort for fatal failures
        StrategyRule(
            failure_types={
                FailureType.CANCELLED,
                FailureType.QUOTA_EXHAUSTED,
            },
            max_severity=FailureSeverity.FATAL,
            max_retries=10,
            strategy=ReplanStrategy.ABORT,
            priority=10,
        ),
    ]

    def __init__(
        self,
        rules: list[StrategyRule] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._rules = rules or list(self.DEFAULT_RULES)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        self._logger = logger or logging.getLogger(__name__)

    def select(
        self,
        failure: FailureContext,
        plan: ExecutionPlan,
        step: PlanStep | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[ReplanStrategy, list[ReplanStrategy]]:
        """
        Select best strategy for failure.

        Args:
            failure: Failure context
            plan: Current plan
            step: Failed step
            context: Additional context

        Returns:
            Tuple of (selected strategy, alternative strategies)
        """
        context = context or {}
        candidates: list[tuple[ReplanStrategy, int]] = []

        severity_order = [
            FailureSeverity.LOW,
            FailureSeverity.MEDIUM,
            FailureSeverity.HIGH,
            FailureSeverity.CRITICAL,
            FailureSeverity.FATAL,
        ]

        for rule in self._rules:
            # Check failure type
            if failure.failure_type not in rule.failure_types:
                continue

            # Check severity
            if severity_order.index(failure.severity) > severity_order.index(rule.max_severity):
                continue

            # Check retry count
            if failure.retry_count > rule.max_retries:
                continue

            # Check conditions
            conditions_met = True
            for key, value in rule.conditions.items():
                if context.get(key) != value:
                    if step and step.metadata.get(key) != value:
                        conditions_met = False
                        break

            if conditions_met:
                candidates.append((rule.strategy, rule.priority))

        if not candidates:
            # Default fallback
            if failure.severity == FailureSeverity.FATAL:
                return ReplanStrategy.ABORT, []
            elif failure.severity == FailureSeverity.CRITICAL:
                return ReplanStrategy.ESCALATE, [ReplanStrategy.ABORT]
            else:
                return ReplanStrategy.RETRY, [ReplanStrategy.SKIP, ReplanStrategy.FULL_REPLAN]

        # Sort by priority
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = candidates[0][0]
        alternatives = [c[0] for c in candidates[1:4]]  # Top 3 alternatives

        return selected, alternatives

    def add_rule(self, rule: StrategyRule) -> None:
        """Add a custom rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)


# =============================================================================
# PLAN MODIFIER
# =============================================================================


class PlanModifier:
    """
    Modifies execution plans based on replan strategies.

    Implements various plan modification operations.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def apply_retry(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        max_retries: int = 3,
    ) -> ExecutionPlan:
        """
        Modify plan to retry failed step.

        Args:
            plan: Original plan
            failure: Failure context
            max_retries: Maximum retries

        Returns:
            Modified plan
        """
        modified = plan.clone()
        step = modified.get_step(failure.step_id)

        if step:
            step.retries = max(step.retries, max_retries)
            step.metadata["retry_after_failure"] = failure.failure_id
            step.metadata["retry_count"] = failure.retry_count + 1

            # Increase timeout for timeout failures
            if failure.failure_type == FailureType.STEP_TIMEOUT:
                step.timeout_seconds *= 1.5

        return modified

    def apply_skip(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
    ) -> ExecutionPlan:
        """
        Modify plan to skip failed step.

        Args:
            plan: Original plan
            failure: Failure context

        Returns:
            Modified plan with step removed
        """
        modified = plan.clone()

        # Remove the failed step
        modified.steps = [s for s in modified.steps if s.step_id != failure.step_id]

        # Update dependencies
        for step in modified.steps:
            if failure.step_id in step.dependencies:
                step.dependencies.remove(failure.step_id)

        modified.metadata["skipped_steps"] = modified.metadata.get("skipped_steps", []) + [failure.step_id]

        return modified

    def apply_substitute(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        substitute_step: PlanStep,
    ) -> ExecutionPlan:
        """
        Substitute failed step with alternative.

        Args:
            plan: Original plan
            failure: Failure context
            substitute_step: Replacement step

        Returns:
            Modified plan with substitution
        """
        modified = plan.clone()
        step_index = modified.get_step_index(failure.step_id)

        if step_index >= 0:
            # Preserve dependencies
            substitute_step.dependencies = modified.steps[step_index].dependencies

            # Replace step
            modified.steps[step_index] = substitute_step

            # Update references
            for step in modified.steps:
                if failure.step_id in step.dependencies:
                    step.dependencies.remove(failure.step_id)
                    step.dependencies.append(substitute_step.step_id)

        modified.metadata["substitutions"] = modified.metadata.get("substitutions", {})
        modified.metadata["substitutions"][failure.step_id] = substitute_step.step_id

        return modified

    def apply_rollback(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        rollback_to_index: int,
    ) -> ExecutionPlan:
        """
        Rollback plan to earlier state.

        Args:
            plan: Original plan
            failure: Failure context
            rollback_to_index: Index to rollback to

        Returns:
            Modified plan from rollback point
        """
        modified = plan.clone()

        # Keep only steps up to rollback point
        modified.steps = [s.clone() for s in plan.steps[:rollback_to_index]]

        modified.metadata["rollback_from"] = failure.step_index
        modified.metadata["rollback_to"] = rollback_to_index
        modified.metadata["rollback_reason"] = failure.failure_id

        return modified

    def apply_decompose(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        sub_steps: list[PlanStep],
    ) -> ExecutionPlan:
        """
        Decompose failed step into smaller steps.

        Args:
            plan: Original plan
            failure: Failure context
            sub_steps: Decomposed steps

        Returns:
            Modified plan with decomposition
        """
        modified = plan.clone()
        step_index = modified.get_step_index(failure.step_id)

        if step_index >= 0:
            original_step = modified.steps[step_index]

            # Set up dependencies for sub-steps
            for i, sub_step in enumerate(sub_steps):
                if i == 0:
                    sub_step.dependencies = original_step.dependencies
                else:
                    sub_step.dependencies = [sub_steps[i - 1].step_id]

            # Replace original step with sub-steps
            modified.steps = (
                modified.steps[:step_index]
                + sub_steps
                + modified.steps[step_index + 1:]
            )

            # Update dependencies pointing to original step
            last_sub_step_id = sub_steps[-1].step_id
            for step in modified.steps:
                if failure.step_id in step.dependencies:
                    step.dependencies.remove(failure.step_id)
                    step.dependencies.append(last_sub_step_id)

        modified.metadata["decomposed_steps"] = modified.metadata.get("decomposed_steps", {})
        modified.metadata["decomposed_steps"][failure.step_id] = [s.step_id for s in sub_steps]

        return modified

    def apply_delegate(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        target_agent_id: str,
    ) -> ExecutionPlan:
        """
        Delegate failed step to another agent.

        Args:
            plan: Original plan
            failure: Failure context
            target_agent_id: Agent to delegate to

        Returns:
            Modified plan with delegation
        """
        modified = plan.clone()
        step = modified.get_step(failure.step_id)

        if step:
            original_agent = step.agent_id
            step.agent_id = target_agent_id
            step.metadata["delegated_from"] = original_agent
            step.metadata["delegation_reason"] = failure.failure_id

        modified.metadata["delegations"] = modified.metadata.get("delegations", {})
        modified.metadata["delegations"][failure.step_id] = target_agent_id

        return modified

    def insert_steps(
        self,
        plan: ExecutionPlan,
        steps: list[PlanStep],
        after_index: int,
    ) -> ExecutionPlan:
        """
        Insert steps into plan.

        Args:
            plan: Original plan
            steps: Steps to insert
            after_index: Index to insert after

        Returns:
            Modified plan with inserted steps
        """
        modified = plan.clone()

        modified.steps = (
            modified.steps[:after_index + 1]
            + steps
            + modified.steps[after_index + 1:]
        )

        return modified

    def remove_steps(
        self,
        plan: ExecutionPlan,
        step_ids: list[str],
    ) -> ExecutionPlan:
        """
        Remove steps from plan.

        Args:
            plan: Original plan
            step_ids: IDs of steps to remove

        Returns:
            Modified plan with steps removed
        """
        modified = plan.clone()

        modified.steps = [s for s in modified.steps if s.step_id not in step_ids]

        # Update dependencies
        for step in modified.steps:
            step.dependencies = [d for d in step.dependencies if d not in step_ids]

        return modified


# =============================================================================
# FALLBACK GENERATOR
# =============================================================================


class FallbackGenerator(Protocol):
    """Protocol for fallback plan generation."""

    async def generate_fallback(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        context: dict[str, Any],
    ) -> ExecutionPlan | None:
        """Generate a fallback plan."""
        ...


class SimpleFallbackGenerator:
    """
    Simple fallback generator using predefined alternatives.

    Maps actions to fallback actions.
    """

    def __init__(
        self,
        fallback_map: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._fallback_map = fallback_map or {}
        self._logger = logger or logging.getLogger(__name__)

    def register_fallback(self, action: str, fallback_action: str) -> None:
        """Register a fallback action."""
        self._fallback_map[action] = fallback_action

    async def generate_fallback(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        context: dict[str, Any],
    ) -> ExecutionPlan | None:
        """Generate fallback by substituting actions."""
        step = plan.get_step(failure.step_id)
        if step is None:
            return None

        fallback_action = self._fallback_map.get(step.action)
        if fallback_action is None:
            return None

        # Create fallback step
        fallback_step = step.clone()
        fallback_step.action = fallback_action
        fallback_step.metadata["fallback_for"] = step.action

        # Create modified plan
        modified = plan.clone()
        step_index = modified.get_step_index(failure.step_id)
        if step_index >= 0:
            modified.steps[step_index] = fallback_step

        return modified


class LLMFallbackGenerator:
    """
    LLM-based fallback generator.

    Uses language model to generate intelligent fallback plans.
    """

    def __init__(
        self,
        llm_client: Any,  # LLM client interface
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)

    async def generate_fallback(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        context: dict[str, Any],
    ) -> ExecutionPlan | None:
        """Generate fallback using LLM."""
        # This would call the LLM to generate a fallback plan
        # Implementation depends on LLM client interface
        self._logger.info("LLM fallback generation not implemented")
        return None


# =============================================================================
# MULTI-AGENT COORDINATOR
# =============================================================================


@dataclass
class AgentCapability:
    """Capability of an agent."""

    agent_id: str
    actions: set[str]
    load: float = 0.0  # 0-1 load factor
    available: bool = True
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiAgentCoordinator:
    """
    Coordinates replanning across multiple agents.

    Handles delegation, load balancing, and agent selection.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._agents: dict[str, AgentCapability] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

    async def register_agent(
        self,
        agent_id: str,
        actions: set[str],
        priority: int = 0,
    ) -> None:
        """Register an agent with its capabilities."""
        async with self._lock:
            self._agents[agent_id] = AgentCapability(
                agent_id=agent_id,
                actions=actions,
                priority=priority,
            )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        async with self._lock:
            self._agents.pop(agent_id, None)

    async def update_agent_status(
        self,
        agent_id: str,
        load: float | None = None,
        available: bool | None = None,
    ) -> None:
        """Update agent status."""
        async with self._lock:
            if agent_id in self._agents:
                if load is not None:
                    self._agents[agent_id].load = load
                if available is not None:
                    self._agents[agent_id].available = available

    async def find_delegate(
        self,
        action: str,
        exclude_agents: set[str] | None = None,
    ) -> str | None:
        """
        Find an agent to delegate an action to.

        Args:
            action: Action to delegate
            exclude_agents: Agents to exclude

        Returns:
            Agent ID or None if no suitable agent
        """
        exclude_agents = exclude_agents or set()
        candidates: list[AgentCapability] = []

        for agent in self._agents.values():
            if agent.agent_id in exclude_agents:
                continue
            if not agent.available:
                continue
            if action not in agent.actions:
                continue

            candidates.append(agent)

        if not candidates:
            return None

        # Sort by load (ascending) then priority (descending)
        candidates.sort(key=lambda a: (a.load, -a.priority))

        return candidates[0].agent_id

    async def get_available_agents(self, action: str) -> list[str]:
        """Get all available agents for an action."""
        return [
            agent.agent_id
            for agent in self._agents.values()
            if agent.available and action in agent.actions
        ]

    async def broadcast_replan(
        self,
        replan_result: ReplanResult,
        affected_agents: set[str],
    ) -> None:
        """Broadcast replan to affected agents."""
        # This would notify agents of plan changes
        self._logger.info(
            f"Broadcasting replan {replan_result.replan_id} to {len(affected_agents)} agents"
        )


# =============================================================================
# REPLANNER
# =============================================================================


@dataclass
class ReplannerConfig:
    """
    Configuration for Replanner.

    Attributes:
        max_replan_attempts: Maximum replan attempts per failure
        recovery_mode: Recovery behavior mode
        enable_learning: Learn from failures
        auto_apply: Automatically apply replans
        require_approval_threshold: Severity requiring approval
    """

    max_replan_attempts: int = 3
    recovery_mode: RecoveryMode = RecoveryMode.BALANCED
    enable_learning: bool = True
    auto_apply: bool = True
    require_approval_threshold: FailureSeverity = FailureSeverity.HIGH


class Replanner:
    """
    Central replanning system.

    Detects failures, selects strategies, modifies plans,
    and coordinates multi-agent workflows.
    """

    def __init__(
        self,
        config: ReplannerConfig | None = None,
        fallback_generator: FallbackGenerator | None = None,
        coordinator: MultiAgentCoordinator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize replanner.

        Args:
            config: Replanner configuration
            fallback_generator: Fallback plan generator
            coordinator: Multi-agent coordinator
            logger: Optional logger
        """
        self._config = config or ReplannerConfig()
        self._fallback_generator = fallback_generator or SimpleFallbackGenerator()
        self._coordinator = coordinator or MultiAgentCoordinator()
        self._logger = logger or logging.getLogger(__name__)

        # Components
        self._detector = FailureDetector(logger)
        self._selector = StrategySelector(logger=logger)
        self._modifier = PlanModifier(logger)

        # State
        self._replan_history: list[ReplanResult] = []
        self._failure_patterns: dict[str, int] = {}  # action -> failure count
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Failure Detection
    # -------------------------------------------------------------------------

    def detect_failure(
        self,
        step_id: str,
        step_index: int,
        error: Exception | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        retry_count: int = 0,
    ) -> FailureContext:
        """
        Detect and classify a failure.

        Args:
            step_id: Step identifier
            step_index: Step index
            error: Exception if available
            error_message: Error message
            result: Step result
            retry_count: Retry count

        Returns:
            FailureContext
        """
        return self._detector.detect(
            step_id=step_id,
            step_index=step_index,
            error=error,
            error_message=error_message,
            result=result,
            retry_count=retry_count,
        )

    # -------------------------------------------------------------------------
    # Replanning
    # -------------------------------------------------------------------------

    async def replan(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        context: dict[str, Any] | None = None,
        preferred_strategy: ReplanStrategy | None = None,
    ) -> ReplanResult:
        """
        Generate a replan for a failure.

        Args:
            plan: Current execution plan
            failure: Failure context
            context: Additional context
            preferred_strategy: Preferred strategy to use

        Returns:
            ReplanResult
        """
        context = context or {}
        replan_id = str(uuid4())

        self._logger.info(
            f"Starting replan {replan_id[:8]} for failure {failure.failure_type.value} "
            f"at step {failure.step_index}"
        )

        # Track failure pattern
        step = plan.get_step(failure.step_id)
        if step and self._config.enable_learning:
            action = step.action
            self._failure_patterns[action] = self._failure_patterns.get(action, 0) + 1

        # Select strategy
        if preferred_strategy:
            strategy = preferred_strategy
            alternatives: list[Any] = []
        else:
            strategy, alternatives = self._selector.select(
                failure=failure,
                plan=plan,
                step=step,
                context=context,
            )

        self._logger.info(f"Selected strategy: {strategy.value}")

        # Create result
        result = ReplanResult(
            replan_id=replan_id,
            status=ReplanStatus.PLANNING,
            strategy=strategy,
            original_plan=plan,
            modified_plan=None,
            failure_context=failure,
            alternatives=alternatives,
        )

        # Apply strategy
        try:
            modified_plan = await self._apply_strategy(
                plan=plan,
                failure=failure,
                strategy=strategy,
                context=context,
            )

            if modified_plan:
                result.modified_plan = modified_plan
                result.status = ReplanStatus.READY
                result.confidence = self._calculate_confidence(strategy, failure)
                result.changes = self._describe_changes(plan, modified_plan)
                result.reasoning = self._generate_reasoning(strategy, failure)
            else:
                result.status = ReplanStatus.FAILED
                result.reasoning = f"Failed to apply strategy {strategy.value}"

        except Exception as e:
            self._logger.error(f"Replan failed: {e}")
            result.status = ReplanStatus.FAILED
            result.reasoning = str(e)

        # Store in history
        async with self._lock:
            self._replan_history.append(result)

        return result

    async def _apply_strategy(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        strategy: ReplanStrategy,
        context: dict[str, Any],
    ) -> ExecutionPlan | None:
        """Apply a replan strategy."""
        if strategy == ReplanStrategy.RETRY:
            return self._modifier.apply_retry(plan, failure)

        elif strategy == ReplanStrategy.SKIP:
            return self._modifier.apply_skip(plan, failure)

        elif strategy == ReplanStrategy.SUBSTITUTE:
            # Try to generate fallback
            fallback = await self._fallback_generator.generate_fallback(
                plan, failure, context
            )
            if fallback:
                return fallback

            # Try to find substitute step
            step = plan.get_step(failure.step_id)
            if step and step.fallback_step_id:
                fallback_step = plan.get_step(step.fallback_step_id)
                if fallback_step:
                    return self._modifier.apply_substitute(
                        plan, failure, fallback_step.clone()
                    )

            return None

        elif strategy == ReplanStrategy.ROLLBACK:
            # Rollback to step before failed step
            rollback_index = max(0, failure.step_index - 1)
            return self._modifier.apply_rollback(plan, failure, rollback_index)

        elif strategy == ReplanStrategy.DECOMPOSE:
            # Would need LLM or predefined decompositions
            # For now, return None
            return None

        elif strategy == ReplanStrategy.DELEGATE:
            step = plan.get_step(failure.step_id)
            if step:
                # Find alternative agent
                exclude = {step.agent_id} if step.agent_id else set()
                delegate_agent = await self._coordinator.find_delegate(
                    step.action, exclude
                )
                if delegate_agent:
                    return self._modifier.apply_delegate(plan, failure, delegate_agent)

            return None

        elif strategy == ReplanStrategy.ESCALATE:
            # Mark plan for human review
            modified = plan.clone()
            modified.metadata["requires_approval"] = True
            modified.metadata["escalation_reason"] = failure.failure_id
            return modified

        elif strategy == ReplanStrategy.FULL_REPLAN:
            # Would need LLM to generate new plan
            # For now, return None
            return None

        elif strategy == ReplanStrategy.ABORT:
            # Return empty plan
            return ExecutionPlan(
                plan_id=str(uuid4()),
                goal=plan.goal,
                steps=[],
                parent_plan_id=plan.plan_id,
                metadata={"aborted": True, "abort_reason": failure.failure_id},
            )

        return None

    def _calculate_confidence(
        self,
        strategy: ReplanStrategy,
        failure: FailureContext,
    ) -> float:
        """Calculate confidence in the replan."""
        base_confidence = {
            ReplanStrategy.RETRY: 0.7,
            ReplanStrategy.SKIP: 0.6,
            ReplanStrategy.SUBSTITUTE: 0.65,
            ReplanStrategy.ROLLBACK: 0.5,
            ReplanStrategy.DECOMPOSE: 0.55,
            ReplanStrategy.DELEGATE: 0.7,
            ReplanStrategy.ESCALATE: 0.8,
            ReplanStrategy.FULL_REPLAN: 0.4,
            ReplanStrategy.ABORT: 1.0,
        }.get(strategy, 0.5)

        # Adjust based on failure
        if failure.is_transient:
            base_confidence += 0.1
        if failure.retry_count > 2:
            base_confidence -= 0.2

        return max(0.0, min(1.0, base_confidence))

    def _describe_changes(
        self,
        original: ExecutionPlan,
        modified: ExecutionPlan,
    ) -> list[str]:
        """Describe changes between plans."""
        changes: list[str] = []

        original_ids = {s.step_id for s in original.steps}
        modified_ids = {s.step_id for s in modified.steps}

        # Removed steps
        removed = original_ids - modified_ids
        if removed:
            changes.append(f"Removed {len(removed)} step(s)")

        # Added steps
        added = modified_ids - original_ids
        if added:
            changes.append(f"Added {len(added)} step(s)")

        # Modified steps
        for step in modified.steps:
            if step.step_id in original_ids:
                original_step = original.get_step(step.step_id)
                if original_step:
                    if step.agent_id != original_step.agent_id:
                        changes.append(f"Delegated step {step.step_id[:8]} to {step.agent_id}")
                    if step.action != original_step.action:
                        changes.append(f"Changed action for step {step.step_id[:8]}")

        # Metadata changes
        if modified.metadata.get("aborted"):
            changes.append("Plan aborted")
        if modified.metadata.get("requires_approval"):
            changes.append("Escalated for approval")

        return changes

    def _generate_reasoning(
        self,
        strategy: ReplanStrategy,
        failure: FailureContext,
    ) -> str:
        """Generate reasoning for the replan."""
        reasons = {
            ReplanStrategy.RETRY: f"Retrying step due to {failure.failure_type.value} failure",
            ReplanStrategy.SKIP: f"Skipping step as it's not critical and failed with {failure.failure_type.value}",
            ReplanStrategy.SUBSTITUTE: f"Substituting with alternative approach after {failure.failure_type.value}",
            ReplanStrategy.ROLLBACK: f"Rolling back to earlier state due to {failure.failure_type.value}",
            ReplanStrategy.DECOMPOSE: f"Breaking down step into smaller parts after {failure.failure_type.value}",
            ReplanStrategy.DELEGATE: f"Delegating to another agent after {failure.failure_type.value}",
            ReplanStrategy.ESCALATE: f"Escalating for human review due to {failure.severity.name} {failure.failure_type.value}",
            ReplanStrategy.FULL_REPLAN: f"Generating new plan after {failure.failure_type.value}",
            ReplanStrategy.ABORT: f"Aborting plan due to {failure.severity.name} {failure.failure_type.value}",
        }

        return reasons.get(strategy, f"Applying {strategy.value} strategy")

    # -------------------------------------------------------------------------
    # Plan Modification
    # -------------------------------------------------------------------------

    async def modify_plan(
        self,
        plan: ExecutionPlan,
        modifications: list[dict[str, Any]],
    ) -> ExecutionPlan:
        """
        Apply multiple modifications to a plan.

        Args:
            plan: Original plan
            modifications: List of modification specs

        Returns:
            Modified plan
        """
        modified = plan.clone()

        for mod in modifications:
            mod_type = mod.get("type")

            if mod_type == "insert":
                steps = [PlanStep.from_dict(s) for s in mod.get("steps", [])]
                after_index = mod.get("after_index", len(modified.steps) - 1)
                modified = self._modifier.insert_steps(modified, steps, after_index)

            elif mod_type == "remove":
                step_ids = mod.get("step_ids", [])
                modified = self._modifier.remove_steps(modified, step_ids)

            elif mod_type == "update":
                step_id = mod.get("step_id")
                updates = mod.get("updates", {})
                step = modified.get_step(step_id)
                if step:
                    for key, value in updates.items():
                        if hasattr(step, key):
                            setattr(step, key, value)

        return modified

    # -------------------------------------------------------------------------
    # Fallback Generation
    # -------------------------------------------------------------------------

    async def generate_fallback(
        self,
        plan: ExecutionPlan,
        failure: FailureContext,
        context: dict[str, Any] | None = None,
    ) -> ExecutionPlan | None:
        """
        Generate a fallback plan.

        Args:
            plan: Original plan
            failure: Failure context
            context: Additional context

        Returns:
            Fallback plan or None
        """
        return await self._fallback_generator.generate_fallback(
            plan, failure, context or {}
        )

    def register_fallback(self, action: str, fallback_action: str) -> None:
        """Register a fallback action mapping."""
        if isinstance(self._fallback_generator, SimpleFallbackGenerator):
            self._fallback_generator.register_fallback(action, fallback_action)

    # -------------------------------------------------------------------------
    # Multi-Agent Coordination
    # -------------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        actions: set[str],
        priority: int = 0,
    ) -> None:
        """Register an agent for delegation."""
        await self._coordinator.register_agent(agent_id, actions, priority)

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        await self._coordinator.unregister_agent(agent_id)

    async def update_agent_status(
        self,
        agent_id: str,
        load: float | None = None,
        available: bool | None = None,
    ) -> None:
        """Update agent status."""
        await self._coordinator.update_agent_status(agent_id, load, available)

    async def find_delegate(
        self,
        action: str,
        exclude_agents: set[str] | None = None,
    ) -> str | None:
        """Find an agent to delegate to."""
        return await self._coordinator.find_delegate(action, exclude_agents)

    # -------------------------------------------------------------------------
    # History and Learning
    # -------------------------------------------------------------------------

    def get_replan_history(
        self,
        limit: int = 100,
        strategy: ReplanStrategy | None = None,
        status: ReplanStatus | None = None,
    ) -> list[ReplanResult]:
        """Get replan history."""
        results = list(reversed(self._replan_history))

        if strategy:
            results = [r for r in results if r.strategy == strategy]

        if status:
            results = [r for r in results if r.status == status]

        return results[:limit]

    def get_failure_patterns(self) -> dict[str, int]:
        """Get failure patterns by action."""
        return dict(self._failure_patterns)

    def get_problematic_actions(self, threshold: int = 3) -> list[str]:
        """Get actions that frequently fail."""
        return [
            action for action, count in self._failure_patterns.items()
            if count >= threshold
        ]

    def clear_history(self) -> None:
        """Clear replan history."""
        self._replan_history.clear()
        self._failure_patterns.clear()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get replanner statistics."""
        total = len(self._replan_history)
        successful = sum(1 for r in self._replan_history if r.is_successful)

        by_strategy: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for result in self._replan_history:
            strategy_name = result.strategy.value
            by_strategy[strategy_name] = by_strategy.get(strategy_name, 0) + 1

            status_name = result.status.name
            by_status[status_name] = by_status.get(status_name, 0) + 1

        return {
            "total_replans": total,
            "successful_replans": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_strategy": by_strategy,
            "by_status": by_status,
            "failure_patterns": dict(self._failure_patterns),
            "problematic_actions": self.get_problematic_actions(),
            "registered_agents": len(self._coordinator._agents),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_replanner(
    max_attempts: int = 3,
    recovery_mode: RecoveryMode = RecoveryMode.BALANCED,
    enable_learning: bool = True,
    auto_apply: bool = True,
    fallback_map: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> Replanner:
    """
    Factory function to create configured Replanner.

    Args:
        max_attempts: Maximum replan attempts
        recovery_mode: Recovery behavior mode
        enable_learning: Enable learning from failures
        auto_apply: Auto-apply replans
        fallback_map: Action to fallback action mapping
        logger: Optional logger

    Returns:
        Configured Replanner
    """
    config = ReplannerConfig(
        max_replan_attempts=max_attempts,
        recovery_mode=recovery_mode,
        enable_learning=enable_learning,
        auto_apply=auto_apply,
    )

    fallback_generator = SimpleFallbackGenerator(fallback_map, logger)
    coordinator = MultiAgentCoordinator(logger)

    return Replanner(
        config=config,
        fallback_generator=fallback_generator,
        coordinator=coordinator,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "FailureType",
    "FailureSeverity",
    "ReplanStrategy",
    "ReplanStatus",
    "RecoveryMode",
    # Failure Detection
    "FailureContext",
    "FailureDetector",
    # Plan Structures
    "PlanStep",
    "ExecutionPlan",
    # Replan Result
    "ReplanResult",
    # Strategy
    "StrategyRule",
    "StrategySelector",
    # Plan Modification
    "PlanModifier",
    # Fallback
    "FallbackGenerator",
    "SimpleFallbackGenerator",
    "LLMFallbackGenerator",
    # Multi-Agent
    "AgentCapability",
    "MultiAgentCoordinator",
    # Replanner
    "ReplannerConfig",
    "Replanner",
    # Factory
    "create_replanner",
]
