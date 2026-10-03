"""
Exceptions - Unified error hierarchy for the AI Agent Platform.

This module provides comprehensive exception classes including:
- Base exception with rich context
- Capability errors
- Execution errors
- Planning errors
- Approval errors
- Budget errors
- Checkpoint errors
- Router errors
- State errors
- Error codes and severity levels
- Error serialization and logging support
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import uuid4


# =============================================================================
# ENUMS
# =============================================================================


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = auto()  # Minor issue, can continue
    MEDIUM = auto()  # Significant issue, may need adjustment
    HIGH = auto()  # Major issue, likely needs intervention
    CRITICAL = auto()  # Severe issue, cannot continue
    FATAL = auto()  # Unrecoverable, must abort


class ErrorCategory(Enum):
    """Categories of errors."""

    CAPABILITY = "capability"
    EXECUTION = "execution"
    PLANNING = "planning"
    APPROVAL = "approval"
    BUDGET = "budget"
    CHECKPOINT = "checkpoint"
    ROUTER = "router"
    STATE = "state"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    INTERNAL = "internal"


class ErrorCode(Enum):
    """Standardized error codes."""

    # General (1000-1099)
    UNKNOWN_ERROR = 1000
    INTERNAL_ERROR = 1001
    NOT_IMPLEMENTED = 1002
    INVALID_ARGUMENT = 1003
    INVALID_STATE = 1004
    TIMEOUT = 1005
    CANCELLED = 1006

    # Capability (1100-1199)
    CAPABILITY_NOT_FOUND = 1100
    CAPABILITY_UNAVAILABLE = 1101
    CAPABILITY_INITIALIZATION_FAILED = 1102
    CAPABILITY_EXECUTION_FAILED = 1103
    CAPABILITY_DEPENDENCY_MISSING = 1104
    CAPABILITY_VERSION_MISMATCH = 1105
    CAPABILITY_DISABLED = 1106
    CAPABILITY_OVERLOADED = 1107

    # Execution (1200-1299)
    EXECUTION_FAILED = 1200
    STEP_FAILED = 1201
    STEP_TIMEOUT = 1202
    STEP_CANCELLED = 1203
    STEP_SKIPPED = 1204
    DEPENDENCY_FAILED = 1205
    PARALLEL_EXECUTION_FAILED = 1206
    RETRY_EXHAUSTED = 1207
    EXECUTION_ABORTED = 1208

    # Planning (1300-1399)
    PLANNING_FAILED = 1300
    INVALID_PLAN = 1301
    PLAN_GENERATION_FAILED = 1302
    PLAN_VALIDATION_FAILED = 1303
    REPLAN_FAILED = 1304
    REPLAN_LIMIT_EXCEEDED = 1305
    NO_VIABLE_PLAN = 1306
    CIRCULAR_DEPENDENCY = 1307

    # Approval (1400-1499)
    APPROVAL_REQUIRED = 1400
    APPROVAL_DENIED = 1401
    APPROVAL_TIMEOUT = 1402
    APPROVAL_CANCELLED = 1403
    ESCALATION_REQUIRED = 1404
    ESCALATION_FAILED = 1405
    DANGEROUS_ACTION_BLOCKED = 1406

    # Budget (1500-1599)
    BUDGET_EXCEEDED = 1500
    TOKEN_LIMIT_EXCEEDED = 1501
    COST_LIMIT_EXCEEDED = 1502
    REQUEST_LIMIT_EXCEEDED = 1503
    RATE_LIMIT_EXCEEDED = 1504
    QUOTA_EXHAUSTED = 1505
    BUDGET_NOT_AVAILABLE = 1506

    # Checkpoint (1600-1699)
    CHECKPOINT_FAILED = 1600
    CHECKPOINT_NOT_FOUND = 1601
    CHECKPOINT_CORRUPTED = 1602
    CHECKPOINT_LOAD_FAILED = 1603
    CHECKPOINT_SAVE_FAILED = 1604
    RECOVERY_FAILED = 1605
    ROLLBACK_FAILED = 1606
    INVALID_CHECKPOINT = 1607

    # Router (1700-1799)
    ROUTING_FAILED = 1700
    NO_MATCHING_CAPABILITY = 1701
    AMBIGUOUS_ROUTING = 1702
    ROUTING_LOOP_DETECTED = 1703
    INVALID_ROUTE = 1704
    ROUTE_NOT_FOUND = 1705

    # State (1800-1899)
    STATE_ERROR = 1800
    STATE_NOT_FOUND = 1801
    STATE_CORRUPTED = 1802
    STATE_TRANSITION_INVALID = 1803
    STATE_LOCK_FAILED = 1804
    STATE_SAVE_FAILED = 1805
    STATE_LOAD_FAILED = 1806
    SESSION_EXPIRED = 1807
    SESSION_NOT_FOUND = 1808

    # Configuration (1900-1999)
    CONFIG_ERROR = 1900
    CONFIG_NOT_FOUND = 1901
    CONFIG_INVALID = 1902
    CONFIG_LOAD_FAILED = 1903

    # Validation (2000-2099)
    VALIDATION_ERROR = 2000
    SCHEMA_VALIDATION_FAILED = 2001
    PRECONDITION_FAILED = 2002
    POSTCONDITION_FAILED = 2003
    CONSTRAINT_VIOLATION = 2004

    # Network (2100-2199)
    NETWORK_ERROR = 2100
    CONNECTION_FAILED = 2101
    CONNECTION_TIMEOUT = 2102
    DNS_ERROR = 2103
    SSL_ERROR = 2104

    # Authentication/Authorization (2200-2299)
    AUTH_ERROR = 2200
    AUTHENTICATION_FAILED = 2201
    AUTHORIZATION_FAILED = 2202
    TOKEN_EXPIRED = 2203
    TOKEN_INVALID = 2204
    PERMISSION_DENIED = 2205

    # Resource (2300-2399)
    RESOURCE_ERROR = 2300
    RESOURCE_NOT_FOUND = 2301
    RESOURCE_UNAVAILABLE = 2302
    RESOURCE_EXHAUSTED = 2303
    RESOURCE_LOCKED = 2304


# =============================================================================
# ERROR CONTEXT
# =============================================================================


@dataclass
class ErrorContext:
    """
    Rich context information for errors.

    Attributes:
        error_id: Unique error identifier
        timestamp: When error occurred
        component: Component that raised the error
        operation: Operation being performed
        session_id: Session identifier
        request_id: Request identifier
        step_id: Step identifier
        plan_id: Plan identifier
        agent_id: Agent identifier
        user_id: User identifier
        metadata: Additional context data
        stack_trace: Error stack trace
        cause: Underlying cause
    """

    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str | None = None
    operation: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    step_id: str | None = None
    plan_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "step_id": self.step_id,
            "plan_id": self.plan_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
            "cause": self.cause,
        }


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class AgentPlatformError(Exception):
    """
    Base exception for all Agent Platform errors.

    Provides rich context, error codes, severity levels,
    and serialization support.
    """

    default_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    default_category: ErrorCategory = ErrorCategory.INTERNAL
    default_severity: ErrorSeverity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize error.

        Args:
            message: Error message
            code: Error code
            category: Error category
            severity: Error severity
            context: Error context
            cause: Underlying exception
            retryable: Whether operation can be retried
            details: Additional error details
        """
        super().__init__(message)

        self.message = message
        self.code = code or self.default_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.retryable = retryable
        self.details = details or {}

        # Capture stack trace
        if self.context.stack_trace is None:
            self.context.stack_trace = traceback.format_exc()

        # Capture cause
        if cause and self.context.cause is None:
            self.context.cause = str(cause)

    @property
    def error_id(self) -> str:
        """Get error ID."""
        return self.context.error_id

    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable."""
        return self.retryable

    @property
    def is_critical(self) -> bool:
        """Check if error is critical or fatal."""
        return self.severity in {ErrorSeverity.CRITICAL, ErrorSeverity.FATAL}

    def with_context(self, **kwargs: Any) -> AgentPlatformError:
        """
        Add context to error.

        Args:
            **kwargs: Context attributes to set

        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
        return self

    def with_details(self, **kwargs: Any) -> AgentPlatformError:
        """
        Add details to error.

        Args:
            **kwargs: Details to add

        Returns:
            Self for chaining
        """
        self.details.update(kwargs)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code.value,
            "code_name": self.code.name,
            "category": self.category.value,
            "severity": self.severity.name,
            "retryable": self.retryable,
            "details": self.details,
            "context": self.context.to_dict(),
        }

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging (excludes stack trace)."""
        result = self.to_dict()
        result["context"].pop("stack_trace", None)
        return result

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.code.name}] {self.message}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code.name}, "
            f"severity={self.severity.name}, "
            f"error_id={self.error_id!r})"
        )


# =============================================================================
# CAPABILITY ERRORS
# =============================================================================


class CapabilityError(AgentPlatformError):
    """Base exception for capability-related errors."""

    default_code = ErrorCode.CAPABILITY_EXECUTION_FAILED
    default_category = ErrorCategory.CAPABILITY
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        capability_id: str | None = None,
        capability_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.capability_id = capability_id
        self.capability_name = capability_name

        if capability_id:
            self.context.metadata["capability_id"] = capability_id
        if capability_name:
            self.context.metadata["capability_name"] = capability_name


class CapabilityNotFoundError(CapabilityError):
    """Raised when a capability is not found."""

    default_code = ErrorCode.CAPABILITY_NOT_FOUND
    default_severity = ErrorSeverity.HIGH

    def __init__(self, capability_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Capability not found: {capability_id}",
            capability_id=capability_id,
            **kwargs,
        )


class CapabilityUnavailableError(CapabilityError):
    """Raised when a capability is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    default_severity = ErrorSeverity.MEDIUM
    retryable = True

    def __init__(
        self,
        capability_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Capability unavailable: {capability_id}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, capability_id=capability_id, **kwargs)
        self.reason = reason


class CapabilityInitializationError(CapabilityError):
    """Raised when capability initialization fails."""

    default_code = ErrorCode.CAPABILITY_INITIALIZATION_FAILED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        capability_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to initialize capability: {capability_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, capability_id=capability_id, **kwargs)


class CapabilityDependencyError(CapabilityError):
    """Raised when capability dependencies are missing."""

    default_code = ErrorCode.CAPABILITY_DEPENDENCY_MISSING
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        capability_id: str,
        missing_dependencies: list[str],
        **kwargs: Any,
    ) -> None:
        deps = ", ".join(missing_dependencies)
        super().__init__(
            f"Missing dependencies for {capability_id}: {deps}",
            capability_id=capability_id,
            **kwargs,
        )
        self.missing_dependencies = missing_dependencies
        self.details["missing_dependencies"] = missing_dependencies


# =============================================================================
# EXECUTION ERRORS
# =============================================================================


class ExecutionError(AgentPlatformError):
    """Base exception for execution-related errors."""

    default_code = ErrorCode.EXECUTION_FAILED
    default_category = ErrorCategory.EXECUTION
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        step_id: str | None = None,
        step_index: int | None = None,
        plan_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.step_id = step_id
        self.step_index = step_index
        self.plan_id = plan_id

        if step_id:
            self.context.step_id = step_id
        if plan_id:
            self.context.plan_id = plan_id
        if step_index is not None:
            self.context.metadata["step_index"] = step_index


class StepExecutionError(ExecutionError):
    """Raised when a step execution fails."""

    default_code = ErrorCode.STEP_FAILED

    def __init__(
        self,
        step_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Step execution failed: {step_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, step_id=step_id, **kwargs)


class StepTimeoutError(ExecutionError):
    """Raised when a step times out."""

    default_code = ErrorCode.STEP_TIMEOUT
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        step_id: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(
            f"Step timed out after {timeout_seconds}s: {step_id}",
            step_id=step_id,
            **kwargs,
        )
        self.timeout_seconds = timeout_seconds
        self.details["timeout_seconds"] = timeout_seconds


class DependencyFailedError(ExecutionError):
    """Raised when a step dependency fails."""

    default_code = ErrorCode.DEPENDENCY_FAILED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        step_id: str,
        failed_dependency: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Dependency failed for step {step_id}: {failed_dependency}",
            step_id=step_id,
            **kwargs,
        )
        self.failed_dependency = failed_dependency
        self.details["failed_dependency"] = failed_dependency


class RetryExhaustedError(ExecutionError):
    """Raised when all retries are exhausted."""

    default_code = ErrorCode.RETRY_EXHAUSTED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        step_id: str,
        attempts: int,
        last_error: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Retry exhausted for step {step_id} after {attempts} attempts"
        if last_error:
            message += f": {last_error}"
        super().__init__(message, step_id=step_id, **kwargs)
        self.attempts = attempts
        self.last_error = last_error
        self.details["attempts"] = attempts


class ExecutionAbortedError(ExecutionError):
    """Raised when execution is aborted."""

    default_code = ErrorCode.EXECUTION_ABORTED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        reason: str = "Execution aborted",
        **kwargs: Any,
    ) -> None:
        super().__init__(reason, **kwargs)


# =============================================================================
# PLANNING ERRORS
# =============================================================================


class PlanningError(AgentPlatformError):
    """Base exception for planning-related errors."""

    default_code = ErrorCode.PLANNING_FAILED
    default_category = ErrorCategory.PLANNING
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        plan_id: str | None = None,
        goal: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.plan_id = plan_id
        self.goal = goal

        if plan_id:
            self.context.plan_id = plan_id
        if goal:
            self.context.metadata["goal"] = goal


class PlanGenerationError(PlanningError):
    """Raised when plan generation fails."""

    default_code = ErrorCode.PLAN_GENERATION_FAILED

    def __init__(
        self,
        goal: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to generate plan for goal: {goal}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, goal=goal, **kwargs)


class PlanValidationError(PlanningError):
    """Raised when plan validation fails."""

    default_code = ErrorCode.PLAN_VALIDATION_FAILED

    def __init__(
        self,
        plan_id: str,
        validation_errors: list[str],
        **kwargs: Any,
    ) -> None:
        errors_str = "; ".join(validation_errors)
        super().__init__(
            f"Plan validation failed: {errors_str}",
            plan_id=plan_id,
            **kwargs,
        )
        self.validation_errors = validation_errors
        self.details["validation_errors"] = validation_errors


class ReplanError(PlanningError):
    """Raised when replanning fails."""

    default_code = ErrorCode.REPLAN_FAILED

    def __init__(
        self,
        plan_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Replanning failed for plan: {plan_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, plan_id=plan_id, **kwargs)


class ReplanLimitExceededError(PlanningError):
    """Raised when replan limit is exceeded."""

    default_code = ErrorCode.REPLAN_LIMIT_EXCEEDED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        plan_id: str,
        attempts: int,
        max_attempts: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Replan limit exceeded for plan {plan_id}: {attempts}/{max_attempts}",
            plan_id=plan_id,
            **kwargs,
        )
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.details["attempts"] = attempts
        self.details["max_attempts"] = max_attempts


class CircularDependencyError(PlanningError):
    """Raised when circular dependency is detected."""

    default_code = ErrorCode.CIRCULAR_DEPENDENCY
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        cycle: list[str],
        **kwargs: Any,
    ) -> None:
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}", **kwargs)
        self.cycle = cycle
        self.details["cycle"] = cycle


class NoViablePlanError(PlanningError):
    """Raised when no viable plan can be generated."""

    default_code = ErrorCode.NO_VIABLE_PLAN
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        goal: str,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"No viable plan for goal: {goal}"
        if constraints:
            message += f" (constraints: {', '.join(constraints)})"
        super().__init__(message, goal=goal, **kwargs)
        self.constraints = constraints or []


# =============================================================================
# APPROVAL ERRORS
# =============================================================================


class ApprovalError(AgentPlatformError):
    """Base exception for approval-related errors."""

    default_code = ErrorCode.APPROVAL_REQUIRED
    default_category = ErrorCategory.APPROVAL
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        action: str | None = None,
        approver: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.action = action
        self.approver = approver

        if action:
            self.context.metadata["action"] = action
        if approver:
            self.context.metadata["approver"] = approver


class ApprovalRequiredError(ApprovalError):
    """Raised when approval is required."""

    default_code = ErrorCode.APPROVAL_REQUIRED
    default_severity = ErrorSeverity.LOW

    def __init__(
        self,
        action: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Approval required for action: {action}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, action=action, **kwargs)


class ApprovalDeniedError(ApprovalError):
    """Raised when approval is denied."""

    default_code = ErrorCode.APPROVAL_DENIED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        action: str,
        reason: str | None = None,
        approver: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Approval denied for action: {action}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, action=action, approver=approver, **kwargs)
        self.reason = reason


class ApprovalTimeoutError(ApprovalError):
    """Raised when approval times out."""

    default_code = ErrorCode.APPROVAL_TIMEOUT
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        action: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(
            f"Approval timed out after {timeout_seconds}s for action: {action}",
            action=action,
            **kwargs,
        )
        self.timeout_seconds = timeout_seconds


class EscalationRequiredError(ApprovalError):
    """Raised when escalation is required."""

    default_code = ErrorCode.ESCALATION_REQUIRED
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        action: str,
        escalation_level: int,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Escalation required (level {escalation_level}) for action: {action}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, action=action, **kwargs)
        self.escalation_level = escalation_level
        self.details["escalation_level"] = escalation_level


class DangerousActionBlockedError(ApprovalError):
    """Raised when a dangerous action is blocked."""

    default_code = ErrorCode.DANGEROUS_ACTION_BLOCKED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        action: str,
        risk_level: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Dangerous action blocked: {action}"
        if risk_level:
            message += f" (risk: {risk_level})"
        super().__init__(message, action=action, **kwargs)
        self.risk_level = risk_level


# =============================================================================
# BUDGET ERRORS
# =============================================================================


class BudgetError(AgentPlatformError):
    """Base exception for budget-related errors."""

    default_code = ErrorCode.BUDGET_EXCEEDED
    default_category = ErrorCategory.BUDGET
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        budget_type: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.budget_type = budget_type
        self.current_value = current_value
        self.limit_value = limit_value

        if budget_type:
            self.context.metadata["budget_type"] = budget_type
        if current_value is not None:
            self.details["current_value"] = current_value
        if limit_value is not None:
            self.details["limit_value"] = limit_value


class BudgetExceededError(BudgetError):
    """Raised when budget is exceeded."""

    default_code = ErrorCode.BUDGET_EXCEEDED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        budget_type: str,
        current: float,
        limit: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Budget exceeded: {budget_type} ({current:.2f}/{limit:.2f})",
            budget_type=budget_type,
            current_value=current,
            limit_value=limit,
            **kwargs,
        )


class TokenLimitExceededError(BudgetError):
    """Raised when token limit is exceeded."""

    default_code = ErrorCode.TOKEN_LIMIT_EXCEEDED

    def __init__(
        self,
        current_tokens: int,
        max_tokens: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Token limit exceeded: {current_tokens}/{max_tokens}",
            budget_type="tokens",
            current_value=float(current_tokens),
            limit_value=float(max_tokens),
            **kwargs,
        )


class CostLimitExceededError(BudgetError):
    """Raised when cost limit is exceeded."""

    default_code = ErrorCode.COST_LIMIT_EXCEEDED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        current_cost: float,
        max_cost: float,
        currency: str = "USD",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Cost limit exceeded: {currency} {current_cost:.2f}/{max_cost:.2f}",
            budget_type="cost",
            current_value=current_cost,
            limit_value=max_cost,
            **kwargs,
        )
        self.currency = currency


class RateLimitExceededError(BudgetError):
    """Raised when rate limit is exceeded."""

    default_code = ErrorCode.RATE_LIMIT_EXCEEDED
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        requests: int,
        limit: int,
        window_seconds: int,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        message = f"Rate limit exceeded: {requests}/{limit} requests per {window_seconds}s"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(
            message,
            budget_type="rate",
            current_value=float(requests),
            limit_value=float(limit),
            **kwargs,
        )
        self.window_seconds = window_seconds
        self.retry_after = retry_after


class QuotaExhaustedError(BudgetError):
    """Raised when quota is exhausted."""

    default_code = ErrorCode.QUOTA_EXHAUSTED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        quota_type: str,
        reset_time: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Quota exhausted: {quota_type}"
        if reset_time:
            message += f" (resets at {reset_time})"
        super().__init__(message, budget_type=quota_type, **kwargs)
        self.reset_time = reset_time


# =============================================================================
# CHECKPOINT ERRORS
# =============================================================================


class CheckpointError(AgentPlatformError):
    """Base exception for checkpoint-related errors."""

    default_code = ErrorCode.CHECKPOINT_FAILED
    default_category = ErrorCategory.CHECKPOINT
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        checkpoint_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.checkpoint_id = checkpoint_id

        if checkpoint_id:
            self.context.metadata["checkpoint_id"] = checkpoint_id


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint is not found."""

    default_code = ErrorCode.CHECKPOINT_NOT_FOUND

    def __init__(self, checkpoint_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Checkpoint not found: {checkpoint_id}",
            checkpoint_id=checkpoint_id,
            **kwargs,
        )


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint is corrupted."""

    default_code = ErrorCode.CHECKPOINT_CORRUPTED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        checkpoint_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Checkpoint corrupted: {checkpoint_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, checkpoint_id=checkpoint_id, **kwargs)


class CheckpointSaveError(CheckpointError):
    """Raised when checkpoint save fails."""

    default_code = ErrorCode.CHECKPOINT_SAVE_FAILED

    def __init__(
        self,
        checkpoint_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        message = "Failed to save checkpoint"
        if checkpoint_id:
            message += f": {checkpoint_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, checkpoint_id=checkpoint_id, **kwargs)


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint load fails."""

    default_code = ErrorCode.CHECKPOINT_LOAD_FAILED

    def __init__(
        self,
        checkpoint_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to load checkpoint: {checkpoint_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, checkpoint_id=checkpoint_id, **kwargs)


class RecoveryError(CheckpointError):
    """Raised when recovery fails."""

    default_code = ErrorCode.RECOVERY_FAILED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        checkpoint_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = "Recovery failed"
        if checkpoint_id:
            message += f" from checkpoint: {checkpoint_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, checkpoint_id=checkpoint_id, **kwargs)


class RollbackError(CheckpointError):
    """Raised when rollback fails."""

    default_code = ErrorCode.ROLLBACK_FAILED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        checkpoint_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Rollback failed to checkpoint: {checkpoint_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, checkpoint_id=checkpoint_id, **kwargs)


# =============================================================================
# ROUTER ERRORS
# =============================================================================


class RouterError(AgentPlatformError):
    """Base exception for router-related errors."""

    default_code = ErrorCode.ROUTING_FAILED
    default_category = ErrorCategory.ROUTER
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        intent: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.intent = intent

        if intent:
            self.context.metadata["intent"] = intent


class NoMatchingCapabilityError(RouterError):
    """Raised when no matching capability is found."""

    default_code = ErrorCode.NO_MATCHING_CAPABILITY

    def __init__(
        self,
        intent: str,
        available_capabilities: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"No matching capability for intent: {intent}"
        super().__init__(message, intent=intent, **kwargs)
        self.available_capabilities = available_capabilities or []
        if available_capabilities:
            self.details["available_capabilities"] = available_capabilities


class AmbiguousRoutingError(RouterError):
    """Raised when routing is ambiguous."""

    default_code = ErrorCode.AMBIGUOUS_ROUTING
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        intent: str,
        matching_capabilities: list[str],
        **kwargs: Any,
    ) -> None:
        caps = ", ".join(matching_capabilities)
        super().__init__(
            f"Ambiguous routing for intent '{intent}': {caps}",
            intent=intent,
            **kwargs,
        )
        self.matching_capabilities = matching_capabilities
        self.details["matching_capabilities"] = matching_capabilities


class RoutingLoopError(RouterError):
    """Raised when routing loop is detected."""

    default_code = ErrorCode.ROUTING_LOOP_DETECTED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        route_path: list[str],
        **kwargs: Any,
    ) -> None:
        path_str = " -> ".join(route_path)
        super().__init__(f"Routing loop detected: {path_str}", **kwargs)
        self.route_path = route_path
        self.details["route_path"] = route_path


class RouteNotFoundError(RouterError):
    """Raised when route is not found."""

    default_code = ErrorCode.ROUTE_NOT_FOUND

    def __init__(
        self,
        route_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Route not found: {route_id}", **kwargs)
        self.route_id = route_id


# =============================================================================
# STATE ERRORS
# =============================================================================


class StateError(AgentPlatformError):
    """Base exception for state-related errors."""

    default_code = ErrorCode.STATE_ERROR
    default_category = ErrorCategory.STATE
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)

        if session_id:
            self.context.session_id = session_id


class StateNotFoundError(StateError):
    """Raised when state is not found."""

    default_code = ErrorCode.STATE_NOT_FOUND

    def __init__(self, session_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"State not found for session: {session_id}",
            session_id=session_id,
            **kwargs,
        )


class StateCorruptedError(StateError):
    """Raised when state is corrupted."""

    default_code = ErrorCode.STATE_CORRUPTED
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        session_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"State corrupted for session: {session_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, session_id=session_id, **kwargs)


class InvalidStateTransitionError(StateError):
    """Raised when state transition is invalid."""

    default_code = ErrorCode.STATE_TRANSITION_INVALID

    def __init__(
        self,
        current_state: str,
        target_state: str,
        allowed_transitions: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Invalid state transition: {current_state} -> {target_state}"
        if allowed_transitions:
            message += f" (allowed: {', '.join(allowed_transitions)})"
        super().__init__(message, **kwargs)
        self.current_state = current_state
        self.target_state = target_state
        self.allowed_transitions = allowed_transitions or []
        self.details["current_state"] = current_state
        self.details["target_state"] = target_state


class StateLockError(StateError):
    """Raised when state lock fails."""

    default_code = ErrorCode.STATE_LOCK_FAILED
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        session_id: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        message = f"Failed to acquire state lock for session: {session_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, session_id=session_id, **kwargs)


class SessionExpiredError(StateError):
    """Raised when session has expired."""

    default_code = ErrorCode.SESSION_EXPIRED
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        session_id: str,
        expired_at: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Session expired: {session_id}"
        if expired_at:
            message += f" (at {expired_at})"
        super().__init__(message, session_id=session_id, **kwargs)
        self.expired_at = expired_at


class SessionNotFoundError(StateError):
    """Raised when session is not found."""

    default_code = ErrorCode.SESSION_NOT_FOUND

    def __init__(self, session_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Session not found: {session_id}",
            session_id=session_id,
            **kwargs,
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(AgentPlatformError):
    """Base exception for configuration errors."""

    default_code = ErrorCode.CONFIG_ERROR
    default_category = ErrorCategory.CONFIGURATION
    default_severity = ErrorSeverity.CRITICAL


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration is not found."""

    default_code = ErrorCode.CONFIG_NOT_FOUND

    def __init__(self, config_key: str, **kwargs: Any) -> None:
        super().__init__(f"Configuration not found: {config_key}", **kwargs)
        self.config_key = config_key


class ConfigInvalidError(ConfigurationError):
    """Raised when configuration is invalid."""

    default_code = ErrorCode.CONFIG_INVALID

    def __init__(
        self,
        config_key: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Invalid configuration: {config_key}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.config_key = config_key


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(AgentPlatformError):
    """Base exception for validation errors."""

    default_code = ErrorCode.VALIDATION_ERROR
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value

        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class PreconditionFailedError(ValidationError):
    """Raised when precondition fails."""

    default_code = ErrorCode.PRECONDITION_FAILED

    def __init__(
        self,
        condition: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Precondition failed: {condition}", **kwargs)
        self.condition = condition


class PostconditionFailedError(ValidationError):
    """Raised when postcondition fails."""

    default_code = ErrorCode.POSTCONDITION_FAILED

    def __init__(
        self,
        condition: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Postcondition failed: {condition}", **kwargs)
        self.condition = condition


# =============================================================================
# NETWORK ERRORS
# =============================================================================


class NetworkError(AgentPlatformError):
    """Base exception for network errors."""

    default_code = ErrorCode.NETWORK_ERROR
    default_category = ErrorCategory.NETWORK
    default_severity = ErrorSeverity.MEDIUM

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ConnectionError(NetworkError):
    """Raised when connection fails."""

    default_code = ErrorCode.CONNECTION_FAILED

    def __init__(
        self,
        host: str,
        port: int | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        target = f"{host}:{port}" if port else host
        message = f"Connection failed to {target}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port


class TimeoutError(NetworkError):
    """Raised when operation times out."""

    default_code = ErrorCode.TIMEOUT
    default_category = ErrorCategory.TIMEOUT

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Operation timed out after {timeout_seconds}s: {operation}",
            **kwargs,
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# =============================================================================
# AUTHENTICATION/AUTHORIZATION ERRORS
# =============================================================================


class AuthError(AgentPlatformError):
    """Base exception for authentication/authorization errors."""

    default_code = ErrorCode.AUTH_ERROR
    default_category = ErrorCategory.AUTHENTICATION
    default_severity = ErrorSeverity.HIGH


class AuthenticationError(AuthError):
    """Raised when authentication fails."""

    default_code = ErrorCode.AUTHENTICATION_FAILED

    def __init__(
        self,
        reason: str = "Authentication failed",
        **kwargs: Any,
    ) -> None:
        super().__init__(reason, **kwargs)


class AuthorizationError(AuthError):
    """Raised when authorization fails."""

    default_code = ErrorCode.AUTHORIZATION_FAILED
    default_category = ErrorCategory.AUTHORIZATION

    def __init__(
        self,
        resource: str,
        action: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Not authorized to {action} on {resource}",
            **kwargs,
        )
        self.resource = resource
        self.action = action


class PermissionDeniedError(AuthError):
    """Raised when permission is denied."""

    default_code = ErrorCode.PERMISSION_DENIED
    default_category = ErrorCategory.AUTHORIZATION

    def __init__(
        self,
        permission: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Permission denied: {permission}", **kwargs)
        self.permission = permission


# =============================================================================
# RESOURCE ERRORS
# =============================================================================


class ResourceError(AgentPlatformError):
    """Base exception for resource errors."""

    default_code = ErrorCode.RESOURCE_ERROR
    default_category = ErrorCategory.RESOURCE
    default_severity = ErrorSeverity.MEDIUM


class ResourceNotFoundError(ResourceError):
    """Raised when resource is not found."""

    default_code = ErrorCode.RESOURCE_NOT_FOUND

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            **kwargs,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ResourceUnavailableError(ResourceError):
    """Raised when resource is unavailable."""

    default_code = ErrorCode.RESOURCE_UNAVAILABLE

    def __init__(
        self,
        resource: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        message = f"Resource unavailable: {resource}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.resource = resource


# =============================================================================
# ERROR HANDLER
# =============================================================================


class ErrorHandler:
    """
    Centralized error handling utilities.

    Provides methods for error wrapping, logging, and conversion.
    """

    @staticmethod
    def wrap(
        error: Exception,
        error_class: type[AgentPlatformError] = AgentPlatformError,
        **kwargs: Any,
    ) -> AgentPlatformError:
        """
        Wrap an exception in an AgentPlatformError.

        Args:
            error: Original exception
            error_class: Error class to use
            **kwargs: Additional error arguments

        Returns:
            Wrapped error
        """
        if isinstance(error, AgentPlatformError):
            return error

        return error_class(
            message=str(error),
            cause=error,
            **kwargs,
        )

    @staticmethod
    def from_http_status(
        status_code: int,
        message: str | None = None,
        **kwargs: Any,
    ) -> AgentPlatformError:
        """
        Create error from HTTP status code.

        Args:
            status_code: HTTP status code
            message: Error message
            **kwargs: Additional error arguments

        Returns:
            Appropriate error
        """
        error_map: dict[int, tuple[type[AgentPlatformError], str]] = {
            400: (ValidationError, "Bad request"),
            401: (AuthenticationError, "Unauthorized"),
            403: (PermissionDeniedError, "Forbidden"),
            404: (ResourceNotFoundError, "Not found"),
            408: (TimeoutError, "Request timeout"),
            429: (RateLimitExceededError, "Too many requests"),
            500: (AgentPlatformError, "Internal server error"),
            502: (NetworkError, "Bad gateway"),
            503: (ResourceUnavailableError, "Service unavailable"),
            504: (TimeoutError, "Gateway timeout"),
        }

        error_class, default_message = error_map.get(
            status_code,
            (AgentPlatformError, f"HTTP error {status_code}"),
        )

        return error_class(message or default_message, **kwargs)

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Check if error is retryable."""
        if isinstance(error, AgentPlatformError):
            return error.is_retryable
        return False

    @staticmethod
    def get_severity(error: Exception) -> ErrorSeverity:
        """Get error severity."""
        if isinstance(error, AgentPlatformError):
            return error.severity
        return ErrorSeverity.MEDIUM


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorCode",
    # Context
    "ErrorContext",
    # Base
    "AgentPlatformError",
    # Capability
    "CapabilityError",
    "CapabilityNotFoundError",
    "CapabilityUnavailableError",
    "CapabilityInitializationError",
    "CapabilityDependencyError",
    # Execution
    "ExecutionError",
    "StepExecutionError",
    "StepTimeoutError",
    "DependencyFailedError",
    "RetryExhaustedError",
    "ExecutionAbortedError",
    # Planning
    "PlanningError",
    "PlanGenerationError",
    "PlanValidationError",
    "ReplanError",
    "ReplanLimitExceededError",
    "CircularDependencyError",
    "NoViablePlanError",
    # Approval
    "ApprovalError",
    "ApprovalRequiredError",
    "ApprovalDeniedError",
    "ApprovalTimeoutError",
    "EscalationRequiredError",
    "DangerousActionBlockedError",
    # Budget
    "BudgetError",
    "BudgetExceededError",
    "TokenLimitExceededError",
    "CostLimitExceededError",
    "RateLimitExceededError",
    "QuotaExhaustedError",
    # Checkpoint
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptedError",
    "CheckpointSaveError",
    "CheckpointLoadError",
    "RecoveryError",
    "RollbackError",
    # Router
    "RouterError",
    "NoMatchingCapabilityError",
    "AmbiguousRoutingError",
    "RoutingLoopError",
    "RouteNotFoundError",
    # State
    "StateError",
    "StateNotFoundError",
    "StateCorruptedError",
    "InvalidStateTransitionError",
    "StateLockError",
    "SessionExpiredError",
    "SessionNotFoundError",
    # Configuration
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigInvalidError",
    # Validation
    "ValidationError",
    "PreconditionFailedError",
    "PostconditionFailedError",
    # Network
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    # Auth
    "AuthError",
    "AuthenticationError",
    "AuthorizationError",
    "PermissionDeniedError",
    # Resource
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceUnavailableError",
    # Handler
    "ErrorHandler",
]
