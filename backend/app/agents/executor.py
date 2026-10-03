"""
Executor - Plan execution engine for the AI Agent Platform.

This module handles the execution of plan steps including:
- Sequential and parallel step execution
- Timeout management
- Failure handling and retries
- Result collection and aggregation
- Execution lifecycle hooks
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from app.agents.models import (
    CapabilityType,
    ExecutionPlan,
    PlanStep,
    SessionState,
    StepResult,
)
from app.providers.provider_router import ProviderRouter, TaskType

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
CapabilityResult = TypeVar("CapabilityResult")


# =============================================================================
# PROTOCOLS
# =============================================================================


class Capability(Protocol[CapabilityResult]):  # type: ignore[misc]
    """
    Protocol for capability implementations.

    Capabilities are modular components that provide specific
    functionality (memory, tools, RAG, web search).
    """

    @property
    def capability_type(self) -> CapabilityType:
        """Return the type of this capability."""
        ...

    @property
    def name(self) -> str:
        """Return the unique name of this capability."""
        ...

    async def execute(
        self,
        action: str,
        parameters: dict[str, Any],
        context: SessionState,
    ) -> CapabilityResult:
        """
        Execute an action with the given parameters.

        Args:
            action: The specific action to perform
            parameters: Action parameters
            context: Current session state

        Returns:
            Capability-specific result
        """
        ...

    async def health_check(self) -> bool:
        """Check if the capability is operational."""
        ...


# =============================================================================
# EXECUTION EVENTS
# =============================================================================


class ExecutionEventType(Enum):
    """Types of execution events for hooks and monitoring."""

    PLAN_STARTED = auto()
    PLAN_COMPLETED = auto()
    PLAN_FAILED = auto()

    STEP_STARTED = auto()
    STEP_COMPLETED = auto()
    STEP_FAILED = auto()
    STEP_RETRYING = auto()
    STEP_SKIPPED = auto()
    STEP_TIMEOUT = auto()


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """
    Event emitted during plan execution.

    Attributes:
        event_type: Type of event
        plan_id: Associated plan ID
        step_id: Associated step ID (if applicable)
        timestamp: When event occurred
        data: Additional event data
    """

    event_type: ExecutionEventType
    plan_id: str
    step_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for event handlers
EventHandler = Callable[[ExecutionEvent], Awaitable[None] | None]


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass
class StepExecutionContext:
    """
    Context for executing a single step.

    Attributes:
        step: The step being executed
        session: Current session state
        previous_results: Results from completed steps
        attempt: Current attempt number (1-based)
        start_time: When execution started
    """

    step: PlanStep
    session: SessionState
    previous_results: dict[str, StepResult]
    attempt: int = 1
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (time.monotonic() - self.start_time) * 1000

    def get_dependency_outputs(self) -> dict[str, Any]:
        """Get outputs from dependency steps."""
        return {
            dep_id: self.previous_results[dep_id].output
            for dep_id in self.step.dependencies
            if dep_id in self.previous_results
            and self.previous_results[dep_id].success
        }


@dataclass
class PlanExecutionContext:
    """
    Context for executing an entire plan.

    Attributes:
        plan: The plan being executed
        session: Current session state
        results: Accumulated step results
        start_time: When execution started
        events: Emitted execution events
    """

    plan: ExecutionPlan
    session: SessionState
    results: list[StepResult] = field(default_factory=list)
    results_by_id: dict[str, StepResult] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)
    events: list[ExecutionEvent] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        return (time.monotonic() - self.start_time) * 1000

    @property
    def successful_steps(self) -> int:
        """Count of successful steps."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_steps(self) -> int:
        """Count of failed steps."""
        return sum(1 for r in self.results if not r.success)

    def add_result(self, result: StepResult) -> None:
        """Add a step result."""
        self.results.append(result)
        self.results_by_id[result.step_id] = result

    def get_result(self, step_id: str) -> StepResult | None:
        """Get result for a specific step."""
        return self.results_by_id.get(step_id)


# =============================================================================
# RETRY STRATEGIES
# =============================================================================


class RetryStrategy(ABC):
    """Abstract base for retry strategies."""

    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if execution should be retried.

        Args:
            attempt: Current attempt number (1-based)
            error: The error that occurred

        Returns:
            True if should retry
        """
        ...

    @abstractmethod
    async def wait_before_retry(self, attempt: int) -> None:
        """
        Wait before retrying.

        Args:
            attempt: Current attempt number
        """
        ...


class NoRetryStrategy(RetryStrategy):
    """Strategy that never retries."""

    def should_retry(self, attempt: int, error: Exception) -> bool:
        return False

    async def wait_before_retry(self, attempt: int) -> None:
        pass


class FixedRetryStrategy(RetryStrategy):
    """
    Retry with fixed delay between attempts.

    Attributes:
        max_attempts: Maximum number of attempts
        delay_seconds: Fixed delay between retries
        retryable_errors: Error types that can be retried
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay_seconds: float = 1.0,
        retryable_errors: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.retryable_errors = retryable_errors or (Exception,)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retryable_errors)

    async def wait_before_retry(self, attempt: int) -> None:
        await asyncio.sleep(self.delay_seconds)


class ExponentialBackoffStrategy(RetryStrategy):
    """
    Retry with exponential backoff.

    Attributes:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        multiplier: Backoff multiplier
        retryable_errors: Error types that can be retried
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        multiplier: float = 2.0,
        retryable_errors: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.retryable_errors = retryable_errors or (Exception,)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retryable_errors)

    async def wait_before_retry(self, attempt: int) -> None:
        delay = min(
            self.base_delay * (self.multiplier ** (attempt - 1)),
            self.max_delay,
        )
        await asyncio.sleep(delay)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ExecutionError(Exception):
    """Base exception for execution errors."""

    def __init__(
        self,
        message: str,
        step_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step_id = step_id
        self.cause = cause


class StepTimeoutError(ExecutionError):
    """Step execution timed out."""

    def __init__(self, step_id: str, timeout_seconds: int) -> None:
        super().__init__(
            f"Step {step_id} timed out after {timeout_seconds}s",
            step_id=step_id,
        )
        self.timeout_seconds = timeout_seconds


class StepExecutionError(ExecutionError):
    """Step execution failed."""

    pass


class CapabilityNotFoundError(ExecutionError):
    """Required capability not available."""

    def __init__(self, capability: CapabilityType, step_id: str) -> None:
        super().__init__(
            f"Capability {capability.value} not found for step {step_id}",
            step_id=step_id,
        )
        self.capability = capability


class DependencyFailedError(ExecutionError):
    """Step dependency failed."""

    def __init__(self, step_id: str, failed_dependency: str) -> None:
        super().__init__(
            f"Step {step_id} dependency {failed_dependency} failed",
            step_id=step_id,
        )
        self.failed_dependency = failed_dependency


# =============================================================================
# EXECUTOR CONFIGURATION
# =============================================================================


@dataclass
class ExecutorConfig:
    """
    Configuration for the Executor.

    Attributes:
        default_timeout_seconds: Default step timeout
        max_parallel_steps: Maximum concurrent step executions
        retry_strategy: Default retry strategy
        fail_fast: Stop on first failure
        collect_events: Whether to collect execution events
        inject_previous_results: Inject dependency results into parameters
    """

    default_timeout_seconds: int = 30
    max_parallel_steps: int = 5
    retry_strategy: RetryStrategy = field(default_factory=lambda: ExponentialBackoffStrategy())
    fail_fast: bool = False
    collect_events: bool = True
    inject_previous_results: bool = True


# =============================================================================
# STEP EXECUTOR
# =============================================================================


class StepExecutor:
    """
    Executes individual plan steps.

    Handles timeout, retries, and result creation for single steps.
    """

    def __init__(
        self,
        capabilities: dict[CapabilityType, Capability[Any]],
        config: ExecutorConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize step executor.

        Args:
            capabilities: Available capability implementations
            config: Executor configuration
            logger: Optional logger
        """
        self._capabilities = capabilities
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        context: StepExecutionContext,
    ) -> StepResult:
        """
        Execute a single step with retries.

        Args:
            context: Step execution context

        Returns:
            StepResult from execution
        """
        step = context.step
        last_error: Exception | None = None

        while True:
            try:
                result = await self._execute_once(context)
                return result

            except Exception as e:
                last_error = e
                self._logger.warning(
                    f"Step {step.step_id} attempt {context.attempt} failed: {e}"
                )

                if self._config.retry_strategy.should_retry(context.attempt, e):
                    self._logger.info(
                        f"Retrying step {step.step_id} (attempt {context.attempt + 1})"
                    )
                    await self._config.retry_strategy.wait_before_retry(context.attempt)
                    context.attempt += 1
                else:
                    break

        # All retries exhausted
        return self._create_failure_result(context, last_error)

    async def _execute_once(
        self,
        context: StepExecutionContext,
    ) -> StepResult:
        """Execute step once without retry logic."""
        step = context.step
        start_time = time.monotonic()

        # Get capability
        capability = self._capabilities.get(step.capability)
        if capability is None:
            raise CapabilityNotFoundError(step.capability, step.step_id)

        # Prepare parameters
        parameters = self._prepare_parameters(context)

        # Execute with timeout
        timeout = step.timeout_seconds or self._config.default_timeout_seconds

        try:
            output = await asyncio.wait_for(
                capability.execute(step.action, parameters, context.session),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise StepTimeoutError(step.step_id, timeout)

        execution_time = (time.monotonic() - start_time) * 1000

        return StepResult.success_result(
            step_id=step.step_id,
            output=output,
            execution_time_ms=execution_time,
            metadata={
                "capability": step.capability.value,
                "action": step.action,
                "attempt": context.attempt,
            },
        )

    def _prepare_parameters(
        self,
        context: StepExecutionContext,
    ) -> dict[str, Any]:
        """Prepare parameters for step execution."""
        parameters = dict(context.step.parameters)

        if self._config.inject_previous_results:
            parameters["_previous_results"] = context.get_dependency_outputs()
            parameters["_session_id"] = context.session.session_id

        return parameters

    def _create_failure_result(
        self,
        context: StepExecutionContext,
        error: Exception | None,
    ) -> StepResult:
        """Create a failure result from an error."""
        execution_time = context.elapsed_ms

        error_message = str(error) if error else "Unknown error"

        return StepResult.failure_result(
            step_id=context.step.step_id,
            error=error_message,
            execution_time_ms=execution_time,
            metadata={
                "capability": context.step.capability.value,
                "action": context.step.action,
                "attempts": context.attempt,
                "error_type": type(error).__name__ if error else "Unknown",
            },
        )


# =============================================================================
# MAIN EXECUTOR
# =============================================================================


class Executor:
    """
    Executes complete execution plans.

    Coordinates step execution, handles dependencies, manages
    parallel execution, and collects results.

    This is the main entry point for plan execution.
    """

    def __init__(
        self,
        capabilities: dict[CapabilityType, Capability[Any]],
        config: ExecutorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            capabilities: Available capability implementations
            config: Executor configuration
            logger: Optional logger
        """
        self._capabilities = capabilities
        self._config = config or ExecutorConfig()
        self._logger = logger or logging.getLogger(__name__)

        self._step_executor = StepExecutor(
            capabilities=capabilities,
            config=self._config,
            logger=logger,
        )

        # Event handlers
        self._event_handlers: list[EventHandler] = []

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def register_capability(
        self,
        capability: Capability[Any],
    ) -> None:
        """Register a capability implementation."""
        self._capabilities[capability.capability_type] = capability

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an execution event handler."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove an execution event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    # -------------------------------------------------------------------------
    # Plan Execution
    # -------------------------------------------------------------------------

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
        context = PlanExecutionContext(plan=plan, session=session)

        # Emit plan started event
        await self._emit_event(
            ExecutionEvent(
                event_type=ExecutionEventType.PLAN_STARTED,
                plan_id=plan.plan_id,
                data={"step_count": plan.step_count},
            ),
            context,
        )

        plan.mark_executing()

        try:
            # Get execution order
            ordered_steps = plan.ordered_steps()

            # Execute steps
            for step in ordered_steps:
                # Check dependencies
                if not self._check_dependencies(step, context):
                    await self._handle_dependency_failure(step, context)
                    if self._config.fail_fast:
                        break
                    continue

                # Execute step
                result = await self._execute_step(step, context)
                context.add_result(result)

                # Update step status
                if result.success:
                    step.mark_completed(result)
                else:
                    step.mark_failed(result)
                    if self._config.fail_fast:
                        break

            # Determine final plan status
            if context.failed_steps > 0:
                plan.mark_failed()
                await self._emit_event(
                    ExecutionEvent(
                        event_type=ExecutionEventType.PLAN_FAILED,
                        plan_id=plan.plan_id,
                        data={
                            "successful_steps": context.successful_steps,
                            "failed_steps": context.failed_steps,
                            "elapsed_ms": context.elapsed_ms,
                        },
                    ),
                    context,
                )
            else:
                plan.mark_completed()
                await self._emit_event(
                    ExecutionEvent(
                        event_type=ExecutionEventType.PLAN_COMPLETED,
                        plan_id=plan.plan_id,
                        data={
                            "successful_steps": context.successful_steps,
                            "elapsed_ms": context.elapsed_ms,
                        },
                    ),
                    context,
                )

        except Exception as e:
            self._logger.exception(f"Plan execution failed: {e}")
            plan.mark_failed()
            raise ExecutionError(
                f"Plan execution failed: {e}",
                cause=e,
            )

        return context.results

    async def execute_plan_parallel(
        self,
        plan: ExecutionPlan,
        session: SessionState,
    ) -> list[StepResult]:
        """
        Execute plan with parallel step execution where possible.

        Steps without dependencies or with satisfied dependencies
        are executed concurrently up to max_parallel_steps.

        Args:
            plan: The execution plan
            session: Current session state

        Returns:
            List of results from all executed steps
        """
        context = PlanExecutionContext(plan=plan, session=session)

        await self._emit_event(
            ExecutionEvent(
                event_type=ExecutionEventType.PLAN_STARTED,
                plan_id=plan.plan_id,
                data={"step_count": plan.step_count, "parallel": True},
            ),
            context,
        )

        plan.mark_executing()

        try:
            pending_steps = set(step.step_id for step in plan.steps)
            completed_steps: set[str] = set()

            while pending_steps:
                # Find ready steps (dependencies satisfied)
                ready_steps = [
                    step
                    for step in plan.steps
                    if step.step_id in pending_steps
                    and all(dep in completed_steps for dep in step.dependencies)
                ]

                if not ready_steps:
                    if pending_steps:
                        # Deadlock or all remaining have failed dependencies
                        self._logger.warning(
                            f"No ready steps but {len(pending_steps)} pending"
                        )
                        break
                    continue

                # Limit parallel execution
                batch = ready_steps[: self._config.max_parallel_steps]

                # Execute batch in parallel
                tasks = [
                    self._execute_step(step, context)
                    for step in batch
                    if self._check_dependencies(step, context)
                ]

                # Handle steps with failed dependencies
                for step in batch:
                    if not self._check_dependencies(step, context):
                        await self._handle_dependency_failure(step, context)
                        pending_steps.discard(step.step_id)

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for step, result in zip(batch, results):
                        if isinstance(result, BaseException):
                            result = StepResult.failure_result(
                                step_id=step.step_id,
                                error=str(result),
                            )

                        context.add_result(result)
                        pending_steps.discard(step.step_id)
                        completed_steps.add(step.step_id)

                        if result.success:
                            step.mark_completed(result)
                        else:
                            step.mark_failed(result)
                            if self._config.fail_fast:
                                pending_steps.clear()
                                break

            # Determine final status
            if context.failed_steps > 0:
                plan.mark_failed()
            else:
                plan.mark_completed()

        except Exception as e:
            self._logger.exception(f"Parallel plan execution failed: {e}")
            plan.mark_failed()
            raise

        return context.results

    # -------------------------------------------------------------------------
    # Step Execution
    # -------------------------------------------------------------------------

    async def _execute_step(
        self,
        step: PlanStep,
        context: PlanExecutionContext,
    ) -> StepResult:
        """Execute a single step within plan context."""
        step.mark_running()

        await self._emit_event(
            ExecutionEvent(
                event_type=ExecutionEventType.STEP_STARTED,
                plan_id=context.plan.plan_id,
                step_id=step.step_id,
                data={
                    "capability": step.capability.value,
                    "action": step.action,
                },
            ),
            context,
        )

        step_context = StepExecutionContext(
            step=step,
            session=context.session,
            previous_results=context.results_by_id,
        )

        result = await self._step_executor.execute(step_context)

        # Emit appropriate event
        if result.success:
            if result.was_skipped:
                event_type = ExecutionEventType.STEP_SKIPPED
            else:
                event_type = ExecutionEventType.STEP_COMPLETED
        else:
            if "timeout" in (result.error or "").lower():
                event_type = ExecutionEventType.STEP_TIMEOUT
            else:
                event_type = ExecutionEventType.STEP_FAILED

        await self._emit_event(
            ExecutionEvent(
                event_type=event_type,
                plan_id=context.plan.plan_id,
                step_id=step.step_id,
                data={
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error,
                },
            ),
            context,
        )

        return result

    async def execute_single_step(
        self,
        step: PlanStep,
        session: SessionState,
        previous_results: dict[str, StepResult] | None = None,
    ) -> StepResult:
        """
        Execute a single step outside of a plan context.

        Useful for testing or ad-hoc step execution.

        Args:
            step: Step to execute
            session: Session state
            previous_results: Optional previous results for dependencies

        Returns:
            StepResult from execution
        """
        step_context = StepExecutionContext(
            step=step,
            session=session,
            previous_results=previous_results or {},
        )

        step.mark_running()
        result = await self._step_executor.execute(step_context)

        if result.success:
            step.mark_completed(result)
        else:
            step.mark_failed(result)

        return result

    # -------------------------------------------------------------------------
    # Dependency Handling
    # -------------------------------------------------------------------------

    def _check_dependencies(
        self,
        step: PlanStep,
        context: PlanExecutionContext,
    ) -> bool:
        """Check if all step dependencies are satisfied."""
        for dep_id in step.dependencies:
            result = context.get_result(dep_id)
            if result is None or not result.success:
                return False
        return True

    async def _handle_dependency_failure(
        self,
        step: PlanStep,
        context: PlanExecutionContext,
    ) -> None:
        """Handle a step whose dependencies failed."""
        # Find which dependency failed
        failed_dep = None
        for dep_id in step.dependencies:
            result = context.get_result(dep_id)
            if result is not None and not result.success:
                failed_dep = dep_id
                break

        # Check for fallback
        if step.fallback_action == "skip":
            reason = f"Dependency {failed_dep} failed" if failed_dep else "Dependency not met"
            step.mark_skipped(reason)
            context.add_result(StepResult.skipped_result(step.step_id, reason))

            await self._emit_event(
                ExecutionEvent(
                    event_type=ExecutionEventType.STEP_SKIPPED,
                    plan_id=context.plan.plan_id,
                    step_id=step.step_id,
                    data={"reason": reason},
                ),
                context,
            )
        else:
            error = f"Dependency {failed_dep} failed" if failed_dep else "Dependency not satisfied"
            step.mark_failed(StepResult.failure_result(step.step_id, error))
            context.add_result(StepResult.failure_result(step.step_id, error))

            await self._emit_event(
                ExecutionEvent(
                    event_type=ExecutionEventType.STEP_FAILED,
                    plan_id=context.plan.plan_id,
                    step_id=step.step_id,
                    data={"error": error, "reason": "dependency_failed"},
                ),
                context,
            )

    # -------------------------------------------------------------------------
    # Fallback Execution
    # -------------------------------------------------------------------------

    async def execute_fallback(
        self,
        step: PlanStep,
        original_error: Exception,
        context: PlanExecutionContext,
    ) -> StepResult:
        """
        Execute fallback action for a failed step.

        Args:
            step: The failed step
            original_error: Error from original execution
            context: Plan execution context

        Returns:
            StepResult from fallback execution
        """
        if step.fallback_action is None:
            return StepResult.failure_result(
                step_id=step.step_id,
                error=f"No fallback available: {original_error}",
            )

        if step.fallback_action == "skip":
            return StepResult.skipped_result(
                step_id=step.step_id,
                reason=f"Skipped after error: {original_error}",
            )

        # Execute fallback action
        self._logger.info(
            f"Executing fallback '{step.fallback_action}' for step {step.step_id}"
        )

        fallback_step = PlanStep(
            step_id=f"{step.step_id}_fallback",
            capability=step.capability,
            action=step.fallback_action,
            parameters=step.parameters,
            timeout_seconds=step.timeout_seconds,
        )

        return await self.execute_single_step(
            fallback_step,
            context.session,
            context.results_by_id,
        )

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    async def _emit_event(
        self,
        event: ExecutionEvent,
        context: PlanExecutionContext,
    ) -> None:
        """Emit an execution event to all handlers."""
        if self._config.collect_events:
            context.events.append(event)

        for handler in self._event_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._logger.warning(f"Event handler error: {e}")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all registered capabilities.

        Returns:
            Dictionary mapping capability names to health status
        """
        health: dict[str, bool] = {}

        for cap_type, capability in self._capabilities.items():
            try:
                health[cap_type.value] = await capability.health_check()
            except Exception:
                health[cap_type.value] = False

        return health

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics and configuration."""
        return {
            "capabilities": list(self._capabilities.keys()),
            "capability_count": len(self._capabilities),
            "config": {
                "default_timeout_seconds": self._config.default_timeout_seconds,
                "max_parallel_steps": self._config.max_parallel_steps,
                "fail_fast": self._config.fail_fast,
            },
            "event_handlers": len(self._event_handlers),
        }


# =============================================================================
# EXECUTION RESULT AGGREGATOR
# =============================================================================


class ResultAggregator:
    """
    Aggregates and analyzes execution results.

    Provides utilities for combining results, extracting
    outputs, and generating summaries.
    """

    @staticmethod
    def get_successful_outputs(results: list[StepResult]) -> list[Any]:
        """Extract outputs from successful steps."""
        return [r.output for r in results if r.success and r.output is not None]

    @staticmethod
    def get_failed_errors(results: list[StepResult]) -> list[str]:
        """Extract error messages from failed steps."""
        return [r.error for r in results if not r.success and r.error]

    @staticmethod
    def get_total_execution_time(results: list[StepResult]) -> float:
        """Calculate total execution time in milliseconds."""
        return sum(r.execution_time_ms for r in results)

    @staticmethod
    def get_outputs_by_capability(
        results: list[StepResult],
    ) -> dict[str, list[Any]]:
        """Group outputs by capability type."""
        by_capability: dict[str, list[Any]] = {}

        for result in results:
            if result.success and result.output is not None:
                capability = result.metadata.get("capability", "unknown")
                if capability not in by_capability:
                    by_capability[capability] = []
                by_capability[capability].append(result.output)

        return by_capability

    @staticmethod
    def create_summary(results: list[StepResult]) -> dict[str, Any]:
        """Create execution summary from results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        skipped = [r for r in results if r.was_skipped]

        return {
            "total_steps": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_execution_time_ms": ResultAggregator.get_total_execution_time(results),
            "errors": ResultAggregator.get_failed_errors(results),
            "capabilities_used": list(
                {r.metadata.get("capability") for r in results if r.metadata}
            ),
        }


# =============================================================================
# PROVIDER CAPABILITY ADAPTER
# =============================================================================


class ProviderCapability:
    """Capability adapter that routes LLM actions through the provider system."""

    def __init__(
        self,
        router: ProviderRouter,
        task_type: TaskType = TaskType.PLANNING,
        name: str = "provider_llm",
        logger: logging.Logger | None = None,
    ) -> None:
        self._router = router
        self._task_type = task_type
        self._name = name
        self._logger = logger or logging.getLogger(__name__)

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.RAG

    @property
    def name(self) -> str:
        return self._name

    async def execute(
        self,
        action: str,
        parameters: dict[str, Any],
        context: SessionState,
    ) -> Any:
        prompt = parameters.get("prompt", "")
        system_prompt = parameters.get("system_prompt")
        temperature = parameters.get("temperature")
        max_tokens = parameters.get("max_tokens")

        provider = self._router.select(self._task_type)
        content, usage = await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": content,
            "model": provider.metadata.model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

    async def health_check(self) -> bool:
        try:
            provider = self._router.select(self._task_type)
            return await provider.health_check()
        except Exception:
            return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_executor(
    capabilities: dict[CapabilityType, Capability[Any]] | None = None,
    config: ExecutorConfig | None = None,
    retry_strategy: RetryStrategy | None = None,
    logger: logging.Logger | None = None,
) -> Executor:
    """
    Factory function to create a configured Executor.

    Args:
        capabilities: Capability implementations
        config: Executor configuration
        retry_strategy: Override retry strategy
        logger: Optional logger

    Returns:
        Configured Executor instance
    """
    if config is None:
        config = ExecutorConfig()

    if retry_strategy is not None:
        config.retry_strategy = retry_strategy

    return Executor(
        capabilities=capabilities or {},
        config=config,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    "Capability",
    # Events
    "ExecutionEventType",
    "ExecutionEvent",
    "EventHandler",
    # Context
    "StepExecutionContext",
    "PlanExecutionContext",
    # Retry Strategies
    "RetryStrategy",
    "NoRetryStrategy",
    "FixedRetryStrategy",
    "ExponentialBackoffStrategy",
    # Exceptions
    "ExecutionError",
    "StepTimeoutError",
    "StepExecutionError",
    "CapabilityNotFoundError",
    "DependencyFailedError",
    # Configuration
    "ExecutorConfig",
    # Executors
    "StepExecutor",
    "Executor",
    # Utilities
    "ResultAggregator",
    # Provider Capability
    "ProviderCapability",
    # Factory
    "create_executor",
]
