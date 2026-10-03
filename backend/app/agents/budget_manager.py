"""
Budget Manager - Resource budget management for the AI Agent Platform.

This module provides comprehensive budget management including:
- Token budgets (input, output, total)
- Cost/money budgets with model pricing
- Execution budgets (requests, steps, retries)
- Latency budgets (per-request, total)
- Model routing based on budget constraints
- Alerts and threshold monitoring
- Usage tracking and reporting
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class BudgetType(Enum):
    """Types of budgets in the system."""

    TOKEN = "token"
    COST = "cost"
    EXECUTION = "execution"
    LATENCY = "latency"
    COMPOSITE = "composite"


class AlertSeverity(Enum):
    """Severity levels for budget alerts."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class AlertType(Enum):
    """Types of budget alerts."""

    THRESHOLD_REACHED = auto()
    BUDGET_EXCEEDED = auto()
    RATE_LIMIT_APPROACHING = auto()
    ANOMALY_DETECTED = auto()
    BUDGET_RESET = auto()


class BudgetPeriod(Enum):
    """Time periods for budget tracking."""

    PER_REQUEST = "per_request"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"
    LIFETIME = "lifetime"


class ModelTier(Enum):
    """Model capability tiers for routing."""

    ECONOMY = auto()  # Cheapest, basic capability
    STANDARD = auto()  # Balanced cost/capability
    PREMIUM = auto()  # High capability, higher cost
    FLAGSHIP = auto()  # Best capability, highest cost


# =============================================================================
# EXCEPTIONS
# =============================================================================


class BudgetError(Exception):
    """Base exception for budget errors."""

    def __init__(self, message: str, budget_type: BudgetType | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.budget_type = budget_type


class BudgetExceededError(BudgetError):
    """Budget has been exceeded."""

    def __init__(
        self,
        message: str,
        budget_type: BudgetType,
        limit: float,
        current: float,
        requested: float,
    ) -> None:
        super().__init__(message, budget_type)
        self.limit = limit
        self.current = current
        self.requested = requested


class InsufficientBudgetError(BudgetError):
    """Insufficient budget for operation."""

    pass


# =============================================================================
# ALERTS
# =============================================================================


@dataclass(frozen=True)
class BudgetAlert:
    """
    Budget alert notification.

    Attributes:
        alert_type: Type of alert
        severity: Alert severity
        budget_type: Which budget triggered alert
        message: Human-readable message
        current_value: Current budget value
        threshold_value: Threshold that was crossed
        timestamp: When alert was generated
        metadata: Additional alert data
    """

    alert_type: AlertType
    severity: AlertSeverity
    budget_type: BudgetType
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for alert handlers
AlertHandler = Callable[[BudgetAlert], Awaitable[None] | None]


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================


@dataclass
class ThresholdConfig:
    """
    Configuration for budget thresholds.

    Attributes:
        warning_percent: Percentage for warning alert (0-100)
        critical_percent: Percentage for critical alert (0-100)
        emergency_percent: Percentage for emergency alert (0-100)
        hard_limit: Whether to enforce hard limit
        auto_downgrade: Auto-downgrade to cheaper options
    """

    warning_percent: float = 75.0
    critical_percent: float = 90.0
    emergency_percent: float = 99.0
    hard_limit: bool = True
    auto_downgrade: bool = False

    def get_threshold_for_limit(self, limit: float, severity: AlertSeverity) -> float:
        """Calculate threshold value for a limit."""
        percent_map = {
            AlertSeverity.INFO: 50.0,
            AlertSeverity.WARNING: self.warning_percent,
            AlertSeverity.CRITICAL: self.critical_percent,
            AlertSeverity.EMERGENCY: self.emergency_percent,
        }
        percent = percent_map.get(severity, self.warning_percent)
        return limit * (percent / 100.0)


# =============================================================================
# MODEL PRICING
# =============================================================================


@dataclass(frozen=True)
class ModelPricing:
    """
    Pricing information for a model.

    Attributes:
        model_id: Model identifier
        input_cost_per_1k: Cost per 1000 input tokens
        output_cost_per_1k: Cost per 1000 output tokens
        tier: Model capability tier
        context_window: Maximum context size
        max_output_tokens: Maximum output tokens
        currency: Currency code (default: USD)
    """

    model_id: str
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    tier: ModelTier = ModelTier.STANDARD
    context_window: int = 8192
    max_output_tokens: int = 4096
    currency: str = "USD"

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Calculate total cost for token usage."""
        input_cost = (Decimal(input_tokens) / 1000) * self.input_cost_per_1k
        output_cost = (Decimal(output_tokens) / 1000) * self.output_cost_per_1k
        return input_cost + output_cost

    def estimate_cost(
        self,
        estimated_input: int,
        estimated_output: int,
    ) -> Decimal:
        """Estimate cost for planned usage."""
        return self.calculate_cost(estimated_input, estimated_output)


# Default NVIDIA NIM pricing (primary models)
NVIDIA_PRICING: dict[str, ModelPricing] = {
    "deepseek-ai/deepseek-v4-pro": ModelPricing(
        model_id="deepseek-ai/deepseek-v4-pro",
        input_cost_per_1k=Decimal("0.0005"),
        output_cost_per_1k=Decimal("0.0015"),
        tier=ModelTier.FLAGSHIP,
        context_window=131072,
        max_output_tokens=16384,
    ),
    "deepseek-ai/deepseek-v4-flash": ModelPricing(
        model_id="deepseek-ai/deepseek-v4-flash",
        input_cost_per_1k=Decimal("0.0001"),
        output_cost_per_1k=Decimal("0.0003"),
        tier=ModelTier.ECONOMY,
        context_window=131072,
        max_output_tokens=16384,
    ),
    "meta/llama-3.3-70b-instruct": ModelPricing(
        model_id="meta/llama-3.3-70b-instruct",
        input_cost_per_1k=Decimal("0.00035"),
        output_cost_per_1k=Decimal("0.0004"),
        tier=ModelTier.STANDARD,
        context_window=128000,
        max_output_tokens=4096,
    ),
    "qwen/qwen3.5-397b-a17b": ModelPricing(
        model_id="qwen/qwen3.5-397b-a17b",
        input_cost_per_1k=Decimal("0.0012"),
        output_cost_per_1k=Decimal("0.0012"),
        tier=ModelTier.PREMIUM,
        context_window=131072,
        max_output_tokens=16384,
    ),
    "meta/llama-3.2-90b-vision-instruct": ModelPricing(
        model_id="meta/llama-3.2-90b-vision-instruct",
        input_cost_per_1k=Decimal("0.0005"),
        output_cost_per_1k=Decimal("0.0005"),
        tier=ModelTier.STANDARD,
        context_window=128000,
        max_output_tokens=512,
    ),
}
        tier=ModelTier.ECONOMY,
        context_window=200000,
        max_output_tokens=4096,
    ),
}


# =============================================================================
# USAGE TRACKING
# =============================================================================


@dataclass
class UsageRecord:
    """
    Record of resource usage.

    Attributes:
        timestamp: When usage occurred
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        cost: Cost incurred
        latency_ms: Request latency
        model_id: Model used
        request_id: Associated request
        metadata: Additional data
    """

    timestamp: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = Decimal("0")
    latency_ms: float = 0.0
    model_id: str = ""
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """
    Tracks resource usage over time.

    Maintains rolling windows for different time periods.
    """

    def __init__(
        self,
        max_history: int = 10000,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize usage tracker.

        Args:
            max_history: Maximum records to keep
            logger: Optional logger
        """
        self._records: list[UsageRecord] = []
        self._max_history = max_history
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

    async def record(self, usage: UsageRecord) -> None:
        """Record a usage event."""
        async with self._lock:
            self._records.append(usage)

            # Trim old records
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history:]

    def get_usage_for_period(
        self,
        period: BudgetPeriod,
        reference_time: datetime | None = None,
    ) -> list[UsageRecord]:
        """Get usage records for a time period."""
        reference_time = reference_time or datetime.now(timezone.utc)

        if period == BudgetPeriod.PER_REQUEST:
            return self._records[-1:] if self._records else []

        # Calculate cutoff time
        cutoff_map = {
            BudgetPeriod.PER_MINUTE: timedelta(minutes=1),
            BudgetPeriod.PER_HOUR: timedelta(hours=1),
            BudgetPeriod.PER_DAY: timedelta(days=1),
            BudgetPeriod.PER_MONTH: timedelta(days=30),
            BudgetPeriod.LIFETIME: timedelta(days=36500),  # ~100 years
        }

        delta = cutoff_map.get(period, timedelta(hours=1))
        cutoff = reference_time - delta

        return [r for r in self._records if r.timestamp >= cutoff]

    def get_totals_for_period(
        self,
        period: BudgetPeriod,
    ) -> dict[str, Any]:
        """Get aggregated totals for a period."""
        records = self.get_usage_for_period(period)

        return {
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "total_cost": sum(r.cost for r in records),
            "total_latency_ms": sum(r.latency_ms for r in records),
            "request_count": len(records),
            "avg_latency_ms": (
                sum(r.latency_ms for r in records) / len(records)
                if records else 0
            ),
        }

    def clear(self) -> None:
        """Clear all usage records."""
        self._records.clear()


# =============================================================================
# BUDGET IMPLEMENTATIONS
# =============================================================================


class Budget(ABC):
    """Abstract base for budget implementations."""

    @property
    @abstractmethod
    def budget_type(self) -> BudgetType:
        """Get budget type."""
        ...

    @property
    @abstractmethod
    def limit(self) -> float:
        """Get budget limit."""
        ...

    @property
    @abstractmethod
    def used(self) -> float:
        """Get amount used."""
        ...

    @property
    def remaining(self) -> float:
        """Get remaining budget."""
        return max(0, self.limit - self.used)

    @property
    def usage_percent(self) -> float:
        """Get usage as percentage."""
        if self.limit == 0:
            return 0.0
        return (self.used / self.limit) * 100

    @abstractmethod
    def can_afford(self, amount: float) -> bool:
        """Check if amount can be afforded."""
        ...

    @abstractmethod
    def consume(self, amount: float) -> None:
        """Consume budget amount."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset budget to initial state."""
        ...


@dataclass
class TokenBudget(Budget):
    """
    Token-based budget.

    Attributes:
        input_limit: Maximum input tokens
        output_limit: Maximum output tokens
        total_limit: Maximum total tokens
        period: Budget period
    """

    input_limit: int = 0
    output_limit: int = 0
    total_limit: int = 1000000
    period: BudgetPeriod = BudgetPeriod.PER_DAY

    _input_used: int = field(default=0, repr=False)
    _output_used: int = field(default=0, repr=False)

    @property
    def budget_type(self) -> BudgetType:
        return BudgetType.TOKEN

    @property
    def limit(self) -> float:
        return float(self.total_limit)

    @property
    def used(self) -> float:
        return float(self._input_used + self._output_used)

    @property
    def input_used(self) -> int:
        return self._input_used

    @property
    def output_used(self) -> int:
        return self._output_used

    @property
    def input_remaining(self) -> int:
        if self.input_limit == 0:
            return self.total_limit - self._input_used
        return max(0, self.input_limit - self._input_used)

    @property
    def output_remaining(self) -> int:
        if self.output_limit == 0:
            return self.total_limit - self._output_used
        return max(0, self.output_limit - self._output_used)

    def can_afford(self, amount: float) -> bool:
        return self.used + amount <= self.total_limit

    def can_afford_tokens(self, input_tokens: int, output_tokens: int) -> bool:
        """Check if specific token amounts can be afforded."""
        # Check individual limits
        if self.input_limit > 0 and self._input_used + input_tokens > self.input_limit:
            return False
        if self.output_limit > 0 and self._output_used + output_tokens > self.output_limit:
            return False

        # Check total limit
        total_new = self._input_used + self._output_used + input_tokens + output_tokens
        return total_new <= self.total_limit

    def consume(self, amount: float) -> None:
        # For generic consume, split evenly
        half = int(amount / 2)
        self._input_used += half
        self._output_used += int(amount) - half

    def consume_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Consume specific token amounts."""
        self._input_used += input_tokens
        self._output_used += output_tokens

    def reset(self) -> None:
        self._input_used = 0
        self._output_used = 0


@dataclass
class CostBudget(Budget):
    """
    Cost/money-based budget.

    Attributes:
        limit_amount: Maximum spend amount
        currency: Currency code
        period: Budget period
    """

    limit_amount: Decimal = Decimal("100.00")
    currency: str = "USD"
    period: BudgetPeriod = BudgetPeriod.PER_MONTH

    _used_amount: Decimal = field(default=Decimal("0"), repr=False)

    @property
    def budget_type(self) -> BudgetType:
        return BudgetType.COST

    @property
    def limit(self) -> float:
        return float(self.limit_amount)

    @property
    def used(self) -> float:
        return float(self._used_amount)

    @property
    def used_decimal(self) -> Decimal:
        return self._used_amount

    @property
    def remaining_decimal(self) -> Decimal:
        return max(Decimal("0"), self.limit_amount - self._used_amount)

    def can_afford(self, amount: float) -> bool:
        return self._used_amount + Decimal(str(amount)) <= self.limit_amount

    def can_afford_decimal(self, amount: Decimal) -> bool:
        return self._used_amount + amount <= self.limit_amount

    def consume(self, amount: float) -> None:
        self._used_amount += Decimal(str(amount))

    def consume_decimal(self, amount: Decimal) -> None:
        self._used_amount += amount

    def reset(self) -> None:
        self._used_amount = Decimal("0")


@dataclass
class ExecutionBudget(Budget):
    """
    Execution-based budget (requests, steps, retries).

    Attributes:
        max_requests: Maximum requests
        max_steps: Maximum execution steps
        max_retries: Maximum retries
        period: Budget period
    """

    max_requests: int = 1000
    max_steps: int = 10000
    max_retries: int = 100
    period: BudgetPeriod = BudgetPeriod.PER_HOUR

    _requests_used: int = field(default=0, repr=False)
    _steps_used: int = field(default=0, repr=False)
    _retries_used: int = field(default=0, repr=False)

    @property
    def budget_type(self) -> BudgetType:
        return BudgetType.EXECUTION

    @property
    def limit(self) -> float:
        return float(self.max_requests)

    @property
    def used(self) -> float:
        return float(self._requests_used)

    @property
    def requests_remaining(self) -> int:
        return max(0, self.max_requests - self._requests_used)

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self._steps_used)

    @property
    def retries_remaining(self) -> int:
        return max(0, self.max_retries - self._retries_used)

    def can_afford(self, amount: float) -> bool:
        return self._requests_used + int(amount) <= self.max_requests

    def can_afford_request(self) -> bool:
        return self._requests_used < self.max_requests

    def can_afford_steps(self, steps: int) -> bool:
        return self._steps_used + steps <= self.max_steps

    def can_afford_retry(self) -> bool:
        return self._retries_used < self.max_retries

    def consume(self, amount: float) -> None:
        self._requests_used += int(amount)

    def consume_request(self) -> None:
        self._requests_used += 1

    def consume_steps(self, steps: int) -> None:
        self._steps_used += steps

    def consume_retry(self) -> None:
        self._retries_used += 1

    def reset(self) -> None:
        self._requests_used = 0
        self._steps_used = 0
        self._retries_used = 0


@dataclass
class LatencyBudget(Budget):
    """
    Latency-based budget.

    Attributes:
        max_request_latency_ms: Maximum per-request latency
        max_total_latency_ms: Maximum total latency
        target_p95_ms: Target P95 latency
        period: Budget period
    """

    max_request_latency_ms: float = 30000.0  # 30 seconds
    max_total_latency_ms: float = 300000.0  # 5 minutes
    target_p95_ms: float = 5000.0  # 5 seconds
    period: BudgetPeriod = BudgetPeriod.PER_REQUEST

    _total_latency_ms: float = field(default=0.0, repr=False)
    _latency_samples: list[float] = field(default_factory=list, repr=False)

    @property
    def budget_type(self) -> BudgetType:
        return BudgetType.LATENCY

    @property
    def limit(self) -> float:
        return self.max_total_latency_ms

    @property
    def used(self) -> float:
        return self._total_latency_ms

    @property
    def p95_latency(self) -> float:
        """Calculate P95 latency from samples."""
        if not self._latency_samples:
            return 0.0

        sorted_samples = sorted(self._latency_samples)
        index = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(index, len(sorted_samples) - 1)]

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def can_afford(self, amount: float) -> bool:
        return self._total_latency_ms + amount <= self.max_total_latency_ms

    def can_afford_request(self, estimated_latency_ms: float) -> bool:
        """Check if estimated latency is within budget."""
        if estimated_latency_ms > self.max_request_latency_ms:
            return False
        return self._total_latency_ms + estimated_latency_ms <= self.max_total_latency_ms

    def consume(self, amount: float) -> None:
        self._total_latency_ms += amount
        self._latency_samples.append(amount)

        # Keep only recent samples
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-1000:]

    def reset(self) -> None:
        self._total_latency_ms = 0.0
        self._latency_samples.clear()


# =============================================================================
# MODEL ROUTER
# =============================================================================


@dataclass
class ModelRoutingConfig:
    """
    Configuration for model routing.

    Attributes:
        preferred_model: Preferred model ID
        fallback_models: Fallback models in order
        max_tier: Maximum model tier to use
        cost_sensitive: Prioritize cost over capability
        latency_sensitive: Prioritize latency over capability
    """

    preferred_model: str = "deepseek-ai/deepseek-v4-pro"
    fallback_models: list[str] = field(default_factory=lambda: ["qwen/qwen3.5-122b-a10b"])
    max_tier: ModelTier = ModelTier.FLAGSHIP
    cost_sensitive: bool = False
    latency_sensitive: bool = False


class ModelRouter:
    """
    Routes requests to appropriate models based on budget constraints.

    Considers cost, latency, and capability requirements.
    """

    def __init__(
        self,
        pricing: dict[str, ModelPricing] | None = None,
        config: ModelRoutingConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize model router.

        Args:
            pricing: Model pricing information
            config: Routing configuration
            logger: Optional logger
        """
        self._pricing = pricing or OPENAI_PRICING
        self._config = config or ModelRoutingConfig()
        self._logger = logger or logging.getLogger(__name__)

    def select_model(
        self,
        cost_budget: CostBudget | None = None,
        token_budget: TokenBudget | None = None,
        latency_budget: LatencyBudget | None = None,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
        min_tier: ModelTier = ModelTier.ECONOMY,
        required_context_window: int = 0,
    ) -> str | None:
        """
        Select best model given budget constraints.

        Args:
            cost_budget: Cost budget constraint
            token_budget: Token budget constraint
            latency_budget: Latency budget constraint
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            min_tier: Minimum required model tier
            required_context_window: Minimum context window size

        Returns:
            Selected model ID or None if no model fits
        """
        candidates: list[tuple[str, ModelPricing, Decimal]] = []

        for model_id, pricing in self._pricing.items():
            # Check tier constraints
            if pricing.tier.value < min_tier.value:
                continue
            if pricing.tier.value > self._config.max_tier.value:
                continue

            # Check context window
            if required_context_window > 0 and pricing.context_window < required_context_window:
                continue

            # Estimate cost
            estimated_cost = pricing.estimate_cost(
                estimated_input_tokens,
                estimated_output_tokens,
            )

            # Check cost budget
            if cost_budget and not cost_budget.can_afford_decimal(estimated_cost):
                continue

            # Check token budget
            if token_budget and not token_budget.can_afford_tokens(
                estimated_input_tokens,
                estimated_output_tokens,
            ):
                continue

            candidates.append((model_id, pricing, estimated_cost))

        if not candidates:
            return None

        # Sort candidates
        if self._config.cost_sensitive:
            # Sort by cost (ascending)
            candidates.sort(key=lambda x: x[2])
        else:
            # Sort by tier (descending) then cost (ascending)
            candidates.sort(key=lambda x: (-x[1].tier.value, x[2]))

        # Check preferred model
        for model_id, _, _ in candidates:
            if model_id == self._config.preferred_model:
                return model_id

        # Return best candidate
        return candidates[0][0] if candidates else None

    def get_pricing(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a model."""
        return self._pricing.get(model_id)

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal | None:
        """Estimate cost for model usage."""
        pricing = self._pricing.get(model_id)
        if pricing is None:
            return None
        return pricing.estimate_cost(input_tokens, output_tokens)

    def list_models_by_tier(self, tier: ModelTier) -> list[str]:
        """List models in a specific tier."""
        return [
            model_id
            for model_id, pricing in self._pricing.items()
            if pricing.tier == tier
        ]


# =============================================================================
# BUDGET MANAGER
# =============================================================================


@dataclass
class BudgetManagerConfig:
    """
    Configuration for BudgetManager.

    Attributes:
        enable_alerts: Enable alert generation
        enable_tracking: Enable usage tracking
        enable_auto_reset: Enable automatic budget resets
        default_thresholds: Default threshold configuration
    """

    enable_alerts: bool = True
    enable_tracking: bool = True
    enable_auto_reset: bool = True
    default_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)


class BudgetManager:
    """
    Central budget management system.

    Manages multiple budget types, tracks usage, generates alerts,
    and provides model routing based on budget constraints.
    """

    def __init__(
        self,
        config: BudgetManagerConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize budget manager.

        Args:
            config: Manager configuration
            logger: Optional logger
        """
        self._config = config or BudgetManagerConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Budgets
        self._token_budget: TokenBudget | None = None
        self._cost_budget: CostBudget | None = None
        self._execution_budget: ExecutionBudget | None = None
        self._latency_budget: LatencyBudget | None = None

        # Components
        self._tracker = UsageTracker(logger=self._logger)
        self._router = ModelRouter(logger=self._logger)
        self._thresholds = self._config.default_thresholds

        # Alert handling
        self._alert_handlers: list[AlertHandler] = []
        self._alerts: list[BudgetAlert] = []
        self._triggered_thresholds: set[tuple[BudgetType, AlertSeverity]] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Reset tracking
        self._last_reset: dict[BudgetPeriod, datetime] = {}

    # -------------------------------------------------------------------------
    # Budget Configuration
    # -------------------------------------------------------------------------

    def set_token_budget(
        self,
        total_limit: int = 1000000,
        input_limit: int = 0,
        output_limit: int = 0,
        period: BudgetPeriod = BudgetPeriod.PER_DAY,
    ) -> None:
        """Configure token budget."""
        self._token_budget = TokenBudget(
            input_limit=input_limit,
            output_limit=output_limit,
            total_limit=total_limit,
            period=period,
        )

    def set_cost_budget(
        self,
        limit_amount: float | Decimal = 100.0,
        currency: str = "USD",
        period: BudgetPeriod = BudgetPeriod.PER_MONTH,
    ) -> None:
        """Configure cost budget."""
        if isinstance(limit_amount, float):
            limit_amount = Decimal(str(limit_amount))

        self._cost_budget = CostBudget(
            limit_amount=limit_amount,
            currency=currency,
            period=period,
        )

    def set_execution_budget(
        self,
        max_requests: int = 1000,
        max_steps: int = 10000,
        max_retries: int = 100,
        period: BudgetPeriod = BudgetPeriod.PER_HOUR,
    ) -> None:
        """Configure execution budget."""
        self._execution_budget = ExecutionBudget(
            max_requests=max_requests,
            max_steps=max_steps,
            max_retries=max_retries,
            period=period,
        )

    def set_latency_budget(
        self,
        max_request_latency_ms: float = 30000.0,
        max_total_latency_ms: float = 300000.0,
        target_p95_ms: float = 5000.0,
        period: BudgetPeriod = BudgetPeriod.PER_REQUEST,
    ) -> None:
        """Configure latency budget."""
        self._latency_budget = LatencyBudget(
            max_request_latency_ms=max_request_latency_ms,
            max_total_latency_ms=max_total_latency_ms,
            target_p95_ms=target_p95_ms,
            period=period,
        )

    def set_thresholds(self, thresholds: ThresholdConfig) -> None:
        """Set threshold configuration."""
        self._thresholds = thresholds

    # -------------------------------------------------------------------------
    # Budget Checking
    # -------------------------------------------------------------------------

    async def check_budget(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: Decimal | None = None,
        estimated_latency_ms: float = 0,
    ) -> dict[str, bool]:
        """
        Check if operation fits within all budgets.

        Args:
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens
            estimated_cost: Estimated cost
            estimated_latency_ms: Estimated latency

        Returns:
            Dictionary mapping budget types to affordability
        """
        results: dict[str, bool] = {}

        if self._token_budget:
            results["token"] = self._token_budget.can_afford_tokens(
                input_tokens, output_tokens
            )

        if self._cost_budget and estimated_cost:
            results["cost"] = self._cost_budget.can_afford_decimal(estimated_cost)

        if self._execution_budget:
            results["execution"] = self._execution_budget.can_afford_request()

        if self._latency_budget and estimated_latency_ms > 0:
            results["latency"] = self._latency_budget.can_afford_request(
                estimated_latency_ms
            )

        return results

    async def require_budget(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: Decimal | None = None,
        estimated_latency_ms: float = 0,
    ) -> None:
        """
        Require budget availability, raising if insufficient.

        Raises:
            BudgetExceededError: If any budget would be exceeded
        """
        checks = await self.check_budget(
            input_tokens, output_tokens, estimated_cost, estimated_latency_ms
        )

        for budget_name, affordable in checks.items():
            if not affordable:
                budget_type = BudgetType(budget_name)
                budget = self._get_budget(budget_type)

                if budget:
                    raise BudgetExceededError(
                        f"{budget_name.title()} budget exceeded",
                        budget_type=budget_type,
                        limit=budget.limit,
                        current=budget.used,
                        requested=float(input_tokens + output_tokens),
                    )

    # -------------------------------------------------------------------------
    # Budget Consumption
    # -------------------------------------------------------------------------

    async def consume(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: Decimal | None = None,
        latency_ms: float = 0,
        model_id: str = "",
        request_id: str = "",
        steps: int = 1,
        is_retry: bool = False,
    ) -> None:
        """
        Consume budget for an operation.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost: Cost incurred
            latency_ms: Latency in milliseconds
            model_id: Model used
            request_id: Request identifier
            steps: Execution steps consumed
            is_retry: Whether this is a retry
        """
        async with self._lock:
            # Consume token budget
            if self._token_budget:
                self._token_budget.consume_tokens(input_tokens, output_tokens)
                await self._check_threshold(self._token_budget)

            # Consume cost budget
            if self._cost_budget and cost:
                self._cost_budget.consume_decimal(cost)
                await self._check_threshold(self._cost_budget)

            # Consume execution budget
            if self._execution_budget:
                self._execution_budget.consume_request()
                self._execution_budget.consume_steps(steps)
                if is_retry:
                    self._execution_budget.consume_retry()
                await self._check_threshold(self._execution_budget)

            # Consume latency budget
            if self._latency_budget:
                self._latency_budget.consume(latency_ms)
                await self._check_threshold(self._latency_budget)

            # Track usage
            if self._config.enable_tracking:
                record = UsageRecord(
                    timestamp=datetime.now(timezone.utc),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost or Decimal("0"),
                    latency_ms=latency_ms,
                    model_id=model_id,
                    request_id=request_id,
                )
                await self._tracker.record(record)

    # -------------------------------------------------------------------------
    # Model Routing
    # -------------------------------------------------------------------------

    def select_model(
        self,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
        min_tier: ModelTier = ModelTier.ECONOMY,
        required_context_window: int = 0,
    ) -> str | None:
        """
        Select best model given current budget constraints.

        Args:
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            min_tier: Minimum model tier
            required_context_window: Required context window

        Returns:
            Selected model ID or None
        """
        return self._router.select_model(
            cost_budget=self._cost_budget,
            token_budget=self._token_budget,
            latency_budget=self._latency_budget,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            min_tier=min_tier,
            required_context_window=required_context_window,
        )

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal | None:
        """Estimate cost for model usage."""
        return self._router.estimate_cost(model_id, input_tokens, output_tokens)

    def configure_router(self, config: ModelRoutingConfig) -> None:
        """Configure model router."""
        self._router = ModelRouter(
            pricing=self._router._pricing,
            config=config,
            logger=self._logger,
        )

    def add_model_pricing(self, pricing: ModelPricing) -> None:
        """Add or update model pricing."""
        self._router._pricing[pricing.model_id] = pricing

    # -------------------------------------------------------------------------
    # Alerts
    # -------------------------------------------------------------------------

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    async def _check_threshold(self, budget: Budget) -> None:
        """Check if budget has crossed any thresholds."""
        if not self._config.enable_alerts:
            return

        usage_percent = budget.usage_percent

        # Check each severity level
        for severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            threshold_key = (budget.budget_type, severity)

            if threshold_key in self._triggered_thresholds:
                continue

            threshold_percent = {
                AlertSeverity.WARNING: self._thresholds.warning_percent,
                AlertSeverity.CRITICAL: self._thresholds.critical_percent,
                AlertSeverity.EMERGENCY: self._thresholds.emergency_percent,
            }.get(severity, 75.0)

            if usage_percent >= threshold_percent:
                await self._emit_alert(
                    BudgetAlert(
                        alert_type=AlertType.THRESHOLD_REACHED,
                        severity=severity,
                        budget_type=budget.budget_type,
                        message=f"{budget.budget_type.value} budget at {usage_percent:.1f}%",
                        current_value=budget.used,
                        threshold_value=budget.limit * (threshold_percent / 100),
                    )
                )
                self._triggered_thresholds.add(threshold_key)

    async def _emit_alert(self, alert: BudgetAlert) -> None:
        """Emit an alert to all handlers."""
        self._alerts.append(alert)

        self._logger.log(
            {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL,
            }.get(alert.severity, logging.WARNING),
            f"Budget alert: {alert.message}",
        )

        for handler in self._alert_handlers:
            try:
                result = handler(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._logger.error(f"Alert handler error: {e}")

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        budget_type: BudgetType | None = None,
        since: datetime | None = None,
    ) -> list[BudgetAlert]:
        """Get alerts matching criteria."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if budget_type:
            alerts = [a for a in alerts if a.budget_type == budget_type]

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
        self._triggered_thresholds.clear()

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    async def reset_budget(self, budget_type: BudgetType) -> None:
        """Reset a specific budget."""
        async with self._lock:
            budget = self._get_budget(budget_type)
            if budget:
                budget.reset()

                # Clear triggered thresholds for this budget
                self._triggered_thresholds = {
                    t for t in self._triggered_thresholds
                    if t[0] != budget_type
                }

                if self._config.enable_alerts:
                    await self._emit_alert(
                        BudgetAlert(
                            alert_type=AlertType.BUDGET_RESET,
                            severity=AlertSeverity.INFO,
                            budget_type=budget_type,
                            message=f"{budget_type.value} budget reset",
                            current_value=0,
                            threshold_value=budget.limit,
                        )
                    )

    async def reset_all_budgets(self) -> None:
        """Reset all budgets."""
        for budget_type in BudgetType:
            if budget_type != BudgetType.COMPOSITE:
                await self.reset_budget(budget_type)

    async def check_auto_reset(self) -> None:
        """Check and perform automatic resets based on periods."""
        if not self._config.enable_auto_reset:
            return

        now = datetime.now(timezone.utc)

        for budget in [
            self._token_budget,
            self._cost_budget,
            self._execution_budget,
            self._latency_budget,
        ]:
            if budget is None:
                continue

            period = budget.period  # type: ignore[attr-defined]
            if period == BudgetPeriod.PER_REQUEST:
                continue

            last_reset = self._last_reset.get(period)

            should_reset = False
            if last_reset is None:
                should_reset = True
            else:
                delta_map = {
                    BudgetPeriod.PER_MINUTE: timedelta(minutes=1),
                    BudgetPeriod.PER_HOUR: timedelta(hours=1),
                    BudgetPeriod.PER_DAY: timedelta(days=1),
                    BudgetPeriod.PER_MONTH: timedelta(days=30),
                }
                delta = delta_map.get(period)
                if delta and now - last_reset >= delta:
                    should_reset = True

            if should_reset:
                await self.reset_budget(budget.budget_type)
                self._last_reset[period] = now

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current budget status."""
        status: dict[str, Any] = {}

        if self._token_budget:
            status["token"] = {
                "limit": self._token_budget.total_limit,
                "used": self._token_budget.used,
                "remaining": self._token_budget.remaining,
                "usage_percent": self._token_budget.usage_percent,
                "input_used": self._token_budget.input_used,
                "output_used": self._token_budget.output_used,
            }

        if self._cost_budget:
            status["cost"] = {
                "limit": float(self._cost_budget.limit_amount),
                "used": float(self._cost_budget.used_decimal),
                "remaining": float(self._cost_budget.remaining_decimal),
                "usage_percent": self._cost_budget.usage_percent,
                "currency": self._cost_budget.currency,
            }

        if self._execution_budget:
            status["execution"] = {
                "requests_limit": self._execution_budget.max_requests,
                "requests_used": self._execution_budget._requests_used,
                "requests_remaining": self._execution_budget.requests_remaining,
                "steps_used": self._execution_budget._steps_used,
                "retries_used": self._execution_budget._retries_used,
            }

        if self._latency_budget:
            status["latency"] = {
                "max_request_ms": self._latency_budget.max_request_latency_ms,
                "total_used_ms": self._latency_budget._total_latency_ms,
                "avg_latency_ms": self._latency_budget.avg_latency,
                "p95_latency_ms": self._latency_budget.p95_latency,
            }

        status["alerts"] = {
            "total": len(self._alerts),
            "by_severity": {
                s.name: len([a for a in self._alerts if a.severity == s])
                for s in AlertSeverity
            },
        }

        return status

    def get_usage_report(
        self,
        period: BudgetPeriod = BudgetPeriod.PER_DAY,
    ) -> dict[str, Any]:
        """Get usage report for a period."""
        return self._tracker.get_totals_for_period(period)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_budget(self, budget_type: BudgetType) -> Budget | None:
        """Get budget by type."""
        mapping: dict[BudgetType, Budget | None] = {
            BudgetType.TOKEN: self._token_budget,
            BudgetType.COST: self._cost_budget,
            BudgetType.EXECUTION: self._execution_budget,
            BudgetType.LATENCY: self._latency_budget,
        }
        return mapping.get(budget_type)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_budget_manager(
    token_limit: int = 1000000,
    cost_limit: float = 100.0,
    requests_limit: int = 1000,
    max_latency_ms: float = 30000.0,
    enable_alerts: bool = True,
    logger: logging.Logger | None = None,
) -> BudgetManager:
    """
    Factory function to create configured BudgetManager.

    Args:
        token_limit: Daily token limit
        cost_limit: Monthly cost limit
        requests_limit: Hourly request limit
        max_latency_ms: Maximum request latency
        enable_alerts: Enable alert generation
        logger: Optional logger

    Returns:
        Configured BudgetManager
    """
    config = BudgetManagerConfig(enable_alerts=enable_alerts)
    manager = BudgetManager(config=config, logger=logger)

    manager.set_token_budget(total_limit=token_limit)
    manager.set_cost_budget(limit_amount=Decimal(str(cost_limit)))
    manager.set_execution_budget(max_requests=requests_limit)
    manager.set_latency_budget(max_request_latency_ms=max_latency_ms)

    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BudgetType",
    "AlertSeverity",
    "AlertType",
    "BudgetPeriod",
    "ModelTier",
    # Exceptions
    "BudgetError",
    "BudgetExceededError",
    "InsufficientBudgetError",
    # Alerts
    "BudgetAlert",
    "AlertHandler",
    "ThresholdConfig",
    # Pricing
    "ModelPricing",
    "OPENAI_PRICING",
    # Usage
    "UsageRecord",
    "UsageTracker",
    # Budgets
    "Budget",
    "TokenBudget",
    "CostBudget",
    "ExecutionBudget",
    "LatencyBudget",
    # Routing
    "ModelRoutingConfig",
    "ModelRouter",
    # Manager
    "BudgetManagerConfig",
    "BudgetManager",
    # Factory
    "create_budget_manager",
]
