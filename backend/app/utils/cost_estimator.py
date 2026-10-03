"""
cost_estimator.py

Cost estimation utilities for OpenAI, Anthropic, and Gemini usage,
covering per-execution cost and monthly cost projections based on
token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Pricing expressed in USD per 1,000 tokens. Values are illustrative
# defaults and should be kept in sync with current provider pricing pages.
_NVIDIA_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "deepseek-ai/deepseek-v4-pro": {"input": 0.0005, "output": 0.0015},
    "deepseek-ai/deepseek-v4-flash": {"input": 0.0001, "output": 0.0003},
    "meta/llama-3.3-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "qwen/qwen3.5-397b-a17b": {"input": 0.0012, "output": 0.0012},
    "meta/llama-3.2-90b-vision-instruct": {"input": 0.0005, "output": 0.0005},
    "nvidia/nv-embed-v1": {"input": 0.00001, "output": 0.00001},
    "openai/whisper-large-v3": {"input": 0.00001, "output": 0.00001},
    "magpie-tts-multilingual": {"input": 0.00001, "output": 0.00001},
    "qwen/qwen3.5-122b-a10b": {"input": 0.0003, "output": 0.0003},
    "meta/llama-3.1-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "qwen/qwen3-next-80b-a3b-instruct": {"input": 0.0003, "output": 0.0003},
    "meta/llama-3.2-11b-vision-instruct": {"input": 0.0001, "output": 0.0001},
}

_OPENAI_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.0100, "output": 0.0300},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

_ANTHROPIC_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "claude-opus-4-7": {"input": 0.0150, "output": 0.0750},
    "claude-sonnet-4-6": {"input": 0.0030, "output": 0.0150},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.0040},
}

_GEMINI_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.0050},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
}

_PRICING_TABLES: Dict[str, Dict[str, Dict[str, float]]] = {
    "nvidia": _NVIDIA_PRICING_PER_1K,
    "openai": _OPENAI_PRICING_PER_1K,
    "anthropic": _ANTHROPIC_PRICING_PER_1K,
    "gemini": _GEMINI_PRICING_PER_1K,
}


class UnknownModelError(Exception):
    """Raised when pricing data is unavailable for a given provider/model."""


@dataclass(frozen=True)
class CostBreakdown:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return round(self.input_cost_usd + self.output_cost_usd, 8)


def get_pricing(provider: str, model: str) -> Dict[str, float]:
    """Look up per-1K-token pricing for a given provider and model."""
    table = _PRICING_TABLES.get(provider.lower())
    if table is None:
        raise UnknownModelError(f"Unknown provider: {provider}")
    pricing = table.get(model)
    if pricing is None:
        raise UnknownModelError(f"Unknown model '{model}' for provider '{provider}'")
    return pricing


def estimate_execution_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> CostBreakdown:
    """Estimate the USD cost of a single execution given token counts."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    pricing = get_pricing(provider, model)
    input_cost = (input_tokens / 1000.0) * pricing["input"]
    output_cost = (output_tokens / 1000.0) * pricing["output"]
    return CostBreakdown(
        provider=provider.lower(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 8),
        output_cost_usd=round(output_cost, 8),
    )


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    call_count: int = 1


def estimate_monthly_cost(usage_records: List[UsageRecord]) -> Dict[str, float]:
    """
    Estimate aggregate monthly cost across multiple usage records.
    Returns a breakdown by provider/model plus a grand total.
    """
    breakdown: Dict[str, float] = {}
    total = 0.0
    for record in usage_records:
        cost = estimate_execution_cost(
            record.provider,
            record.model,
            record.input_tokens * record.call_count,
            record.output_tokens * record.call_count,
        )
        key = f"{record.provider}:{record.model}"
        breakdown[key] = round(breakdown.get(key, 0.0) + cost.total_cost_usd, 8)
        total += cost.total_cost_usd
    breakdown["total"] = round(total, 8)
    return breakdown


def project_monthly_cost(
    daily_input_tokens: int,
    daily_output_tokens: int,
    provider: str,
    model: str,
    *,
    days_in_month: int = 30,
) -> CostBreakdown:
    """Project a monthly cost from average daily token usage."""
    return estimate_execution_cost(
        provider,
        model,
        daily_input_tokens * days_in_month,
        daily_output_tokens * days_in_month,
    )


def compare_model_costs(
    input_tokens: int,
    output_tokens: int,
    candidates: Optional[List[tuple[str, str]]] = None,
) -> List[CostBreakdown]:
    """Compare estimated cost across a set of (provider, model) candidates."""
    pairs = candidates or [
        (provider, model)
        for provider, table in _PRICING_TABLES.items()
        for model in table
    ]
    results = [
        estimate_execution_cost(provider, model, input_tokens, output_tokens)
        for provider, model in pairs
    ]
    return sorted(results, key=lambda r: r.total_cost_usd)


def cheapest_model(
    input_tokens: int,
    output_tokens: int,
    candidates: Optional[List[tuple[str, str]]] = None,
) -> CostBreakdown:
    """Return the cheapest (provider, model) option for the given token usage."""
    results = compare_model_costs(input_tokens, output_tokens, candidates)
    if not results:
        raise UnknownModelError("No pricing candidates available")
    return results[0]
