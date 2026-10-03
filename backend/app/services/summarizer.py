from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.providers.provider_registry import ProviderRegistry
from app.providers.provider_router import ProviderRouter, TaskType

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    summary: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


class Summarizer:
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        router: Optional[ProviderRouter] = None,
    ) -> None:
        self._registry = registry
        self._router = router

    @classmethod
    def from_registry(cls, registry: ProviderRegistry) -> Summarizer:
        return cls(registry=registry)

    async def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        format: str = "concise",
        **kwargs: Any,
    ) -> SummarizationResult:
        provider = self._resolve_provider()
        prompt = self._build_prompt(text, format)

        t0 = time.monotonic()
        content, usage = await provider.generate(
            prompt=prompt,
            max_tokens=max_length,
            temperature=temperature,
            **kwargs,
        )
        latency = (time.monotonic() - t0) * 1000

        return SummarizationResult(
            summary=content,
            model=provider.metadata.model,
            tokens_used=usage.total_tokens,
            latency_ms=latency,
        )

    async def summarize_batch(
        self,
        texts: list[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        format: str = "concise",
        **kwargs: Any,
    ) -> list[SummarizationResult]:
        results: list[SummarizationResult] = []
        for text in texts:
            result = await self.summarize(
                text=text,
                max_length=max_length,
                temperature=temperature,
                format=format,
                **kwargs,
            )
            results.append(result)
        return results

    def _resolve_provider(self) -> Any:
        if self._router:
            return self._router.select(TaskType.SUMMARIZATION)
        if self._registry:
            return self._registry.get("qwen")
        raise RuntimeError("No ProviderRegistry or ProviderRouter configured")

    @staticmethod
    def _build_prompt(text: str, format: str = "concise") -> str:
        if format == "concise":
            return f"Summarize the following text concisely:\n\n{text}"
        if format == "bullet_points":
            return f"Summarize the following text as bullet points:\n\n{text}"
        if format == "detailed":
            return f"Provide a detailed summary of the following text:\n\n{text}"
        return f"Summarize the following text ({format}):\n\n{text}"
