from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Optional


class ProviderType(Enum):
    CHAT = "chat"
    VISION = "vision"
    EMBED = "embed"
    SPEECH = "speech"
    TTS = "tts"


class ProviderError(Exception):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 8192
    top_p: float = 0.95
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class UsageMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ProviderMetadata:
    name: str
    provider_type: ProviderType
    model: str
    version: str = "1.0.0"
    supports_streaming: bool = False
    supports_tools: bool = False


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._metadata = self._build_metadata()
        self._total_usage = UsageMetrics()
        self._request_count = 0
        self._error_count = 0

    @property
    def config(self) -> ProviderConfig:
        return self._config

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    @abstractmethod
    def _build_metadata(self) -> ProviderMetadata:
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[str, UsageMetrics]:
        ...

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def health_check(self) -> bool:
        try:
            await self.generate("ping", max_tokens=1)
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self._metadata.name,
            "model": self._metadata.model,
            "total_usage": {
                "prompt_tokens": self._total_usage.prompt_tokens,
                "completion_tokens": self._total_usage.completion_tokens,
                "total_tokens": self._total_usage.total_tokens,
                "cost_usd": self._total_usage.cost_usd,
            },
            "request_count": self._request_count,
            "error_count": self._error_count,
        }

    def _record_usage(self, usage: UsageMetrics) -> None:
        self._total_usage = UsageMetrics(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
            total_tokens=self._total_usage.total_tokens + usage.total_tokens,
            cost_usd=self._total_usage.cost_usd + usage.cost_usd,
            latency_ms=self._total_usage.latency_ms + usage.latency_ms,
        )
        self._request_count += 1

    def _make_request_id(self) -> str:
        return f"{self._metadata.name}-{uuid.uuid4().hex[:12]}"
