from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfig,
    ProviderError,
    ProviderMetadata,
    ProviderType,
    UsageMetrics,
)


_DEFAULT_MODEL = "nvidia/nv-embed-v1"


class EmbedProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_EMBED_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            timeout_seconds=kwargs.get("timeout_seconds", 60.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="embed",
            provider_type=ProviderType.EMBED,
            model=self._model,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[str, UsageMetrics]:
        texts = kwargs.pop("texts", [prompt])
        input_type = kwargs.pop("input_type", "passage")

        t0 = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                resp = await asyncio.wait_for(
                    self._client.embeddings.create(
                        model=self._config.model,
                        input=texts,
                        extra_body={"input_type": input_type},
                    ),
                    timeout=self._config.timeout_seconds,
                )
                latency = (time.monotonic() - t0) * 1000
                embeddings = [item.embedding for item in resp.data]
                usage = UsageMetrics(
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    total_tokens=resp.usage.total_tokens if resp.usage else 0,
                    latency_ms=latency,
                )
                self._record_usage(usage)
                return embeddings, usage

            except asyncio.TimeoutError:
                last_error = ProviderError(f"Embed request timed out after {self._config.timeout_seconds}s")
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        raise ProviderError(f"Embed request failed after {self._config.max_retries} retries: {last_error}") from last_error

    async def health_check(self) -> bool:
        try:
            await asyncio.wait_for(
                self._client.embeddings.create(
                    model=self._config.model,
                    input=["ping"],
                    extra_body={"input_type": "query"},
                ),
                timeout=10.0,
            )
            return True
        except Exception:
            return False
