from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Optional

from openai import AsyncOpenAI

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfig,
    ProviderError,
    ProviderMetadata,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderType,
    UsageMetrics,
)


class NVIDIAProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        import httpx
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=10.0,
                pool=10.0,
            ),
            max_retries=0,
        )

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="nvidia",
            provider_type=ProviderType.CHAT,
            model=self._config.model,
            supports_streaming=True,
            supports_tools=True,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[str, UsageMetrics]:
        messages = self._build_messages(prompt, system_prompt)
        t0 = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._config.model,
                        messages=messages,
                        temperature=temperature or self._config.temperature,
                        max_tokens=max_tokens or self._config.max_tokens,
                        top_p=self._config.top_p,
                        **kwargs,
                    ),
                    timeout=self._config.timeout_seconds,
                )
                latency = (time.monotonic() - t0) * 1000
                content = resp.choices[0].message.content or ""
                usage = UsageMetrics(
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    total_tokens=resp.usage.total_tokens if resp.usage else 0,
                    latency_ms=latency,
                )
                self._record_usage(usage)
                return content, usage

            except asyncio.TimeoutError:
                last_error = ProviderTimeoutError(f"Request timed out after {self._config.timeout_seconds}s")
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if "rate" in str(exc).lower():
                    last_error = ProviderRateLimitError(str(exc))
                    if attempt < self._config.max_retries - 1:
                        await asyncio.sleep(self._config.retry_delay * (attempt + 1) * 2)
                elif attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        self._error_count += 1
        raise ProviderError(f"All {self._config.max_retries} retries failed: {last_error}") from last_error

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, system_prompt)
        try:
            stream = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=temperature or self._config.temperature,
                max_tokens=max_tokens or self._config.max_tokens,
                top_p=self._config.top_p,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            raise ProviderError(f"Stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            await asyncio.wait_for(
                self._client.models.list(),
                timeout=10.0,
            )
            return True
        except Exception:
            return False

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
