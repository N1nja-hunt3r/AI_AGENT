from __future__ import annotations

import asyncio
import base64
import os
import time
from typing import Any, Optional

import httpx

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfig,
    ProviderError,
    ProviderMetadata,
    ProviderType,
    UsageMetrics,
)


_DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"


class VisionProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_VISION_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            temperature=kwargs.get("temperature", 1.0),
            max_tokens=kwargs.get("max_tokens", 512),
            top_p=kwargs.get("top_p", 1.0),
            timeout_seconds=kwargs.get("timeout_seconds", 60.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="vision",
            provider_type=ProviderType.VISION,
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
        image_b64 = kwargs.pop("image_b64", None)
        image_path = kwargs.pop("image_path", None)

        if image_path:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature or self._config.temperature,
            "top_p": self._config.top_p,
        }

        t0 = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                resp = await asyncio.wait_for(
                    self._client.post(
                        "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {self._config.api_key}"},
                    ),
                    timeout=self._config.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                latency = (time.monotonic() - t0) * 1000
                content = data["choices"][0]["message"]["content"]
                usage_data = data.get("usage", {})
                usage = UsageMetrics(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                    latency_ms=latency,
                )
                self._record_usage(usage)
                return content, usage

            except httpx.TimeoutException:
                last_error = ProviderError(f"Vision request timed out after {self._config.timeout_seconds}s")
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        raise ProviderError(f"Vision request failed after {self._config.max_retries} retries: {last_error}") from last_error

    async def health_check(self) -> bool:
        try:
            await asyncio.wait_for(
                self._client.get("/models", headers={"Authorization": f"Bearer {self._config.api_key}"}),
                timeout=10.0,
            )
            return True
        except Exception:
            return False
