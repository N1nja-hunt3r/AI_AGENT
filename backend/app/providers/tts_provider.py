from __future__ import annotations

import asyncio
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


_DEFAULT_MODEL = "magpie-tts-multilingual"


class TTSProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_TTS_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            timeout_seconds=kwargs.get("timeout_seconds", 120.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="magpie-tts",
            provider_type=ProviderType.TTS,
            model=self._model,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[bytes, UsageMetrics]:
        voice = kwargs.pop("voice", None)
        language = kwargs.pop("language", "en")

        payload: dict[str, Any] = {
            "model": self._config.model,
            "input": prompt,
            "language": language,
        }
        if voice:
            payload["voice"] = voice

        t0 = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                resp = await asyncio.wait_for(
                    self._client.post(
                        "/audio/speech",
                        json=payload,
                        headers={"Authorization": f"Bearer {self._config.api_key}"},
                    ),
                    timeout=self._config.timeout_seconds,
                )
                resp.raise_for_status()
                latency = (time.monotonic() - t0) * 1000
                audio_data = resp.content
                usage = UsageMetrics(latency_ms=latency)
                self._record_usage(usage)
                return audio_data, usage

            except httpx.TimeoutException:
                last_error = ProviderError(f"TTS request timed out after {self._config.timeout_seconds}s")
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        raise ProviderError(f"TTS request failed after {self._config.max_retries} retries: {last_error}") from last_error

    async def health_check(self) -> bool:
        return self._config.api_key is not None and len(self._config.api_key) > 0
