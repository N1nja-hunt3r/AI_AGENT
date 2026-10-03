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


_DEFAULT_MODEL = "openai/whisper-large-v3"


class SpeechProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_STT_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            timeout_seconds=kwargs.get("timeout_seconds", 120.0),
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
            name="whisper",
            provider_type=ProviderType.SPEECH,
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
        audio_file = kwargs.pop("audio_file", None)
        audio_bytes = kwargs.pop("audio_bytes", None)
        language = kwargs.pop("language", None)

        if audio_file is None and audio_bytes is None:
            raise ValueError("Either audio_file or audio_bytes is required")

        t0 = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                if audio_file:
                    with open(audio_file, "rb") as f:
                        file_content = f.read()
                    file_name = audio_file.split("/")[-1]
                else:
                    file_content = audio_bytes
                    file_name = "audio.wav"

                resp = await asyncio.wait_for(
                    self._client.audio.transcriptions.create(
                        model=self._config.model,
                        file=(file_name, file_content),
                        language=language,
                    ),
                    timeout=self._config.timeout_seconds,
                )
                latency = (time.monotonic() - t0) * 1000
                usage = UsageMetrics(latency_ms=latency)
                self._record_usage(usage)
                return resp.text, usage

            except asyncio.TimeoutError:
                last_error = ProviderError(f"Speech request timed out after {self._config.timeout_seconds}s")
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
            except Exception as exc:
                last_error = exc
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        raise ProviderError(f"Speech request failed after {self._config.max_retries} retries: {last_error}") from last_error

    async def health_check(self) -> bool:
        return self._config.api_key is not None and len(self._config.api_key) > 0
