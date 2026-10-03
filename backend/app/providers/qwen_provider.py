from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Optional

from app.providers.base_provider import ProviderConfig, ProviderMetadata, ProviderType
from app.providers.nvidia_provider import NVIDIAProvider


_DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"


class QwenProvider(NVIDIAProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_QWEN_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 16384),
            top_p=kwargs.get("top_p", 0.95),
            timeout_seconds=kwargs.get("timeout_seconds", 60.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        super().__init__(config)
        self._extra_gen_kwargs: dict[str, Any] = {
            k: kwargs[k]
            for k in ("top_k", "presence_penalty", "repetition_penalty")
            if k in kwargs
        }

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="qwen",
            provider_type=ProviderType.CHAT,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )

    async def generate(self, prompt: str, system_prompt: str | None = None, temperature: float | None = None, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, Any]:
        kwargs = {**self._extra_gen_kwargs, **kwargs}
        return await super().generate(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        kwargs = {**self._extra_gen_kwargs, **kwargs}
        async for chunk in super().stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk
