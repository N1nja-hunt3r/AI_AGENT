from __future__ import annotations

import os
from typing import Any

from app.providers.base_provider import ProviderConfig, ProviderMetadata, ProviderType
from app.providers.nvidia_provider import NVIDIAProvider


_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"


class LlamaProvider(NVIDIAProvider):
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self._model = kwargs.get("model") or os.environ.get("NVIDIA_LLAMA_MODEL") or _DEFAULT_MODEL
        base_url = kwargs.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=self._model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.7),
            timeout_seconds=kwargs.get("timeout_seconds", 60.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        super().__init__(config)

    def _build_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="llama",
            provider_type=ProviderType.CHAT,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )
