from __future__ import annotations

import logging
from typing import Any

from app.providers.base_provider import BaseProvider, ProviderType
from app.providers.deepseek_provider import DeepSeekProvider
from app.providers.deepseek_flash_provider import DeepSeekFlashProvider
from app.providers.embed_provider import EmbedProvider
from app.providers.llama_provider import LlamaProvider
from app.providers.qwen_provider import QwenProvider
from app.providers.speech_provider import SpeechProvider
from app.providers.tts_provider import TTSProvider
from app.providers.vision_provider import VisionProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, name: str, provider: BaseProvider) -> None:
        self._providers[name] = provider
        logger.info("Registered provider: %s (%s)", name, provider.metadata.model)

    def get(self, name: str) -> BaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(f"Provider '{name}' not found. Available: {list(self._providers.keys())}")
        return provider

    def get_by_type(self, provider_type: ProviderType) -> list[BaseProvider]:
        return [p for p in self._providers.values() if p.metadata.provider_type == provider_type]

    def list_providers(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "type": p.metadata.provider_type.value,
                "model": p.metadata.model,
                "supports_streaming": p.metadata.supports_streaming,
            }
            for name, p in self._providers.items()
        ]

    def initialize_defaults(self, api_key_or_settings: Any, **overrides: Any) -> None:
        if hasattr(api_key_or_settings, "get_deepseek_key"):
            s = api_key_or_settings
            self.register("deepseek", DeepSeekProvider(s.get_deepseek_key(), **overrides.get("deepseek", {})))
            self.register("flash", DeepSeekFlashProvider(s.get_fast_key(), **overrides.get("flash", {})))
            self.register("llama", LlamaProvider(s.get_llama_key(), **overrides.get("llama", {})))
            self.register("qwen", QwenProvider(s.get_qwen_key(), **overrides.get("qwen", {})))
            self.register("vision", VisionProvider(s.get_vision_key(), **overrides.get("vision", {})))
            self.register("embed", EmbedProvider(s.get_embed_key(), **overrides.get("embed", {})))
            self.register("whisper", SpeechProvider(s.get_stt_key(), **overrides.get("whisper", {})))
            self.register("tts", TTSProvider(s.get_tts_key(), **overrides.get("tts", {})))
        else:
            key = str(api_key_or_settings)
            self.register("deepseek", DeepSeekProvider(key, **overrides.get("deepseek", {})))
            self.register("flash", DeepSeekFlashProvider(key, **overrides.get("flash", {})))
            self.register("llama", LlamaProvider(key, **overrides.get("llama", {})))
            self.register("qwen", QwenProvider(key, **overrides.get("qwen", {})))
            self.register("vision", VisionProvider(key, **overrides.get("vision", {})))
            self.register("embed", EmbedProvider(key, **overrides.get("embed", {})))
            self.register("whisper", SpeechProvider(key, **overrides.get("whisper", {})))
            self.register("tts", TTSProvider(key, **overrides.get("tts", {})))

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", name, exc)
                results[name] = False
        return results

    async def health_check(self, name: str) -> bool:
        provider = self.get(name)
        try:
            return await provider.health_check()
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", name, exc)
            return False

    def get_stats_all(self) -> dict[str, Any]:
        return {name: p.get_stats() for name, p in self._providers.items()}
