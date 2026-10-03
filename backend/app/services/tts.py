from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.providers.provider_registry import ProviderRegistry
from app.providers.provider_router import ProviderRouter, TaskType

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    audio_data: bytes
    model: str
    duration_seconds: float = 0.0
    latency_ms: float = 0.0


class TextToSpeech:
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        router: Optional[ProviderRouter] = None,
    ) -> None:
        self._registry = registry
        self._router = router

    @classmethod
    def from_registry(cls, registry: ProviderRegistry) -> TextToSpeech:
        return cls(registry=registry)

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs: Any,
    ) -> TTSResult:
        provider = self._resolve_provider()

        t0 = time.monotonic()
        audio_data, usage = await provider.generate(
            prompt=text,
            voice=voice,
            language=language,
            **kwargs,
        )
        latency = (time.monotonic() - t0) * 1000

        return TTSResult(
            audio_data=audio_data,
            model=provider.metadata.model,
            latency_ms=latency,
        )

    def _resolve_provider(self) -> Any:
        if self._router:
            return self._router.select(TaskType.TEXT_TO_SPEECH)
        if self._registry:
            return self._registry.get("tts")
        raise RuntimeError("No ProviderRegistry or ProviderRouter configured")
