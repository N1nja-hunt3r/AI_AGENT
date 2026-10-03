from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.providers.provider_registry import ProviderRegistry
from app.providers.provider_router import ProviderRouter, TaskType

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    model: str
    language: Optional[str] = None
    latency_ms: float = 0.0


class SpeechToText:
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        router: Optional[ProviderRouter] = None,
    ) -> None:
        self._registry = registry
        self._router = router

    @classmethod
    def from_registry(cls, registry: ProviderRegistry) -> SpeechToText:
        return cls(registry=registry)

    async def transcribe(
        self,
        audio_file: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        provider = self._resolve_provider()

        t0 = time.monotonic()
        text, usage = await provider.generate(
            prompt="",
            audio_file=audio_file,
            audio_bytes=audio_bytes,
            language=language,
            **kwargs,
        )
        latency = (time.monotonic() - t0) * 1000

        return TranscriptionResult(
            text=text,
            model=provider.metadata.model,
            language=language,
            latency_ms=latency,
        )

    def _resolve_provider(self) -> Any:
        if self._router:
            return self._router.select(TaskType.SPEECH_TO_TEXT)
        if self._registry:
            return self._registry.get("whisper")
        raise RuntimeError("No ProviderRegistry or ProviderRouter configured")
