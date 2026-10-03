"""
wake_word.py

Wake word detection service for "Hey ASPIRE" activation.
Provides backend API for wake word configuration and detection state.
Actual audio-level detection runs on the client; the server manages
activation state and coordinates the response flow.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class WakeWordState(Enum):
    DISABLED = "disabled"
    LISTENING = "listening"
    DETECTED = "detected"
    ACTIVATED = "activated"
    ERROR = "error"


@dataclass
class WakeWordConfig:
    enabled: bool = True
    wake_word: str = "hey aspire"
    sensitivity: float = 0.7
    cooldown_seconds: float = 2.0
    require_activation_sound: bool = True


ActivationCallback = Callable[[str, float], None]


class WakeWordService:
    def __init__(self, config: Optional[WakeWordConfig] = None) -> None:
        self._config = config or WakeWordConfig()
        self._state = WakeWordState.DISABLED if not config or not config.enabled else WakeWordState.LISTENING
        self._last_activation: float = 0.0
        self._activation_count: int = 0
        self._callbacks: list[ActivationCallback] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> WakeWordState:
        return self._state

    @property
    def config(self) -> WakeWordConfig:
        return self._config

    async def activate(self, confidence: float = 1.0) -> bool:
        now = time.time()
        if now - self._last_activation < self._config.cooldown_seconds:
            logger.debug("Wake word activation on cooldown")
            return False

        async with self._lock:
            self._state = WakeWordState.DETECTED
            self._last_activation = now
            self._activation_count += 1

        logger.info("Wake word detected (confidence=%.2f, count=%d)", confidence, self._activation_count)

        for callback in self._callbacks:
            try:
                callback(self._config.wake_word, confidence)
            except Exception:
                logger.exception("Wake word callback error")

        async with self._lock:
            self._state = WakeWordState.ACTIVATED

        return True

    async def deactivate(self) -> None:
        async with self._lock:
            self._state = WakeWordState.LISTENING if self._config.enabled else WakeWordState.DISABLED

    async def set_enabled(self, enabled: bool) -> None:
        async with self._lock:
            self._config.enabled = enabled
            self._state = WakeWordState.LISTENING if enabled else WakeWordState.DISABLED

    def on_activation(self, callback: ActivationCallback) -> None:
        self._callbacks.append(callback)

    def reset_count(self) -> None:
        self._activation_count = 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "state": self._state.value,
            "wake_word": self._config.wake_word,
            "last_activation": self._last_activation,
            "activation_count": self._activation_count,
            "cooldown_seconds": self._config.cooldown_seconds,
        }


wake_word_service = WakeWordService()
