from __future__ import annotations

import logging
from enum import Enum

from app.providers.base_provider import BaseProvider
from app.providers.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


class TaskType(Enum):
    CHAT = "chat"
    PLANNING = "planning"
    MANAGEMENT = "management"
    CODING = "coding"
    REVIEW = "review"
    RESEARCH = "research"
    MEMORY = "memory"
    SUMMARIZATION = "summarization"
    EXECUTION = "execution"
    VISION = "vision"
    EMBEDDING = "embedding"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    FALLBACK = "fallback"
    REASONING = "reasoning"


TASK_ROUTING: dict[TaskType, str] = {
    # Fast chat / small talk → deepseek flash (smallest, fastest)
    TaskType.CHAT: "flash",
    # Brain / conversation / planning → most capable reasoning model
    TaskType.PLANNING: "deepseek",
    TaskType.MANAGEMENT: "deepseek",
    TaskType.REASONING: "deepseek",
    # Coding → Llama (strong code generation)
    TaskType.CODING: "llama",
    TaskType.REVIEW: "llama",
    TaskType.EXECUTION: "llama",
    # Research / memory / summarization → Qwen (largest context)
    TaskType.RESEARCH: "qwen",
    TaskType.MEMORY: "qwen",
    TaskType.SUMMARIZATION: "qwen",
    # Specialized providers
    TaskType.VISION: "vision",
    TaskType.EMBEDDING: "embed",
    TaskType.SPEECH_TO_TEXT: "whisper",
    TaskType.TEXT_TO_SPEECH: "tts",
    TaskType.FALLBACK: "deepseek",
}

# Per-provider fallback mapping (primary provider → fallback provider name)
FALLBACK_ROUTING: dict[str, str] = {
    "deepseek": "fallback:deepseek",
    "flash": "fallback:deepseek",
    "llama": "fallback:llama",
    "qwen": "fallback:qwen",
    "vision": "fallback:vision",
    "whisper": "fallback:whisper",
    "tts": "fallback:tts",
    "embed": "embed",  # No fallback for embeddings
}


class ProviderRouter:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def select(self, task_type: TaskType) -> BaseProvider:
        provider_name = TASK_ROUTING.get(task_type, TASK_ROUTING[TaskType.FALLBACK])
        return self._registry.get(provider_name)

    def select_with_fallback(self, task_type: TaskType) -> list[BaseProvider]:
        primary_name = TASK_ROUTING.get(task_type, TASK_ROUTING[TaskType.FALLBACK])
        primary = self._registry.get(primary_name)

        # Per-model fallback (the frontend should never know a fallback was used)
        fallback_name = FALLBACK_ROUTING.get(primary_name)
        providers = [primary]

        if fallback_name and fallback_name != primary_name:
            try:
                fallback = self._registry.get(fallback_name)
                # Only add if the fallback is a different model
                if fallback.metadata.model != primary.metadata.model:
                    providers.append(fallback)
                    logger.info(
                        "Router: %s → primary=%s, fallback=%s",
                        task_type.value, primary.metadata.model, fallback.metadata.model,
                    )
                else:
                    logger.info(
                        "Router: %s → %s (no fallback needed, same model)",
                        task_type.value, primary.metadata.model,
                    )
            except KeyError:
                logger.warning(
                    "Router: fallback provider '%s' not registered for %s",
                    fallback_name, primary_name,
                )

        return providers

    def get_provider_name(self, task_type: TaskType) -> str:
        return TASK_ROUTING.get(task_type, TASK_ROUTING[TaskType.FALLBACK])

    def get_fallback_name(self, provider_name: str) -> str | None:
        return FALLBACK_ROUTING.get(provider_name)
