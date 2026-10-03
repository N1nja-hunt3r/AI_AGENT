from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Optional

from app.agents.controller import (
    AgentController,
    AgentRequest,
    AgentResponse,
    Capability,
    CapabilityType,
    IntentType,
    LLMClient,
    SessionState,
    StateBackend,
)
from app.providers.base_provider import ProviderConfig
from app.providers.deepseek_provider import DeepSeekProvider
from app.providers.embed_provider import EmbedProvider
from app.providers.llama_provider import LlamaProvider
from app.providers.nvidia_provider import NVIDIAProvider
from app.providers.provider_registry import ProviderRegistry
from app.providers.provider_router import ProviderRouter, TaskType
from app.providers.qwen_provider import QwenProvider
from app.providers.speech_provider import SpeechProvider
from app.providers.tts_provider import TTSProvider
from app.providers.vision_provider import VisionProvider
from app.capabilities.computer_capability import ComputerCapability
from app.capabilities.file_capability import FileCapability
from app.capabilities.coding_capability import CodingCapability
from app.capabilities.vision_capability import VisionCapability

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


class EnvSettings:
    def get_deepseek_key(self) -> str:
        return os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_fast_key(self) -> str:
        return os.environ.get("NVIDIA_FAST_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_llama_key(self) -> str:
        return os.environ.get("NVIDIA_LLAMA_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_qwen_key(self) -> str:
        return os.environ.get("NVIDIA_QWEN_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_vision_key(self) -> str:
        return os.environ.get("NVIDIA_VISION_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_embed_key(self) -> str:
        return os.environ.get("NVIDIA_EMBED_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_stt_key(self) -> str:
        return os.environ.get("NVIDIA_STT_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    def get_tts_key(self) -> str:
        return os.environ.get("NVIDIA_TTS_API_KEY") or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")

    # ── Fallback key getters ─────────────────────────────────────────────────

    def get_deepseek_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_DEEPSEEK_FALLBACK_API_KEY") or self.get_deepseek_key()

    def get_llama_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_LLAMA_FALLBACK_API_KEY") or self.get_llama_key()

    def get_qwen_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_QWEN_FALLBACK_API_KEY") or self.get_qwen_key()

    def get_vision_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_VISION_FALLBACK_API_KEY") or self.get_vision_key()

    def get_stt_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_STT_FALLBACK_API_KEY") or self.get_stt_key()

    def get_tts_fallback_key(self) -> str:
        return os.environ.get("NVIDIA_TTS_FALLBACK_API_KEY") or self.get_tts_key()

    # ── Model getters (from centralized env vars) ────────────────────────────

    def get_deepseek_model(self) -> str:
        return os.environ.get("NVIDIA_DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4-pro")

    def get_fast_model(self) -> str:
        return os.environ.get("NVIDIA_FAST_MODEL", "deepseek-ai/deepseek-v4-flash")

    def get_llama_model(self) -> str:
        return os.environ.get("NVIDIA_LLAMA_MODEL", "meta/llama-3.3-70b-instruct")

    def get_qwen_model(self) -> str:
        return os.environ.get("NVIDIA_QWEN_MODEL", "qwen/qwen3.5-397b-a17b")

    def get_vision_model(self) -> str:
        return os.environ.get("NVIDIA_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")

    def get_embed_model(self) -> str:
        return os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v1")

    def get_stt_model(self) -> str:
        return os.environ.get("NVIDIA_STT_MODEL", "openai/whisper-large-v3")

    def get_tts_model(self) -> str:
        return os.environ.get("NVIDIA_TTS_MODEL", "magpie-tts-multilingual")

    # ── Fallback model getters ───────────────────────────────────────────────

    def get_deepseek_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_DEEPSEEK_FALLBACK_MODEL", "qwen/qwen3.5-122b-a10b")

    def get_llama_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_LLAMA_FALLBACK_MODEL", "meta/llama-3.1-70b-instruct")

    def get_qwen_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_QWEN_FALLBACK_MODEL", "qwen/qwen3-next-80b-a3b-instruct")

    def get_vision_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_VISION_FALLBACK_MODEL", "meta/llama-3.2-11b-vision-instruct")

    def get_stt_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_STT_FALLBACK_MODEL", "parakeet-1.1b-rnnt-multilingual-asr")

    def get_tts_fallback_model(self) -> str:
        return os.environ.get("NVIDIA_TTS_FALLBACK_MODEL", "chatterbox-multilingual-tts")


def init_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    settings = EnvSettings()
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    # ── Primary providers ────────────────────────────────────────────────────
    registry.initialize_defaults(
        settings,
        deepseek={"temperature": 0.3, "max_tokens": 16384, "base_url": base_url, "model": settings.get_deepseek_model()},
        llama={"temperature": 0.2, "max_tokens": 4096, "base_url": base_url, "model": settings.get_llama_model()},
        qwen={"temperature": 0.6, "max_tokens": 16384, "base_url": base_url, "model": settings.get_qwen_model()},
        vision={"temperature": 1.0, "max_tokens": 512, "base_url": base_url, "model": settings.get_vision_model()},
        embed={"base_url": base_url, "model": settings.get_embed_model()},
        whisper={"base_url": base_url, "model": settings.get_stt_model()},
        tts={"base_url": base_url, "model": settings.get_tts_model()},
    )

    # ── Fallback providers (registered with "fallback:" prefix) ──────────────
    enable_fallbacks = os.environ.get("NVIDIA_ENABLE_FALLBACKS", "true").lower() == "true"
    if enable_fallbacks:
        registry.register("fallback:deepseek", DeepSeekProvider(
            settings.get_deepseek_fallback_key(),
            model=settings.get_deepseek_fallback_model(),
            base_url=base_url, temperature=0.3, max_tokens=16384,
        ))
        registry.register("fallback:llama", LlamaProvider(
            settings.get_llama_fallback_key(),
            model=settings.get_llama_fallback_model(),
            base_url=base_url, temperature=0.2, max_tokens=4096,
        ))
        registry.register("fallback:qwen", QwenProvider(
            settings.get_qwen_fallback_key(),
            model=settings.get_qwen_fallback_model(),
            base_url=base_url, temperature=0.6, max_tokens=16384,
        ))
        registry.register("fallback:vision", VisionProvider(
            settings.get_vision_fallback_key(),
            model=settings.get_vision_fallback_model(),
            base_url=base_url, temperature=1.0, max_tokens=512,
        ))
        registry.register("fallback:whisper", SpeechProvider(
            settings.get_stt_fallback_key(),
            model=settings.get_stt_fallback_model(),
            base_url=base_url,
        ))
        registry.register("fallback:tts", TTSProvider(
            settings.get_tts_fallback_key(),
            model=settings.get_tts_fallback_model(),
            base_url=base_url,
        ))

    logger.info("Provider registry initialized with %d providers", len(registry.list_providers()))
    return registry


INTENT_TO_TASK: dict[IntentType, TaskType] = {
    IntentType.QUERY: TaskType.RESEARCH,
    IntentType.ACTION: TaskType.PLANNING,
    IntentType.CONVERSATION: TaskType.RESEARCH,
    IntentType.ANALYSIS: TaskType.RESEARCH,
    IntentType.CREATION: TaskType.CODING,
    IntentType.UNKNOWN: TaskType.FALLBACK,
}

import re

SIMPLE_CHAT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|howdy|greetings|yo|sup)\b", re.IGNORECASE),
    re.compile(r"^(good morning|good evening|good afternoon|good day)\b", re.IGNORECASE),
    re.compile(r"^how are you", re.IGNORECASE),
    re.compile(r"^what('s| is) (up|new|going on)\b", re.IGNORECASE),
    re.compile(r"^nice to meet you", re.IGNORECASE),
    re.compile(r"^thanks?( you)?$", re.IGNORECASE),
    re.compile(r"^ok(ay)?$", re.IGNORECASE),
    re.compile(r"^bye|goodbye|see you|talk later", re.IGNORECASE),
]

SIMPLE_ROUTING: dict[str, TaskType] = {
    "write": TaskType.CODING,
    "code": TaskType.CODING,
    "program": TaskType.CODING,
    "debug": TaskType.CODING,
    "refactor": TaskType.CODING,
    "create": TaskType.CODING,
    "build": TaskType.CODING,
    "implement": TaskType.CODING,
    "fix": TaskType.CODING,
    "function": TaskType.CODING,
    "script": TaskType.CODING,
    "research": TaskType.RESEARCH,
    "summarize": TaskType.RESEARCH,
    "summarise": TaskType.RESEARCH,
    "explain": TaskType.RESEARCH,
    "analyze": TaskType.RESEARCH,
    "analyse": TaskType.RESEARCH,
    "what": TaskType.RESEARCH,
    "why": TaskType.RESEARCH,
    "how": TaskType.RESEARCH,
    "tell me": TaskType.RESEARCH,
    "remember": TaskType.MEMORY,
    "memorize": TaskType.MEMORY,
    "memorise": TaskType.MEMORY,
    "save": TaskType.MEMORY,
    "recall": TaskType.MEMORY,
    "search": TaskType.MEMORY,
    "find": TaskType.MEMORY,
    "plan": TaskType.PLANNING,
    "schedule": TaskType.PLANNING,
    "organize": TaskType.PLANNING,
    "organise": TaskType.PLANNING,
    "review": TaskType.REVIEW,
    "check": TaskType.REVIEW,
    "image": TaskType.VISION,
    "screenshot": TaskType.VISION,
    "ocr": TaskType.VISION,
    "picture": TaskType.VISION,
    "see": TaskType.VISION,
}


def classify_task(content: str) -> TaskType:
    lower = content.lower().strip()

    for pattern in SIMPLE_CHAT_PATTERNS:
        if pattern.match(lower):
            return TaskType.CHAT

    for keyword, task in SIMPLE_ROUTING.items():
        if lower.startswith(keyword):
            return task

    words = lower.split()
    word_count = len(words)
    total_chars = len(lower)

    if word_count <= 4 and total_chars <= 30:
        return TaskType.CHAT

    return TaskType.RESEARCH


class RouterLLMClient(LLMClient):
    def __init__(self, router: ProviderRouter) -> None:
        self._router = router

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        task_type = classify_task(prompt)
        providers = self._router.select_with_fallback(task_type)
        last_error: Optional[Exception] = None
        for provider in providers:
            try:
                text, usage = await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text
            except Exception as exc:
                logger.warning("Provider %s failed: %s", provider.metadata.name, exc)
                last_error = exc
                continue
        raise last_error or RuntimeError("No provider available")

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        import time
        task_type = classify_task(prompt)
        providers = self._router.select_with_fallback(task_type)
        provider_names = [p.metadata.name for p in providers]
        logger.info(
            "RouterLLMClient.stream task_type=%s providers=%s prompt_len=%d",
            task_type.value, provider_names, len(prompt),
        )
        last_error: Optional[Exception] = None
        for provider in providers:
            t0 = time.monotonic()
            try:
                logger.info("RouterLLMClient.stream trying provider=%s model=%s", provider.metadata.name, provider.metadata.model)
                async for chunk in provider.stream(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
                elapsed = time.monotonic() - t0
                logger.info("RouterLLMClient.stream success provider=%s elapsed_s=%.2f", provider.metadata.name, elapsed)
                return
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "RouterLLMClient.stream failed provider=%s elapsed_s=%.2f error=%s",
                    provider.metadata.name, elapsed, exc,
                )
                last_error = exc
                continue
        raise last_error or RuntimeError("No provider available for streaming")

    async def complete_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        import json

        schema_desc = ", ".join(f'"{k}": {v}' for k, v in response_schema.items())
        structured_prompt = (
            f"{prompt}\n\nRespond ONLY with a valid JSON object containing these fields:\n"
            f"{{{schema_desc}}}\nNo other text."
        )
        result = await self.complete(
            prompt=structured_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[-1]
            result = result.rsplit("\n", 1)[0]
        if result.startswith("```json"):
            result = result[7:]
            result = result.rsplit("```", 1)[0]
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"result": result, "intent_type": "conversation", "confidence": 0.5, "entities": {}, "complexity_score": 5, "requires_context": False, "requires_current_info": False}


class SimpleStateBackend(StateBackend):
    def __init__(self) -> None:
        self._store: dict[str, SessionState] = {}

    async def load(self, session_id: str) -> SessionState | None:
        return self._store.get(session_id)

    async def save(self, state: SessionState) -> None:
        self._store[state.session_id] = state

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class NoopCapability(Capability[Any]):
    def __init__(self, cap_type: CapabilityType) -> None:
        self._type = cap_type

    @property
    def capability_type(self) -> CapabilityType:
        return self._type

    @property
    def name(self) -> str:
        return self._type.value

    async def execute(self, action: str, parameters: dict[str, Any], context: SessionState) -> Any:
        return {"action": action, "params": parameters}

    async def health_check(self) -> bool:
        return True


def create_orchestrator(
    registry: Optional[ProviderRegistry] = None,
) -> AgentController:
    if registry is None:
        registry = init_provider_registry()
    router = ProviderRouter(registry)
    llm_client = RouterLLMClient(router)
    state_backend = SimpleStateBackend()
    capabilities: dict[CapabilityType, Capability[Any]] = {
        CapabilityType.MEMORY: NoopCapability(CapabilityType.MEMORY),
        CapabilityType.TOOLS: NoopCapability(CapabilityType.TOOLS),
        CapabilityType.RAG: NoopCapability(CapabilityType.RAG),
        CapabilityType.WEB_SEARCH: NoopCapability(CapabilityType.WEB_SEARCH),
        CapabilityType.COMPUTER: ComputerCapability(),
        CapabilityType.FILES: FileCapability(),
        CapabilityType.CODING: CodingCapability(llm_service=llm_client),
        CapabilityType.VISION: VisionCapability(vision_service=registry.get("vision")),
    }
    controller = AgentController(
        llm_client=llm_client,
        state_backend=state_backend,
        capabilities=capabilities,
    )
    logger.info("Agent orchestrator created with router to %d providers", len(registry.list_providers()))
    return controller


def create_llm_client(
    registry: Optional[ProviderRegistry] = None,
) -> RouterLLMClient:
    if registry is None:
        registry = init_provider_registry()
    router = ProviderRouter(registry)
    return RouterLLMClient(router)
