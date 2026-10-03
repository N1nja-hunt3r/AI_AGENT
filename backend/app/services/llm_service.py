"""
llm_service.py - LLM abstraction layer for the AI Operating System.

Responsibilities: chat completion, streaming, tool calling, structured output,
token counting, cost tracking, retries, fallback models, context management,
health checks, async execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelProvider(str, Enum):
    NVIDIA = "nvidia"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LOCAL = "local"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_claude_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_gemini_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CompletionResponse:
    content: str
    finish_reason: FinishReason
    usage: Usage
    model: str
    provider: ModelProvider
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Optional[Any] = None
    latency_ms: float = 0.0


@dataclass
class StreamChunk:
    delta: str
    finish_reason: Optional[FinishReason] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Usage] = None


@dataclass
class ModelConfig:
    model_id: str
    provider: ModelProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    context_window: int = 128_000
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_structured_output: bool = False
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMServiceConfig:
    primary: ModelConfig
    fallbacks: List[ModelConfig] = field(default_factory=list)
    max_context_tokens: int = 100_000
    budget_limit_usd: Optional[float] = None
    enable_cost_tracking: bool = True
    enable_health_checks: bool = True
    health_check_interval: float = 60.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    pass

class ProviderError(LLMServiceError):
    def __init__(self, message: str, provider: ModelProvider, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

class RateLimitError(ProviderError):
    pass

class ContextLengthError(LLMServiceError):
    pass

class BudgetExceededError(LLMServiceError):
    pass

class AllProvidersFailedError(LLMServiceError):
    pass

class HealthCheckError(LLMServiceError):
    pass


# ---------------------------------------------------------------------------
# Cost registry
# ---------------------------------------------------------------------------

COST_TABLE: Dict[str, Tuple[float, float]] = {
    # (input $/1M tokens, output $/1M tokens)
    # ── NVIDIA NIM primary models ────────────────────────────────────────────
    "deepseek-ai/deepseek-v4-pro": (0.50, 1.50),
    "deepseek-ai/deepseek-v4-flash": (0.10, 0.30),
    "meta/llama-3.3-70b-instruct": (0.35, 0.40),
    "qwen/qwen3.5-397b-a17b": (1.20, 1.20),
    "meta/llama-3.2-90b-vision-instruct": (0.50, 0.50),
    "nvidia/nv-embed-v1": (0.01, 0.01),
    "openai/whisper-large-v3": (0.01, 0.01),
    "magpie-tts-multilingual": (0.01, 0.01),
    # ── NVIDIA NIM fallback models ───────────────────────────────────────────
    "qwen/qwen3.5-122b-a10b": (0.30, 0.30),
    "meta/llama-3.1-70b-instruct": (0.35, 0.40),
    "qwen/qwen3-next-80b-a3b-instruct": (0.30, 0.30),
    "meta/llama-3.2-11b-vision-instruct": (0.10, 0.10),
    "parakeet-1.1b-rnnt-multilingual-asr": (0.01, 0.01),
    "chatterbox-multilingual-tts": (0.01, 0.01),
    # ── Third-party (kept for compatibility) ─────────────────────────────────
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.25, 1.25),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.35, 1.05),
    "gemini-2.0-flash": (0.10, 0.40),
}


def _compute_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    base = model_id.split(":")[0]
    for key, (inp, out) in COST_TABLE.items():
        if key in base:
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
    return 0.0


# ---------------------------------------------------------------------------
# Abstract provider backend
# ---------------------------------------------------------------------------

class ProviderBackend(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        ...

    @abstractmethod
    async def count_tokens(self, messages: List[Message]) -> int:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIBackend(ProviderBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import openai  # type: ignore
            self._client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

    def _to_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in messages]

    def _parse_finish(self, reason: Optional[str]) -> FinishReason:
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason or "", FinishReason.UNKNOWN)

    def _parse_tool_calls(self, raw_calls: Any) -> List[ToolCall]:
        result: List[ToolCall] = []
        if not raw_calls:
            return result
        for tc in raw_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return result

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        t0 = time.monotonic()
        params: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": self._to_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if tools:
            params["tools"] = [t.to_openai_dict() for t in tools]
            params["tool_choice"] = "auto"
        if response_schema and self.config.supports_structured_output:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema},
            }

        try:
            resp = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.OPENAI) from exc

        choice = resp.choices[0]
        u = resp.usage
        usage = Usage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
            cost_usd=_compute_cost(self.config.model_id, u.prompt_tokens, u.completion_tokens),
        )
        return CompletionResponse(
            content=choice.message.content or "",
            finish_reason=self._parse_finish(choice.finish_reason),
            usage=usage,
            model=self.config.model_id,
            provider=ModelProvider.OPENAI,
            tool_calls=self._parse_tool_calls(choice.message.tool_calls),
            raw=resp,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        params: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": self._to_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            params["tools"] = [t.to_openai_dict() for t in tools]
            params["tool_choice"] = "auto"

        try:
            async with self._client.chat.completions.stream(**params) as stream:
                async for event in stream:
                    choice = event.choices[0] if event.choices else None
                    if choice is None:
                        continue
                    delta = choice.delta.content or ""
                    finish = self._parse_finish(choice.finish_reason) if choice.finish_reason else None
                    usage: Optional[Usage] = None
                    if event.usage:
                        u = event.usage
                        usage = Usage(
                            prompt_tokens=u.prompt_tokens,
                            completion_tokens=u.completion_tokens,
                            total_tokens=u.total_tokens,
                            cost_usd=_compute_cost(self.config.model_id, u.prompt_tokens, u.completion_tokens),
                        )
                    yield StreamChunk(delta=delta, finish_reason=finish, usage=usage)
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.OPENAI) from exc

    async def count_tokens(self, messages: List[Message]) -> int:
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.encoding_for_model(self.config.model_id)
            total = 0
            for m in messages:
                total += 4 + len(enc.encode(m.content))
            return total
        except Exception:
            return sum(len(m.content.split()) * 4 // 3 for m in messages)

    async def health_check(self) -> bool:
        try:
            await self._client.models.retrieve(self.config.model_id)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Claude (Anthropic) backend
# ---------------------------------------------------------------------------

class ClaudeBackend(ProviderBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import anthropic  # type: ignore
            self._client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        except ImportError as exc:
            raise ImportError("anthropic package required: pip install anthropic") from exc

    def _split_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        conv: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conv.append({"role": m.role, "content": m.content})
        return system, conv

    def _parse_tool_calls(self, blocks: Any) -> List[ToolCall]:
        result: List[ToolCall] = []
        for b in blocks or []:
            if getattr(b, "type", None) == "tool_use":
                result.append(ToolCall(id=b.id, name=b.name, arguments=b.input or {}))
        return result

    def _get_text(self, blocks: Any) -> str:
        parts: List[str] = []
        for b in blocks or []:
            if getattr(b, "type", None) == "text":
                parts.append(b.text)
        return "".join(parts)

    def _parse_finish(self, reason: str) -> FinishReason:
        mapping = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALL,
            "stop_sequence": FinishReason.STOP,
        }
        return mapping.get(reason, FinishReason.UNKNOWN)

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        t0 = time.monotonic()
        system, conv = self._split_messages(messages)
        params: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": conv,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [t.to_claude_dict() for t in tools]

        try:
            resp = await self._client.messages.create(**params)
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.CLAUDE) from exc

        u = resp.usage
        usage = Usage(
            prompt_tokens=u.input_tokens,
            completion_tokens=u.output_tokens,
            total_tokens=u.input_tokens + u.output_tokens,
            cost_usd=_compute_cost(self.config.model_id, u.input_tokens, u.output_tokens),
        )
        return CompletionResponse(
            content=self._get_text(resp.content),
            finish_reason=self._parse_finish(resp.stop_reason or ""),
            usage=usage,
            model=self.config.model_id,
            provider=ModelProvider.CLAUDE,
            tool_calls=self._parse_tool_calls(resp.content),
            raw=resp,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Anthropic."""
        system, conv = self._split_messages(messages)
        params: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": conv,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [t.to_claude_dict() for t in tools]

        try:
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)
                final = await stream.get_final_message()
                u = final.usage
                usage = Usage(
                    prompt_tokens=u.input_tokens,
                    completion_tokens=u.output_tokens,
                    total_tokens=u.input_tokens + u.output_tokens,
                    cost_usd=_compute_cost(self.config.model_id, u.input_tokens, u.output_tokens),
                )
                yield StreamChunk(
                    delta="",
                    finish_reason=self._parse_finish(final.stop_reason or ""),
                    usage=usage,
                )
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.CLAUDE) from exc

    async def count_tokens(self, messages: List[Message]) -> int:
        system, conv = self._split_messages(messages)
        try:
            resp = await self._client.messages.count_tokens(
                model=self.config.model_id,
                system=system or "",
                messages=conv,
            )
            return resp.input_tokens
        except Exception:
            return sum(len(m.content.split()) * 4 // 3 for m in messages)

    async def health_check(self) -> bool:
        try:
            await self._client.models.retrieve(self.config.model_id)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class GeminiBackend(ProviderBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=config.api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(config.model_id)
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package required: pip install google-generativeai"
            ) from exc

    def _to_gemini_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        history: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            elif m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})
        return system, history

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        t0 = time.monotonic()
        system, history = self._to_gemini_messages(messages)

        gen_config = self._genai.types.GenerationConfig(
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        try:
            if system:
                model = self._genai.GenerativeModel(
                    self.config.model_id, system_instruction=system
                )
            else:
                model = self._model

            gemini_tools = None
            if tools:
                gemini_tools = [
                    self._genai.protos.Tool(
                        function_declarations=[
                            self._genai.protos.FunctionDeclaration(**t.to_gemini_dict())
                            for t in tools
                        ]
                    )
                ]

            resp = await asyncio.to_thread(
                model.generate_content,
                history,
                generation_config=gen_config,
                tools=gemini_tools,
            )
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.GEMINI) from exc

        text = resp.text if hasattr(resp, "text") else ""
        pt = resp.usage_metadata.prompt_token_count if hasattr(resp, "usage_metadata") else 0
        ct = resp.usage_metadata.candidates_token_count if hasattr(resp, "usage_metadata") else 0
        usage = Usage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            cost_usd=_compute_cost(self.config.model_id, pt, ct),
        )
        return CompletionResponse(
            content=text,
            finish_reason=FinishReason.STOP,
            usage=usage,
            model=self.config.model_id,
            provider=ModelProvider.GEMINI,
            raw=resp,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        system, history = self._to_gemini_messages(messages)
        gen_config = self._genai.types.GenerationConfig(
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        try:
            model = (
                self._genai.GenerativeModel(self.config.model_id, system_instruction=system)
                if system
                else self._model
            )
            response = await asyncio.to_thread(
                model.generate_content,
                history,
                generation_config=gen_config,
                stream=True,
            )
            for chunk in response:
                delta = chunk.text if hasattr(chunk, "text") else ""
                yield StreamChunk(delta=delta)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.GEMINI) from exc

    async def count_tokens(self, messages: List[Message]) -> int:
        try:
            _, history = self._to_gemini_messages(messages)
            result = await asyncio.to_thread(self._model.count_tokens, history)
            return result.total_tokens
        except Exception:
            return sum(len(m.content.split()) * 4 // 3 for m in messages)

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._genai.get_model, f"models/{self.config.model_id}")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(ProviderBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import httpx  # type: ignore
            self._http = httpx.AsyncClient(
                base_url=config.base_url or "http://localhost:11434",
                timeout=config.timeout,
            )
        except ImportError as exc:
            raise ImportError("httpx package required: pip install httpx") from exc

    def _to_ollama_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        t0 = time.monotonic()
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]

        try:
            resp = await self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.OLLAMA) from exc

        msg = data.get("message", {})
        pt = data.get("prompt_eval_count", 0)
        ct = data.get("eval_count", 0)
        usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        return CompletionResponse(
            content=msg.get("content", ""),
            finish_reason=FinishReason.STOP,
            usage=usage,
            model=self.config.model_id,
            provider=ModelProvider.OLLAMA,
            raw=data,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": {
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    delta = data.get("message", {}).get("content", "")
                    done = data.get("done", False)
                    finish = FinishReason.STOP if done else None
                    usage: Optional[Usage] = None
                    if done:
                        pt = data.get("prompt_eval_count", 0)
                        ct = data.get("eval_count", 0)
                        usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
                    yield StreamChunk(delta=delta, finish_reason=finish, usage=usage)
        except Exception as exc:
            raise ProviderError(str(exc), ModelProvider.OLLAMA) from exc

    async def count_tokens(self, messages: List[Message]) -> int:
        return sum(len(m.content.split()) * 4 // 3 for m in messages)

    async def health_check(self) -> bool:
        try:
            resp = await self._http.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKEND_MAP: Dict[ModelProvider, Type[ProviderBackend]] = {
    ModelProvider.OPENAI: OpenAIBackend,
    ModelProvider.CLAUDE: ClaudeBackend,
    ModelProvider.GEMINI: GeminiBackend,
    ModelProvider.OLLAMA: OllamaBackend,
}


def _build_backend(config: ModelConfig) -> ProviderBackend:
    cls = _BACKEND_MAP.get(config.provider)
    if cls is None:
        raise LLMServiceError(f"Unsupported provider: {config.provider}")
    return cls(config)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(
    fn: Callable[[], Any],
    max_retries: int,
    retry_delay: float,
    provider: ModelProvider,
) -> Any:
    last_exc: Exception = LLMServiceError("No attempts made")
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except RateLimitError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** attempt)
            logger.warning("Rate limit on %s, retry %d/%d in %.1fs", provider, attempt + 1, max_retries, wait)
            await asyncio.sleep(wait)
        except ProviderError as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                break
        except Exception:
            raise
    raise last_exc


# ---------------------------------------------------------------------------
# Context manager (trim to fit context window)
# ---------------------------------------------------------------------------

def _trim_messages(
    messages: List[Message],
    token_count: int,
    max_tokens: int,
) -> List[Message]:
    if token_count <= max_tokens:
        return messages
    # Always keep system message and last user message
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    while len(rest) > 1:
        rest.pop(0)
        estimated = sum(len(m.content.split()) * 4 // 3 for m in system + rest)
        if estimated <= max_tokens:
            break
    return system + rest


# ---------------------------------------------------------------------------
# LLMService
# ---------------------------------------------------------------------------

class LLMService:
    """
    Abstraction layer between the AI Operating System and language model providers.

    Compatible with: context_engine.py, executor.py, planner.py,
                     memory_capability.py, budget_manager.py
    """

    def __init__(self, config: LLMServiceConfig) -> None:
        self._config = config
        self._primary: ProviderBackend = _build_backend(config.primary)
        self._fallbacks: List[ProviderBackend] = [_build_backend(c) for c in config.fallbacks]
        self._active: ProviderBackend = self._primary
        self._active_config: ModelConfig = config.primary

        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._request_count: int = 0
        self._error_count: int = 0

        self._health_status: Dict[str, bool] = {}
        self._health_lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task[None]] = None

        if config.enable_health_checks:
            try:
                loop = asyncio.get_running_loop()
                self._health_task = loop.create_task(self._health_loop())
            except RuntimeError:
                pass  # No event loop yet; caller must call start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background tasks. Call once inside an async context."""
        if self._config.enable_health_checks and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Cancel background tasks."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Single-turn completion with automatic retry and fallback."""
        messages = await self._prepare_messages(messages)
        self._check_budget()
        backends = self._get_backend_chain(model_override)
        last_error: Exception = LLMServiceError("No backends available")

        for backend, cfg in backends:
            try:
                resp = await _with_retry(
                    lambda b=backend: b.complete(messages, tools, response_schema, **kwargs),  # type: ignore[misc]
                    cfg.max_retries,
                    cfg.retry_delay,
                    cfg.provider,
                )
                self._record_usage(resp.usage)
                return resp
            except (ProviderError, LLMServiceError) as exc:
                logger.warning("Backend %s failed: %s", cfg.provider, exc)
                last_error = exc
                self._error_count += 1
                continue

        raise AllProvidersFailedError(f"All providers failed. Last: {last_error}") from last_error

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        on_tool_call: Optional[Callable[[ToolCall], Any]] = None,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Multi-turn agentic chat loop with tool execution support.
        Compatible with executor.py tool dispatch conventions.
        """
        history = list(messages)
        last_resp: Optional[CompletionResponse] = None

        for _ in range(max_turns):
            resp = await self.complete(history, tools=tools, **kwargs)
            last_resp = resp
            history.append(Message(role="assistant", content=resp.content, tool_calls=[
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in resp.tool_calls
            ] if resp.tool_calls else None))

            if not resp.tool_calls:
                break

            for tc in resp.tool_calls:
                if on_tool_call:
                    result = await on_tool_call(tc) if asyncio.iscoroutinefunction(on_tool_call) else on_tool_call(tc)
                    tool_result = str(result) if result is not None else "null"
                else:
                    tool_result = json.dumps({"error": "no tool handler registered"})
                history.append(Message(role="tool", content=tool_result, tool_call_id=tc.id))

        return last_resp  # type: ignore[return-value]

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming completion with fallback on first-chunk failure."""
        messages = await self._prepare_messages(messages)
        self._check_budget()
        backends = self._get_backend_chain(None)
        last_error: Exception = LLMServiceError("No backends available")

        for backend, cfg in backends:
            if not cfg.supports_streaming:
                continue
            try:
                gen = backend.stream(messages, tools, **kwargs)
                first = True
                async for chunk in gen:
                    first = False
                    if chunk.usage:
                        self._record_usage(chunk.usage)
                    yield chunk
                return
            except (ProviderError, LLMServiceError) as exc:
                if first:
                    logger.warning("Streaming backend %s failed: %s", cfg.provider, exc)
                    last_error = exc
                    self._error_count += 1
                    continue
                raise

        raise AllProvidersFailedError(f"All streaming providers failed. Last: {last_error}") from last_error

    async def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens using the active backend."""
        return await self._active.count_tokens(messages)

    async def estimate_cost(
        self,
        messages: List[Message],
        expected_output_tokens: int = 500,
        model_id: Optional[str] = None,
    ) -> float:
        """Estimate cost in USD for a given request."""
        input_tokens = await self.count_tokens(messages)
        mid = model_id or self._active_config.model_id
        return _compute_cost(mid, input_tokens, expected_output_tokens)

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of all configured backends."""
        results: Dict[str, Any] = {}
        backends_to_check = [
            (self._primary, self._config.primary),
            *zip(self._fallbacks, self._config.fallbacks),
        ]
        for backend, cfg in backends_to_check:
            key = f"{cfg.provider.value}/{cfg.model_id}"
            try:
                ok = await asyncio.wait_for(backend.health_check(), timeout=10.0)
                results[key] = {"healthy": ok, "provider": cfg.provider, "model": cfg.model_id}
            except Exception as exc:
                results[key] = {"healthy": False, "error": str(exc), "provider": cfg.provider}
        return results

    def switch_model(self, config: ModelConfig) -> None:
        """Hot-swap the active model at runtime."""
        new_backend = _build_backend(config)
        self._active = new_backend
        self._active_config = config
        logger.info("Switched active model to %s/%s", config.provider, config.model_id)

    # ------------------------------------------------------------------
    # Budget / stats interface (for budget_manager.py)
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def reset_stats(self) -> None:
        self._total_cost = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._error_count = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "total_tokens": self._total_tokens,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "active_model": self._active_config.model_id,
            "active_provider": self._active_config.provider.value,
        }

    # ------------------------------------------------------------------
    # Context engine interface (for context_engine.py)
    # ------------------------------------------------------------------

    def get_context_window(self) -> int:
        return self._active_config.context_window

    def get_max_tokens(self) -> int:
        return self._active_config.max_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _prepare_messages(self, messages: List[Message]) -> List[Message]:
        token_count = await self._active.count_tokens(messages)
        max_ctx = self._config.max_context_tokens
        if token_count > max_ctx:
            logger.warning("Context %d tokens exceeds limit %d, trimming.", token_count, max_ctx)
            messages = _trim_messages(messages, token_count, max_ctx)
        return messages

    def _check_budget(self) -> None:
        if self._config.budget_limit_usd is not None:
            if self._total_cost >= self._config.budget_limit_usd:
                raise BudgetExceededError(
                    f"Budget limit ${self._config.budget_limit_usd:.4f} exceeded "
                    f"(spent ${self._total_cost:.4f})"
                )

    def _record_usage(self, usage: Usage) -> None:
        self._total_cost += usage.cost_usd
        self._total_tokens += usage.total_tokens
        self._request_count += 1

    def _get_backend_chain(
        self, model_override: Optional[str]
    ) -> List[Tuple[ProviderBackend, ModelConfig]]:
        if model_override:
            cfg = ModelConfig(
                model_id=model_override,
                provider=self._active_config.provider,
                api_key=self._active_config.api_key,
                base_url=self._active_config.base_url,
                max_tokens=self._active_config.max_tokens,
                temperature=self._active_config.temperature,
                timeout=self._active_config.timeout,
                max_retries=self._active_config.max_retries,
                retry_delay=self._active_config.retry_delay,
            )
            return [(_build_backend(cfg), cfg)]
        chain: List[Tuple[ProviderBackend, ModelConfig]] = [
            (self._active, self._active_config)
        ]
        for fb, fc in zip(self._fallbacks, self._config.fallbacks):
            chain.append((fb, fc))
        return chain

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.health_check_interval)
            async with self._health_lock:
                for backend, cfg in [(self._primary, self._config.primary)] + list(
                    zip(self._fallbacks, self._config.fallbacks)
                ):
                    key = f"{cfg.provider.value}/{cfg.model_id}"
                    try:
                        ok = await asyncio.wait_for(backend.health_check(), timeout=10.0)
                        self._health_status[key] = ok
                        if not ok:
                            logger.warning("Health check failed for %s", key)
                    except Exception as exc:
                        self._health_status[key] = False
                        logger.error("Health check error for %s: %s", key, exc)
