"""
llm_connector.py - Production-grade LLM connector with multi-provider support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    pass  # type: ignore[import]

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional provider SDK imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore

try:
    from anthropic import AsyncAnthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None  # type: ignore

try:
    import google.generativeai as _genai_mod  # type: ignore[import]
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    _genai_mod = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    NVIDIA = "nvidia"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    UNKNOWN = "unknown"


# Provider pricing per 1 000 tokens (prompt, completion)
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    # ── NVIDIA NIM ───────────────────────────────────────────────────────────
    "deepseek-ai/deepseek-v4-pro": (0.0005, 0.0015),
    "deepseek-ai/deepseek-v4-flash": (0.0001, 0.0003),
    "meta/llama-3.3-70b-instruct": (0.00035, 0.0004),
    "qwen/qwen3.5-397b-a17b": (0.0012, 0.0012),
    "meta/llama-3.2-90b-vision-instruct": (0.0005, 0.0005),
    "nvidia/nv-embed-v1": (0.00001, 0.00001),
    "openai/whisper-large-v3": (0.00001, 0.00001),
    "magpie-tts-multilingual": (0.00001, 0.00001),
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    # Claude
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    # Gemini
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash-exp": (0.0, 0.0),
    # Ollama (local – free)
    "ollama": (0.0, 0.0),
}

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 1.0  # seconds
_DEFAULT_TIMEOUT = 120.0
_OLLAMA_BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    finish_reason: FinishReason
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latency_ms: float = 0.0


@dataclass
class ProviderConfig:
    provider: Provider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_delay: float = _DEFAULT_RETRY_DELAY
    timeout: float = _DEFAULT_TIMEOUT
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMConnectorError(Exception):
    """Base connector error."""


class ProviderUnavailableError(LLMConnectorError):
    """Raised when a provider cannot be reached."""


class AllProvidersFailedError(LLMConnectorError):
    """Raised when every provider in the fallback chain fails."""


class BudgetExceededError(LLMConnectorError):
    """Raised when estimated cost would exceed budget."""


# ---------------------------------------------------------------------------
# Cost calculator
# ---------------------------------------------------------------------------

def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = model.lower()
    pricing = _DEFAULT_PRICING.get(key)
    if pricing is None:
        # Try prefix match
        for k, v in _DEFAULT_PRICING.items():
            if key.startswith(k) or k.startswith(key.split("-")[0]):
                pricing = v
                break
    if pricing is None:
        pricing = (0.0, 0.0)
    prompt_cost = (prompt_tokens / 1000) * pricing[0]
    completion_cost = (completion_tokens / 1000) * pricing[1]
    return round(prompt_cost + completion_cost, 8)


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

class _OpenAIAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _OPENAI_AVAILABLE:
            raise LLMConnectorError("openai package not installed. pip install openai")
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**kwargs)
        self.config = config

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            **self.config.extra,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        if stream and not tools:
            return self._stream(params)
        return await self._complete(params)

    async def _complete(self, params: Dict[str, Any]) -> LLMResponse:
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(**params)
        latency = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = _calculate_cost(self.config.model, prompt_tokens, completion_tokens)
        finish = _map_finish_reason(choice.finish_reason)
        return LLMResponse(
            text=text,
            provider=Provider.OPENAI,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            finish_reason=finish,
            tool_calls=tool_calls,
            raw=response,
            latency_ms=latency,
        )

    async def _stream(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        params["stream"] = True
        async with await self._client.chat.completions.create(**params) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


class _ClaudeAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise LLMConnectorError("anthropic package not installed. pip install anthropic")
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = AsyncAnthropic(api_key=api_key, timeout=config.timeout)
        self.config = config

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        system, claude_messages = _split_system_messages(messages)
        params: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": claude_messages,
            **self.config.extra,
            **kwargs,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = _convert_tools_to_claude(tools)
        if stream and not tools:
            return self._stream(params)
        return await self._complete(params)

    async def _complete(self, params: Dict[str, Any]) -> LLMResponse:
        t0 = time.perf_counter()
        response = await self._client.messages.create(**params)
        latency = (time.perf_counter() - t0) * 1000
        text = ""
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        cost = _calculate_cost(self.config.model, prompt_tokens, completion_tokens)
        finish_map = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALL,
        }
        finish = finish_map.get(response.stop_reason or "", FinishReason.UNKNOWN)
        return LLMResponse(
            text=text,
            provider=Provider.CLAUDE,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            finish_reason=finish,
            tool_calls=tool_calls,
            raw=response,
            latency_ms=latency,
        )

    async def _stream(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


class _GeminiAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _GEMINI_AVAILABLE:
            raise LLMConnectorError(
                "google-generativeai package not installed. pip install google-generativeai"
            )
        api_key = config.api_key or os.environ.get("GEMINI_API_KEY", "")
        _genai_mod.configure(api_key=api_key)
        self._model_name = config.model
        self.config = config

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        system_instruction, gemini_contents = _convert_messages_to_gemini(messages)
        model_kwargs: Dict[str, Any] = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        if tools:
            model_kwargs["tools"] = _convert_tools_to_gemini(tools)
        model = _genai_mod.GenerativeModel(self._model_name, **model_kwargs)
        gen_config = _genai_mod.GenerationConfig(
            max_output_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )
        if stream and not tools:
            return await self._stream(model, gemini_contents, gen_config)
        return await self._complete(model, gemini_contents, gen_config)

    async def _complete(
        self, model: Any, contents: List[Any], gen_config: Any
    ) -> LLMResponse:
        t0 = time.perf_counter()
        response = await asyncio.to_thread(
            model.generate_content, contents, generation_config=gen_config
        )
        latency = (time.perf_counter() - t0) * 1000
        text = ""
        tool_calls: List[ToolCall] = []
        try:
            text = response.text or ""
        except Exception:
            pass
        for candidate in response.candidates or []:
            for part in candidate.content.parts if candidate.content else []:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(id=str(uuid.uuid4()), name=fc.name, arguments=args)
                    )
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        cost = _calculate_cost(self.config.model, prompt_tokens, completion_tokens)
        return LLMResponse(
            text=text,
            provider=Provider.GEMINI,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            finish_reason=FinishReason.STOP if not tool_calls else FinishReason.TOOL_CALL,
            tool_calls=tool_calls,
            raw=response,
            latency_ms=latency,
        )

    async def _stream(
        self, model: Any, contents: List[Any], gen_config: Any
    ) -> AsyncGenerator[str, None]:
        def _sync_gen():
            for chunk in model.generate_content(
                contents, generation_config=gen_config, stream=True
            ):
                try:
                    yield chunk.text
                except Exception:
                    yield ""

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def _run():
            for text in _sync_gen():
                loop.call_soon_threadsafe(queue.put_nowait, text)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        await asyncio.to_thread(_run)

        async def _gen():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return _gen()

    async def health_check(self) -> bool:
        try:
            models = await asyncio.to_thread(lambda: list(_genai_mod.list_models()))
            return len(models) > 0
        except Exception:
            return False


class _OllamaAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._base_url = config.base_url or os.environ.get("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 4096),
            },
        }
        if tools:
            payload["tools"] = tools
        if stream and not tools:
            return self._stream(payload)
        return await self._complete(payload)

    async def _complete(self, payload: Dict[str, Any]) -> LLMResponse:
        payload["stream"] = False
        t0 = time.perf_counter()
        resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = resp.json()
        msg = data.get("message", {})
        text = msg.get("content", "")
        tool_calls: List[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(id=str(uuid.uuid4()), name=fn.get("name", ""), arguments=args)
            )
        prompt_eval = data.get("prompt_eval_count", 0) or 0
        eval_count = data.get("eval_count", 0) or 0
        total = prompt_eval + eval_count
        cost = 0.0
        finish = FinishReason.TOOL_CALL if tool_calls else FinishReason.STOP
        return LLMResponse(
            text=text,
            provider=Provider.OLLAMA,
            model=self.config.model,
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=total,
            cost_usd=cost,
            finish_reason=finish,
            tool_calls=tool_calls,
            raw=data,
            latency_ms=latency,
        )

    async def _stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        payload["stream"] = True
        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Helper converters
# ---------------------------------------------------------------------------

def _map_finish_reason(reason: Optional[str]) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALL,
        "function_call": FinishReason.TOOL_CALL,
        "content_filter": FinishReason.STOP,
    }
    return mapping.get(reason or "", FinishReason.UNKNOWN)


def _split_system_messages(
    messages: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    other: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            other.append(m)
    return "\n".join(system_parts), other


def _convert_tools_to_claude(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def _convert_tools_to_gemini(tools: List[Dict[str, Any]]) -> List[Any]:
    if not _GEMINI_AVAILABLE:
        return []
    declarations = []
    for t in tools:
        fn = t.get("function", t)
        declarations.append(
            _genai_mod.protos.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=_openapi_to_gemini_schema(fn.get("parameters", {})),
            )
        )
    return [_genai_mod.protos.Tool(function_declarations=declarations)]


def _openapi_to_gemini_schema(schema: Dict[str, Any]) -> Any:
    if not _GEMINI_AVAILABLE:
        return None
    return _genai_mod.protos.Schema(
        type=_genai_mod.protos.Type.OBJECT,
        properties={
            k: _genai_mod.protos.Schema(
                type=_map_type(v.get("type", "string")),
                description=v.get("description", ""),
            )
            for k, v in schema.get("properties", {}).items()
        },
        required=schema.get("required", []),
    )


def _map_type(t: str) -> Any:
    if not _GEMINI_AVAILABLE:
        return None
    type_map = {
        "string": _genai_mod.protos.Type.STRING,
        "number": _genai_mod.protos.Type.NUMBER,
        "integer": _genai_mod.protos.Type.INTEGER,
        "boolean": _genai_mod.protos.Type.BOOLEAN,
        "array": _genai_mod.protos.Type.ARRAY,
        "object": _genai_mod.protos.Type.OBJECT,
    }
    return type_map.get(t, _genai_mod.protos.Type.STRING)


def _convert_messages_to_gemini(
    messages: List[Dict[str, Any]]
) -> Tuple[Optional[str], List[Any]]:
    if not _GEMINI_AVAILABLE:
        return None, []
    system_instruction = None
    contents = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    return system_instruction, contents


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_AdapterType = Union[_OpenAIAdapter, _ClaudeAdapter, _GeminiAdapter, _OllamaAdapter]


def _build_adapter(config: ProviderConfig) -> _AdapterType:
    if config.provider == Provider.OPENAI:
        return _OpenAIAdapter(config)
    elif config.provider == Provider.CLAUDE:
        return _ClaudeAdapter(config)
    elif config.provider == Provider.GEMINI:
        return _GeminiAdapter(config)
    elif config.provider == Provider.OLLAMA:
        return _OllamaAdapter(config)
    else:
        raise LLMConnectorError(f"Unknown provider: {config.provider}")


# ---------------------------------------------------------------------------
# LLMConnector – Singleton
# ---------------------------------------------------------------------------

class LLMConnector:
    """
    Production-grade multi-provider LLM connector.

    Usage
    -----
    connector = LLMConnector.get_instance()
    response = await connector.complete(messages=[{"role": "user", "content": "Hello"}])
    """

    _instance: Optional["LLMConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._adapters: Dict[str, _AdapterType] = {}
        self._fallback_chain: List[str] = []
        self._token_usage: TokenUsage = TokenUsage()
        self._budget_usd: Optional[float] = None
        self._response_hooks: List[Callable[[LLMResponse], None]] = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "LLMConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "LLMConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        configs: List[ProviderConfig],
        fallback_order: Optional[List[str]] = None,
        budget_usd: Optional[float] = None,
    ) -> "LLMConnector":
        """
        Configure the connector with one or more provider configs.
        configs: list of ProviderConfig objects.
        fallback_order: list of adapter keys (e.g. ["openai-gpt-4o", "claude-claude-3-5-sonnet-20241022"]).
        budget_usd: optional hard budget cap.
        """
        for cfg in configs:
            key = f"{cfg.provider.value}-{cfg.model}"
            self._adapters[key] = _build_adapter(cfg)
            if key not in self._fallback_chain:
                self._fallback_chain.append(key)

        if fallback_order:
            self._fallback_chain = [k for k in fallback_order if k in self._adapters]
            # append any not in explicit order
            for k in self._adapters:
                if k not in self._fallback_chain:
                    self._fallback_chain.append(k)

        if budget_usd is not None:
            self._budget_usd = budget_usd

        self._initialized = True
        logger.info(
            "LLMConnector configured. Adapters=%s FallbackChain=%s Budget=%s",
            list(self._adapters.keys()),
            self._fallback_chain,
            self._budget_usd,
        )
        return self

    def configure_from_env(self) -> "LLMConnector":
        """Auto-configure from environment variables."""
        configs: List[ProviderConfig] = []
        if os.environ.get("OPENAI_API_KEY") and _OPENAI_AVAILABLE:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            configs.append(ProviderConfig(provider=Provider.OPENAI, model=model))

        if os.environ.get("ANTHROPIC_API_KEY") and _ANTHROPIC_AVAILABLE:
            model = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            configs.append(ProviderConfig(provider=Provider.CLAUDE, model=model))

        if os.environ.get("GEMINI_API_KEY") and _GEMINI_AVAILABLE:
            model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            configs.append(ProviderConfig(provider=Provider.GEMINI, model=model))

        if os.environ.get("OLLAMA_ENABLED", "").lower() in ("1", "true", "yes"):
            model = os.environ.get("OLLAMA_MODEL", "llama3")
            configs.append(ProviderConfig(provider=Provider.OLLAMA, model=model))

        if not configs:
            logger.warning("No LLM providers configured from environment.")
            return self

        budget = os.environ.get("LLM_BUDGET_USD")
        budget_usd = float(budget) if budget else None
        return self.configure(configs, budget_usd=budget_usd)

    def set_budget(self, budget_usd: float) -> None:
        self._budget_usd = budget_usd

    def add_response_hook(self, hook: Callable[[LLMResponse], None]) -> None:
        """Register a callback invoked after every successful response."""
        self._response_hooks.append(hook)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        provider_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Complete a chat request.

        Parameters
        ----------
        messages      : OpenAI-style message list.
        provider_key  : specific adapter key; if None uses fallback chain.
        tools         : tool/function definitions (OpenAI format).
        stream        : return an async generator of text chunks.
        use_fallback  : attempt next provider on failure.
        **kwargs      : forwarded to the adapter (temperature, max_tokens, …).
        """
        if not self._initialized:
            self.configure_from_env()

        self._check_budget()

        if provider_key:
            adapter = self._adapters.get(provider_key)
            if not adapter:
                raise LLMConnectorError(f"Unknown provider key: {provider_key}")
            return await self._complete_with_retry(adapter, messages, tools, stream, **kwargs)

        if use_fallback:
            return await self._complete_with_fallback(messages, tools, stream, **kwargs)

        # Use first adapter
        if not self._fallback_chain:
            raise LLMConnectorError("No adapters configured.")
        adapter = self._adapters[self._fallback_chain[0]]
        return await self._complete_with_retry(adapter, messages, tools, stream, **kwargs)

    async def _complete_with_retry(
        self,
        adapter: _AdapterType,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        cfg: ProviderConfig = adapter.config  # type: ignore[attr-defined]
        last_exc: Optional[Exception] = None
        for attempt in range(cfg.max_retries):
            try:
                result = await adapter.complete(
                    messages=messages, tools=tools, stream=stream, **kwargs
                )
                if not stream:
                    assert isinstance(result, LLMResponse)
                    self._record_usage(result)
                return result
            except (BudgetExceededError,) as exc:
                raise exc
            except Exception as exc:
                last_exc = exc
                wait = cfg.retry_delay * (2 ** attempt)
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    cfg.max_retries,
                    exc,
                    wait,
                )
                if attempt < cfg.max_retries - 1:
                    await asyncio.sleep(wait)
        raise ProviderUnavailableError(
            f"Provider {cfg.provider}/{cfg.model} failed after {cfg.max_retries} attempts: {last_exc}"
        ) from last_exc

    async def _complete_with_fallback(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        if not self._fallback_chain:
            raise LLMConnectorError("No adapters configured.")
        errors: List[str] = []
        for key in self._fallback_chain:
            adapter = self._adapters[key]
            try:
                result = await self._complete_with_retry(
                    adapter, messages, tools, stream, **kwargs
                )
                return result
            except BudgetExceededError:
                raise
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                logger.warning("Fallback: %s failed, trying next. Error: %s", key, exc)
        raise AllProvidersFailedError(
            "All providers failed:\n" + "\n".join(errors)
        )

    # ------------------------------------------------------------------
    # Streaming helper (non-blocking wrapper)
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        provider_key: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Convenience wrapper that always returns an async generator."""
        result = await self.complete(
            messages=messages,
            provider_key=provider_key,
            stream=True,
            use_fallback=provider_key is None,
            **kwargs,
        )
        if isinstance(result, LLMResponse):
            async def _wrap():
                yield result.text
            return _wrap()
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tool calling helpers
    # ------------------------------------------------------------------

    async def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_executor: Optional[Callable[[ToolCall], Any]] = None,
        max_rounds: int = 10,
        provider_key: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Agentic loop: call LLM, execute tool calls, feed results back until done.

        Parameters
        ----------
        messages      : conversation history.
        tools         : tool definitions.
        tool_executor : async or sync callable(ToolCall) -> result string.
        max_rounds    : safety cap on tool call rounds.
        """
        current_messages = list(messages)
        last_response: Optional[LLMResponse] = None

        for _round in range(max_rounds):
            response = await self.complete(
                messages=current_messages,
                provider_key=provider_key,
                tools=tools,
                stream=False,
                **kwargs,
            )
            assert isinstance(response, LLMResponse)
            last_response = response

            if not response.tool_calls:
                break

            # Append assistant message with tool calls
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.text}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
            current_messages.append(assistant_msg)

            # Execute tools
            if tool_executor:
                for tc in response.tool_calls:
                    try:
                        if asyncio.iscoroutinefunction(tool_executor):
                            result = await tool_executor(tc)
                        else:
                            result = await asyncio.to_thread(tool_executor, tc)
                        result_str = result if isinstance(result, str) else json.dumps(result)
                    except Exception as exc:
                        result_str = f"Tool error: {exc}"
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                    )
            else:
                break  # No executor; return tool call response

        return last_response or LLMResponse(
            text="",
            provider="unknown",
            model="unknown",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            finish_reason=FinishReason.ERROR,
        )

    # ------------------------------------------------------------------
    # Budget & token tracking
    # ------------------------------------------------------------------

    def _check_budget(self) -> None:
        if self._budget_usd is not None:
            if self._token_usage.total_cost_usd >= self._budget_usd:
                raise BudgetExceededError(
                    f"Budget of ${self._budget_usd:.4f} exceeded "
                    f"(spent ${self._token_usage.total_cost_usd:.4f})"
                )

    def _record_usage(self, response: LLMResponse) -> None:
        self._token_usage.prompt_tokens += response.prompt_tokens
        self._token_usage.completion_tokens += response.completion_tokens
        self._token_usage.total_tokens += response.total_tokens
        self._token_usage.total_cost_usd += response.cost_usd
        self._token_usage.request_count += 1
        logger.debug(
            "Token usage recorded: +%d tokens, +$%.6f | Total: %d tokens, $%.4f",
            response.total_tokens,
            response.cost_usd,
            self._token_usage.total_tokens,
            self._token_usage.total_cost_usd,
        )
        for hook in self._response_hooks:
            try:
                hook(response)
            except Exception as exc:
                logger.warning("Response hook raised: %s", exc)

    @property
    def token_usage(self) -> TokenUsage:
        """Cumulative token and cost usage."""
        return self._token_usage

    def reset_usage(self) -> None:
        self._token_usage = TokenUsage()

    def remaining_budget(self) -> Optional[float]:
        if self._budget_usd is None:
            return None
        return max(0.0, self._budget_usd - self._token_usage.total_cost_usd)

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return _calculate_cost(model, prompt_tokens, completion_tokens)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self, provider_key: Optional[str] = None) -> Dict[str, bool]:
        """
        Run health checks.
        Returns dict of {adapter_key: is_healthy}.
        """
        keys = [provider_key] if provider_key else list(self._adapters.keys())
        results: Dict[str, bool] = {}
        tasks = {k: self._adapters[k].health_check() for k in keys if k in self._adapters}
        checks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, outcome in zip(tasks.keys(), checks):
            if isinstance(outcome, Exception):
                results[key] = False
                logger.warning("Health check failed for %s: %s", key, outcome)
            else:
                results[key] = bool(outcome)
        return results

    async def get_healthy_providers(self) -> List[str]:
        statuses = await self.health_check()
        return [k for k, healthy in statuses.items() if healthy]

    # ------------------------------------------------------------------
    # Context helpers (for context_engine.py)
    # ------------------------------------------------------------------

    def count_tokens_estimate(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)

    def messages_token_estimate(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens_estimate(content) + 4  # role overhead
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self.count_tokens_estimate(str(part))
        return total + 3  # reply priming

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())

    def get_primary_adapter_key(self) -> Optional[str]:
        return self._fallback_chain[0] if self._fallback_chain else None

    def get_primary_model(self) -> Optional[str]:
        key = self.get_primary_adapter_key()
        if key and key in self._adapters:
            return self._adapters[key].config.model  # type: ignore[attr-defined]
        return None

    def get_primary_provider(self) -> Optional[str]:
        key = self.get_primary_adapter_key()
        if key:
            parts = key.split("-", 1)
            return parts[0] if parts else None
        return None

    def set_fallback_order(self, order: List[str]) -> None:
        self._fallback_chain = [k for k in order if k in self._adapters]

    def remove_adapter(self, key: str) -> None:
        self._adapters.pop(key, None)
        if key in self._fallback_chain:
            self._fallback_chain.remove(key)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LLMConnector":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            if hasattr(adapter, "aclose"):
                try:
                    await adapter.aclose()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMConnector(adapters={list(self._adapters.keys())}, "
            f"requests={self._token_usage.request_count}, "
            f"cost=${self._token_usage.total_cost_usd:.4f})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_connector() -> LLMConnector:
    """Return the global singleton LLMConnector, auto-configured from env."""
    connector = LLMConnector.get_instance()
    if not connector._initialized:
        connector.configure_from_env()
    return connector


async def complete(
    messages: List[Dict[str, Any]],
    provider_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    **kwargs: Any,
) -> Union[LLMResponse, AsyncGenerator[str, None]]:
    """Module-level shortcut for LLMConnector.get_instance().complete(...)."""
    return await get_connector().complete(
        messages=messages,
        provider_key=provider_key,
        tools=tools,
        stream=stream,
        **kwargs,
    )


async def stream_complete(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Module-level streaming shortcut."""
    return await get_connector().stream(messages=messages, **kwargs)


async def health_check() -> Dict[str, bool]:
    """Module-level health check."""
    return await get_connector().health_check()
