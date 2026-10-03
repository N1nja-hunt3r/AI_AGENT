"""
token_counter.py

Token counting and estimation utilities for OpenAI, Anthropic, and
Gemini models, with lightweight in-memory caching of results.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> List[int]: ...


@dataclass(frozen=True)
class TokenCountResult:
    provider: str
    model: str
    token_count: int
    estimated: bool


class _ResultCache:
    """Thread-safe bounded cache for token count results."""

    def __init__(self, max_size: int = 5000) -> None:
        self._max_size = max_size
        self._store: Dict[str, TokenCountResult] = {}
        self._lock = threading.RLock()

    def _key(self, provider: str, model: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{provider}:{model}:{digest}"

    def get(self, provider: str, model: str, text: str) -> Optional[TokenCountResult]:
        with self._lock:
            return self._store.get(self._key(provider, model, text))

    def set(self, provider: str, model: str, text: str, result: TokenCountResult) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                self._store.pop(next(iter(self._store)))
            self._store[self._key(provider, model, text)] = result


_cache = _ResultCache()


def _estimate_tokens_fallback(text: str) -> int:
    """Heuristic fallback: approximate tokens as ~4 characters or 0.75 * words."""
    if not text:
        return 0
    char_estimate = max(1, len(text) // 4)
    word_count = len(_WORD_RE.findall(text))
    word_estimate = max(1, int(word_count * 1.3))
    return max(char_estimate, word_estimate)


def count_tokens_openai(text: str, model: str = "deepseek-ai/deepseek-v4-pro") -> TokenCountResult:
    """Count tokens for an OpenAI model, using tiktoken if available."""
    cached_result = _cache.get("openai", model, text)
    if cached_result is not None:
        return cached_result

    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        count = len(encoding.encode(text))
        result = TokenCountResult("openai", model, count, estimated=False)
    except ImportError:
        logger.debug("tiktoken not installed; falling back to heuristic estimate")
        result = TokenCountResult("openai", model, _estimate_tokens_fallback(text), estimated=True)

    _cache.set("openai", model, text, result)
    return result


def count_tokens_anthropic(text: str, model: str = "claude-sonnet-4-6") -> TokenCountResult:
    """
    Estimate tokens for an Anthropic model.
    Anthropic's tokenizer is not bundled here; uses a calibrated heuristic
    (~3.5 characters per token for English text) unless a client-provided
    counter is supplied externally.
    """
    cached_result = _cache.get("anthropic", model, text)
    if cached_result is not None:
        return cached_result

    char_estimate = max(1, int(len(text) / 3.5))
    result = TokenCountResult("anthropic", model, char_estimate, estimated=True)
    _cache.set("anthropic", model, text, result)
    return result


def count_tokens_gemini(text: str, model: str = "gemini-1.5-pro") -> TokenCountResult:
    """
    Estimate tokens for a Gemini model using a calibrated heuristic
    (~4 characters per token), since Gemini's tokenizer requires a live
    API call to count exactly.
    """
    cached_result = _cache.get("gemini", model, text)
    if cached_result is not None:
        return cached_result

    char_estimate = max(1, len(text) // 4)
    result = TokenCountResult("gemini", model, char_estimate, estimated=True)
    _cache.set("gemini", model, text, result)
    return result


_PROVIDER_FUNCS = {
    "openai": count_tokens_openai,
    "anthropic": count_tokens_anthropic,
    "gemini": count_tokens_gemini,
}


def count_tokens(text: str, provider: str, model: str) -> TokenCountResult:
    """Dispatch token counting to the appropriate provider-specific function."""
    func = _PROVIDER_FUNCS.get(provider.lower())
    if func is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return func(text, model)


def count_message_tokens(
    messages: List[Dict[str, Any]],
    provider: str,
    model: str,
    *,
    overhead_per_message: int = 4,
) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = " ".join(text_parts)
        else:
            text = str(content)
        total += count_tokens(text, provider, model).token_count + overhead_per_message
    return total


def clear_token_cache() -> None:
    """Clear the module-level token count cache."""
    global _cache
    _cache = _ResultCache()
