"""Async web search module: Tavily, Brave, and SerpAPI providers with
caching, history tracking, result ranking, and summarization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class SearchError(Exception):
    """Base error for web search operations."""


class ProviderError(SearchError):
    """Raised when a search provider request fails."""


class ProviderAuthError(ProviderError):
    """Raised when a search provider rejects credentials."""


class ProviderRateLimitError(ProviderError):
    """Raised when a search provider enforces rate limiting."""


class ProviderTimeoutError(ProviderError):
    """Raised when a search provider request times out."""


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float
    source: str
    published_at: Optional[str] = None
    raw: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResponse:
    query: str
    provider: str
    results: List[SearchResult]
    cached: bool
    latency_ms: float
    timestamp: float


@dataclass(frozen=True)
class SearchHistoryEntry:
    query: str
    provider: str
    result_count: int
    cached: bool
    latency_ms: float
    timestamp: float


@dataclass(frozen=True)
class Summary:
    text: str
    source_count: int
    top_urls: List[str]


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------

class SearchProvider(ABC):
    """Abstract base for web search providers."""

    name: str = "base"

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty.")
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        ...

    @staticmethod
    def _raise_for_status(exc: httpx.HTTPStatusError, provider_name: str) -> None:
        status = exc.response.status_code
        if status in (401, 403):
            raise ProviderAuthError(f"{provider_name} authentication failed: {exc}") from exc
        if status == 429:
            raise ProviderRateLimitError(f"{provider_name} rate limit exceeded: {exc}") from exc
        raise ProviderError(f"{provider_name} request failed with status {status}: {exc}") from exc


class TavilyProvider(SearchProvider):
    name = "tavily"
    BASE_URL = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        payload = {"api_key": self.api_key, "query": query, "max_results": max_results}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.BASE_URL, json=payload)
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"Tavily search timed out: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                self._raise_for_status(exc, "Tavily")
            except httpx.HTTPError as exc:
                raise ProviderError(f"Tavily request failed: {exc}") from exc

        data = response.json()
        results: List[SearchResult] = []
        for index, item in enumerate(data.get("results", [])):
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    score=float(item.get("score", max(0.0, 1.0 - index * 0.01))),
                    source=self.name,
                    published_at=item.get("published_date"),
                    raw=item,
                )
            )
        return results


class BraveProvider(SearchProvider):
    name = "brave"
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        params = {"q": query, "count": max_results}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.BASE_URL, headers=headers, params=params)
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"Brave search timed out: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                self._raise_for_status(exc, "Brave")
            except httpx.HTTPError as exc:
                raise ProviderError(f"Brave request failed: {exc}") from exc

        data = response.json()
        web_results = data.get("web", {}).get("results", [])
        results: List[SearchResult] = []
        for index, item in enumerate(web_results):
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("description", "")),
                    score=max(0.0, 1.0 - index * 0.01),
                    source=self.name,
                    published_at=item.get("age"),
                    raw=item,
                )
            )
        return results


class SerpApiProvider(SearchProvider):
    name = "serpapi"
    BASE_URL = "https://serpapi.com/search"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": max_results,
            "engine": "google",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"SerpAPI search timed out: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                self._raise_for_status(exc, "SerpAPI")
            except httpx.HTTPError as exc:
                raise ProviderError(f"SerpAPI request failed: {exc}") from exc

        data = response.json()
        organic_results = data.get("organic_results", [])
        results: List[SearchResult] = []
        for index, item in enumerate(organic_results):
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("link", "")),
                    snippet=str(item.get("snippet", "")),
                    score=max(0.0, 1.0 - index * 0.01),
                    source=self.name,
                    published_at=item.get("date"),
                    raw=item,
                )
            )
        return results


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------

class SearchCache:
    """TTL + LRU cache for search results, keyed by provider/query/max_results."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: "OrderedDict[str, Tuple[float, List[SearchResult]]]" = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def make_key(provider: str, query: str, max_results: int) -> str:
        normalized = " ".join(query.strip().lower().split())
        digest_input = f"{provider}:{normalized}:{max_results}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()

    async def get(self, key: str) -> Optional[List[SearchResult]]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, results = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return list(results)

    async def set(self, key: str, results: List[SearchResult]) -> None:
        async with self._lock:
            self._store[key] = (time.monotonic() + self.ttl_seconds, list(results))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------

class SearchHistory:
    """Bounded, thread-safe-within-event-loop log of past search requests."""

    def __init__(self, max_entries: int = 500) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self._entries: Deque[SearchHistoryEntry] = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()

    async def add(self, entry: SearchHistoryEntry) -> None:
        async with self._lock:
            self._entries.append(entry)

    async def recent(self, limit: int = 20) -> List[SearchHistoryEntry]:
        async with self._lock:
            entries = list(self._entries)
        return entries[-limit:]

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------

def rank_results(
    results: Sequence[SearchResult], max_results: Optional[int] = None
) -> List[SearchResult]:
    """Deduplicate by URL (keeping the highest-scoring entry) and sort by score."""
    deduped: Dict[str, SearchResult] = {}
    for result in results:
        key = result.url.strip().lower().rstrip("/")
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or result.score > existing.score:
            deduped[key] = result

    ranked = sorted(deduped.values(), key=lambda r: r.score, reverse=True)
    return ranked[:max_results] if max_results is not None else ranked


# ----------------------------------------------------------------------
# Summarization
# ----------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "and", "or", "but", "with", "as",
        "at", "by", "from", "that", "this", "it", "its", "into", "their",
        "his", "her", "they", "he", "she", "we", "you", "i", "not", "no",
    }
)


def _split_sentences(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


def _term_frequencies(text: str) -> Counter:
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
    return Counter(words)


def _score_sentence(sentence: str, term_freq: Counter) -> float:
    words = [w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS]
    if not words:
        return 0.0
    return sum(term_freq[w] for w in words) / len(words)


def summarize_results(
    results: Sequence[SearchResult], max_chars: int = 600, max_sources: int = 5
) -> Summary:
    """Produce an extractive summary from the snippets of the top results."""
    if not results:
        return Summary(text="", source_count=0, top_urls=[])

    top = list(results[:max_sources])
    combined = " ".join(r.snippet.strip() for r in top if r.snippet)
    sentences = _split_sentences(combined)

    if not sentences:
        return Summary(text=combined[:max_chars].strip(), source_count=len(top), top_urls=[r.url for r in top])

    term_freq = _term_frequencies(combined)
    ranked_sentences = sorted(sentences, key=lambda s: _score_sentence(s, term_freq), reverse=True)

    selected: List[str] = []
    used_chars = 0
    for sentence in ranked_sentences:
        if used_chars + len(sentence) > max_chars and selected:
            break
        selected.append(sentence)
        used_chars += len(sentence)

    ordered = [s for s in sentences if s in selected]
    text = " ".join(ordered).strip()
    return Summary(text=text, source_count=len(top), top_urls=[r.url for r in top])


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

class WebSearchService:
    """Coordinates multiple search providers with caching, history, ranking,
    and summarization.
    """

    def __init__(
        self,
        providers: Sequence[SearchProvider],
        cache: Optional[SearchCache] = None,
        history: Optional[SearchHistory] = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one search provider must be configured.")
        self._providers: Dict[str, SearchProvider] = {p.name: p for p in providers}
        self._cache = cache or SearchCache()
        self._history = history or SearchHistory()

    def _resolve_provider(self, provider: Optional[str]) -> SearchProvider:
        if provider is None:
            return next(iter(self._providers.values()))
        resolved = self._providers.get(provider)
        if resolved is None:
            raise ProviderError(f"Unknown search provider '{provider}'.")
        return resolved

    async def search(
        self,
        query: str,
        provider: Optional[str] = None,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> SearchResponse:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        selected_provider = self._resolve_provider(provider)
        cache_key = SearchCache.make_key(selected_provider.name, query, max_results)
        start = time.monotonic()

        cached_results = await self._cache.get(cache_key) if use_cache else None
        cached = cached_results is not None

        if cached_results is not None:
            results = cached_results
        else:
            try:
                raw_results = await selected_provider.search(query, max_results)
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize provider failures
                raise ProviderError(f"{selected_provider.name} search failed: {exc}") from exc

            results = rank_results(raw_results, max_results)
            if use_cache:
                await self._cache.set(cache_key, results)

        latency_ms = (time.monotonic() - start) * 1000.0
        timestamp = time.time()
        response = SearchResponse(
            query=query,
            provider=selected_provider.name,
            results=results,
            cached=cached,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )
        await self._history.add(
            SearchHistoryEntry(
                query=query,
                provider=selected_provider.name,
                result_count=len(results),
                cached=cached,
                latency_ms=latency_ms,
                timestamp=timestamp,
            )
        )
        return response

    async def search_all(
        self, query: str, max_results: int = 10, use_cache: bool = True
    ) -> List[SearchResponse]:
        """Query every configured provider concurrently, skipping failures."""
        names = list(self._providers.keys())
        outcomes = await asyncio.gather(
            *(
                self.search(query, provider=name, max_results=max_results, use_cache=use_cache)
                for name in names
            ),
            return_exceptions=True,
        )

        responses: List[SearchResponse] = []
        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning("Provider '%s' failed: %s", name, outcome)
                continue
            responses.append(outcome)
        return responses

    def summarize(
        self, response: SearchResponse, max_chars: int = 600, max_sources: int = 5
    ) -> Summary:
        return summarize_results(response.results, max_chars=max_chars, max_sources=max_sources)

    async def history_log(self, limit: int = 20) -> List[SearchHistoryEntry]:
        return await self._history.recent(limit)

    async def clear_cache(self) -> None:
        await self._cache.clear()

    async def clear_history(self) -> None:
        await self._history.clear()
