from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class SearchProvider(str, Enum):
    SERPER = "serper"
    BRAVE = "brave"
    SERPAPI = "serpapi"
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"


class SearchType(str, Enum):
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    ACADEMIC = "academic"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int = 0
    score: float = 0.0
    published_date: Optional[str] = None
    source: Optional[str] = None
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "rank": self.rank,
            "score": self.score,
            "published_date": self.published_date,
            "source": self.source,
        }


@dataclass
class SearchResponse:
    query: str
    results: List[SearchResult]
    summary: Optional[str] = None
    total_results: int = 0
    provider: Optional[str] = None
    search_type: SearchType = SearchType.WEB
    latency_ms: float = 0.0
    cached: bool = False
    searched_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "total_results": self.total_results,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


@dataclass
class SearchHistoryEntry:
    query: str
    response: SearchResponse
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchServiceConfig:
    provider: SearchProvider = SearchProvider.SERPER
    api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    max_results: int = 10
    timeout: float = 15.0
    cache_ttl: float = 3600.0
    cache_max_size: int = 500
    enable_summarization: bool = True
    summarization_model: str = "deepseek-ai/deepseek-v4-pro"
    summarization_api_key: Optional[str] = None
    max_retries: int = 2
    retry_delay: float = 1.0
    history_limit: int = 200
    rerank_results: bool = True


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class SearchCache:
    def __init__(self, ttl: float, max_size: int) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: Dict[str, Tuple[SearchResponse, float]] = {}
        self._lock = asyncio.Lock()

    def _key(self, query: str, search_type: SearchType, num_results: int) -> str:
        raw = f"{query}|{search_type.value}|{num_results}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def get(self, query: str, search_type: SearchType, num_results: int) -> Optional[SearchResponse]:
        async with self._lock:
            key = self._key(query, search_type, num_results)
            entry = self._store.get(key)
            if entry is None:
                return None
            resp, stored_at = entry
            if time.time() - stored_at > self._ttl:
                del self._store[key]
                return None
            resp.cached = True
            return resp

    async def set(self, query: str, search_type: SearchType, num_results: int, resp: SearchResponse) -> None:
        async with self._lock:
            if len(self._store) >= self._max_size:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]
            key = self._key(query, search_type, num_results)
            self._store[key] = (resp, time.time())

    async def invalidate(self, query: str) -> None:
        async with self._lock:
            keys = [k for k in self._store if query.lower() in k]
            for k in keys:
                del self._store[k]

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

class BaseSearchBackend:
    async def search(
        self, query: str, num_results: int, search_type: SearchType, **kwargs: Any
    ) -> List[SearchResult]:
        raise NotImplementedError


class SerperBackend(BaseSearchBackend):
    BASE_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str, timeout: float) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(
        self, query: str, num_results: int, search_type: SearchType, **kwargs: Any
    ) -> List[SearchResult]:
        import httpx
        endpoint = "https://google.serper.dev/news" if search_type == SearchType.NEWS else self.BASE_URL
        payload = {"q": query, "num": num_results}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                endpoint,
                json=payload,
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        items = data.get("organic", data.get("news", []))
        for i, item in enumerate(items[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                rank=i + 1,
                published_date=item.get("date"),
                source=item.get("source"),
                provider="serper",
            ))
        return results


class BraveBackend(BaseSearchBackend):
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: float) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(
        self, query: str, num_results: int, search_type: SearchType, **kwargs: Any
    ) -> List[SearchResult]:
        import httpx
        params = {"q": query, "count": num_results}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self.BASE_URL,
                params=params,
                headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        for i, item in enumerate(data.get("web", {}).get("results", [])[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                rank=i + 1,
                published_date=item.get("page_age"),
                provider="brave",
            ))
        return results


class TavilyBackend(BaseSearchBackend):
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(
        self, query: str, num_results: int, search_type: SearchType, **kwargs: Any
    ) -> List[SearchResult]:
        import httpx
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self.BASE_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        for i, item in enumerate(data.get("results", [])[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                rank=i + 1,
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
                provider="tavily",
            ))
        return results


class DuckDuckGoBackend(BaseSearchBackend):
    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    async def search(
        self, query: str, num_results: int, search_type: SearchType, **kwargs: Any
    ) -> List[SearchResult]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError("pip install duckduckgo-search") from exc
        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(
            None, lambda: DDGS().text(query, max_results=num_results)
        )
        results: List[SearchResult] = []
        for i, item in enumerate(items[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                rank=i + 1,
                provider="duckduckgo",
            ))
        return results


def _build_backend(config: SearchServiceConfig) -> BaseSearchBackend:
    if config.provider == SearchProvider.SERPER:
        if not config.api_key:
            raise ValueError("SERPER api_key required")
        return SerperBackend(config.api_key, config.timeout)
    if config.provider == SearchProvider.BRAVE:
        key = config.brave_api_key or config.api_key
        if not key:
            raise ValueError("Brave api_key required")
        return BraveBackend(key, config.timeout)
    if config.provider == SearchProvider.TAVILY:
        key = config.tavily_api_key or config.api_key
        if not key:
            raise ValueError("Tavily api_key required")
        return TavilyBackend(key, config.timeout)
    if config.provider == SearchProvider.DUCKDUCKGO:
        return DuckDuckGoBackend(config.timeout)
    raise ValueError(f"Unsupported provider: {config.provider}")


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class ResultRanker:
    DOMAIN_BOOST: Dict[str, float] = {
        "arxiv.org": 0.15,
        "github.com": 0.10,
        "wikipedia.org": 0.08,
        "stackoverflow.com": 0.08,
        "docs.python.org": 0.10,
        "nature.com": 0.12,
    }

    def rank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        query_words = set(query.lower().split())
        for result in results:
            base_score = 1.0 / (result.rank + 1)
            text = (result.title + " " + result.snippet).lower()
            match_score = sum(1 for w in query_words if w in text) / max(len(query_words), 1)
            domain_boost = 0.0
            for domain, boost in self.DOMAIN_BOOST.items():
                if domain in result.url:
                    domain_boost = boost
                    break
            has_date = 0.05 if result.published_date else 0.0
            result.score = round(0.4 * base_score + 0.4 * match_score + 0.15 * domain_boost + 0.05 * has_date, 4)

        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class SearchSummarizer:
    def __init__(self, model: str, api_key: Optional[str]) -> None:
        self._model = model
        self._api_key = api_key

    async def summarize(self, query: str, results: List[SearchResult], max_results: int = 5) -> str:
        snippets = "\n\n".join(
            f"[{r.rank}] {r.title}\n{r.snippet}"
            for r in results[:max_results]
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Search results:\n{snippets}\n\n"
            "Provide a concise, factual summary of the above results that directly answers the query. "
            "Cite sources by number [1], [2] etc. Be brief and accurate."
        )
        try:
            import os
            from openai import AsyncOpenAI
            base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            api_key = self._api_key or os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
            resp = await client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or snippets[:800]
        except Exception as exc:
            logger.warning("Summarization failed: %s", exc)
            return snippets[:800]


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(fn: Any, max_retries: int, delay: float) -> Any:
    last_exc: Exception = Exception("No attempts")
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))
    raise last_exc


# ---------------------------------------------------------------------------
# SearchService
# ---------------------------------------------------------------------------

class SearchService:
    """
    Web search abstraction with caching, result ranking, summarization, and history.
    """

    def __init__(self, config: Optional[SearchServiceConfig] = None) -> None:
        self._config = config or SearchServiceConfig()
        self._backend = _build_backend(self._config)
        self._cache = SearchCache(self._config.cache_ttl, self._config.cache_max_size)
        self._ranker = ResultRanker()
        self._summarizer = (
            SearchSummarizer(self._config.summarization_model, self._config.summarization_api_key)
            if self._config.enable_summarization
            else None
        )
        self._history: List[SearchHistoryEntry] = []
        self._total_searches: int = 0
        self._cache_hits: int = 0

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: SearchType = SearchType.WEB,
        use_cache: bool = True,
        summarize: bool = True,
        rerank: bool = True,
        **kwargs: Any,
    ) -> SearchResponse:
        n = num_results or self._config.max_results
        if use_cache:
            cached = await self._cache.get(query, search_type, n)
            if cached:
                self._cache_hits += 1
                self._record_history(query, cached)
                return cached

        t0 = time.monotonic()
        results = await _with_retry(
            lambda: self._backend.search(query, n, search_type, **kwargs),
            self._config.max_retries,
            self._config.retry_delay,
        )

        if rerank and self._config.rerank_results:
            results = self._ranker.rank(query, results)

        summary: Optional[str] = None
        if summarize and self._config.enable_summarization and self._summarizer and results:
            summary = await self._summarizer.summarize(query, results)

        latency = (time.monotonic() - t0) * 1000
        response = SearchResponse(
            query=query,
            results=results,
            summary=summary,
            total_results=len(results),
            provider=self._config.provider.value,
            search_type=search_type,
            latency_ms=round(latency, 2),
            cached=False,
        )

        if use_cache:
            await self._cache.set(query, search_type, n, response)

        self._total_searches += 1
        self._record_history(query, response)
        return response

    async def search_news(self, query: str, **kwargs: Any) -> SearchResponse:
        return await self.search(query, search_type=SearchType.NEWS, **kwargs)

    async def search_batch(
        self,
        queries: List[str],
        parallel: bool = True,
        **kwargs: Any,
    ) -> List[SearchResponse]:
        if parallel:
            return list(await asyncio.gather(*[self.search(q, **kwargs) for q in queries]))
        results: List[SearchResponse] = []
        for q in queries:
            results.append(await self.search(q, **kwargs))
        return results

    async def search_and_extract(
        self,
        query: str,
        fields: List[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        response = await self.search(query, **kwargs)
        extracted: List[Dict[str, Any]] = []
        for r in response.results:
            entry: Dict[str, Any] = {}
            for f in fields:
                entry[f] = getattr(r, f, r.metadata.get(f))
            extracted.append(entry)
        return extracted

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def invalidate_cache(self, query: Optional[str] = None) -> None:
        if query:
            await self._cache.invalidate(query)
        else:
            await self._cache.clear()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(
        self,
        limit: int = 20,
        query_filter: Optional[str] = None,
    ) -> List[SearchHistoryEntry]:
        hist = self._history
        if query_filter:
            hist = [h for h in hist if query_filter.lower() in h.query.lower()]
        return hist[-limit:]

    def clear_history(self) -> None:
        self._history.clear()

    def get_recent_queries(self, n: int = 10) -> List[str]:
        return [h.query for h in self._history[-n:]]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_searches": self._total_searches,
            "cache_hits": self._cache_hits,
            "cache_size": self._cache.size,
            "cache_hit_rate": round(self._cache_hits / max(self._total_searches, 1), 3),
            "history_count": len(self._history),
            "provider": self._config.provider.value,
        }

    # ------------------------------------------------------------------
    # Provider switching
    # ------------------------------------------------------------------

    def switch_provider(self, config: SearchServiceConfig) -> None:
        self._config = config
        self._backend = _build_backend(config)
        logger.info("Switched search provider to %s", config.provider.value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_history(self, query: str, response: SearchResponse) -> None:
        self._history.append(SearchHistoryEntry(query=query, response=response))
        if len(self._history) > self._config.history_limit:
            self._history = self._history[-self._config.history_limit:]
