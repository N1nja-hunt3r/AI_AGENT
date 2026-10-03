"""
Web Capability - Internet access for the AI Operating System.

This module provides the WebCapability class which handles:
- Web search across multiple providers
- URL fetching with content extraction
- HTML parsing and content cleaning
- Page summarization
- Query optimization
- Search history tracking
- Result caching

Supports multiple search providers:
- Mock (default, for development/testing)
- Tavily (future, for AI-optimized search)
- SerpAPI (future, for Google search)
- Brave Search (future, for privacy-focused search)
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

if TYPE_CHECKING:
    pass

# Import from kernel modules
from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CapabilityType,
)
from app.agents.exceptions import (
    CapabilityError,
    ValidationError,
)


# =============================================================================
# ENUMS
# =============================================================================


class SearchProvider(Enum):
    """Supported search providers."""

    MOCK = "mock"
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    BRAVE = "brave"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"


class ContentType(Enum):
    """Types of web content."""

    HTML = "html"
    JSON = "json"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class WebAction(Enum):
    """Actions supported by the web capability."""

    SEARCH = "search"
    FETCH = "fetch"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    OPTIMIZE_QUERY = "optimize_query"
    HISTORY = "history"
    CLEAR_CACHE = "clear_cache"
    PROVIDERS = "providers"


class SearchType(Enum):
    """Types of search queries."""

    GENERAL = "general"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    ACADEMIC = "academic"
    CODE = "code"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SearchResult:
    """
    A single search result.

    Attributes:
        title: Result title
        url: Result URL
        snippet: Text snippet/description
        content: Full extracted content (if fetched)
        score: Relevance score (0-1)
        position: Position in search results
        source: Source domain
        published_date: Publication date (if available)
        metadata: Additional metadata
    """

    title: str
    url: str
    snippet: str
    content: str | None = None
    score: float = 0.0
    position: int = 0
    source: str = ""
    published_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract source from URL if not provided."""
        if not self.source and self.url:
            try:
                parsed = urllib.parse.urlparse(self.url)
                self.source = parsed.netloc
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "position": self.position,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            snippet=data.get("snippet", ""),
            content=data.get("content"),
            score=data.get("score", 0.0),
            position=data.get("position", 0),
            source=data.get("source", ""),
            published_date=(
                datetime.fromisoformat(data["published_date"])
                if data.get("published_date")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResults:
    """
    Collection of search results.

    Attributes:
        query: Original search query
        results: List of search results
        total_results: Total results available
        search_time: Time taken for search
        provider: Search provider used
        search_type: Type of search performed
        metadata: Additional metadata
    """

    query: str
    results: list[SearchResult]
    total_results: int = 0
    search_time: float = 0.0
    provider: str = ""
    search_type: SearchType = SearchType.GENERAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time,
            "provider": self.provider,
            "search_type": self.search_type.value,
            "metadata": self.metadata,
        }


@dataclass
class FetchedContent:
    """
    Fetched web content.

    Attributes:
        url: Source URL
        content: Raw content
        content_type: Type of content
        title: Page title
        extracted_text: Clean extracted text
        links: Extracted links
        images: Extracted image URLs
        fetch_time: Time taken to fetch
        status_code: HTTP status code
        headers: Response headers
        metadata: Additional metadata
    """

    url: str
    content: str
    content_type: ContentType = ContentType.HTML
    title: str = ""
    extracted_text: str = ""
    links: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    fetch_time: float = 0.0
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "content_type": self.content_type.value,
            "title": self.title,
            "extracted_text": self.extracted_text,
            "links": self.links,
            "images": self.images,
            "fetch_time": self.fetch_time,
            "status_code": self.status_code,
            "metadata": self.metadata,
        }


@dataclass
class SearchHistoryEntry:
    """
    Search history entry.

    Attributes:
        query: Search query
        timestamp: When search was performed
        provider: Provider used
        result_count: Number of results
        search_type: Type of search
        agent_id: Agent that performed search
        session_id: Session ID
    """

    query: str
    timestamp: datetime
    provider: str
    result_count: int
    search_type: SearchType = SearchType.GENERAL
    agent_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "result_count": self.result_count,
            "search_type": self.search_type.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }


@dataclass
class WebResult:
    """
    Result of a web operation.

    Attributes:
        success: Whether operation succeeded
        data: Result data
        error: Error message if failed
        cached: Whether result was from cache
        execution_time: Time taken
        metadata: Additional metadata
    """

    success: bool
    data: Any = None
    error: str | None = None
    cached: bool = False
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SEARCH PROVIDER BASE
# =============================================================================


class BaseSearchProvider(ABC):
    """
    Abstract base class for search providers.

    All providers must implement:
    - search: Perform web search
    - health_check: Verify provider availability
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name."""
        ...

    @property
    @abstractmethod
    def provider_type(self) -> SearchProvider:
        """Get provider type."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform web search."""
        ...

    async def fetch(self, url: str, **kwargs: Any) -> FetchedContent:
        """Fetch content from URL (optional implementation)."""
        raise NotImplementedError("Fetch not supported by this provider")

    async def health_check(self) -> bool:
        """Check if provider is available."""
        return True

    @property
    def supports_fetch(self) -> bool:
        """Whether provider supports URL fetching."""
        return False

    @property
    def rate_limit(self) -> int:
        """Requests per minute limit."""
        return 60


# =============================================================================
# MOCK SEARCH PROVIDER
# =============================================================================


class MockSearchProvider(BaseSearchProvider):
    """
    Mock search provider for development and testing.

    Provides realistic mock search results without external API calls.
    """

    # Mock search database
    MOCK_DATABASE: dict[str, list[dict[str, Any]]] = {
        "python": [
            {
                "title": "Welcome to Python.org",
                "url": "https://www.python.org/",
                "snippet": "The official home of the Python Programming Language. Python is a programming language that lets you work quickly and integrate systems more effectively.",
            },
            {
                "title": "Python Tutorial - W3Schools",
                "url": "https://www.w3schools.com/python/",
                "snippet": "Well organized and easy to understand Web building tutorials with lots of examples of how to use HTML, CSS, JavaScript, SQL, Python, PHP, Bootstrap, Java, XML and more.",
            },
            {
                "title": "Python (programming language) - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "snippet": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            },
        ],
        "machine learning": [
            {
                "title": "Machine Learning | Coursera",
                "url": "https://www.coursera.org/learn/machine-learning",
                "snippet": "Machine learning is the science of getting computers to act without being explicitly programmed. Learn from Stanford University.",
            },
            {
                "title": "Machine Learning - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "snippet": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
            },
        ],
        "artificial intelligence": [
            {
                "title": "What is Artificial Intelligence (AI)? | IBM",
                "url": "https://www.ibm.com/topics/artificial-intelligence",
                "snippet": "Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.",
            },
            {
                "title": "Artificial Intelligence - MIT Technology Review",
                "url": "https://www.technologyreview.com/topic/artificial-intelligence/",
                "snippet": "The latest AI news, research, and breakthroughs from MIT Technology Review.",
            },
        ],
        "weather": [
            {
                "title": "Weather.com - Local Weather Forecasts",
                "url": "https://weather.com/",
                "snippet": "Get the latest weather news and forecasts from The Weather Channel. Find local weather forecasts for cities around the world.",
            },
            {
                "title": "National Weather Service",
                "url": "https://www.weather.gov/",
                "snippet": "NOAA National Weather Service. Official weather forecasts, warnings, and observations for the United States.",
            },
        ],
    }

    # Mock page content
    MOCK_PAGES: dict[str, dict[str, Any]] = {
        "https://www.python.org/": {
            "title": "Welcome to Python.org",
            "content": """
            <html>
            <head><title>Welcome to Python.org</title></head>
            <body>
            <h1>Python</h1>
            <p>Python is a programming language that lets you work quickly and integrate systems more effectively.</p>
            <h2>Get Started</h2>
            <p>Whether you're new to programming or an experienced developer, it's easy to learn and use Python.</p>
            <ul>
                <li><a href="/downloads/">Download Python</a></li>
                <li><a href="/doc/">Documentation</a></li>
                <li><a href="/community/">Community</a></li>
            </ul>
            <h2>Python is powerful</h2>
            <p>Python is a versatile language used in web development, data science, artificial intelligence, and more.</p>
            </body>
            </html>
            """,
        },
        "https://en.wikipedia.org/wiki/Python_(programming_language)": {
            "title": "Python (programming language) - Wikipedia",
            "content": """
            <html>
            <head><title>Python (programming language) - Wikipedia</title></head>
            <body>
            <h1>Python (programming language)</h1>
            <p>Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.</p>
            <h2>History</h2>
            <p>Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI).</p>
            <h2>Features</h2>
            <p>Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms.</p>
            </body>
            </html>
            """,
        },
    }

    def __init__(self, latency: float = 0.1) -> None:
        """
        Initialize mock provider.

        Args:
            latency: Simulated network latency in seconds
        """
        self._latency = latency

    @property
    def name(self) -> str:
        return "Mock Search"

    @property
    def provider_type(self) -> SearchProvider:
        return SearchProvider.MOCK

    @property
    def supports_fetch(self) -> bool:
        return True

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform mock search."""
        start_time = time.time()

        # Simulate network latency
        await asyncio.sleep(self._latency)

        query_lower = query.lower()
        results: list[SearchResult] = []

        # Find matching results
        for keyword, mock_results in self.MOCK_DATABASE.items():
            if keyword in query_lower:
                for i, result in enumerate(mock_results):
                    results.append(SearchResult(
                        title=result["title"],
                        url=result["url"],
                        snippet=result["snippet"],
                        score=1.0 - (i * 0.1),
                        position=len(results) + 1,
                    ))

        # Add generic result if no matches
        if not results:
            results.append(SearchResult(
                title=f"Search results for: {query}",
                url=f"https://search.example.com/q={urllib.parse.quote(query)}",
                snippet=f"Information about {query} from various sources on the web.",
                score=0.5,
                position=1,
            ))

        # Limit results
        results = results[:num_results]

        return SearchResults(
            query=query,
            results=results,
            total_results=len(results) * 10,  # Simulate more results available
            search_time=time.time() - start_time,
            provider=self.name,
            search_type=search_type,
        )

    async def fetch(self, url: str, **kwargs: Any) -> FetchedContent:
        """Fetch mock page content."""
        start_time = time.time()

        # Simulate network latency
        await asyncio.sleep(self._latency)

        # Check mock pages
        if url in self.MOCK_PAGES:
            page = self.MOCK_PAGES[url]
            content = page["content"]
            title = page["title"]
        else:
            # Generate mock content
            title = f"Page: {url}"
            content = f"""
            <html>
            <head><title>{title}</title></head>
            <body>
            <h1>{title}</h1>
            <p>This is mock content for the URL: {url}</p>
            <p>In production, this would contain the actual page content.</p>
            </body>
            </html>
            """

        return FetchedContent(
            url=url,
            content=content,
            content_type=ContentType.HTML,
            title=title,
            fetch_time=time.time() - start_time,
            status_code=200,
        )


# =============================================================================
# TAVILY PROVIDER (Future Implementation)
# =============================================================================


class TavilyProvider(BaseSearchProvider):
    """
    Tavily search provider for AI-optimized search.

    Tavily provides search results optimized for LLM consumption
    with built-in content extraction and summarization.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.tavily.com",
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize Tavily provider.

        Args:
            api_key: Tavily API key
            base_url: API base URL
            timeout: HTTP request timeout in seconds
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "Tavily"

    @property
    def provider_type(self) -> SearchProvider:
        return SearchProvider.TAVILY

    @property
    def supports_fetch(self) -> bool:
        return True

    @property
    def rate_limit(self) -> int:
        return 100  # Tavily rate limit

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform Tavily search."""
        import time
        import httpx
        from datetime import datetime

        start_time = time.monotonic()
        url = f"{self._base_url}/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": False,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            except httpx.TimeoutException:
                raise ValueError("Tavily search timed out")
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    raise ValueError(f"Tavily authentication failed: {exc}")
                if status == 429:
                    raise ValueError(f"Tavily rate limit exceeded: {exc}")
                raise ValueError(f"Tavily request failed with status {status}: {exc}")
            except httpx.HTTPError as exc:
                raise ValueError(f"Tavily request failed: {exc}")

        data = response.json()
        results: list[SearchResult] = []
        for idx, item in enumerate(data.get("results", [])[:num_results]):
            pub_date = None
            if item.get("published_date"):
                try:
                    pub_date = datetime.fromisoformat(item["published_date"])
                except (ValueError, TypeError):
                    pass
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                    position=idx + 1,
                    source="tavily",
                    published_date=pub_date,
                    metadata={"score": item.get("score")},
                )
            )

        search_time = time.monotonic() - start_time
        return SearchResults(
            query=query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            provider="tavily",
            search_type=search_type,
        )

    async def fetch(self, url: str, **kwargs: Any) -> FetchedContent:
        """Fetch and extract content via Tavily."""
        import time
        import httpx

        start_time = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ValueError(f"Tavily fetch failed for {url}: {exc}")

        content = response.text
        fetch_time = time.monotonic() - start_time
        return FetchedContent(
            url=url,
            content=content,
            content_type=ContentType.HTML,
            fetch_time=fetch_time,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def health_check(self) -> bool:
        """Check Tavily API availability."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self._base_url}/search",
                    json={"api_key": self._api_key, "query": "health", "max_results": 1},
                )
                return resp.status_code == 200
        except Exception:
            return False


# =============================================================================
# SERPAPI PROVIDER (Future Implementation)
# =============================================================================


class SerpAPIProvider(BaseSearchProvider):
    """
    SerpAPI provider for Google search results.

    SerpAPI provides structured Google search results
    with support for various search types.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://serpapi.com",
    ) -> None:
        """
        Initialize SerpAPI provider.

        Args:
            api_key: SerpAPI key
            base_url: API base URL
        """
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "SerpAPI"

    @property
    def provider_type(self) -> SearchProvider:
        return SearchProvider.SERPAPI

    @property
    def rate_limit(self) -> int:
        return 100

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform SerpAPI search."""
        raise NotImplementedError("SerpAPI provider not yet implemented")

    async def health_check(self) -> bool:
        """Check SerpAPI availability."""
        return False


# =============================================================================
# BRAVE SEARCH PROVIDER (Future Implementation)
# =============================================================================


class BraveSearchProvider(BaseSearchProvider):
    """
    Brave Search provider for privacy-focused search.

    Brave Search provides independent search results
    without tracking or profiling.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.search.brave.com",
    ) -> None:
        """
        Initialize Brave Search provider.

        Args:
            api_key: Brave Search API key
            base_url: API base URL
        """
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "Brave Search"

    @property
    def provider_type(self) -> SearchProvider:
        return SearchProvider.BRAVE

    @property
    def rate_limit(self) -> int:
        return 60

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform Brave search."""
        raise NotImplementedError("Brave Search provider not yet implemented")

    async def health_check(self) -> bool:
        """Check Brave Search availability."""
        return False


# =============================================================================
# CONTENT EXTRACTOR
# =============================================================================


class ContentExtractor:
    """
    Extracts clean content from HTML pages.

    Features:
    - HTML tag removal
    - Script/style removal
    - Link extraction
    - Image extraction
    - Text normalization
    """

    # Tags to completely remove (including content)
    REMOVE_TAGS: set[str] = {
        "script", "style", "noscript", "iframe", "svg",
        "canvas", "video", "audio", "map", "object", "embed",
    }

    # Tags that typically contain navigation/boilerplate
    BOILERPLATE_TAGS: set[str] = {
        "nav", "header", "footer", "aside", "menu",
        "sidebar", "advertisement", "ad", "banner",
    }

    def __init__(
        self,
        remove_boilerplate: bool = True,
        extract_links: bool = True,
        extract_images: bool = True,
        max_content_length: int = 50000,
    ) -> None:
        """
        Initialize content extractor.

        Args:
            remove_boilerplate: Remove navigation/header/footer
            extract_links: Extract links from content
            extract_images: Extract image URLs
            max_content_length: Maximum content length to process
        """
        self._remove_boilerplate = remove_boilerplate
        self._extract_links = extract_links
        self._extract_images = extract_images
        self._max_content_length = max_content_length

    def extract(self, html_content: str, base_url: str = "") -> dict[str, Any]:
        """
        Extract clean content from HTML.

        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Dictionary with extracted content
        """
        # Truncate if too long
        if len(html_content) > self._max_content_length:
            html_content = html_content[:self._max_content_length]

        # Extract title
        title = self._extract_title(html_content)

        # Extract links
        links = self._extract_links_from_html(html_content, base_url) if self._extract_links else []

        # Extract images
        images = self._extract_images_from_html(html_content, base_url) if self._extract_images else []

        # Remove unwanted tags
        clean_html = self._remove_tags(html_content)

        # Extract text
        text = self._html_to_text(clean_html)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        return {
            "title": title,
            "text": text,
            "links": links,
            "images": images,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    def _extract_title(self, html_content: str) -> str:
        """Extract page title."""
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())

        # Try <h1> tag
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())

        return ""

    def _extract_links_from_html(self, html_content: str, base_url: str) -> list[str]:
        """Extract links from HTML."""
        links: list[str] = []
        pattern = r'<a[^>]+href=["\']([^"\']+)["\']'

        for match in re.finditer(pattern, html_content, re.IGNORECASE):
            href = match.group(1)
            # Resolve relative URLs
            if base_url and not href.startswith(("http://", "https://", "//")):
                href = urllib.parse.urljoin(base_url, href)
            if href.startswith(("http://", "https://")):
                links.append(href)

        return list(set(links))[:50]  # Limit to 50 unique links

    def _extract_images_from_html(self, html_content: str, base_url: str) -> list[str]:
        """Extract image URLs from HTML."""
        images: list[str] = []
        pattern = r'<img[^>]+src=["\']([^"\']+)["\']'

        for match in re.finditer(pattern, html_content, re.IGNORECASE):
            src = match.group(1)
            # Resolve relative URLs
            if base_url and not src.startswith(("http://", "https://", "//", "data:")):
                src = urllib.parse.urljoin(base_url, src)
            if src.startswith(("http://", "https://")):
                images.append(src)

        return list(set(images))[:20]  # Limit to 20 unique images

    def _remove_tags(self, html_content: str) -> str:
        """Remove unwanted HTML tags."""
        content = html_content

        # Remove tags that should be completely removed (with content)
        for tag in self.REMOVE_TAGS:
            pattern = rf"<{tag}[^>]*>.*?</{tag}>"
            content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL)

        # Remove boilerplate sections
        if self._remove_boilerplate:
            for tag in self.BOILERPLATE_TAGS:
                pattern = rf"<{tag}[^>]*>.*?</{tag}>"
                content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL)

        # Remove HTML comments
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        return content

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text."""
        # Replace block elements with newlines
        block_tags = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]
        for tag in block_tags:
            html_content = re.sub(rf"</?{tag}[^>]*>", "\n", html_content, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", html_content)

        # Decode HTML entities
        text = html.unescape(text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return text.strip()


# =============================================================================
# CONTENT SUMMARIZER
# =============================================================================


class ContentSummarizer:
    """
    Summarizes web content.

    Provides extractive summarization by selecting
    the most important sentences from content.
    """

    def __init__(
        self,
        max_sentences: int = 5,
        max_length: int = 1000,
    ) -> None:
        """
        Initialize summarizer.

        Args:
            max_sentences: Maximum sentences in summary
            max_length: Maximum character length
        """
        self._max_sentences = max_sentences
        self._max_length = max_length

    def summarize(self, text: str, query: str | None = None) -> str:
        """
        Summarize text content.

        Args:
            text: Text to summarize
            query: Optional query for relevance scoring

        Returns:
            Summarized text
        """
        if not text:
            return ""

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return text[:self._max_length]

        # Score sentences
        scored = self._score_sentences(sentences, query)

        # Select top sentences
        selected = sorted(scored, key=lambda x: x[1], reverse=True)[:self._max_sentences]

        # Restore original order
        selected = sorted(selected, key=lambda x: sentences.index(x[0]))

        # Build summary
        summary = " ".join(s[0] for s in selected)

        # Truncate if needed
        if len(summary) > self._max_length:
            summary = summary[:self._max_length - 3] + "..."

        return summary

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Filter short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def _score_sentences(
        self,
        sentences: list[str],
        query: str | None = None,
    ) -> list[tuple[str, float]]:
        """Score sentences by importance."""
        scored: list[tuple[str, float]] = []

        # Build word frequency
        all_words = " ".join(sentences).lower().split()
        word_freq: dict[str, int] = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        query_words = set(query.lower().split()) if query else set()

        for i, sentence in enumerate(sentences):
            score = 0.0
            words = sentence.lower().split()

            # Position score (earlier sentences often more important)
            position_score = 1.0 / (i + 1)
            score += position_score * 0.3

            # Length score (prefer medium-length sentences)
            length_score = min(len(words) / 20, 1.0)
            score += length_score * 0.2

            # Word frequency score
            freq_score = sum(word_freq.get(w, 0) for w in words) / (len(words) + 1)
            score += min(freq_score / 10, 1.0) * 0.3

            # Query relevance score
            if query_words:
                overlap = len(set(words) & query_words)
                query_score = overlap / len(query_words)
                score += query_score * 0.5

            scored.append((sentence, score))

        return scored


# =============================================================================
# QUERY OPTIMIZER
# =============================================================================


class QueryOptimizer:
    """
    Optimizes search queries for better results.

    Features:
    - Query expansion
    - Stop word removal
    - Query reformulation
    - Synonym expansion
    """

    # Common stop words
    STOP_WORDS: set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "if", "or", "because",
        "until", "while", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "i", "me", "my", "myself",
        "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    }

    # Query type patterns
    QUERY_PATTERNS: dict[str, list[str]] = {
        "definition": ["what is", "define", "meaning of", "definition of"],
        "how_to": ["how to", "how do", "how can", "steps to", "guide to"],
        "comparison": ["vs", "versus", "compared to", "difference between"],
        "list": ["list of", "top", "best", "examples of"],
        "location": ["where is", "location of", "find"],
        "time": ["when", "date of", "year of"],
        "reason": ["why", "reason for", "cause of"],
    }

    def __init__(
        self,
        remove_stop_words: bool = True,
        expand_synonyms: bool = False,
    ) -> None:
        """
        Initialize query optimizer.

        Args:
            remove_stop_words: Remove common stop words
            expand_synonyms: Expand query with synonyms
        """
        self._remove_stop_words = remove_stop_words
        self._expand_synonyms = expand_synonyms

    def optimize(self, query: str) -> dict[str, Any]:
        """
        Optimize a search query.

        Args:
            query: Original query

        Returns:
            Dictionary with optimized query and metadata
        """
        original = query.strip()

        # Detect query type
        query_type = self._detect_query_type(original)

        # Clean query
        cleaned = self._clean_query(original)

        # Remove stop words (optional)
        if self._remove_stop_words:
            keywords = self._extract_keywords(cleaned)
        else:
            keywords = cleaned.split()

        # Build optimized query
        optimized = " ".join(keywords)

        # Generate alternative queries
        alternatives = self._generate_alternatives(original, query_type)

        return {
            "original": original,
            "optimized": optimized,
            "keywords": keywords,
            "query_type": query_type,
            "alternatives": alternatives,
        }

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query."""
        query_lower = query.lower()

        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return query_type

        return "general"

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        query = " ".join(query.split())

        # Remove special characters (keep alphanumeric and spaces)
        query = re.sub(r"[^\w\s]", " ", query)

        return query.strip()

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        words = query.lower().split()
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 1]
        return keywords if keywords else words

    def _generate_alternatives(self, query: str, query_type: str) -> list[str]:
        """Generate alternative query formulations."""
        alternatives: list[str] = []
        query_lower = query.lower()

        # Add quoted exact match
        if " " in query and not query.startswith('"'):
            alternatives.append(f'"{query}"')

        # Type-specific alternatives
        if query_type == "definition":
            base = re.sub(r"^(what is|define|meaning of|definition of)\s*", "", query_lower)
            alternatives.append(f"{base} definition")
            alternatives.append(f"{base} explained")

        elif query_type == "how_to":
            base = re.sub(r"^(how to|how do|how can|steps to|guide to)\s*", "", query_lower)
            alternatives.append(f"{base} tutorial")
            alternatives.append(f"{base} guide")

        elif query_type == "comparison":
            alternatives.append(query_lower.replace(" vs ", " versus "))
            alternatives.append(query_lower.replace(" versus ", " compared to "))

        return alternatives[:3]


# =============================================================================
# SEARCH CACHE
# =============================================================================


class SearchCache:
    """
    Cache for search results and fetched content.

    Features:
    - TTL-based expiration
    - LRU eviction
    - Separate caches for search and fetch
    """

    def __init__(
        self,
        max_search_entries: int = 1000,
        max_fetch_entries: int = 500,
        search_ttl: int = 3600,  # 1 hour
        fetch_ttl: int = 86400,  # 24 hours
    ) -> None:
        """
        Initialize cache.

        Args:
            max_search_entries: Max cached search results
            max_fetch_entries: Max cached fetched pages
            search_ttl: Search cache TTL in seconds
            fetch_ttl: Fetch cache TTL in seconds
        """
        self._search_cache: dict[str, tuple[SearchResults, float]] = {}
        self._fetch_cache: dict[str, tuple[FetchedContent, float]] = {}
        self._max_search = max_search_entries
        self._max_fetch = max_fetch_entries
        self._search_ttl = search_ttl
        self._fetch_ttl = fetch_ttl

    def get_search(self, cache_key: str) -> SearchResults | None:
        """Get cached search results."""
        if cache_key not in self._search_cache:
            return None

        results, cached_at = self._search_cache[cache_key]
        if time.time() - cached_at > self._search_ttl:
            del self._search_cache[cache_key]
            return None

        return results

    def set_search(self, cache_key: str, results: SearchResults) -> None:
        """Cache search results."""
        # Evict if at capacity
        if len(self._search_cache) >= self._max_search:
            self._evict_oldest(self._search_cache)

        self._search_cache[cache_key] = (results, time.time())

    def get_fetch(self, url: str) -> FetchedContent | None:
        """Get cached fetched content."""
        if url not in self._fetch_cache:
            return None

        content, cached_at = self._fetch_cache[url]
        if time.time() - cached_at > self._fetch_ttl:
            del self._fetch_cache[url]
            return None

        return content

    def set_fetch(self, url: str, content: FetchedContent) -> None:
        """Cache fetched content."""
        if len(self._fetch_cache) >= self._max_fetch:
            self._evict_oldest(self._fetch_cache)

        self._fetch_cache[url] = (content, time.time())

    def _evict_oldest(self, cache: dict[str, tuple[Any, float]]) -> None:
        """Evict oldest entry from cache."""
        if not cache:
            return

        oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
        del cache[oldest_key]

    def clear(self) -> None:
        """Clear all caches."""
        self._search_cache.clear()
        self._fetch_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "search_entries": len(self._search_cache),
            "fetch_entries": len(self._fetch_cache),
            "search_max": self._max_search,
            "fetch_max": self._max_fetch,
        }


# =============================================================================
# SEARCH HISTORY
# =============================================================================


class SearchHistory:
    """
    Tracks search history for analytics and optimization.

    Features:
    - Query tracking
    - Agent/session filtering
    - Analytics
    """

    def __init__(
        self,
        max_entries: int = 10000,
        retention_days: int = 30,
    ) -> None:
        """
        Initialize search history.

        Args:
            max_entries: Maximum history entries
            retention_days: Days to retain history
        """
        self._history: list[SearchHistoryEntry] = []
        self._max_entries = max_entries
        self._retention_days = retention_days

    def record(
        self,
        query: str,
        provider: str,
        result_count: int,
        search_type: SearchType = SearchType.GENERAL,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record a search."""
        entry = SearchHistoryEntry(
            query=query,
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            result_count=result_count,
            search_type=search_type,
            agent_id=agent_id,
            session_id=session_id,
        )

        self._history.append(entry)

        # Trim if needed
        if len(self._history) > self._max_entries:
            self._history = self._history[-self._max_entries:]

    def get_history(
        self,
        limit: int = 100,
        agent_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> list[SearchHistoryEntry]:
        """Get search history."""
        results = self._history

        # Filter by agent
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]

        # Filter by session
        if session_id:
            results = [e for e in results if e.session_id == session_id]

        # Filter by time
        if since:
            results = [e for e in results if e.timestamp >= since]

        # Return most recent
        return list(reversed(results[-limit:]))

    def get_popular_queries(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most popular queries."""
        query_counts: dict[str, int] = {}
        for entry in self._history:
            query_lower = entry.query.lower()
            query_counts[query_lower] = query_counts.get(query_lower, 0) + 1

        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_queries[:limit]

    def cleanup_old(self) -> int:
        """Remove entries older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        original_count = len(self._history)
        self._history = [e for e in self._history if e.timestamp >= cutoff]
        return original_count - len(self._history)

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get history statistics."""
        if not self._history:
            return {
                "total_searches": 0,
                "unique_queries": 0,
                "providers_used": {},
                "search_types": {},
            }

        providers: dict[str, int] = {}
        search_types: dict[str, int] = {}
        unique_queries: set[str] = set()

        for entry in self._history:
            providers[entry.provider] = providers.get(entry.provider, 0) + 1
            search_types[entry.search_type.value] = search_types.get(entry.search_type.value, 0) + 1
            unique_queries.add(entry.query.lower())

        return {
            "total_searches": len(self._history),
            "unique_queries": len(unique_queries),
            "providers_used": providers,
            "search_types": search_types,
            "oldest_entry": self._history[0].timestamp.isoformat() if self._history else None,
            "newest_entry": self._history[-1].timestamp.isoformat() if self._history else None,
        }


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Rate limiter for API calls.

    Uses sliding window algorithm.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self._limit = requests_per_minute
        self._requests: list[float] = []

    def check(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60

        # Remove old requests
        self._requests = [t for t in self._requests if t > minute_ago]

        return len(self._requests) < self._limit

    def record(self) -> None:
        """Record a request."""
        self._requests.append(time.time())

    def wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.check():
            return 0.0

        now = time.time()
        minute_ago = now - 60

        # Find oldest request in window
        oldest = min(t for t in self._requests if t > minute_ago)
        return oldest + 60 - now


# =============================================================================
# WEB CAPABILITY
# =============================================================================


class WebCapability(Capability):
    """
    Web access capability for the AI Operating System.

    Provides unified interface for:
    - Web search across multiple providers
    - URL fetching and content extraction
    - Page summarization
    - Query optimization
    - Search history tracking

    Compatible with:
    - CapabilityRegistry for registration
    - Executor for action execution
    - ContextEngine for context preparation
    """

    def __init__(
        self,
        provider: BaseSearchProvider | None = None,
        extractor: ContentExtractor | None = None,
        summarizer: ContentSummarizer | None = None,
        optimizer: QueryOptimizer | None = None,
        cache: SearchCache | None = None,
        history: SearchHistory | None = None,
        default_agent_id: str | None = None,
        default_session_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize web capability.

        Args:
            provider: Search provider (default: MockSearchProvider)
            extractor: Content extractor
            summarizer: Content summarizer
            optimizer: Query optimizer
            cache: Search cache
            history: Search history tracker
            default_agent_id: Default agent ID
            default_session_id: Default session ID
            logger: Optional logger
        """
        self._provider = provider or MockSearchProvider()
        self._extractor = extractor or ContentExtractor()
        self._summarizer = summarizer or ContentSummarizer()
        self._optimizer = optimizer or QueryOptimizer()
        self._cache = cache or SearchCache()
        self._history = history or SearchHistory()
        self._default_agent_id = default_agent_id
        self._default_session_id = default_session_id
        self._logger = logger or logging.getLogger(__name__)
        self._rate_limiter = RateLimiter(self._provider.rate_limit)
        self._initialized = False

        # Additional providers (for fallback/comparison)
        self._providers: dict[str, BaseSearchProvider] = {
            self._provider.provider_type.value: self._provider,
        }

        # Action handlers
        self._action_handlers: dict[WebAction, Callable[..., Any]] = {
            WebAction.SEARCH: self._handle_search,
            WebAction.FETCH: self._handle_fetch,
            WebAction.EXTRACT: self._handle_extract,
            WebAction.SUMMARIZE: self._handle_summarize,
            WebAction.OPTIMIZE_QUERY: self._handle_optimize_query,
            WebAction.HISTORY: self._handle_history,
            WebAction.CLEAR_CACHE: self._handle_clear_cache,
            WebAction.PROVIDERS: self._handle_providers,
        }

    # -------------------------------------------------------------------------
    # Capability Interface
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return CapabilityMetadata(
            name="web",
            version="1.0.0",
            capability_type=CapabilityType.WEB,  # type: ignore[attr-defined]
            description="Web access capability for search, fetch, and content extraction",
            actions=tuple(action.value for action in WebAction),
            required_permissions=("web:search", "web:fetch"),
            config_schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["mock", "tavily", "serpapi", "brave"]},
                    "cache_enabled": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            tags=("web", "search", "fetch", "internet", "content"),
        )

    async def _do_initialize(self) -> None:
        self._logger.debug("WebCapability._do_initialize")

    async def _do_shutdown(self) -> None:
        self._logger.debug("WebCapability._do_shutdown")

    async def _do_execute(self, context) -> CapabilityResult:
        return CapabilityResult(success=True, status=CapabilityStatus.SUCCESS)  # type: ignore[attr-defined]

    async def initialize(self) -> None:
        """Initialize the capability."""
        if self._initialized:
            return

        self._initialized = True
        self._logger.info(f"WebCapability initialized with provider: {self._provider.name}")

    async def shutdown(self) -> None:
        """Shutdown the capability."""
        if not self._initialized:
            return

        self._cache.clear()
        self._initialized = False
        self._logger.info("WebCapability shutdown")

    async def health_check(self) -> bool:
        """Check capability health."""
        try:
            return await self._provider.health_check()
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    async def execute(  # type: ignore[override]
        self,
        action: str,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> CapabilityResult:
        """
        Execute a web action.

        Args:
            action: Action to perform
            parameters: Action parameters
            context: Execution context

        Returns:
            CapabilityResult with operation outcome
        """
        start_time = time.time()
        context = context or {}

        try:
            # Validate action
            try:
                web_action = WebAction(action)
            except ValueError:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Get handler
            handler = self._action_handlers.get(web_action)
            if handler is None:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"No handler for action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Execute handler
            result = await handler(parameters, context)

            return CapabilityResult(
                success=result.success,
                status=CapabilityStatus.COMPLETED if result.success else CapabilityStatus.FAILED,
                data=result.data,
                error=result.error,
                execution_time=time.time() - start_time,
                metadata={
                    "cached": result.cached,
                    **result.metadata,
                },
            )

        except ValidationError as e:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=f"Validation error: {e}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            self._logger.exception(f"Web action failed: {e}")
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    async def _handle_search(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle search action."""
        query = parameters.get("query")
        if not query:
            raise ValidationError("query is required")

        num_results = parameters.get("num_results", 10)
        search_type_str = parameters.get("search_type", "general")
        use_cache = parameters.get("use_cache", True)
        optimize = parameters.get("optimize_query", False)

        try:
            search_type = SearchType(search_type_str)
        except ValueError:
            search_type = SearchType.GENERAL

        # Optimize query if requested
        if optimize:
            optimized = self._optimizer.optimize(query)
            query = optimized["optimized"]

        # Check cache
        cache_key = self._get_cache_key(query, num_results, search_type)
        if use_cache:
            cached = self._cache.get_search(cache_key)
            if cached:
                return WebResult(
                    success=True,
                    data=cached.to_dict(),
                    cached=True,
                    metadata={"provider": cached.provider},
                )

        # Check rate limit
        if not self._rate_limiter.check():
            wait_time = self._rate_limiter.wait_time()
            return WebResult(
                success=False,
                error=f"Rate limited. Try again in {wait_time:.1f} seconds",
                metadata={"wait_time": wait_time},
            )

        # Perform search
        self._rate_limiter.record()
        results = await self._provider.search(
            query=query,
            num_results=num_results,
            search_type=search_type,
        )

        # Cache results
        if use_cache:
            self._cache.set_search(cache_key, results)

        # Record history
        self._history.record(
            query=query,
            provider=self._provider.name,
            result_count=len(results.results),
            search_type=search_type,
            agent_id=context.get("agent_id") or self._default_agent_id,
            session_id=context.get("session_id") or self._default_session_id,
        )

        return WebResult(
            success=True,
            data=results.to_dict(),
            metadata={"provider": self._provider.name},
        )

    async def _handle_fetch(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle fetch action."""
        url = parameters.get("url")
        if not url:
            raise ValidationError("url is required")

        use_cache = parameters.get("use_cache", True)
        extract_content = parameters.get("extract", True)

        # Validate URL
        if not url.startswith(("http://", "https://")):
            raise ValidationError("Invalid URL: must start with http:// or https://")

        # Check cache
        if use_cache:
            cached = self._cache.get_fetch(url)
            if cached:
                return WebResult(
                    success=True,
                    data=cached.to_dict(),
                    cached=True,
                )

        # Check if provider supports fetch
        if not self._provider.supports_fetch:
            return WebResult(
                success=False,
                error=f"Provider {self._provider.name} does not support URL fetching",
            )

        # Check rate limit
        if not self._rate_limiter.check():
            wait_time = self._rate_limiter.wait_time()
            return WebResult(
                success=False,
                error=f"Rate limited. Try again in {wait_time:.1f} seconds",
            )

        # Fetch content
        self._rate_limiter.record()
        content = await self._provider.fetch(url)

        # Extract content if requested
        if extract_content and content.content_type == ContentType.HTML:
            extracted = self._extractor.extract(content.content, url)
            content.title = extracted["title"] or content.title
            content.extracted_text = extracted["text"]
            content.links = extracted["links"]
            content.images = extracted["images"]

        # Cache content
        if use_cache:
            self._cache.set_fetch(url, content)

        return WebResult(
            success=True,
            data=content.to_dict(),
        )

    async def _handle_extract(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle extract action."""
        html_content = parameters.get("html")
        url = parameters.get("url", "")

        if not html_content:
            raise ValidationError("html content is required")

        extracted = self._extractor.extract(html_content, url)

        return WebResult(
            success=True,
            data=extracted,
        )

    async def _handle_summarize(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle summarize action."""
        text = parameters.get("text")
        url = parameters.get("url")
        query = parameters.get("query")

        # If URL provided, fetch and extract first
        if url and not text:
            fetch_result = await self._handle_fetch(
                {"url": url, "extract": True},
                context,
            )
            if not fetch_result.success:
                return fetch_result
            text = fetch_result.data.get("extracted_text", "")

        if not text:
            raise ValidationError("text or url is required")

        summary = self._summarizer.summarize(text, query)

        return WebResult(
            success=True,
            data={
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
            },
        )

    async def _handle_optimize_query(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle query optimization."""
        query = parameters.get("query")
        if not query:
            raise ValidationError("query is required")

        optimized = self._optimizer.optimize(query)

        return WebResult(
            success=True,
            data=optimized,
        )

    async def _handle_history(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle history retrieval."""
        limit = parameters.get("limit", 100)
        agent_id = parameters.get("agent_id") or context.get("agent_id")
        session_id = parameters.get("session_id") or context.get("session_id")
        include_stats = parameters.get("include_stats", False)

        history = self._history.get_history(
            limit=limit,
            agent_id=agent_id,
            session_id=session_id,
        )

        data: dict[str, Any] = {
            "entries": [e.to_dict() for e in history],
            "count": len(history),
        }

        if include_stats:
            data["stats"] = self._history.get_stats()
            data["popular_queries"] = self._history.get_popular_queries()

        return WebResult(
            success=True,
            data=data,
        )

    async def _handle_clear_cache(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle cache clearing."""
        stats_before = self._cache.get_stats()
        self._cache.clear()

        return WebResult(
            success=True,
            data={
                "cleared": True,
                "entries_cleared": stats_before["search_entries"] + stats_before["fetch_entries"],
            },
        )

    async def _handle_providers(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> WebResult:
        """Handle provider listing."""
        providers = []
        for name, provider in self._providers.items():
            providers.append({
                "name": provider.name,
                "type": provider.provider_type.value,
                "supports_fetch": provider.supports_fetch,
                "rate_limit": provider.rate_limit,
                "is_active": provider == self._provider,
            })

        return WebResult(
            success=True,
            data={
                "providers": providers,
                "active_provider": self._provider.name,
            },
        )

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: SearchType = SearchType.GENERAL,
        use_cache: bool = True,
    ) -> SearchResults:
        """
        Perform web search.

        Args:
            query: Search query
            num_results: Number of results
            search_type: Type of search
            use_cache: Whether to use cache

        Returns:
            SearchResults
        """
        result = await self._handle_search(
            {
                "query": query,
                "num_results": num_results,
                "search_type": search_type.value,
                "use_cache": use_cache,
            },
            {},
        )

        if not result.success:
            raise CapabilityError(result.error or "Search failed")

        return SearchResults(
            query=result.data["query"],
            results=[SearchResult.from_dict(r) for r in result.data["results"]],
            total_results=result.data["total_results"],
            search_time=result.data["search_time"],
            provider=result.data["provider"],
            search_type=SearchType(result.data["search_type"]),
        )

    async def fetch(
        self,
        url: str,
        extract: bool = True,
        use_cache: bool = True,
    ) -> FetchedContent:
        """
        Fetch content from URL.

        Args:
            url: URL to fetch
            extract: Whether to extract content
            use_cache: Whether to use cache

        Returns:
            FetchedContent
        """
        result = await self._handle_fetch(
            {"url": url, "extract": extract, "use_cache": use_cache},
            {},
        )

        if not result.success:
            raise CapabilityError(result.error or "Fetch failed")

        data = result.data
        return FetchedContent(
            url=data["url"],
            content="",  # Raw content not returned in dict
            content_type=ContentType(data["content_type"]),
            title=data["title"],
            extracted_text=data["extracted_text"],
            links=data["links"],
            images=data["images"],
            fetch_time=data["fetch_time"],
            status_code=data["status_code"],
        )

    async def search_and_fetch(
        self,
        query: str,
        num_results: int = 3,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """
        Search and fetch top results.

        Args:
            query: Search query
            num_results: Number of results to fetch
            summarize: Whether to summarize content

        Returns:
            Dictionary with search results and fetched content
        """
        # Search
        search_results = await self.search(query, num_results=num_results)

        # Fetch each result
        fetched: list[dict[str, Any]] = []
        for result in search_results.results[:num_results]:
            try:
                content = await self.fetch(result.url)
                entry: dict[str, Any] = {
                    "title": result.title,
                    "url": result.url,
                    "content": content.extracted_text,
                }

                if summarize:
                    summary = self._summarizer.summarize(content.extracted_text, query)
                    entry["summary"] = summary

                fetched.append(entry)
            except Exception as e:
                self._logger.warning(f"Failed to fetch {result.url}: {e}")

        return {
            "query": query,
            "results": fetched,
            "total_fetched": len(fetched),
        }

    def add_provider(self, provider: BaseSearchProvider) -> None:
        """Add a search provider."""
        self._providers[provider.provider_type.value] = provider
        self._logger.info(f"Added provider: {provider.name}")

    def set_active_provider(self, provider_type: str) -> bool:
        """Set the active search provider."""
        if provider_type not in self._providers:
            return False

        self._provider = self._providers[provider_type]
        self._rate_limiter = RateLimiter(self._provider.rate_limit)
        self._logger.info(f"Active provider set to: {self._provider.name}")
        return True

    def get_context(
        self,
        query: str | None = None,
        recent_searches: int = 5,
    ) -> dict[str, Any]:
        """
        Get web context for context engine integration.

        Args:
            query: Optional query for context
            recent_searches: Number of recent searches to include

        Returns:
            Context dictionary
        """
        history = self._history.get_history(limit=recent_searches)

        return {
            "recent_searches": [e.to_dict() for e in history],
            "active_provider": self._provider.name,
            "cache_stats": self._cache.get_stats(),
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_cache_key(
        self,
        query: str,
        num_results: int,
        search_type: SearchType,
    ) -> str:
        """Generate cache key for search."""
        key_str = f"{query}:{num_results}:{search_type.value}"
        return hashlib.md5(key_str.encode()).hexdigest()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_web_capability(
    provider_type: str = "mock",
    api_key: str | None = None,
    cache_enabled: bool = True,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> WebCapability:
    """
    Factory function to create WebCapability with specified provider.

    Args:
        provider_type: Provider type ("mock", "tavily", "serpapi", "brave")
        api_key: API key for provider
        cache_enabled: Whether to enable caching
        logger: Optional logger
        **kwargs: Additional configuration

    Returns:
        Configured WebCapability
    """
    logger = logger or logging.getLogger(__name__)

    # Create provider
    if provider_type == "mock":
        provider = MockSearchProvider(latency=kwargs.get("latency", 0.1))
    elif provider_type == "tavily":
        if not api_key:
            raise ValueError("API key required for Tavily provider")
        provider = TavilyProvider(api_key=api_key)  # type: ignore[assignment]
    elif provider_type == "serpapi":
        if not api_key:
            raise ValueError("API key required for SerpAPI provider")
        provider = SerpAPIProvider(api_key=api_key)  # type: ignore[assignment]
    elif provider_type == "brave":
        if not api_key:
            raise ValueError("API key required for Brave Search provider")
        provider = BraveSearchProvider(api_key=api_key)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    # Create cache
    cache = SearchCache() if cache_enabled else None

    return WebCapability(
        provider=provider,
        cache=cache,
        default_agent_id=kwargs.get("default_agent_id"),
        default_session_id=kwargs.get("default_session_id"),
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SearchProvider",
    "ContentType",
    "WebAction",
    "SearchType",
    # Data Classes
    "SearchResult",
    "SearchResults",
    "FetchedContent",
    "SearchHistoryEntry",
    "WebResult",
    # Providers
    "BaseSearchProvider",
    "MockSearchProvider",
    "TavilyProvider",
    "SerpAPIProvider",
    "BraveSearchProvider",
    # Components
    "ContentExtractor",
    "ContentSummarizer",
    "QueryOptimizer",
    "SearchCache",
    "SearchHistory",
    "RateLimiter",
    # Capability
    "WebCapability",
    # Factory
    "create_web_capability",
]
