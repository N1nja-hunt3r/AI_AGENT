from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from app.multi_agents.base_agent import (
    Agent,
    AgentMetadata,
    AgentPriority,
    AgentTask,
    HealthCheckResult,
    HealthStatus,
)


class ResearchError(Exception):
    pass


class NoResultsError(ResearchError):
    pass


class SourceType(Enum):
    WEB = "web"
    DOCUMENT = "document"
    API = "api"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    source_type: SourceType = SourceType.WEB
    published_at_epoch: Optional[float] = None
    raw_content: Optional[str] = None


@dataclass(frozen=True)
class RankedSource:
    result: SearchResult
    relevance_score: float
    authority_score: float
    recency_score: float
    combined_score: float


@dataclass(frozen=True)
class ExtractedFact:
    statement: str
    source_url: str
    confidence: float
    extracted_at_epoch: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Citation:
    citation_id: str
    source_url: str
    title: str
    accessed_at_epoch: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ResearchSummary:
    query: str
    summary_text: str
    facts: tuple[ExtractedFact, ...]
    citations: tuple[Citation, ...]
    sources_considered: int
    generated_at_epoch: float = field(default_factory=time.time)


WebSearchFn = Callable[[str, int], Awaitable[list[SearchResult]]]
SummarizerFn = Callable[[str, list[SearchResult]], Awaitable[str]]
FactExtractorFn = Callable[[SearchResult], Awaitable[list[ExtractedFact]]]


class SourceRanker:
    def __init__(
        self,
        trusted_domains: Optional[set[str]] = None,
        recency_half_life_days: float = 180.0,
    ) -> None:
        self._trusted_domains = trusted_domains or set()
        self._recency_half_life_days = recency_half_life_days

    def rank(self, query: str, results: list[SearchResult]) -> list[RankedSource]:
        ranked = [self._score(query, result) for result in results]
        return sorted(ranked, key=lambda r: r.combined_score, reverse=True)

    def _score(self, query: str, result: SearchResult) -> RankedSource:
        relevance = self._relevance_score(query, result)
        authority = self._authority_score(result)
        recency = self._recency_score(result)
        combined = relevance * 0.5 + authority * 0.3 + recency * 0.2
        return RankedSource(
            result=result,
            relevance_score=relevance,
            authority_score=authority,
            recency_score=recency,
            combined_score=combined,
        )

    @staticmethod
    def _relevance_score(query: str, result: SearchResult) -> float:
        query_terms = {term.lower() for term in query.split() if term}
        if not query_terms:
            return 0.0
        haystack = f"{result.title} {result.snippet}".lower()
        matches = sum(1 for term in query_terms if term in haystack)
        return min(1.0, matches / len(query_terms))

    def _authority_score(self, result: SearchResult) -> float:
        domain = self._extract_domain(result.url)
        if domain in self._trusted_domains:
            return 1.0
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.9
        if domain.endswith(".org"):
            return 0.6
        return 0.4

    def _recency_score(self, result: SearchResult) -> float:
        if result.published_at_epoch is None:
            return 0.5
        age_days = max(0.0, (time.time() - result.published_at_epoch) / 86400.0)
        return 0.5 ** (age_days / self._recency_half_life_days)

    @staticmethod
    def _extract_domain(url: str) -> str:
        cleaned = url.split("//")[-1]
        domain = cleaned.split("/")[0]
        return domain.lower()


class ResearchAgent(Agent):
    def __init__(
        self,
        web_search_fn: WebSearchFn,
        summarizer_fn: Optional[SummarizerFn] = None,
        fact_extractor_fn: Optional[FactExtractorFn] = None,
        ranker: Optional[SourceRanker] = None,
        metadata: Optional[AgentMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or AgentMetadata(
                name="research_agent",
                version="1.0.0",
                description="Performs web research, ranks sources, summarizes, and cites findings",
                priority=AgentPriority.NORMAL,
                capabilities=("research", "web_search", "summarization"),
                timeout_seconds=90.0,
                approval_required=False,
            )
        )
        self._web_search_fn = web_search_fn
        self._summarizer_fn = summarizer_fn
        self._fact_extractor_fn = fact_extractor_fn
        self._ranker = ranker or SourceRanker()
        self._knowledge_collection: list[ResearchSummary] = []

    async def _on_initialize(self) -> None:
        self._knowledge_collection = []

    async def _on_shutdown(self) -> None:
        pass

    async def _on_health_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            agent_name=self.name,
            status=HealthStatus.HEALTHY,
            latency_seconds=None,
            checked_at_epoch=time.time(),
            detail=f"collected_summaries={len(self._knowledge_collection)}",
        )

    async def _on_execute(self, task: AgentTask) -> Any:
        action = task.action
        params = task.parameters

        if action == "research":
            return await self.research(
                str(params["query"]), int(params.get("max_results", 10))
            )
        if action == "search":
            return await self.search(str(params["query"]), int(params.get("max_results", 10)))
        if action == "rank_sources":
            return self._ranker.rank(str(params["query"]), list(params["results"]))
        if action == "get_collection":
            return self.get_knowledge_collection()

        raise ValueError(f"unknown research action: {action}")

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        results = await self._web_search_fn(query, max_results)
        if not results:
            raise NoResultsError(f"no results found for query '{query}'")
        return results

    async def research(self, query: str, max_results: int = 10) -> ResearchSummary:
        results = await self.search(query, max_results)
        ranked_sources = self._ranker.rank(query, results)
        top_sources = ranked_sources[: min(len(ranked_sources), max_results)]
        top_results = [rs.result for rs in top_sources]

        facts: list[ExtractedFact] = []
        if self._fact_extractor_fn is not None:
            for result in top_results:
                extracted = await self._fact_extractor_fn(result)
                facts.extend(extracted)

        if self._summarizer_fn is not None:
            summary_text = await self._summarizer_fn(query, top_results)
        else:
            summary_text = self._default_summarize(query, top_results)

        citations = self._generate_citations(top_results)

        summary = ResearchSummary(
            query=query,
            summary_text=summary_text,
            facts=tuple(facts),
            citations=citations,
            sources_considered=len(results),
        )
        self._knowledge_collection.append(summary)
        return summary

    @staticmethod
    def _default_summarize(query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"No information was found for '{query}'."
        snippets = "; ".join(result.snippet for result in results[:5] if result.snippet)
        return f"Findings related to '{query}': {snippets}".strip()

    @staticmethod
    def _generate_citations(results: list[SearchResult]) -> tuple[Citation, ...]:
        citations = []
        for index, result in enumerate(results, start=1):
            citations.append(
                Citation(
                    citation_id=f"cite-{index}",
                    source_url=result.url,
                    title=result.title,
                )
            )
        return tuple(citations)

    def get_knowledge_collection(self) -> list[ResearchSummary]:
        return list(self._knowledge_collection)

    def clear_knowledge_collection(self) -> None:
        self._knowledge_collection.clear()
