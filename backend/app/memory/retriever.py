"""
retriever.py
============
Memory retrieval engine for the AI Operating System.

Responsibilities
----------------
- Unified similarity search across short-term and long-term memory.
- Multi-factor ranking: semantic similarity + recency + access frequency.
- Configurable scoring weights per retrieval profile.
- Context preparation: format ranked memories into LLM-ready context blocks.
- Async throughout; zero blocking calls on the hot path.

Compatible with
---------------
- short_term.py   (ShortTermMemory)
- long_term.py    (LongTermMemory)
- memory_capability.py
- context_engine.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.providers.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemorySource(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    COMBINED = "combined"


class RankingProfile(str, Enum):
    BALANCED = "balanced"       # equal weight: similarity + recency + frequency
    RECENCY = "recency"         # prioritise newest memories
    SEMANTIC = "semantic"       # pure vector similarity
    FREQUENCY = "frequency"     # most-accessed memories first
    PRECISE = "precise"         # high min_score, strict cutoff


# ---------------------------------------------------------------------------
# Scoring weights per profile
# ---------------------------------------------------------------------------

_PROFILE_WEIGHTS: Dict[RankingProfile, Dict[str, float]] = {
    RankingProfile.BALANCED:   {"similarity": 0.5, "recency": 0.3, "frequency": 0.2},
    RankingProfile.RECENCY:    {"similarity": 0.2, "recency": 0.6, "frequency": 0.2},
    RankingProfile.SEMANTIC:   {"similarity": 0.8, "recency": 0.1, "frequency": 0.1},
    RankingProfile.FREQUENCY:  {"similarity": 0.3, "recency": 0.2, "frequency": 0.5},
    RankingProfile.PRECISE:    {"similarity": 0.9, "recency": 0.05, "frequency": 0.05},
}

_PROFILE_MIN_SCORE: Dict[RankingProfile, float] = {
    RankingProfile.BALANCED:  0.10,
    RankingProfile.RECENCY:   0.05,
    RankingProfile.SEMANTIC:  0.15,
    RankingProfile.FREQUENCY: 0.05,
    RankingProfile.PRECISE:   0.40,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RawMemory:
    """
    Normalised representation of a memory from any source.
    Populated by source adapters before scoring.
    """

    memory_id: str
    source: MemorySource
    role: str = ""                      # short-term: user/assistant/system
    content: str = ""                   # primary text payload
    description: str = ""              # long-term description / summary
    key: str = ""                       # long-term key slug
    category: str = ""                 # long-term category
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: str = ""                # ISO-8601
    access_count: int = 0
    token_count: int = 0


@dataclass
class ScoredMemory:
    """A RawMemory annotated with its composite retrieval score."""

    memory: RawMemory
    similarity_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory.memory_id,
            "source": self.memory.source.value,
            "content": self.memory.content or self.memory.description,
            "role": self.memory.role,
            "key": self.memory.key,
            "category": self.memory.category,
            "tags": self.memory.tags,
            "timestamp": self.memory.timestamp,
            "similarity_score": round(self.similarity_score, 4),
            "recency_score": round(self.recency_score, 4),
            "frequency_score": round(self.frequency_score, 4),
            "composite_score": round(self.composite_score, 4),
            "metadata": self.memory.metadata,
        }


@dataclass
class RetrievalResult:
    """Final output of a retrieval call."""

    query: str
    source: MemorySource
    profile: RankingProfile
    scored_memories: List[ScoredMemory]
    context_block: str = ""            # formatted, LLM-ready context string
    total_tokens: int = 0
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "source": self.source.value,
            "profile": self.profile.value,
            "count": len(self.scored_memories),
            "total_tokens": self.total_tokens,
            "retrieved_at": self.retrieved_at,
            "memories": [m.to_dict() for m in self.scored_memories],
            "context_block": self.context_block,
        }


# ---------------------------------------------------------------------------
# Embedder (mirrors long_term.py; avoid circular import)
# ---------------------------------------------------------------------------


class _Embedder:
    """Lightweight bag-of-words embedder. Replace with tiktoken/ST in prod."""

    _DIM: int = 512

    def _vectorise(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._DIM
        for token in tokens:
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._vectorise, text)

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, min(1.0, dot))


class ProviderEmbedder:
    """Embedder backed by the NVIDIA Embed provider."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._provider = registry.get("embed")

    async def embed(self, text: str) -> List[float]:
        embeddings, _ = await self._provider.generate(prompt=text, texts=[text], input_type="query")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return [0.0] * 512


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> float:
    """Return POSIX timestamp from ISO-8601 string; 0.0 on failure."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def _recency_score(timestamp: str, half_life_hours: float = 24.0) -> float:
    """
    Exponential decay: score = exp(-lambda * age_hours).
    score = 1.0 for brand-new; ~0.5 at half_life_hours; approaches 0 for old.
    """
    ts = _parse_iso(timestamp)
    if ts == 0.0:
        return 0.5
    age_hours = max(0.0, (time.time() - ts) / 3600.0)
    lam = math.log(2) / max(half_life_hours, 1e-6)
    return math.exp(-lam * age_hours)


def _frequency_score(access_count: int, saturation: int = 50) -> float:
    """
    Sigmoid-style normalisation: score approaches 1 as access_count → saturation.
    """
    if access_count <= 0:
        return 0.0
    return access_count / (access_count + saturation)


def _composite(
    similarity: float,
    recency: float,
    frequency: float,
    weights: Dict[str, float],
) -> float:
    return (
        weights["similarity"] * similarity
        + weights["recency"] * recency
        + weights["frequency"] * frequency
    )


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


async def _adapt_short_term(
    messages: List[Dict[str, Any]],
) -> List[RawMemory]:
    """Convert short_term.py message dicts → RawMemory list."""
    out: List[RawMemory] = []
    for msg in messages:
        out.append(RawMemory(
            memory_id=msg.get("message_id", ""),
            source=MemorySource.SHORT_TERM,
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", ""),
            access_count=0,
            token_count=msg.get("token_count", 0),
            metadata=msg.get("metadata", {}),
        ))
    return out


async def _adapt_long_term(
    records: List[Any],   # List[SearchResult] or List[MemoryRecord]
) -> List[RawMemory]:
    """Convert long_term.py SearchResult / MemoryRecord objects → RawMemory list."""
    out: List[RawMemory] = []
    for item in records:
        # Support both SearchResult (has .record) and bare MemoryRecord
        record = getattr(item, "record", item)
        pre_score = getattr(item, "score", None)
        out.append(RawMemory(
            memory_id=record.memory_id,
            source=MemorySource.LONG_TERM,
            content=str(record.value) if record.value is not None else "",
            description=record.description,
            key=record.key,
            category=record.category.value if hasattr(record.category, "value") else str(record.category),
            tags=record.tags,
            embedding=record.embedding,
            timestamp=record.updated_at,
            access_count=record.access_count,
            metadata={**record.metadata, "_pre_score": pre_score},
        ))
    return out


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Unified retrieval engine over short-term and long-term memory stores.

    Parameters
    ----------
    short_term:      ShortTermMemory instance (optional).
    long_term:       LongTermMemory instance (optional).
    half_life_hours: Recency decay half-life in hours (default 24).
    freq_saturation: Access count at which frequency score ≈ 0.5 (default 50).
    context_sep:     Separator inserted between context blocks.
    max_context_tokens: Soft token cap for prepared context (0 = unlimited).
    """

    def __init__(
        self,
        short_term: Any = None,
        long_term: Any = None,
        half_life_hours: float = 24.0,
        freq_saturation: int = 50,
        context_sep: str = "\n---\n",
        max_context_tokens: int = 4_000,
        embed_registry: Optional[ProviderRegistry] = None,
    ) -> None:
        self._st = short_term
        self._lt = long_term
        if embed_registry is not None:
            self._embedder = ProviderEmbedder(embed_registry)
        else:
            self._embedder = _Embedder()
        self._half_life = half_life_hours
        self._freq_sat = freq_saturation
        self._context_sep = context_sep
        self._max_context_tokens = max_context_tokens

    @staticmethod
    async def _empty_list() -> list[Any]:
        return []

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    async def _score(
        self,
        query_embedding: List[float],
        memories: List[RawMemory],
        weights: Dict[str, float],
        min_score: float,
    ) -> List[ScoredMemory]:
        scored: List[ScoredMemory] = []

        for mem in memories:
            # Similarity
            if mem.embedding:
                sim = _Embedder.cosine(query_embedding, mem.embedding)
            else:
                # Embed on-the-fly for short-term messages
                text = mem.content or mem.description
                if text:
                    emb = await self._embedder.embed(text)
                    sim = _Embedder.cosine(query_embedding, emb)
                else:
                    sim = 0.0

            # Use pre-computed score from long_term search if available and higher
            pre = mem.metadata.get("_pre_score")
            if pre is not None:
                sim = max(sim, float(pre))

            rec = _recency_score(mem.timestamp, self._half_life)
            freq = _frequency_score(mem.access_count, self._freq_sat)
            comp = _composite(sim, rec, freq, weights)

            if comp >= min_score:
                scored.append(ScoredMemory(
                    memory=mem,
                    similarity_score=sim,
                    recency_score=rec,
                    frequency_score=freq,
                    composite_score=comp,
                ))

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Context preparation
    # ------------------------------------------------------------------

    def _prepare_context(
        self,
        scored: List[ScoredMemory],
        include_scores: bool = False,
    ) -> Tuple[str, int]:
        """
        Format scored memories into a single LLM-ready context string.
        Returns (context_block, estimated_token_count).
        """
        parts: List[str] = []
        total_tokens = 0

        for sm in scored:
            mem = sm.memory
            text = mem.content or mem.description or mem.key

            if mem.source == MemorySource.SHORT_TERM:
                header = f"[{mem.role.upper()}]"
            else:
                cat = f"[{mem.category.upper()}]" if mem.category else "[MEMORY]"
                key_part = f" {mem.key}:" if mem.key else ""
                header = f"{cat}{key_part}"

            score_suffix = f" (score={sm.composite_score:.2f})" if include_scores else ""
            block = f"{header}{score_suffix} {text}"

            est_tokens = max(1, len(block.split()))
            if self._max_context_tokens > 0 and total_tokens + est_tokens > self._max_context_tokens:
                break

            parts.append(block)
            total_tokens += est_tokens

        return self._context_sep.join(parts), total_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        source: MemorySource = MemorySource.COMBINED,
        profile: RankingProfile = RankingProfile.BALANCED,
        top_k: int = 10,
        min_score: Optional[float] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_scores: bool = False,
        st_max_messages: int = 50,
        lt_top_k: int = 30,
    ) -> RetrievalResult:
        """
        Retrieve and rank memories relevant to *query*.

        Parameters
        ----------
        query:           Natural-language retrieval query.
        session_id:      Required when source includes SHORT_TERM.
        source:          Which memory stores to search.
        profile:         Scoring weight profile.
        top_k:           Maximum memories to return after ranking.
        min_score:       Override profile default minimum composite score.
        category:        Long-term category filter.
        tags:            Long-term tag filter.
        include_scores:  Embed score annotations in context_block.
        st_max_messages: Max short-term messages to consider.
        lt_top_k:        Max long-term candidates to pull before re-ranking.

        Returns
        -------
        RetrievalResult with ranked ScoredMemory list and context_block.
        """
        weights = _PROFILE_WEIGHTS[profile]
        effective_min = min_score if min_score is not None else _PROFILE_MIN_SCORE[profile]

        # Embed query once
        query_embedding = await self._embedder.embed(query)

        gather_tasks = []

        # Short-term fetch
        if source in (MemorySource.SHORT_TERM, MemorySource.COMBINED):
            if self._st is not None and session_id:
                gather_tasks.append(
                    self._st.get_recent_context(session_id, max_messages=st_max_messages)
                )
            else:
                gather_tasks.append(self._empty_list())

        # Long-term fetch
        if source in (MemorySource.LONG_TERM, MemorySource.COMBINED):
            if self._lt is not None:
                gather_tasks.append(
                    self._lt.search(
                        query,
                        category=category,
                        tags=tags,
                        top_k=lt_top_k,
                        min_score=0.0,
                        semantic=True,
                    )
                )
            else:
                gather_tasks.append(self._empty_list())

        raw_memories: List[RawMemory] = []

        if source == MemorySource.SHORT_TERM:
            st_msgs = await gather_tasks[0]
            raw_memories = await _adapt_short_term(st_msgs)

        elif source == MemorySource.LONG_TERM:
            lt_records = await gather_tasks[0]
            raw_memories = await _adapt_long_term(lt_records)

        else:  # COMBINED
            results = await asyncio.gather(*gather_tasks, return_exceptions=True)
            st_data = results[0] if not isinstance(results[0], BaseException) else []
            lt_data = results[1] if not isinstance(results[1], BaseException) else []
            st_raw = await _adapt_short_term(st_data)  # type: ignore[arg-type]
            lt_raw = await _adapt_long_term(lt_data)  # type: ignore[arg-type]
            raw_memories = st_raw + lt_raw

        # Score and rank
        scored = await self._score(query_embedding, raw_memories, weights, effective_min)
        scored = scored[:top_k]

        # Prepare context
        context_block, total_tokens = self._prepare_context(scored, include_scores)

        return RetrievalResult(
            query=query,
            source=source,
            profile=profile,
            scored_memories=scored,
            context_block=context_block,
            total_tokens=total_tokens,
        )

    async def similarity_search(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[ScoredMemory]:
        """
        Ad-hoc similarity search over a caller-supplied list of memory dicts.
        Each dict must contain at least a 'content' or 'description' field.

        Useful for context_engine.py to re-rank an already fetched set.
        """
        query_embedding = await self._embedder.embed(query)

        raw: List[RawMemory] = []
        for item in memories:
            raw.append(RawMemory(
                memory_id=item.get("memory_id", ""),
                source=MemorySource(item.get("source", MemorySource.SHORT_TERM)),
                content=item.get("content", item.get("description", "")),
                role=item.get("role", ""),
                key=item.get("key", ""),
                category=item.get("category", ""),
                tags=item.get("tags", []),
                timestamp=item.get("timestamp", ""),
                access_count=item.get("access_count", 0),
                token_count=item.get("token_count", 0),
                metadata=item.get("metadata", {}),
            ))

        weights = _PROFILE_WEIGHTS[RankingProfile.SEMANTIC]
        scored = await self._score(query_embedding, raw, weights, min_score)
        return scored[:top_k]

    async def rank(
        self,
        query: str,
        memories: List[ScoredMemory],
        *,
        profile: RankingProfile = RankingProfile.BALANCED,
    ) -> List[ScoredMemory]:
        """
        Re-rank an existing ScoredMemory list against a (possibly new) query.
        Preserves original similarity scores; recomputes composite only.
        """
        weights = _PROFILE_WEIGHTS[profile]
        query_embedding = await self._embedder.embed(query)

        for sm in memories:
            text = sm.memory.content or sm.memory.description
            if text:
                emb = await self._embedder.embed(text)
                sm.similarity_score = _Embedder.cosine(query_embedding, emb)
            sm.recency_score = _recency_score(sm.memory.timestamp, self._half_life)
            sm.frequency_score = _frequency_score(sm.memory.access_count, self._freq_sat)
            sm.composite_score = _composite(
                sm.similarity_score, sm.recency_score, sm.frequency_score, weights
            )

        memories.sort(key=lambda s: s.composite_score, reverse=True)
        return memories

    def prepare_context(
        self,
        scored: List[ScoredMemory],
        *,
        include_scores: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        """
        Public wrapper: format a ranked ScoredMemory list into context text.

        Returns
        -------
        (context_block, estimated_token_count)
        """
        orig = self._max_context_tokens
        if max_tokens is not None:
            self._max_context_tokens = max_tokens
        result = self._prepare_context(scored, include_scores)
        self._max_context_tokens = orig
        return result

    async def health_check(self) -> Dict[str, Any]:
        st_ok = self._st is not None
        lt_ok = self._lt is not None
        return {
            "status": "healthy",
            "short_term_connected": st_ok,
            "long_term_connected": lt_ok,
            "embedder": type(self._embedder).__name__,
            "half_life_hours": self._half_life,
            "freq_saturation": self._freq_sat,
            "max_context_tokens": self._max_context_tokens,
            "profiles_available": [p.value for p in RankingProfile],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_retriever(
    short_term: Any = None,
    long_term: Any = None,
    *,
    half_life_hours: float = 24.0,
    freq_saturation: int = 50,
    max_context_tokens: int = 4_000,
    context_sep: str = "\n---\n",
    embed_registry: Optional[ProviderRegistry] = None,
) -> Retriever:
    """
    Instantiate a Retriever wired to the supplied memory stores.

    Parameters
    ----------
    short_term:          ShortTermMemory instance or None.
    long_term:           LongTermMemory instance or None.
    half_life_hours:     Recency decay half-life.
    freq_saturation:     Access count at frequency score ≈ 0.5.
    max_context_tokens:  Soft token cap for context_block output.
    context_sep:         Separator between context entries.
    embed_registry:      ProviderRegistry with 'embed' provider registered.
    """
    return Retriever(
        short_term=short_term,
        long_term=long_term,
        half_life_hours=half_life_hours,
        freq_saturation=freq_saturation,
        max_context_tokens=max_context_tokens,
        context_sep=context_sep,
        embed_registry=embed_registry,
    )
