"""Memory Agent: stores, retrieves, ranks, summarizes, and injects contextual memory."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set


logger = logging.getLogger("memory_agent")


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class MemoryEntry:
    memory_id: str
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    tags: Set[str] = field(default_factory=set)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class MemoryQueryResult:
    entry: MemoryEntry
    score: float


_STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of",
    "in", "on", "for", "with", "this", "that", "it", "as", "at", "by", "be",
}


def _tokenize(text: str) -> List[str]:
    return [
        w for w in re.findall(r"[a-zA-Z0-9']+", text.lower())
        if w not in _STOPWORDS and len(w) > 1
    ]


class MemoryAgent:
    """Agent responsible for managing the memory store of an agent system."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        max_entries: int = 10000,
        decay_half_life_seconds: float = 86400.0 * 7,
    ) -> None:
        self.agent_id: str = agent_id or f"memory-{uuid.uuid4().hex[:8]}"
        self.max_entries = max_entries
        self.decay_half_life_seconds = decay_half_life_seconds
        self._store: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
        self._metrics: Dict[str, int] = {
            "stored": 0,
            "retrieved": 0,
            "evicted": 0,
            "summaries": 0,
            "deleted": 0,
        }

    # ---------- storage ----------

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        tags: Optional[Sequence[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        memory_id: Optional[str] = None,
    ) -> MemoryEntry:
        entry_id = memory_id or f"mem-{uuid.uuid4().hex[:12]}"
        entry = MemoryEntry(
            memory_id=entry_id,
            content=content,
            memory_type=memory_type,
            tags=set(tags or []),
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
            embedding=embedding,
        )
        self._store[entry_id] = entry
        self._metrics["stored"] += 1

        if len(self._store) > self.max_entries:
            self._evict()

        return entry

    async def store_async(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        tags: Optional[Sequence[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        memory_id: Optional[str] = None,
    ) -> MemoryEntry:
        async with self._lock:
            return self.store(
                content, memory_type, tags, importance, metadata, embedding, memory_id
            )

    def _evict(self) -> None:
        if not self._store:
            return
        weakest_id = min(self._store, key=lambda mid: self._retention_score(self._store[mid]))
        del self._store[weakest_id]
        self._metrics["evicted"] += 1

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            self._metrics["deleted"] += 1
            return True
        return False

    async def delete_async(self, memory_id: str) -> bool:
        async with self._lock:
            return self.delete(memory_id)

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        entry = self._store.get(memory_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    async def get_async(self, memory_id: str) -> Optional[MemoryEntry]:
        return await asyncio.to_thread(self.get, memory_id)

    # ---------- retrieval ----------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[MemoryQueryResult]:
        candidates = list(self._store.values())

        if memory_type is not None:
            candidates = [e for e in candidates if e.memory_type == memory_type]

        if tags:
            tag_set = set(tags)
            candidates = [e for e in candidates if tag_set.intersection(e.tags)]

        ranked = self.rank(candidates, query)
        results = [r for r in ranked if r.score >= min_score][:top_k]

        for r in results:
            r.entry.access_count += 1
            r.entry.last_accessed = time.time()

        self._metrics["retrieved"] += len(results)
        return results

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[MemoryQueryResult]:
        return await asyncio.to_thread(
            self.retrieve, query, top_k, memory_type, tags, min_score
        )

    # ---------- ranking ----------

    def _retention_score(self, entry: MemoryEntry) -> float:
        age = max(0.0, time.time() - entry.created_at)
        decay = 0.5 ** (age / self.decay_half_life_seconds) if self.decay_half_life_seconds > 0 else 1.0
        access_bonus = min(0.3, entry.access_count * 0.02)
        return entry.importance * 0.6 + decay * 0.3 + access_bonus

    def _similarity(self, query_tokens: List[str], entry: MemoryEntry) -> float:
        entry_tokens = _tokenize(entry.content)
        if not entry_tokens or not query_tokens:
            return 0.0
        q_counter = Counter(query_tokens)
        e_counter = Counter(entry_tokens)
        overlap = sum((q_counter & e_counter).values())
        union = len(set(query_tokens) | set(entry_tokens))
        jaccard = overlap / union if union else 0.0
        tag_overlap = len(set(query_tokens) & {t.lower() for t in entry.tags})
        return jaccard + 0.05 * tag_overlap

    def rank(self, memories: Sequence[MemoryEntry], query: str = "") -> List[MemoryQueryResult]:
        query_tokens = _tokenize(query) if query else []
        scored: List[MemoryQueryResult] = []

        for entry in memories:
            similarity = self._similarity(query_tokens, entry) if query_tokens else 0.0
            retention = self._retention_score(entry)
            score = (0.7 * similarity + 0.3 * retention) if query_tokens else retention
            scored.append(MemoryQueryResult(entry=entry, score=round(score, 5)))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    async def rank_async(
        self, memories: Sequence[MemoryEntry], query: str = ""
    ) -> List[MemoryQueryResult]:
        return await asyncio.to_thread(self.rank, memories, query)

    # ---------- summarization ----------

    def summarize(self, memories: Sequence[MemoryEntry], max_chars: int = 800) -> str:
        if not memories:
            return ""

        sorted_memories = sorted(memories, key=lambda e: e.importance, reverse=True)
        lines: List[str] = []
        total_len = 0

        for entry in sorted_memories:
            snippet = entry.content.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            line = f"- [{entry.memory_type.value}] {snippet}"
            if total_len + len(line) > max_chars:
                break
            lines.append(line)
            total_len += len(line)

        self._metrics["summaries"] += 1
        return "\n".join(lines)

    async def summarize_async(self, memories: Sequence[MemoryEntry], max_chars: int = 800) -> str:
        return await asyncio.to_thread(self.summarize, memories, max_chars)

    def summarize_query(self, query: str, top_k: int = 10, max_chars: int = 800) -> str:
        results = self.retrieve(query, top_k=top_k)
        return self.summarize([r.entry for r in results], max_chars=max_chars)

    async def summarize_query_async(
        self, query: str, top_k: int = 10, max_chars: int = 800
    ) -> str:
        results = await self.retrieve_async(query, top_k=top_k)
        return await self.summarize_async([r.entry for r in results], max_chars=max_chars)

    # ---------- context injection ----------

    def inject_context(
        self,
        prompt: str,
        query: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 800,
        header: str = "Relevant memory context:",
    ) -> str:
        effective_query = query if query is not None else prompt
        results = self.retrieve(effective_query, top_k=top_k)
        if not results:
            return prompt

        summary = self.summarize([r.entry for r in results], max_chars=max_chars)
        if not summary:
            return prompt

        return f"{header}\n{summary}\n\n{prompt}"

    async def inject_context_async(
        self,
        prompt: str,
        query: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 800,
        header: str = "Relevant memory context:",
    ) -> str:
        effective_query = query if query is not None else prompt
        results = await self.retrieve_async(effective_query, top_k=top_k)
        if not results:
            return prompt

        summary = await self.summarize_async([r.entry for r in results], max_chars=max_chars)
        if not summary:
            return prompt

        return f"{header}\n{summary}\n\n{prompt}"

    # ---------- bulk operations ----------

    def all_memories(self) -> List[MemoryEntry]:
        return list(self._store.values())

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return len(self._store)

    # ---------- health ----------

    def health_check(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "metrics": dict(self._metrics),
            "timestamp": time.time(),
        }

    async def health_check_async(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self.health_check)
