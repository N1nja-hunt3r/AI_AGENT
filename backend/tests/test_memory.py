"""
test_memory.py
Tests for the memory store: short-term and long-term tiers, promotion,
retrieval, and compression.
"""

from __future__ import annotations

import pytest

from conftest import MockMemoryStore


def test_add_short_term_memory(mock_memory: MockMemoryStore) -> None:
    record = mock_memory.add_short("user prefers concise answers")
    assert record in mock_memory.short_term
    assert record.tier == "short"


def test_add_long_term_memory(mock_memory: MockMemoryStore) -> None:
    record = mock_memory.add_long("user's name is Priya")
    assert record in mock_memory.long_term
    assert record.tier == "long"


def test_promote_moves_short_term_record_to_long_term(mock_memory: MockMemoryStore) -> None:
    record = mock_memory.add_short("recurring preference: dark mode")
    promoted = mock_memory.promote(record.id)

    assert promoted is True
    assert record.id not in [r.id for r in mock_memory.short_term]
    assert record.id in [r.id for r in mock_memory.long_term]


def test_promote_missing_record_returns_false(mock_memory: MockMemoryStore) -> None:
    assert mock_memory.promote("missing-id") is False


def test_retrieve_matches_query_content(mock_memory: MockMemoryStore) -> None:
    mock_memory.add_long("user works in finance")
    mock_memory.add_long("user enjoys hiking")

    results = mock_memory.retrieve("finance")
    assert any("finance" in record.content for record in results)


def test_retrieve_respects_limit(mock_memory: MockMemoryStore) -> None:
    for i in range(5):
        mock_memory.add_long(f"fact number {i}")

    results = mock_memory.retrieve("fact", limit=2)
    assert len(results) == 2


def test_retrieve_falls_back_to_recent_when_no_match(mock_memory: MockMemoryStore) -> None:
    mock_memory.add_long("unrelated entry")
    results = mock_memory.retrieve("nonexistent topic", limit=1)
    assert len(results) == 1


def test_compress_long_term_joins_entries(mock_memory: MockMemoryStore) -> None:
    mock_memory.add_long("fact one")
    mock_memory.add_long("fact two")

    compressed = mock_memory.compress(tier="long")
    assert "fact one" in compressed and "fact two" in compressed


def test_compress_short_term_is_independent_of_long_term(mock_memory: MockMemoryStore) -> None:
    mock_memory.add_short("transient note")
    mock_memory.add_long("durable fact")

    assert "transient note" in mock_memory.compress(tier="short")
    assert "transient note" not in mock_memory.compress(tier="long")


@pytest.mark.asyncio
async def test_memory_operations_are_usable_in_async_context(mock_memory: MockMemoryStore) -> None:
    async def add_and_retrieve():
        mock_memory.add_short("async preference noted")
        return mock_memory.retrieve("async")

    results = await add_and_retrieve()
    assert results