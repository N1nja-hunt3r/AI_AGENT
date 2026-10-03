"""
test_rag.py
Tests for the retrieval-augmented generation pipeline: chunking,
embedding, vector retrieval, and reranking.
"""

from __future__ import annotations

from typing import Callable, List

import pytest

from conftest import MockRetriever, MockVectorDB


@pytest.mark.asyncio
async def test_chunker_splits_text_into_bounded_chunks(chunker: Callable[[str, int], List[str]]) -> None:
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunker(text, 50)

    assert len(chunks) == 3
    assert all(len(chunk.split()) <= 50 for chunk in chunks)


def test_chunker_handles_empty_text(chunker: Callable[[str, int], List[str]]) -> None:
    assert chunker("", 10) == [""]


def test_embed_fn_is_deterministic(mock_embed_fn: Callable[[str], List[float]]) -> None:
    first = mock_embed_fn("hello world")
    second = mock_embed_fn("hello world")
    assert first == second


def test_embed_fn_differs_for_different_text(mock_embed_fn: Callable[[str], List[float]]) -> None:
    assert mock_embed_fn("alpha") != mock_embed_fn("a completely different sentence")


@pytest.mark.asyncio
async def test_retriever_returns_upserted_documents(
    mock_retriever: MockRetriever, mock_vector_db: MockVectorDB
) -> None:
    await mock_vector_db.upsert("doc1", [0.1, 0.2], {"text": "revenue grew this quarter"})
    results = await mock_retriever.retrieve("revenue", top_k=1)

    assert results
    assert results[0]["id"] == "doc1"


@pytest.mark.asyncio
async def test_retriever_respects_top_k(mock_retriever: MockRetriever, mock_vector_db: MockVectorDB) -> None:
    for i in range(5):
        await mock_vector_db.upsert(f"doc{i}", [float(i)], {"text": f"chunk {i}"})

    results = await mock_retriever.retrieve("chunk", top_k=2)
    assert len(results) == 2


def test_rerank_orders_by_keyword_overlap(reranker) -> None:
    candidates = [
        {"metadata": {"text": "unrelated content about weather"}},
        {"metadata": {"text": "quarterly revenue and profit report"}},
    ]
    ranked = reranker("revenue profit", candidates)
    assert ranked[0]["metadata"]["text"].startswith("quarterly")


def test_rerank_returns_all_candidates(reranker) -> None:
    candidates = [{"metadata": {"text": "a"}}, {"metadata": {"text": "b"}}]
    ranked = reranker("a", candidates)
    assert len(ranked) == len(candidates)


@pytest.mark.asyncio
async def test_full_rag_pipeline_chunk_embed_retrieve_rerank(
    chunker: Callable[[str, int], List[str]],
    mock_embed_fn: Callable[[str], List[float]],
    mock_vector_db: MockVectorDB,
    reranker,
) -> None:
    text = "revenue grew in q1. profit declined in q2. headcount increased in q3."
    chunks = chunker(text, 6)

    for i, chunk in enumerate(chunks):
        await mock_vector_db.upsert(f"c{i}", mock_embed_fn(chunk), {"text": chunk})

    query_vector = mock_embed_fn("profit")
    candidates = await mock_vector_db.query(query_vector, top_k=len(chunks))
    ranked = reranker("profit", candidates)

    assert ranked
    assert any("profit" in candidate["metadata"]["text"] for candidate in ranked)