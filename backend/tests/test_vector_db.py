"""
test_vector_db.py
Tests for the vector database abstraction across backend flavors
(Chroma, FAISS, Pinecone, Qdrant) and the retriever built on top of it.
"""

from __future__ import annotations

from typing import Callable, List

import pytest

from conftest import MockRetriever, MockVectorDB

BACKENDS = ["chroma", "faiss", "pinecone", "qdrant"]


@pytest.fixture(params=BACKENDS)
def vector_db(request: pytest.FixtureRequest, vector_db_factory: Callable[[str], MockVectorDB]) -> MockVectorDB:
    return vector_db_factory(request.param)


@pytest.mark.asyncio
async def test_upsert_and_query_round_trip_per_backend(vector_db: MockVectorDB) -> None:
    await vector_db.upsert("doc1", [0.1, 0.2, 0.3], {"text": "hello world", "backend": vector_db.backend})
    results = await vector_db.query([0.1, 0.2, 0.3], top_k=1)

    assert results
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"]["backend"] == vector_db.backend


@pytest.mark.asyncio
async def test_query_respects_top_k_per_backend(vector_db: MockVectorDB) -> None:
    for i in range(5):
        await vector_db.upsert(f"doc{i}", [float(i)], {"text": f"chunk {i}"})

    results = await vector_db.query([0.0], top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete_removes_document_per_backend(vector_db: MockVectorDB) -> None:
    await vector_db.upsert("doc1", [0.1], {"text": "to be removed"})
    deleted = await vector_db.delete("doc1")

    assert deleted is True
    results = await vector_db.query([0.1], top_k=5)
    assert all(r["id"] != "doc1" for r in results)


@pytest.mark.asyncio
async def test_delete_unknown_document_returns_false(vector_db: MockVectorDB) -> None:
    assert await vector_db.delete("never-existed") is False


def test_factory_assigns_requested_backend_label(vector_db_factory: Callable[[str], MockVectorDB]) -> None:
    db = vector_db_factory("qdrant")
    assert db.backend == "qdrant"


@pytest.mark.asyncio
async def test_retriever_works_against_any_backend(
    vector_db_factory: Callable[[str], MockVectorDB], mock_embed_fn: Callable[[str], List[float]]
) -> None:
    for backend in BACKENDS:
        db = vector_db_factory(backend)
        retriever = MockRetriever(db, mock_embed_fn)
        await db.upsert("doc1", mock_embed_fn("revenue report"), {"text": "revenue report", "backend": backend})

        results = await retriever.retrieve("revenue report")
        assert results
        assert results[0]["metadata"]["backend"] == backend


@pytest.mark.asyncio
async def test_retriever_returns_empty_when_index_is_empty(
    mock_retriever: MockRetriever,
) -> None:
    results = await mock_retriever.retrieve("anything")
    assert results == []