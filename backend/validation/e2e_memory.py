import asyncio
import uuid
import pytest
import httpx
import numpy as np

BASE_URL = "http://localhost:8000"
MEMORY_STORE_ENDPOINT = f"{BASE_URL}/memory/store"
MEMORY_RETRIEVE_ENDPOINT = f"{BASE_URL}/memory/retrieve"
MEMORY_SEARCH_ENDPOINT = f"{BASE_URL}/memory/search"
EMBEDDINGS_ENDPOINT = f"{BASE_URL}/memory/embeddings"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
def unique_session():
    return f"mem-test-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_store_memory(async_client: httpx.AsyncClient, unique_session: str):
    payload = {
        "session_id": unique_session,
        "content": "The Eiffel Tower is located in Paris, France.",
        "metadata": {"source": "test", "topic": "geography"},
    }
    response = await async_client.post(MEMORY_STORE_ENDPOINT, json=payload)
    assert response.status_code in (200, 201)
    data = response.json()
    assert "id" in data or "memory_id" in data or "status" in data


@pytest.mark.asyncio
async def test_retrieve_memory(async_client: httpx.AsyncClient, unique_session: str):
    content = "Python is a high-level programming language."
    store_payload = {"session_id": unique_session, "content": content}
    store_r = await async_client.post(MEMORY_STORE_ENDPOINT, json=store_payload)
    assert store_r.status_code in (200, 201)
    memory_id = store_r.json().get("id") or store_r.json().get("memory_id")

    retrieve_payload = {"session_id": unique_session, "memory_id": memory_id}
    retrieve_r = await async_client.post(MEMORY_RETRIEVE_ENDPOINT, json=retrieve_payload)
    assert retrieve_r.status_code == 200
    data = retrieve_r.json()
    retrieved_content = data.get("content") or data.get("memory") or ""
    assert "python" in retrieved_content.lower() or "programming" in retrieved_content.lower()


@pytest.mark.asyncio
async def test_semantic_retrieval(async_client: httpx.AsyncClient, unique_session: str):
    documents = [
        "Cats are domesticated mammals often kept as pets.",
        "Dogs are loyal animals and common household pets.",
        "The stock market crashed in 2008 due to financial crisis.",
    ]
    for doc in documents:
        await async_client.post(MEMORY_STORE_ENDPOINT, json={"session_id": unique_session, "content": doc})

    search_payload = {"session_id": unique_session, "query": "What animals are popular pets?", "top_k": 2}
    response = await async_client.post(MEMORY_SEARCH_ENDPOINT, json=search_payload)
    assert response.status_code == 200
    results = response.json().get("results") or response.json().get("memories") or []
    assert len(results) > 0
    combined = " ".join(r.get("content", "") for r in results).lower()
    assert "cat" in combined or "dog" in combined or "pet" in combined


@pytest.mark.asyncio
async def test_similarity_search(async_client: httpx.AsyncClient, unique_session: str):
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
    ]
    for doc in docs:
        await async_client.post(MEMORY_STORE_ENDPOINT, json={"session_id": unique_session, "content": doc})

    payload = {"session_id": unique_session, "query": "neural networks and AI", "top_k": 2, "min_score": 0.5}
    response = await async_client.post(MEMORY_SEARCH_ENDPOINT, json=payload)
    assert response.status_code == 200
    results = response.json().get("results") or []
    for r in results:
        score = r.get("score") or r.get("similarity") or r.get("distance")
        assert score is not None, "Similarity score missing from results"


@pytest.mark.asyncio
async def test_embeddings_generated(async_client: httpx.AsyncClient):
    payload = {"text": "Generate an embedding for this sentence."}
    response = await async_client.post(EMBEDDINGS_ENDPOINT, json=payload)
    assert response.status_code == 200
    data = response.json()
    embedding = data.get("embedding") or data.get("vector") or data.get("embeddings")
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(v, (int, float)) for v in embedding)


@pytest.mark.asyncio
async def test_embedding_dimensionality(async_client: httpx.AsyncClient):
    texts = ["Hello world", "Another sentence for embedding"]
    embeddings = []
    for text in texts:
        r = await async_client.post(EMBEDDINGS_ENDPOINT, json={"text": text})
        assert r.status_code == 200
        emb = r.json().get("embedding") or r.json().get("vector") or []
        embeddings.append(emb)
    assert len(embeddings[0]) == len(embeddings[1]), "Embedding dimensions inconsistent"


@pytest.mark.asyncio
async def test_embedding_cosine_similarity(async_client: httpx.AsyncClient):
    r1 = await async_client.post(EMBEDDINGS_ENDPOINT, json={"text": "I love cats and dogs."})
    r2 = await async_client.post(EMBEDDINGS_ENDPOINT, json={"text": "Pets are wonderful companions."})
    r3 = await async_client.post(EMBEDDINGS_ENDPOINT, json={"text": "The stock market is volatile today."})

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    e1 = r1.json().get("embedding") or r1.json().get("vector")
    e2 = r2.json().get("embedding") or r2.json().get("vector")
    e3 = r3.json().get("embedding") or r3.json().get("vector")
    sim_related = cosine(e1, e2)
    sim_unrelated = cosine(e1, e3)
    assert sim_related > sim_unrelated, "Semantic similarity not working correctly"


@pytest.mark.asyncio
async def test_chroma_store_and_query(async_client: httpx.AsyncClient, unique_session: str):
    chroma_endpoint = f"{BASE_URL}/memory/chroma/store"
    chroma_query = f"{BASE_URL}/memory/chroma/query"

    store_r = await async_client.post(chroma_endpoint, json={
        "collection": unique_session,
        "documents": ["Chroma is a vector database.", "It supports semantic search."],
        "ids": [f"{unique_session}-1", f"{unique_session}-2"],
    })
    assert store_r.status_code in (200, 201)

    query_r = await async_client.post(chroma_query, json={
        "collection": unique_session,
        "query": "vector database",
        "top_k": 2,
    })
    assert query_r.status_code == 200
    results = query_r.json().get("results") or []
    assert len(results) > 0


@pytest.mark.asyncio
async def test_postgresql_memory_persistence(async_client: httpx.AsyncClient, unique_session: str):
    pg_store = f"{BASE_URL}/memory/postgres/store"
    pg_retrieve = f"{BASE_URL}/memory/postgres/retrieve"

    store_r = await async_client.post(pg_store, json={
        "session_id": unique_session,
        "content": "PostgreSQL memory persistence test.",
        "metadata": {"test": True},
    })
    assert store_r.status_code in (200, 201)
    record_id = store_r.json().get("id") or store_r.json().get("record_id")

    retrieve_r = await async_client.post(pg_retrieve, json={
        "session_id": unique_session,
        "id": record_id,
    })
    assert retrieve_r.status_code == 200
    content = retrieve_r.json().get("content") or ""
    assert "postgresql" in content.lower() or "persistence" in content.lower()


@pytest.mark.asyncio
async def test_memory_update(async_client: httpx.AsyncClient, unique_session: str):
    store_r = await async_client.post(MEMORY_STORE_ENDPOINT, json={
        "session_id": unique_session,
        "content": "Initial content.",
    })
    memory_id = store_r.json().get("id") or store_r.json().get("memory_id")

    update_r = await async_client.put(f"{BASE_URL}/memory/{memory_id}", json={
        "content": "Updated content after modification."
    })
    assert update_r.status_code in (200, 204)


@pytest.mark.asyncio
async def test_memory_delete(async_client: httpx.AsyncClient, unique_session: str):
    store_r = await async_client.post(MEMORY_STORE_ENDPOINT, json={
        "session_id": unique_session,
        "content": "This memory will be deleted.",
    })
    memory_id = store_r.json().get("id") or store_r.json().get("memory_id")

    delete_r = await async_client.delete(f"{BASE_URL}/memory/{memory_id}")
    assert delete_r.status_code in (200, 204)

    retrieve_r = await async_client.post(MEMORY_RETRIEVE_ENDPOINT, json={
        "session_id": unique_session,
        "memory_id": memory_id,
    })
    assert retrieve_r.status_code in (404, 200)
    if retrieve_r.status_code == 200:
        content = retrieve_r.json().get("content")
        assert content is None or content == ""