import asyncio
import io
import os
import uuid
import pytest
import httpx

BASE_URL = "http://localhost:8000"
RAG_UPLOAD = f"{BASE_URL}/rag/upload"
RAG_QUERY = f"{BASE_URL}/rag/query"
RAG_CHUNKS = f"{BASE_URL}/rag/chunks"
RAG_EMBED = f"{BASE_URL}/rag/embed"
RAG_STORE = f"{BASE_URL}/rag/store"
RAG_RETRIEVE = f"{BASE_URL}/rag/retrieve"
RAG_RERANK = f"{BASE_URL}/rag/rerank"
RAG_ANSWER = f"{BASE_URL}/rag/answer"

SAMPLE_PDF_CONTENT = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 120>>stream
BT /F1 12 Tf 100 700 Td (Artificial intelligence is transforming industries.) Tj 0 -20 Td (Machine learning enables computers to learn from data.) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000446 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
537
%%EOF"""


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.fixture
def doc_id():
    return f"rag-doc-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_upload_pdf(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    data = {"doc_id": doc_id}
    response = await async_client.post(RAG_UPLOAD, files=files, data=data)
    assert response.status_code in (200, 201), f"Upload failed: {response.text}"
    resp_data = response.json()
    assert "doc_id" in resp_data or "id" in resp_data or "status" in resp_data


@pytest.mark.asyncio
async def test_chunk_document(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    assert upload_r.status_code in (200, 201)
    uploaded_id = upload_r.json().get("doc_id") or upload_r.json().get("id") or doc_id

    chunk_r = await async_client.post(RAG_CHUNKS, json={"doc_id": uploaded_id, "chunk_size": 256, "overlap": 32})
    assert chunk_r.status_code == 200
    chunks = chunk_r.json().get("chunks") or chunk_r.json().get("data") or []
    assert len(chunks) > 0, "No chunks returned"
    for chunk in chunks:
        assert "content" in chunk or "text" in chunk


@pytest.mark.asyncio
async def test_embed_chunks(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id

    embed_r = await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    assert embed_r.status_code == 200
    data = embed_r.json()
    assert data.get("embedded") or data.get("count") or data.get("status") == "success"


@pytest.mark.asyncio
async def test_store_embeddings(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})

    store_r = await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})
    assert store_r.status_code in (200, 201)


@pytest.mark.asyncio
async def test_query_and_retrieve(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})

    retrieve_r = await async_client.post(RAG_RETRIEVE, json={
        "query": "What is artificial intelligence?",
        "doc_ids": [uploaded_id],
        "top_k": 3,
    })
    assert retrieve_r.status_code == 200
    results = retrieve_r.json().get("results") or retrieve_r.json().get("chunks") or []
    assert len(results) > 0


@pytest.mark.asyncio
async def test_rerank_results(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})

    retrieve_r = await async_client.post(RAG_RETRIEVE, json={
        "query": "machine learning",
        "doc_ids": [uploaded_id],
        "top_k": 5,
    })
    candidates = retrieve_r.json().get("results") or retrieve_r.json().get("chunks") or []

    rerank_r = await async_client.post(RAG_RERANK, json={
        "query": "machine learning data",
        "candidates": candidates,
        "top_k": 3,
    })
    assert rerank_r.status_code == 200
    reranked = rerank_r.json().get("results") or rerank_r.json().get("ranked") or []
    assert len(reranked) > 0
    scores = [r.get("score") or r.get("relevance_score") for r in reranked]
    valid_scores = [s for s in scores if s is not None]
    if len(valid_scores) > 1:
        assert valid_scores == sorted(valid_scores, reverse=True), "Results not sorted by score"


@pytest.mark.asyncio
async def test_rag_answer_generation(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})

    answer_r = await async_client.post(RAG_ANSWER, json={
        "question": "What is artificial intelligence?",
        "doc_ids": [uploaded_id],
        "top_k": 3,
    })
    assert answer_r.status_code == 200
    data = answer_r.json()
    answer = data.get("answer") or data.get("response") or data.get("content") or ""
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_rag_citations_present(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})

    answer_r = await async_client.post(RAG_ANSWER, json={
        "question": "What does machine learning do?",
        "doc_ids": [uploaded_id],
        "top_k": 3,
        "include_citations": True,
    })
    assert answer_r.status_code == 200
    data = answer_r.json()
    citations = data.get("citations") or data.get("sources") or data.get("references") or []
    assert len(citations) > 0, f"No citations returned. Response: {data}"
    for citation in citations:
        assert "doc_id" in citation or "source" in citation or "chunk_id" in citation


@pytest.mark.asyncio
async def test_rag_full_pipeline(async_client: httpx.AsyncClient):
    full_doc_id = f"full-pipeline-{uuid.uuid4().hex[:8]}"
    files = {"file": (f"{full_doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}

    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": full_doc_id})
    assert upload_r.status_code in (200, 201)
    uploaded_id = upload_r.json().get("doc_id") or full_doc_id

    chunk_r = await async_client.post(RAG_CHUNKS, json={"doc_id": uploaded_id})
    assert chunk_r.status_code == 200

    embed_r = await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    assert embed_r.status_code == 200

    store_r = await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})
    assert store_r.status_code in (200, 201)

    answer_r = await async_client.post(RAG_ANSWER, json={
        "question": "Explain machine learning.",
        "doc_ids": [uploaded_id],
        "top_k": 3,
        "include_citations": True,
    })
    assert answer_r.status_code == 200
    data = answer_r.json()
    assert data.get("answer") or data.get("response")
    assert data.get("citations") or data.get("sources")


@pytest.mark.asyncio
async def test_rag_query_endpoint(async_client: httpx.AsyncClient, doc_id: str):
    files = {"file": (f"{doc_id}.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")}
    upload_r = await async_client.post(RAG_UPLOAD, files=files, data={"doc_id": doc_id})
    uploaded_id = upload_r.json().get("doc_id") or doc_id
    await async_client.post(RAG_EMBED, json={"doc_id": uploaded_id})
    await async_client.post(RAG_STORE, json={"doc_id": uploaded_id})

    query_r = await async_client.post(RAG_QUERY, json={
        "query": "artificial intelligence transformation",
        "doc_ids": [uploaded_id],
    })
    assert query_r.status_code == 200
    data = query_r.json()
    assert data.get("results") or data.get("answer") or data.get("response")