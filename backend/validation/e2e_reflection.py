import asyncio
import pytest
import httpx

BASE_URL = "http://localhost:8000"
REFLECT_ANALYZE = f"{BASE_URL}/reflection/analyze"
REFLECT_SUGGEST = f"{BASE_URL}/reflection/suggest"
REFLECT_EVALUATE = f"{BASE_URL}/reflection/evaluate"
REFLECT_GENERATE = f"{BASE_URL}/reflection/generate"
REFLECT_FAILURES = f"{BASE_URL}/reflection/failures"

SAMPLE_FAILURES = [
    {
        "test": "test_chat_latency",
        "error": "AssertionError: Latency 12500ms exceeds 10000ms",
        "context": "Chat endpoint response time exceeded threshold.",
    },
    {
        "test": "test_memory_retrieval",
        "error": "AssertionError: Expected 'Alice' in response",
        "context": "Memory retrieval returned empty string.",
    },
    {
        "test": "test_rag_citations",
        "error": "AssertionError: No citations returned",
        "context": "RAG pipeline did not attach citations to answer.",
    },
]

SAMPLE_OUTPUT = {
    "input": "What is the capital of France?",
    "expected": "Paris",
    "actual": "The capital city of France is Paris, located in northern France.",
    "score": None,
}


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.mark.asyncio
async def test_analyze_failures(async_client: httpx.AsyncClient):
    payload = {"failures": SAMPLE_FAILURES}
    response = await async_client.post(REFLECT_ANALYZE, json=payload)
    assert response.status_code == 200, f"Analysis failed: {response.text}"
    data = response.json()
    assert "analysis" in data or "insights" in data or "summary" in data


@pytest.mark.asyncio
async def test_failure_analysis_content(async_client: httpx.AsyncClient):
    payload = {"failures": SAMPLE_FAILURES}
    response = await async_client.post(REFLECT_ANALYZE, json=payload)
    assert response.status_code == 200
    data = response.json()
    analysis = data.get("analysis") or data.get("insights") or data.get("summary") or ""
    assert len(str(analysis)) > 50, "Analysis is too brief"


@pytest.mark.asyncio
async def test_suggest_improvements(async_client: httpx.AsyncClient):
    payload = {
        "failures": SAMPLE_FAILURES,
        "context": "Production chatbot with memory and RAG capabilities.",
    }
    response = await async_client.post(REFLECT_SUGGEST, json=payload)
    assert response.status_code == 200
    data = response.json()
    suggestions = data.get("suggestions") or data.get("improvements") or data.get("recommendations") or []
    assert len(suggestions) > 0, "No suggestions returned"


@pytest.mark.asyncio
async def test_suggestions_are_actionable(async_client: httpx.AsyncClient):
    payload = {"failures": SAMPLE_FAILURES[:1]}
    response = await async_client.post(REFLECT_SUGGEST, json=payload)
    assert response.status_code == 200
    data = response.json()
    suggestions = data.get("suggestions") or data.get("improvements") or []
    if isinstance(suggestions, list) and len(suggestions) > 0:
        first = suggestions[0]
        text = first.get("description") or first.get("action") or str(first)
        assert len(text) > 10
    elif isinstance(suggestions, str):
        assert len(suggestions) > 10


@pytest.mark.asyncio
async def test_evaluate_output(async_client: httpx.AsyncClient):
    payload = SAMPLE_OUTPUT
    response = await async_client.post(REFLECT_EVALUATE, json=payload)
    assert response.status_code == 200
    data = response.json()
    score = data.get("score") or data.get("rating") or data.get("evaluation")
    assert score is not None


@pytest.mark.asyncio
async def test_evaluate_output_score_range(async_client: httpx.AsyncClient):
    payload = SAMPLE_OUTPUT
    response = await async_client.post(REFLECT_EVALUATE, json=payload)
    assert response.status_code == 200
    data = response.json()
    score = data.get("score") or data.get("rating")
    if isinstance(score, (int, float)):
        assert 0 <= score <= 10 or 0.0 <= score <= 1.0, f"Score {score} out of expected range"


@pytest.mark.asyncio
async def test_generate_reflection(async_client: httpx.AsyncClient):
    payload = {
        "session_id": "reflection-test-001",
        "failures": SAMPLE_FAILURES,
        "outputs": [SAMPLE_OUTPUT],
        "context": "E2E test suite run on 2024-01-01",
    }
    response = await async_client.post(REFLECT_GENERATE, json=payload)
    assert response.status_code == 200
    data = response.json()
    reflection = data.get("reflection") or data.get("report") or data.get("content") or ""
    assert len(str(reflection)) > 100


@pytest.mark.asyncio
async def test_reflection_includes_failures(async_client: httpx.AsyncClient):
    payload = {"failures": SAMPLE_FAILURES, "outputs": [], "session_id": "reflect-002"}
    response = await async_client.post(REFLECT_GENERATE, json=payload)
    assert response.status_code == 200
    reflection = str(response.json())
    assert any(
        keyword in reflection.lower()
        for keyword in ("latency", "memory", "citation", "failure", "error", "fix", "improve")
    )


@pytest.mark.asyncio
async def test_analyze_single_failure(async_client: httpx.AsyncClient):
    payload = {"failures": [SAMPLE_FAILURES[0]]}
    response = await async_client.post(REFLECT_ANALYZE, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("analysis") or data.get("insights") or data.get("summary")


@pytest.mark.asyncio
async def test_reflection_failures_list(async_client: httpx.AsyncClient):
    for failure in SAMPLE_FAILURES:
        r = await async_client.post(REFLECT_FAILURES, json={"failure": failure})
        assert r.status_code in (200, 201)

    list_r = await async_client.get(REFLECT_FAILURES)
    assert list_r.status_code == 200
    failures = list_r.json().get("failures") or list_r.json().get("data") or []
    assert len(failures) >= len(SAMPLE_FAILURES)


@pytest.mark.asyncio
async def test_evaluate_multiple_outputs(async_client: httpx.AsyncClient):
    outputs = [
        {"input": "What is 2+2?", "expected": "4", "actual": "4"},
        {"input": "Capital of Germany?", "expected": "Berlin", "actual": "The capital is Berlin."},
        {"input": "Water boiling point?", "expected": "100°C", "actual": "0°C"},
    ]
    tasks = [async_client.post(REFLECT_EVALUATE, json=o) for o in outputs]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    scores = []
    for r in responses:
        assert not isinstance(r, Exception)
        assert r.status_code == 200
        score = r.json().get("score") or r.json().get("rating") or 0
        scores.append(float(score) if score else 0)
    assert scores[0] > scores[2], "Correct answer should score higher than incorrect"