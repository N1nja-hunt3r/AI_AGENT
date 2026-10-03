import asyncio
import time
import pytest
import httpx
from typing import Any

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is 2 + 2?",
    "Summarize the theory of relativity.",
]


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
async def test_chat_endpoint_reachable(async_client: httpx.AsyncClient):
    response = await async_client.get(f"{BASE_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", SAMPLE_PROMPTS)
async def test_chat_response_exists(async_client: httpx.AsyncClient, prompt: str):
    payload = {"message": prompt, "session_id": "test-session-001"}
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data or "message" in data or "content" in data
    content = data.get("response") or data.get("message") or data.get("content")
    assert content is not None
    assert len(str(content)) > 0


@pytest.mark.asyncio
async def test_chat_provider_called(async_client: httpx.AsyncClient):
    payload = {"message": "Hello, who are you?", "session_id": "test-session-002"}
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    assert response.status_code == 200
    data = response.json()
    provider_keys = {"provider", "model", "llm_provider", "used_provider"}
    found = any(k in data for k in provider_keys)
    if not found and "metadata" in data:
        meta = data["metadata"]
        found = any(k in meta for k in provider_keys)
    assert found, f"No provider info in response: {data}"


@pytest.mark.asyncio
async def test_chat_memory_updated(async_client: httpx.AsyncClient):
    session_id = "test-memory-session-001"
    payload1 = {"message": "My name is Alice.", "session_id": session_id}
    r1 = await async_client.post(CHAT_ENDPOINT, json=payload1)
    assert r1.status_code == 200

    payload2 = {"message": "What is my name?", "session_id": session_id}
    r2 = await async_client.post(CHAT_ENDPOINT, json=payload2)
    assert r2.status_code == 200
    data2 = r2.json()
    content = (
        data2.get("response") or data2.get("message") or data2.get("content") or ""
    )
    assert "alice" in content.lower(), f"Memory not updated, response: {content}"


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", SAMPLE_PROMPTS[:3])
async def test_chat_response_latency(async_client: httpx.AsyncClient, prompt: str):
    MAX_LATENCY_MS = 10000
    payload = {"message": prompt, "session_id": "test-latency-001"}
    start = time.perf_counter()
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    assert elapsed_ms < MAX_LATENCY_MS, f"Latency {elapsed_ms:.1f}ms exceeds {MAX_LATENCY_MS}ms"


@pytest.mark.asyncio
async def test_chat_token_counting(async_client: httpx.AsyncClient):
    payload = {"message": "Count tokens for this message.", "session_id": "test-tokens-001"}
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    assert response.status_code == 200
    data = response.json()
    token_keys = {"tokens", "token_count", "usage", "input_tokens", "output_tokens", "total_tokens"}
    found = any(k in data for k in token_keys)
    if not found and "metadata" in data:
        found = any(k in data["metadata"] for k in token_keys)
    assert found, f"No token count in response: {data}"


@pytest.mark.asyncio
async def test_chat_cost_tracking(async_client: httpx.AsyncClient):
    payload = {"message": "Track cost for this request.", "session_id": "test-cost-001"}
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    assert response.status_code == 200
    data = response.json()
    cost_keys = {"cost", "cost_usd", "price", "total_cost", "billing"}
    found = any(k in data for k in cost_keys)
    if not found and "metadata" in data:
        found = any(k in data["metadata"] for k in cost_keys)
    assert found, f"No cost tracking in response: {data}"


@pytest.mark.asyncio
async def test_chat_concurrent_requests(async_client: httpx.AsyncClient):
    prompts = ["Tell me a joke.", "What is AI?", "Define machine learning."]
    tasks = [
        async_client.post(CHAT_ENDPOINT, json={"message": p, "session_id": f"concurrent-{i}"})
        for i, p in enumerate(prompts)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for r in responses:
        assert not isinstance(r, Exception), f"Request raised exception: {r}"
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_chat_empty_prompt_rejected(async_client: httpx.AsyncClient):
    payload = {"message": "", "session_id": "test-empty-001"}
    response = await async_client.post(CHAT_ENDPOINT, json=payload)
    assert response.status_code in (400, 422), f"Expected 4xx for empty prompt, got {response.status_code}"


@pytest.mark.asyncio
async def test_chat_session_isolation(async_client: httpx.AsyncClient):
    await async_client.post(CHAT_ENDPOINT, json={"message": "My secret is 42.", "session_id": "session-A"})
    r2 = await async_client.post(CHAT_ENDPOINT, json={"message": "What is my secret?", "session_id": "session-B"})
    assert r2.status_code == 200
    content = r2.json().get("response") or r2.json().get("message") or r2.json().get("content") or ""
    assert "42" not in content, "Session isolation failed: secret leaked across sessions"


@pytest.mark.asyncio
async def test_chat_latency_p95(async_client: httpx.AsyncClient):
    latencies = []
    for i in range(10):
        payload = {"message": f"Ping {i}", "session_id": f"latency-p95-{i}"}
        start = time.perf_counter()
        r = await async_client.post(CHAT_ENDPOINT, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        latencies.append(elapsed_ms)
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies))]
    assert p95 < 15000, f"p95 latency {p95:.1f}ms too high"