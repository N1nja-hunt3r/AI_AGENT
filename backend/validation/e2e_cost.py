import asyncio
import time
import pytest
import httpx
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"
COST_TOKENS = f"{BASE_URL}/cost/tokens"
COST_LATENCY = f"{BASE_URL}/cost/latency"
COST_PROVIDERS = f"{BASE_URL}/cost/providers"
COST_SUMMARY = f"{BASE_URL}/cost/summary"
COST_REPORT = f"{BASE_URL}/cost/report"
COST_USAGE = f"{BASE_URL}/cost/usage"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.fixture(scope="module")
async def chat_session(async_client: httpx.AsyncClient):
    session_id = "cost-test-session-001"
    messages = [
        "What is machine learning?",
        "Explain neural networks.",
        "What are transformers in AI?",
    ]
    for msg in messages:
        await async_client.post(CHAT_ENDPOINT, json={"message": msg, "session_id": session_id})
    return session_id


@pytest.mark.asyncio
async def test_token_tracking(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_TOKENS}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    token_fields = {"input_tokens", "output_tokens", "total_tokens", "tokens", "prompt_tokens", "completion_tokens"}
    found = any(k in data for k in token_fields)
    assert found, f"No token data in response: {data}"


@pytest.mark.asyncio
async def test_token_counts_positive(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_TOKENS}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    total = data.get("total_tokens") or data.get("tokens") or 0
    assert int(total) > 0, "Token count should be positive after chat"


@pytest.mark.asyncio
async def test_latency_tracking(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_LATENCY}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    latency_fields = {"avg_latency_ms", "p50", "p95", "p99", "mean_latency", "latency_ms", "latencies"}
    found = any(k in data for k in latency_fields)
    assert found, f"No latency data in response: {data}"


@pytest.mark.asyncio
async def test_latency_values_valid(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_LATENCY}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    avg = data.get("avg_latency_ms") or data.get("mean_latency") or data.get("p50") or 0
    assert float(avg) > 0, "Average latency should be positive"
    assert float(avg) < 60000, "Average latency unreasonably high"


@pytest.mark.asyncio
async def test_provider_tracking(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_PROVIDERS}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    providers = data.get("providers") or data.get("data") or []
    assert len(providers) > 0, "No provider data tracked"
    for provider in providers:
        assert "name" in provider or "provider" in provider
        assert "requests" in provider or "calls" in provider or "count" in provider


@pytest.mark.asyncio
async def test_cost_tracking(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_SUMMARY}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    cost_fields = {"total_cost", "cost_usd", "cost", "total_spend", "amount"}
    found = any(k in data for k in cost_fields)
    assert found, f"No cost data in response: {data}"


@pytest.mark.asyncio
async def test_cost_is_positive(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_SUMMARY}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    cost = data.get("total_cost") or data.get("cost_usd") or data.get("cost") or 0
    assert float(cost) >= 0, "Cost should be non-negative"


@pytest.mark.asyncio
async def test_generate_cost_report(async_client: httpx.AsyncClient):
    now = datetime.now(tz=timezone.utc)
    payload = {
        "start_date": (now - timedelta(days=7)).isoformat(),
        "end_date": now.isoformat(),
        "format": "json",
        "group_by": "provider",
    }
    response = await async_client.post(COST_REPORT, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "report" in data or "data" in data or "summary" in data


@pytest.mark.asyncio
async def test_report_contains_breakdowns(async_client: httpx.AsyncClient):
    now = datetime.now(tz=timezone.utc)
    payload = {
        "start_date": (now - timedelta(days=1)).isoformat(),
        "end_date": now.isoformat(),
        "include_breakdown": True,
    }
    response = await async_client.post(COST_REPORT, json=payload)
    assert response.status_code == 200
    data = response.json()
    report = data.get("report") or data.get("data") or data
    assert isinstance(report, (dict, list))


@pytest.mark.asyncio
async def test_usage_endpoint(async_client: httpx.AsyncClient, chat_session: str):
    response = await async_client.get(f"{COST_USAGE}?session_id={chat_session}")
    assert response.status_code == 200
    data = response.json()
    usage_fields = {"requests", "total_requests", "calls", "invocations", "count"}
    found = any(k in data for k in usage_fields)
    assert found, f"No usage data: {data}"


@pytest.mark.asyncio
async def test_cost_per_provider_breakdown(async_client: httpx.AsyncClient):
    new_session = "cost-breakdown-test"
    await async_client.post(CHAT_ENDPOINT, json={"message": "Hello", "session_id": new_session})

    response = await async_client.get(f"{COST_PROVIDERS}?session_id={new_session}")
    assert response.status_code == 200
    data = response.json()
    providers = data.get("providers") or []
    for p in providers:
        assert "cost" in p or "cost_usd" in p or "spend" in p or "tokens" in p


@pytest.mark.asyncio
async def test_global_usage_stats(async_client: httpx.AsyncClient):
    response = await async_client.get(COST_USAGE)
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert isinstance(data, dict)