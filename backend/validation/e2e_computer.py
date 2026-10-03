import asyncio
import base64
import time
import pytest
import httpx

BASE_URL = "http://localhost:8000"
COMPUTER_BROWSER_OPEN = f"{BASE_URL}/computer/browser/open"
COMPUTER_BROWSER_NAVIGATE = f"{BASE_URL}/computer/browser/navigate"
COMPUTER_BROWSER_SEARCH = f"{BASE_URL}/computer/browser/search"
COMPUTER_BROWSER_SCREENSHOT = f"{BASE_URL}/computer/browser/screenshot"
COMPUTER_BROWSER_ANALYZE = f"{BASE_URL}/computer/browser/analyze"
COMPUTER_BROWSER_CLOSE = f"{BASE_URL}/computer/browser/close"
COMPUTER_BROWSER_PIPELINE = f"{BASE_URL}/computer/browser/pipeline"


def is_valid_base64_image(data: str) -> bool:
    try:
        if data.startswith("data:image"):
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        return len(decoded) > 100
    except Exception:
        return False


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


@pytest.fixture
async def browser_session(async_client: httpx.AsyncClient):
    open_r = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True})
    assert open_r.status_code in (200, 201), f"Browser open failed: {open_r.text}"
    session_id = open_r.json().get("session_id") or open_r.json().get("browser_id") or open_r.json().get("id")
    yield session_id
    await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": session_id})


@pytest.mark.asyncio
async def test_open_browser(async_client: httpx.AsyncClient):
    response = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True, "browser": "chromium"})
    assert response.status_code in (200, 201), f"Browser open failed: {response.text}"
    data = response.json()
    session_id = data.get("session_id") or data.get("browser_id") or data.get("id")
    assert session_id is not None

    await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": session_id})


@pytest.mark.asyncio
async def test_search_web(async_client: httpx.AsyncClient, browser_session: str):
    payload = {
        "session_id": browser_session,
        "query": "Python programming language",
        "engine": "google",
    }
    response = await async_client.post(COMPUTER_BROWSER_SEARCH, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data or "url" in data or "status" in data


@pytest.mark.asyncio
async def test_navigate_to_url(async_client: httpx.AsyncClient, browser_session: str):
    payload = {"session_id": browser_session, "url": "https://example.com"}
    response = await async_client.post(COMPUTER_BROWSER_NAVIGATE, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "success" or data.get("url") or data.get("title")


@pytest.mark.asyncio
async def test_take_screenshot(async_client: httpx.AsyncClient, browser_session: str):
    await async_client.post(COMPUTER_BROWSER_NAVIGATE, json={
        "session_id": browser_session,
        "url": "https://example.com",
    })
    response = await async_client.post(COMPUTER_BROWSER_SCREENSHOT, json={"session_id": browser_session})
    assert response.status_code == 200

    content_type = response.headers.get("content-type", "")
    if "image" in content_type:
        assert len(response.content) > 500
    else:
        data = response.json()
        screenshot = data.get("screenshot") or data.get("image") or data.get("data") or ""
        if isinstance(screenshot, str) and len(screenshot) > 0:
            assert is_valid_base64_image(screenshot) or screenshot.startswith("http")
        else:
            assert data.get("url") or data.get("path"), f"No screenshot data: {data}"


@pytest.mark.asyncio
async def test_analyze_page(async_client: httpx.AsyncClient, browser_session: str):
    await async_client.post(COMPUTER_BROWSER_NAVIGATE, json={
        "session_id": browser_session,
        "url": "https://example.com",
    })
    response = await async_client.post(COMPUTER_BROWSER_ANALYZE, json={
        "session_id": browser_session,
        "task": "What is the main heading on this page?",
    })
    assert response.status_code == 200
    data = response.json()
    analysis = data.get("analysis") or data.get("description") or data.get("response") or data.get("content") or ""
    assert len(analysis) > 0


@pytest.mark.asyncio
async def test_close_browser(async_client: httpx.AsyncClient):
    open_r = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True})
    assert open_r.status_code in (200, 201)
    session_id = open_r.json().get("session_id") or open_r.json().get("id")

    close_r = await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": session_id})
    assert close_r.status_code in (200, 204)


@pytest.mark.asyncio
async def test_full_browser_pipeline(async_client: httpx.AsyncClient):
    payload = {
        "steps": [
            {"action": "open", "headless": True},
            {"action": "navigate", "url": "https://example.com"},
            {"action": "screenshot"},
            {"action": "analyze", "task": "Describe the page content."},
            {"action": "close"},
        ]
    }
    response = await async_client.post(COMPUTER_BROWSER_PIPELINE, json=payload)
    assert response.status_code == 200
    data = response.json()
    results = data.get("results") or data.get("steps") or []
    assert len(results) > 0


@pytest.mark.asyncio
async def test_search_and_analyze(async_client: httpx.AsyncClient):
    open_r = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True})
    session_id = open_r.json().get("session_id") or open_r.json().get("id")
    try:
        search_r = await async_client.post(COMPUTER_BROWSER_SEARCH, json={
            "session_id": session_id,
            "query": "OpenAI GPT",
        })
        assert search_r.status_code == 200

        screenshot_r = await async_client.post(COMPUTER_BROWSER_SCREENSHOT, json={"session_id": session_id})
        assert screenshot_r.status_code == 200

        analyze_r = await async_client.post(COMPUTER_BROWSER_ANALYZE, json={
            "session_id": session_id,
            "task": "Summarize the search results.",
        })
        assert analyze_r.status_code == 200
    finally:
        await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": session_id})


@pytest.mark.asyncio
async def test_screenshot_latency(async_client: httpx.AsyncClient, browser_session: str):
    await async_client.post(COMPUTER_BROWSER_NAVIGATE, json={
        "session_id": browser_session,
        "url": "https://example.com",
    })
    start = time.perf_counter()
    response = await async_client.post(COMPUTER_BROWSER_SCREENSHOT, json={"session_id": browser_session})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    assert elapsed_ms < 10000, f"Screenshot latency {elapsed_ms:.0f}ms too high"


@pytest.mark.asyncio
async def test_browser_session_isolation(async_client: httpx.AsyncClient):
    r1 = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True})
    r2 = await async_client.post(COMPUTER_BROWSER_OPEN, json={"headless": True})
    s1 = r1.json().get("session_id") or r1.json().get("id")
    s2 = r2.json().get("session_id") or r2.json().get("id")
    assert s1 != s2, "Browser sessions should have unique IDs"
    await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": s1})
    await async_client.post(COMPUTER_BROWSER_CLOSE, json={"session_id": s2})