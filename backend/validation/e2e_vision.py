import asyncio
import base64
import io
import struct
import zlib
import pytest
import httpx

BASE_URL = "http://localhost:8000"
VISION_DESCRIBE = f"{BASE_URL}/vision/describe"
VISION_OCR = f"{BASE_URL}/vision/ocr"
VISION_BROWSER = f"{BASE_URL}/vision/browser"
VISION_DESKTOP = f"{BASE_URL}/vision/desktop"
VISION_UPLOAD = f"{BASE_URL}/vision/upload"


def generate_png_bytes(width: int = 100, height: int = 100) -> bytes:
    def make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_png_chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    for _ in range(height):
        row = b"\x00" + b"\xFF\x00\x00" * width
        raw_rows += row
    compressed = zlib.compress(raw_rows)
    idat = make_png_chunk(b"IDAT", compressed)
    iend = make_png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.fixture(scope="session")
def sample_png() -> bytes:
    return generate_png_bytes()


@pytest.mark.asyncio
async def test_upload_image(async_client: httpx.AsyncClient, sample_png: bytes):
    files = {"image": ("test.png", io.BytesIO(sample_png), "image/png")}
    response = await async_client.post(VISION_UPLOAD, files=files)
    assert response.status_code in (200, 201), f"Image upload failed: {response.text}"
    data = response.json()
    assert "image_id" in data or "id" in data or "url" in data or "status" in data


@pytest.mark.asyncio
async def test_describe_image(async_client: httpx.AsyncClient, sample_png: bytes):
    files = {"image": ("test.png", io.BytesIO(sample_png), "image/png")}
    response = await async_client.post(VISION_DESCRIBE, files=files)
    assert response.status_code == 200
    data = response.json()
    description = data.get("description") or data.get("caption") or data.get("content") or data.get("response") or ""
    assert len(description) > 0, "Image description is empty"


@pytest.mark.asyncio
async def test_describe_image_base64(async_client: httpx.AsyncClient, sample_png: bytes):
    b64 = base64.b64encode(sample_png).decode()
    payload = {"image_base64": b64, "mime_type": "image/png"}
    response = await async_client.post(VISION_DESCRIBE, json=payload)
    assert response.status_code == 200
    data = response.json()
    description = data.get("description") or data.get("content") or data.get("response") or ""
    assert len(description) > 0


@pytest.mark.asyncio
async def test_ocr_image(async_client: httpx.AsyncClient, sample_png: bytes):
    files = {"image": ("ocr_test.png", io.BytesIO(sample_png), "image/png")}
    data = {"mode": "ocr"}
    response = await async_client.post(VISION_OCR, files=files, data=data)
    assert response.status_code == 200
    resp_data = response.json()
    assert "text" in resp_data or "ocr_result" in resp_data or "content" in resp_data


@pytest.mark.asyncio
async def test_ocr_returns_string(async_client: httpx.AsyncClient, sample_png: bytes):
    files = {"image": ("ocr_test.png", io.BytesIO(sample_png), "image/png")}
    response = await async_client.post(VISION_OCR, files=files)
    assert response.status_code == 200
    text = response.json().get("text") or response.json().get("ocr_result") or response.json().get("content") or ""
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_browser_analysis(async_client: httpx.AsyncClient):
    payload = {
        "url": "https://example.com",
        "task": "Describe what you see on this webpage.",
        "capture_screenshot": True,
    }
    response = await async_client.post(VISION_BROWSER, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "description" in data or "analysis" in data or "content" in data or "screenshot" in data


@pytest.mark.asyncio
async def test_browser_screenshot_captured(async_client: httpx.AsyncClient):
    payload = {"url": "https://example.com", "capture_screenshot": True}
    response = await async_client.post(VISION_BROWSER, json=payload)
    assert response.status_code == 200
    data = response.json()
    screenshot = data.get("screenshot") or data.get("screenshot_url") or data.get("image")
    assert screenshot is not None, "No screenshot in browser analysis response"


@pytest.mark.asyncio
async def test_desktop_analysis(async_client: httpx.AsyncClient):
    payload = {"task": "Capture and describe the current desktop state."}
    response = await async_client.post(VISION_DESKTOP, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "description" in data or "analysis" in data or "screenshot" in data or "content" in data


@pytest.mark.asyncio
async def test_vision_describe_with_prompt(async_client: httpx.AsyncClient, sample_png: bytes):
    files = {"image": ("test.png", io.BytesIO(sample_png), "image/png")}
    data = {"prompt": "What colors do you see in this image?"}
    response = await async_client.post(VISION_DESCRIBE, files=files, data=data)
    assert response.status_code == 200
    content = response.json().get("description") or response.json().get("response") or ""
    assert len(content) > 0


@pytest.mark.asyncio
async def test_invalid_image_rejected(async_client: httpx.AsyncClient):
    files = {"image": ("bad.png", io.BytesIO(b"not an image"), "image/png")}
    response = await async_client.post(VISION_DESCRIBE, files=files)
    assert response.status_code in (400, 422, 500), f"Expected error for invalid image, got {response.status_code}"


@pytest.mark.asyncio
async def test_vision_concurrent_requests(async_client: httpx.AsyncClient, sample_png: bytes):
    tasks = [
        async_client.post(VISION_DESCRIBE, files={"image": ("t.png", io.BytesIO(sample_png), "image/png")})
        for _ in range(3)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for r in responses:
        assert not isinstance(r, Exception)
        assert r.status_code == 200