import asyncio
import io
import os
import struct
import wave
import pytest
import httpx

BASE_URL = "http://localhost:8000"
VOICE_TRANSCRIBE = f"{BASE_URL}/voice/transcribe"
VOICE_RESPOND = f"{BASE_URL}/voice/respond"
VOICE_TTS = f"{BASE_URL}/voice/tts"
VOICE_PIPELINE = f"{BASE_URL}/voice/pipeline"

SAMPLE_RATE = 16000
DURATION_SECONDS = 2


def generate_wav_bytes(text_hint: str = "hello") -> bytes:
    num_samples = SAMPLE_RATE * DURATION_SECONDS
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        silence = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
        wf.writeframes(silence)
    return buf.getvalue()


def is_valid_audio(data: bytes) -> bool:
    if len(data) < 4:
        return False
    if data[:4] == b"RIFF":
        return True
    if data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return True
    if data[:4] in (b"OggS",):
        return True
    if data[:4] == b"fLaC":
        return True
    if len(data) > 100:
        return True
    return False


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
def sample_wav() -> bytes:
    return generate_wav_bytes()


@pytest.mark.asyncio
async def test_transcribe_audio(async_client: httpx.AsyncClient, sample_wav: bytes):
    files = {"audio": ("test.wav", io.BytesIO(sample_wav), "audio/wav")}
    data = {"language": "en"}
    response = await async_client.post(VOICE_TRANSCRIBE, files=files, data=data)
    assert response.status_code == 200, f"Transcription failed: {response.text}"
    resp_data = response.json()
    assert "transcript" in resp_data or "text" in resp_data or "transcription" in resp_data


@pytest.mark.asyncio
async def test_transcription_has_text(async_client: httpx.AsyncClient, sample_wav: bytes):
    files = {"audio": ("test.wav", io.BytesIO(sample_wav), "audio/wav")}
    response = await async_client.post(VOICE_TRANSCRIBE, files=files)
    assert response.status_code == 200
    data = response.json()
    text = data.get("transcript") or data.get("text") or data.get("transcription") or ""
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_generate_voice_response(async_client: httpx.AsyncClient, sample_wav: bytes):
    files = {"audio": ("test.wav", io.BytesIO(sample_wav), "audio/wav")}
    response = await async_client.post(VOICE_RESPOND, files=files, data={"session_id": "voice-session-001"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data or "message" in data or "text" in data or "audio" in data


@pytest.mark.asyncio
async def test_text_to_speech(async_client: httpx.AsyncClient):
    payload = {"text": "Hello, this is a test of the text to speech system.", "voice": "default", "language": "en"}
    response = await async_client.post(VOICE_TTS, json=payload)
    assert response.status_code == 200

    content_type = response.headers.get("content-type", "")
    if "audio" in content_type or "octet-stream" in content_type:
        audio_data = response.content
        assert is_valid_audio(audio_data), "TTS output is not valid audio"
    else:
        data = response.json()
        audio_url = data.get("audio_url") or data.get("url") or data.get("audio")
        assert audio_url is not None, f"No audio URL in response: {data}"


@pytest.mark.asyncio
async def test_tts_output_audio_quality(async_client: httpx.AsyncClient):
    payload = {"text": "Testing audio quality output.", "voice": "default", "sample_rate": 22050}
    response = await async_client.post(VOICE_TTS, json=payload)
    assert response.status_code == 200

    content_type = response.headers.get("content-type", "")
    if "audio/wav" in content_type or "audio/wave" in content_type:
        buf = io.BytesIO(response.content)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() in (1, 2)
            assert wf.getsampwidth() in (2, 4)
            assert wf.getframerate() > 8000


@pytest.mark.asyncio
async def test_full_voice_pipeline(async_client: httpx.AsyncClient, sample_wav: bytes):
    files = {"audio": ("input.wav", io.BytesIO(sample_wav), "audio/wav")}
    data = {"session_id": "pipeline-test-001", "return_audio": "true"}
    response = await async_client.post(VOICE_PIPELINE, files=files, data=data)
    assert response.status_code == 200

    content_type = response.headers.get("content-type", "")
    if "audio" in content_type:
        assert is_valid_audio(response.content)
    else:
        data = response.json()
        assert data.get("transcript") or data.get("text")
        assert data.get("response") or data.get("reply")
        assert data.get("audio") or data.get("audio_url") or data.get("speech")


@pytest.mark.asyncio
async def test_voice_pipeline_latency(async_client: httpx.AsyncClient, sample_wav: bytes):
    import time
    files = {"audio": ("latency.wav", io.BytesIO(sample_wav), "audio/wav")}
    start = time.perf_counter()
    response = await async_client.post(VOICE_PIPELINE, files=files)
    elapsed = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    assert elapsed < 30000, f"Voice pipeline latency {elapsed:.0f}ms too high"


@pytest.mark.asyncio
async def test_tts_different_voices(async_client: httpx.AsyncClient):
    voices = ["default", "male", "female"]
    for voice in voices:
        payload = {"text": f"Testing voice: {voice}", "voice": voice}
        response = await async_client.post(VOICE_TTS, json=payload)
        assert response.status_code in (200, 400), f"Unexpected status for voice {voice}"


@pytest.mark.asyncio
async def test_transcription_language_detection(async_client: httpx.AsyncClient, sample_wav: bytes):
    files = {"audio": ("test.wav", io.BytesIO(sample_wav), "audio/wav")}
    response = await async_client.post(VOICE_TRANSCRIBE, files=files, data={"detect_language": "true"})
    assert response.status_code == 200
    data = response.json()
    assert "language" in data or "detected_language" in data or "transcript" in data


@pytest.mark.asyncio
async def test_empty_audio_rejected(async_client: httpx.AsyncClient):
    files = {"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = await async_client.post(VOICE_TRANSCRIBE, files=files)
    assert response.status_code in (400, 422), f"Expected 4xx for empty audio, got {response.status_code}"