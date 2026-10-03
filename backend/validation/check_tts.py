import asyncio
import io
import os
import platform
import subprocess
import time
import wave
import httpx

MAGPIE_URL = os.getenv("MAGPIE_URL", "http://localhost:8000")
TTS_ENDPOINT = f"{MAGPIE_URL}/voice/tts"
MAGPIE_WS_URL = os.getenv("MAGPIE_WS_URL", "ws://localhost:8000/voice/stream")
OUTPUT_WAV = "check_tts_output.wav"

TEST_TEXT = "Hello! This is a Magpie TTS connection test. Audio generation is working correctly."


async def connect_magpie_http() -> httpx.AsyncClient:
    client = httpx.AsyncClient(timeout=60.0)
    health_url = f"{MAGPIE_URL}/health"
    try:
        r = await client.get(health_url)
        if r.status_code == 200:
            print(f"[OK] Magpie connected at {MAGPIE_URL}")
        else:
            print(f"[WARN] Magpie health returned {r.status_code}")
    except Exception as e:
        print(f"[WARN] Could not reach Magpie at {MAGPIE_URL}: {e}")
    return client


def save_wav(audio_bytes: bytes, path: str) -> bool:
    if audio_bytes[:4] == b"RIFF":
        with open(path, "wb") as f:
            f.write(audio_bytes)
        print(f"[OK] WAV saved to {path} ({len(audio_bytes):,} bytes)")
        return True

    buf = io.BytesIO()
    sample_rate = 22050
    num_channels = 1
    sample_width = 2
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    wav_bytes = buf.getvalue()
    with open(path, "wb") as f:
        f.write(wav_bytes)
    print(f"[OK] WAV wrapped and saved to {path} ({len(wav_bytes):,} bytes)")
    return True


def play_audio(path: str) -> bool:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path], timeout=30, check=True)
            print(f"[OK] Audio played on macOS: {path}")
            return True
        elif system == "Linux":
            for player in ("aplay", "paplay", "mpg123", "ffplay"):
                if subprocess.run(["which", player], capture_output=True).returncode == 0:
                    args = [player, path] if player != "ffplay" else ["ffplay", "-nodisp", "-autoexit", path]
                    subprocess.run(args, timeout=30, check=True)
                    print(f"[OK] Audio played with {player}: {path}")
                    return True
            print("[WARN] No audio player found on Linux. Skipping playback.")
            return False
        elif system == "Windows":
            subprocess.run(["start", path], shell=True, timeout=30, check=True)
            print(f"[OK] Audio played on Windows: {path}")
            return True
        else:
            print(f"[WARN] Unknown OS: {system}. Skipping playback.")
            return False
    except subprocess.TimeoutExpired:
        print("[WARN] Audio playback timed out.")
        return False
    except Exception as e:
        print(f"[WARN] Audio playback failed: {e}")
        return False


async def generate_speech_http(client: httpx.AsyncClient) -> tuple[bytes, float]:
    payload = {
        "text": TEST_TEXT,
        "voice": "default",
        "language": "en",
        "sample_rate": 22050,
        "format": "wav",
    }
    print(f"[...] Generating speech via {TTS_ENDPOINT}")
    start = time.perf_counter()
    response = await client.post(TTS_ENDPOINT, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise RuntimeError(f"TTS request failed with status {response.status_code}: {response.text}")

    content_type = response.headers.get("content-type", "")
    if "audio" in content_type or "octet-stream" in content_type:
        audio_bytes = response.content
    else:
        data = response.json()
        audio_url = data.get("audio_url") or data.get("url")
        if audio_url:
            print(f"[...] Fetching audio from URL: {audio_url}")
            fetch_r = await client.get(audio_url)
            audio_bytes = fetch_r.content
        elif data.get("audio"):
            import base64
            audio_bytes = base64.b64decode(data["audio"])
        else:
            raise RuntimeError(f"No audio data in response: {data}")

    return audio_bytes, latency_ms


async def run_tts_check():
    print("=" * 60)
    print("Magpie TTS Connection Check")
    print("=" * 60)

    client = await connect_magpie_http()
    results = {"connected": True, "generated": False, "saved": False, "played": False, "latency_ms": None}

    try:
        audio_bytes, latency_ms = await generate_speech_http(client)
        results["generated"] = True
        results["latency_ms"] = latency_ms
        print(f"[OK] Speech generated: {len(audio_bytes):,} bytes in {latency_ms:.0f}ms")

        if latency_ms > 10000:
            print(f"[WARN] Latency {latency_ms:.0f}ms exceeds 10000ms threshold")
        else:
            print(f"[OK] Latency within threshold: {latency_ms:.0f}ms")

        results["saved"] = save_wav(audio_bytes, OUTPUT_WAV)

        if results["saved"] and os.path.exists(OUTPUT_WAV):
            file_size = os.path.getsize(OUTPUT_WAV)
            assert file_size > 100, f"Output WAV too small: {file_size} bytes"
            print(f"[OK] WAV file verified: {file_size:,} bytes")

            try:
                with wave.open(OUTPUT_WAV, "rb") as wf:
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    frames = wf.getnframes()
                    duration = frames / rate
                    print(f"[OK] WAV info: {channels}ch, {rate}Hz, {duration:.2f}s")
            except Exception as e:
                print(f"[WARN] Could not read WAV properties: {e}")

            results["played"] = play_audio(OUTPUT_WAV)

    except Exception as e:
        print(f"[ERROR] TTS check failed: {e}")
        results["error"] = str(e)
    finally:
        await client.aclose()

    print("\n" + "=" * 60)
    print("Results:")
    for k, v in results.items():
        status = "[OK]" if v else "[FAIL]" if v is False else "[--]"
        print(f"  {status} {k}: {v}")
    print("=" * 60)

    assert results["connected"], "Could not connect to Magpie"
    assert results["generated"], "Speech generation failed"
    assert results["saved"], "WAV file save failed"
    assert results["latency_ms"] is not None and results["latency_ms"] < 15000, (
        f"Latency {results.get('latency_ms')}ms out of range"
    )
    print("\n[PASS] All TTS checks passed!")
    return results


if __name__ == "__main__":
    asyncio.run(run_tts_check())