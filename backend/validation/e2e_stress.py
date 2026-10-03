import asyncio
import statistics
import time
import psutil
import pytest
import httpx

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=100)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def make_chat_request(client: httpx.AsyncClient, idx: int) -> dict:
    payload = {"message": f"Stress test request number {idx}. What is {idx} + {idx}?", "session_id": f"stress-{idx}"}
    start = time.perf_counter()
    try:
        response = await client.post(CHAT_ENDPOINT, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "index": idx,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "success": response.status_code == 200,
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "index": idx,
            "status_code": None,
            "latency_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def summarize(results: list[dict]) -> dict:
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in successes]
    return {
        "total": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / len(results) if results else 0,
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1] if len(latencies) >= 20 else None,
        "latency_p99_ms": sorted(latencies)[int(0.99 * len(latencies)) - 1] if len(latencies) >= 100 else None,
        "latency_max_ms": max(latencies) if latencies else None,
    }


@pytest.mark.asyncio
async def test_health_endpoint_before_stress():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(HEALTH_ENDPOINT)
        assert r.status_code == 200, "Service unhealthy before stress test"


@pytest.mark.asyncio
async def test_concurrent_users_10():
    N = 10
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        tasks = [make_chat_request(client, i) for i in range(N)]
        results = await asyncio.gather(*tasks)
    summary = summarize(results)
    assert summary["success_rate"] >= 0.9, f"Success rate {summary['success_rate']:.1%} below 90% for {N} users"


@pytest.mark.asyncio
async def test_concurrent_users_50():
    N = 50
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        tasks = [make_chat_request(client, i) for i in range(N)]
        results = await asyncio.gather(*tasks)
    summary = summarize(results)
    assert summary["success_rate"] >= 0.85, f"Success rate {summary['success_rate']:.1%} below 85% for {N} users"


@pytest.mark.asyncio
async def test_100_requests():
    N = 100
    BATCH = 20
    all_results = []
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        for batch_start in range(0, N, BATCH):
            batch = [make_chat_request(client, batch_start + i) for i in range(BATCH)]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
            await asyncio.sleep(0.5)
    summary = summarize(all_results)
    assert summary["successes"] >= 80, f"Only {summary['successes']}/100 requests succeeded"
    assert summary["latency_p95_ms"] is None or summary["latency_p95_ms"] < 30000


@pytest.mark.asyncio
async def test_500_requests():
    N = 500
    BATCH = 25
    all_results = []
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        for batch_start in range(0, N, BATCH):
            batch = [make_chat_request(client, batch_start + i) for i in range(BATCH)]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
            await asyncio.sleep(1.0)
    summary = summarize(all_results)
    assert summary["success_rate"] >= 0.80, f"500-request success rate {summary['success_rate']:.1%} too low"
    print(f"\n[500 requests] mean={summary['latency_mean_ms']:.0f}ms failures={summary['failures']}")


@pytest.mark.asyncio
async def test_1000_requests():
    N = 1000
    BATCH = 20
    all_results = []
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        for batch_start in range(0, N, BATCH):
            batch = [make_chat_request(client, batch_start + i) for i in range(BATCH)]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
            await asyncio.sleep(1.5)
    summary = summarize(all_results)
    assert summary["success_rate"] >= 0.75, f"1000-request success rate {summary['success_rate']:.1%} too low"
    print(f"\n[1000 requests] mean={summary['latency_mean_ms']:.0f}ms failures={summary['failures']}")


@pytest.mark.asyncio
async def test_measure_latency_under_load():
    N = 50
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        tasks = [make_chat_request(client, i) for i in range(N)]
        results = await asyncio.gather(*tasks)
    summary = summarize(results)
    assert summary["latency_mean_ms"] is not None
    assert summary["latency_mean_ms"] < 15000, f"Mean latency {summary['latency_mean_ms']:.0f}ms too high under load"
    print(f"\n[Latency] mean={summary['latency_mean_ms']:.0f}ms p50={summary['latency_p50_ms']:.0f}ms")


@pytest.mark.asyncio
async def test_measure_failure_rate():
    N = 100
    BATCH = 20
    all_results = []
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        for b in range(0, N, BATCH):
            batch = [make_chat_request(client, b + i) for i in range(BATCH)]
            all_results.extend(await asyncio.gather(*batch))
            await asyncio.sleep(0.5)
    failure_rate = sum(1 for r in all_results if not r["success"]) / len(all_results)
    assert failure_rate < 0.20, f"Failure rate {failure_rate:.1%} exceeds 20%"
    print(f"\n[Failures] rate={failure_rate:.1%} count={int(failure_rate * N)}/{N}")


@pytest.mark.asyncio
async def test_measure_memory_usage():
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    N = 50
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        tasks = [make_chat_request(client, i) for i in range(N)]
        await asyncio.gather(*tasks)

    await asyncio.sleep(2)
    mem_after = process.memory_info().rss / 1024 / 1024
    delta_mb = mem_after - mem_before
    print(f"\n[Memory] before={mem_before:.1f}MB after={mem_after:.1f}MB delta={delta_mb:.1f}MB")
    assert delta_mb < 500, f"Memory grew by {delta_mb:.1f}MB during stress — possible leak"


@pytest.mark.asyncio
async def test_health_endpoint_after_stress():
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(HEALTH_ENDPOINT)
        assert r.status_code == 200, "Service unhealthy after stress test"


@pytest.mark.asyncio
async def test_ramp_up_load():
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        for concurrency in [5, 10, 20, 40]:
            tasks = [make_chat_request(client, i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            summary = summarize(results)
            assert summary["success_rate"] >= 0.80, (
                f"At concurrency={concurrency}, success rate {summary['success_rate']:.1%} too low"
            )
            await asyncio.sleep(2)