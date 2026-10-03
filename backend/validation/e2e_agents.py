import asyncio
import time
import uuid
import pytest
import httpx

BASE_URL = "http://localhost:8000"
AGENTS_SPAWN = f"{BASE_URL}/agents/spawn"
AGENTS_DELEGATE = f"{BASE_URL}/agents/delegate"
AGENTS_STATUS = f"{BASE_URL}/agents/status"
AGENTS_RESULT = f"{BASE_URL}/agents/result"
AGENTS_ARTIFACTS = f"{BASE_URL}/agents/artifacts"
AGENTS_STOP = f"{BASE_URL}/agents/stop"

POLL_INTERVAL = 2
MAX_WAIT_SECONDS = 120


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


async def wait_for_completion(client: httpx.AsyncClient, agent_id: str, timeout: int = MAX_WAIT_SECONDS) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = await client.get(f"{AGENTS_STATUS}/{agent_id}")
        assert r.status_code == 200
        status = r.json().get("status") or r.json().get("state") or ""
        if status.lower() in ("completed", "done", "finished", "success"):
            return r.json()
        if status.lower() in ("failed", "error", "cancelled"):
            pytest.fail(f"Agent {agent_id} failed with status: {status}")
        await asyncio.sleep(POLL_INTERVAL)
    pytest.fail(f"Agent {agent_id} did not complete within {timeout}s")


@pytest.mark.asyncio
async def test_spawn_single_agent(async_client: httpx.AsyncClient):
    payload = {
        "agent_type": "research",
        "task": "Research the history of Python programming language.",
        "config": {"max_steps": 5, "timeout": 60},
    }
    response = await async_client.post(AGENTS_SPAWN, json=payload)
    assert response.status_code in (200, 201), f"Spawn failed: {response.text}"
    data = response.json()
    assert "agent_id" in data or "id" in data
    agent_id = data.get("agent_id") or data.get("id")
    assert agent_id is not None

    stop_r = await async_client.post(f"{AGENTS_STOP}/{agent_id}")
    assert stop_r.status_code in (200, 204)


@pytest.mark.asyncio
async def test_spawn_multiple_agents(async_client: httpx.AsyncClient):
    tasks = [
        {"agent_type": "summarizer", "task": f"Summarize task {i}"}
        for i in range(3)
    ]
    spawn_tasks = [async_client.post(AGENTS_SPAWN, json=t) for t in tasks]
    responses = await asyncio.gather(*spawn_tasks, return_exceptions=True)
    agent_ids = []
    for r in responses:
        assert not isinstance(r, Exception)
        assert r.status_code in (200, 201)
        aid = r.json().get("agent_id") or r.json().get("id")
        agent_ids.append(aid)
    assert len(set(agent_ids)) == len(agent_ids), "Duplicate agent IDs returned"

    for aid in agent_ids:
        await async_client.post(f"{AGENTS_STOP}/{aid}")


@pytest.mark.asyncio
async def test_delegate_task_to_agent(async_client: httpx.AsyncClient):
    spawn_r = await async_client.post(AGENTS_SPAWN, json={"agent_type": "general", "task": "Idle agent"})
    assert spawn_r.status_code in (200, 201)
    agent_id = spawn_r.json().get("agent_id") or spawn_r.json().get("id")

    delegate_r = await async_client.post(AGENTS_DELEGATE, json={
        "agent_id": agent_id,
        "task": "Calculate the sum of numbers from 1 to 100.",
        "priority": "high",
    })
    assert delegate_r.status_code in (200, 202)
    data = delegate_r.json()
    assert "task_id" in data or "job_id" in data or "status" in data


@pytest.mark.asyncio
async def test_wait_for_agent_completion(async_client: httpx.AsyncClient):
    spawn_r = await async_client.post(AGENTS_SPAWN, json={
        "agent_type": "compute",
        "task": "Compute: what is 15 * 23?",
        "config": {"auto_start": True},
    })
    assert spawn_r.status_code in (200, 201)
    agent_id = spawn_r.json().get("agent_id") or spawn_r.json().get("id")

    final_status = await wait_for_completion(async_client, agent_id)
    assert final_status.get("status") in ("completed", "done", "finished", "success")


@pytest.mark.asyncio
async def test_verify_agent_output(async_client: httpx.AsyncClient):
    spawn_r = await async_client.post(AGENTS_SPAWN, json={
        "agent_type": "qa",
        "task": "Answer: What is the capital of Japan?",
        "config": {"auto_start": True},
    })
    assert spawn_r.status_code in (200, 201)
    agent_id = spawn_r.json().get("agent_id") or spawn_r.json().get("id")

    await wait_for_completion(async_client, agent_id)

    result_r = await async_client.get(f"{AGENTS_RESULT}/{agent_id}")
    assert result_r.status_code == 200
    result = result_r.json()
    output = result.get("output") or result.get("result") or result.get("response") or ""
    assert "tokyo" in str(output).lower(), f"Expected 'Tokyo' in output, got: {output}"


@pytest.mark.asyncio
async def test_verify_agent_artifacts(async_client: httpx.AsyncClient):
    spawn_r = await async_client.post(AGENTS_SPAWN, json={
        "agent_type": "writer",
        "task": "Write a short poem about clouds and save it as artifact.",
        "config": {"auto_start": True, "save_artifacts": True},
    })
    assert spawn_r.status_code in (200, 201)
    agent_id = spawn_r.json().get("agent_id") or spawn_r.json().get("id")

    await wait_for_completion(async_client, agent_id)

    artifacts_r = await async_client.get(f"{AGENTS_ARTIFACTS}/{agent_id}")
    assert artifacts_r.status_code == 200
    artifacts = artifacts_r.json().get("artifacts") or []
    assert len(artifacts) > 0, "No artifacts produced by agent"
    for artifact in artifacts:
        assert "content" in artifact or "url" in artifact or "path" in artifact
        artifact_type = artifact.get("type") or artifact.get("mime_type") or ""
        assert artifact_type != "" or "content" in artifact


@pytest.mark.asyncio
async def test_agent_status_transitions(async_client: httpx.AsyncClient):
    spawn_r = await async_client.post(AGENTS_SPAWN, json={
        "agent_type": "general",
        "task": "Simple task for status tracking.",
        "config": {"auto_start": True},
    })
    agent_id = spawn_r.json().get("agent_id") or spawn_r.json().get("id")

    statuses_seen = set()
    for _ in range(10):
        status_r = await async_client.get(f"{AGENTS_STATUS}/{agent_id}")
        current = status_r.json().get("status") or ""
        statuses_seen.add(current.lower())
        if current.lower() in ("completed", "done", "failed", "error"):
            break
        await asyncio.sleep(1)

    valid_statuses = {"pending", "running", "started", "processing", "completed", "done", "failed", "error", "finished"}
    assert statuses_seen.issubset(valid_statuses), f"Unexpected statuses: {statuses_seen - valid_statuses}"


@pytest.mark.asyncio
async def test_agent_parallel_execution(async_client: httpx.AsyncClient):
    spawn_tasks = [
        async_client.post(AGENTS_SPAWN, json={
            "agent_type": "compute",
            "task": f"Task {i}: Return the number {i * 10}",
            "config": {"auto_start": True},
        })
        for i in range(5)
    ]
    responses = await asyncio.gather(*spawn_tasks)
    agent_ids = [r.json().get("agent_id") or r.json().get("id") for r in responses]

    completion_tasks = [wait_for_completion(async_client, aid) for aid in agent_ids]
    results = await asyncio.gather(*completion_tasks, return_exceptions=True)
    successful = [r for r in results if not isinstance(r, Exception)]
    assert len(successful) >= 4, f"Too many agents failed: {len(agent_ids) - len(successful)} failures"