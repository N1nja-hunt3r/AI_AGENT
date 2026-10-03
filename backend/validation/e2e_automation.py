import asyncio
import time
import uuid
import pytest
import httpx
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8000"
AUTOMATION_CREATE = f"{BASE_URL}/automation/tasks"
AUTOMATION_SCHEDULE = f"{BASE_URL}/automation/schedule"
AUTOMATION_EXECUTE = f"{BASE_URL}/automation/execute"
AUTOMATION_STATUS = f"{BASE_URL}/automation/tasks"
AUTOMATION_NOTIFY = f"{BASE_URL}/automation/notifications"
AUTOMATION_COMPLETE = f"{BASE_URL}/automation/complete"

POLL_INTERVAL = 2
MAX_WAIT = 90


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


def future_timestamp(seconds: int = 60) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(seconds=seconds)).isoformat()


async def poll_task(client: httpx.AsyncClient, task_id: str, timeout: int = MAX_WAIT) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = await client.get(f"{AUTOMATION_STATUS}/{task_id}")
        assert r.status_code == 200
        data = r.json()
        status = (data.get("status") or data.get("state") or "").lower()
        if status in ("completed", "done", "finished", "success"):
            return data
        if status in ("failed", "error", "cancelled"):
            pytest.fail(f"Task {task_id} failed: {status}")
        await asyncio.sleep(POLL_INTERVAL)
    pytest.fail(f"Task {task_id} timed out after {timeout}s")


@pytest.mark.asyncio
async def test_create_task(async_client: httpx.AsyncClient):
    payload = {
        "name": "E2E Test Task",
        "type": "compute",
        "action": "echo",
        "params": {"message": "Hello from automation"},
        "description": "Created by e2e test",
    }
    response = await async_client.post(AUTOMATION_CREATE, json=payload)
    assert response.status_code in (200, 201), f"Task creation failed: {response.text}"
    data = response.json()
    task_id = data.get("task_id") or data.get("id")
    assert task_id is not None


@pytest.mark.asyncio
async def test_schedule_task(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Scheduled Task",
        "type": "compute",
        "action": "echo",
        "params": {"message": "Scheduled execution"},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")

    schedule_r = await async_client.post(AUTOMATION_SCHEDULE, json={
        "task_id": task_id,
        "scheduled_at": future_timestamp(10),
        "timezone": "UTC",
    })
    assert schedule_r.status_code in (200, 201)
    data = schedule_r.json()
    assert data.get("scheduled") or data.get("schedule_id") or data.get("status") == "scheduled"


@pytest.mark.asyncio
async def test_execute_task(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Execute Task",
        "type": "compute",
        "action": "echo",
        "params": {"message": "Immediate execution"},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")

    execute_r = await async_client.post(f"{AUTOMATION_EXECUTE}/{task_id}")
    assert execute_r.status_code in (200, 202)
    data = execute_r.json()
    assert data.get("started") or data.get("status") or data.get("execution_id")


@pytest.mark.asyncio
async def test_notification_sent_on_completion(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Notification Task",
        "type": "compute",
        "action": "echo",
        "params": {"message": "Notify on complete"},
        "notifications": {"on_complete": True, "channel": "webhook"},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")
    await async_client.post(f"{AUTOMATION_EXECUTE}/{task_id}")
    await poll_task(async_client, task_id)

    notify_r = await async_client.get(f"{AUTOMATION_NOTIFY}?task_id={task_id}")
    assert notify_r.status_code == 200
    notifications = notify_r.json().get("notifications") or notify_r.json().get("data") or []
    assert len(notifications) > 0


@pytest.mark.asyncio
async def test_verify_task_completion(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Completion Verify Task",
        "type": "compute",
        "action": "compute_sum",
        "params": {"numbers": [1, 2, 3, 4, 5]},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")
    await async_client.post(f"{AUTOMATION_EXECUTE}/{task_id}")
    final = await poll_task(async_client, task_id)

    assert final.get("status") in ("completed", "done", "finished", "success")
    result = final.get("result") or final.get("output") or {}
    assert result is not None


@pytest.mark.asyncio
async def test_task_status_lifecycle(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Lifecycle Task",
        "type": "compute",
        "action": "sleep",
        "params": {"seconds": 2},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")

    status_r = await async_client.get(f"{AUTOMATION_STATUS}/{task_id}")
    initial_status = status_r.json().get("status") or ""
    assert initial_status.lower() in ("pending", "created", "queued", "idle")

    await async_client.post(f"{AUTOMATION_EXECUTE}/{task_id}")
    await poll_task(async_client, task_id)

    final_r = await async_client.get(f"{AUTOMATION_STATUS}/{task_id}")
    final_status = final_r.json().get("status") or ""
    assert final_status.lower() in ("completed", "done", "finished", "success")


@pytest.mark.asyncio
async def test_concurrent_task_creation(async_client: httpx.AsyncClient):
    tasks = [
        async_client.post(AUTOMATION_CREATE, json={
            "name": f"Concurrent Task {i}",
            "type": "compute",
            "action": "echo",
            "params": {"index": i},
        })
        for i in range(5)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    ids = []
    for r in responses:
        assert not isinstance(r, Exception)
        assert r.status_code in (200, 201)
        tid = r.json().get("task_id") or r.json().get("id")
        ids.append(tid)
    assert len(set(ids)) == len(ids), "Duplicate task IDs"


@pytest.mark.asyncio
async def test_cron_schedule(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Cron Task",
        "type": "recurring",
        "action": "health_check",
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")

    cron_r = await async_client.post(AUTOMATION_SCHEDULE, json={
        "task_id": task_id,
        "cron": "*/5 * * * *",
        "timezone": "UTC",
    })
    assert cron_r.status_code in (200, 201)
    data = cron_r.json()
    assert data.get("cron") or data.get("schedule") or data.get("next_run")


@pytest.mark.asyncio
async def test_task_cancellation(async_client: httpx.AsyncClient):
    create_r = await async_client.post(AUTOMATION_CREATE, json={
        "name": "Cancel Task",
        "type": "compute",
        "action": "sleep",
        "params": {"seconds": 30},
    })
    task_id = create_r.json().get("task_id") or create_r.json().get("id")
    await async_client.post(f"{AUTOMATION_EXECUTE}/{task_id}")
    await asyncio.sleep(1)

    cancel_r = await async_client.delete(f"{AUTOMATION_STATUS}/{task_id}")
    assert cancel_r.status_code in (200, 202, 204)

    status_r = await async_client.get(f"{AUTOMATION_STATUS}/{task_id}")
    status = status_r.json().get("status") or ""
    assert status.lower() in ("cancelled", "canceled", "stopped", "aborted")