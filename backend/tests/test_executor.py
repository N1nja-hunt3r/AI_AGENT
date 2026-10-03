"""
test_executor.py
Tests for the Executor: execute(), retry(), rollback(), artifact tracking,
and approval requests.
"""

from __future__ import annotations

import pytest

from conftest import Executor, MockSecurity, Plan, PlanNode


@pytest.mark.asyncio
async def test_execute_runs_nodes_in_dependency_order(executor: Executor) -> None:
    plan = Plan(
        objective="x",
        nodes=[
            PlanNode(id="A", description="first"),
            PlanNode(id="B", description="second", depends_on=["A"]),
        ],
    )
    results = await executor.execute(plan)

    assert [r.node_id for r in results] == ["A", "B"]
    assert all(r.status == "success" for r in results)
    assert executor.artifacts["A"] == "first"


@pytest.mark.asyncio
async def test_execute_records_artifact_per_node(executor: Executor) -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="generate report")])
    await executor.execute(plan)
    assert executor.artifacts["A"] == "generate report"


@pytest.mark.asyncio
async def test_execute_falls_back_when_handler_fails(executor: Executor) -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="risky", fallback="use default")])

    def boom() -> None:
        raise RuntimeError("network error")

    results = await executor.execute(plan, handlers={"A": boom})

    assert results[0].status == "failed"
    assert results[0].error == "network error"
    assert executor.artifacts["A"] == "use default"


@pytest.mark.asyncio
async def test_execute_without_fallback_leaves_artifact_unset(executor: Executor) -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="risky")])

    def boom() -> None:
        raise RuntimeError("no fallback configured")

    await executor.execute(plan, handlers={"A": boom})
    assert "A" not in executor.artifacts


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures(executor: Executor) -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise TimeoutError("transient")
        return "recovered"

    result = await executor.retry(flaky)
    assert result == "recovered"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_retry_exhausts_attempts_and_raises(executor: Executor) -> None:
    def always_fails() -> None:
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError):
        await executor.retry(always_fails)


@pytest.mark.asyncio
async def test_rollback_removes_artifact_and_records_it(executor: Executor) -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="write file")])
    await executor.execute(plan)
    assert "A" in executor.artifacts

    rolled_back = await executor.rollback("A")
    assert rolled_back is True
    assert "A" not in executor.artifacts
    assert "A" in executor.rolled_back


@pytest.mark.asyncio
async def test_rollback_on_missing_artifact_returns_false(executor: Executor) -> None:
    assert await executor.rollback("missing") is False


@pytest.mark.asyncio
async def test_request_approval_creates_pending_record(executor: Executor, mock_security: MockSecurity) -> None:
    approval_id = await executor.request_approval("delete production data")

    assert approval_id in mock_security.pending_approvals
    assert mock_security.is_approved(approval_id) is False

    mock_security.approve(approval_id)
    assert mock_security.is_approved(approval_id) is True