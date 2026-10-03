"""
test_controller.py
Tests for the Controller orchestration layer: plan creation, execution,
replanning on failure, context tracking, and error propagation.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from conftest import Controller, Executor, Planner, Router


@pytest.mark.asyncio
async def test_handle_creates_and_executes_plan(controller: Controller) -> None:
    result = await controller.handle("draft a summary")

    assert result["objective"] == "draft a summary"
    assert result["results"]
    assert all(r.status in ("success", "failed") for r in result["results"])


@pytest.mark.asyncio
async def test_handle_returns_results_in_dependency_order(controller: Controller) -> None:
    result = await controller.handle("gather data and build the model and ship the report")

    node_ids = [r.node_id for r in result["results"]]
    assert node_ids == sorted(node_ids, key=lambda n: int("".join(filter(str.isdigit, n)) or 0))


@pytest.mark.asyncio
async def test_handle_routes_objective_to_expected_capability(controller: Controller) -> None:
    result = await controller.handle("what is today's latest news")
    assert result["path"] == "web"


@pytest.mark.asyncio
async def test_controller_tracks_context_after_handling(controller: Controller) -> None:
    await controller.handle("remember my preference for dark mode")

    assert controller.context["last_objective"] == "remember my preference for dark mode"
    assert controller.context["last_path"] == "memory"


@pytest.mark.asyncio
async def test_handle_replans_on_node_failure(controller: Controller) -> None:
    def failing_handler() -> None:
        raise RuntimeError("transient network error")

    result = await controller.handle("step one and step two", handlers={"T1": failing_handler})

    failed = [r for r in result["results"] if r.status == "failed"]
    assert failed
    assert failed[0].error == "transient network error"
    assert any(r.node_id == "T1-r" for r in result["results"])
    assert any(r.node_id == "T1-r" and r.status == "success" for r in result["results"])


@pytest.mark.asyncio
async def test_handle_preserves_unaffected_nodes_after_replan(controller: Controller) -> None:
    def failing_handler() -> None:
        raise RuntimeError("boom")

    result = await controller.handle("alpha and beta and gamma", handlers={"T2": failing_handler})

    succeeded_first_pass = [r for r in result["results"] if r.node_id == "T1" and r.status == "success"]
    assert succeeded_first_pass


@pytest.mark.asyncio
async def test_handle_with_no_handlers_succeeds_end_to_end(controller: Controller) -> None:
    result = await controller.handle("plan and execute and review")
    assert all(r.status == "success" for r in result["results"])


@pytest.mark.asyncio
async def test_controller_components_are_independently_wired(
    planner: Planner, executor: Executor, router: Router
) -> None:
    controller = Controller(planner, executor, router)
    result = await controller.handle("calculate the monthly total")
    assert result["path"] == "tool"