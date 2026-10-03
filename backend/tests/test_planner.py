"""
test_planner.py
Tests for the Planner: goal decomposition, dependency graph correctness,
fallback assignment, and replanning after a node failure.
"""

from __future__ import annotations

import pytest

from conftest import Plan, PlanNode, Planner


@pytest.mark.asyncio
async def test_decompose_creates_sequential_dependency_chain(planner: Planner) -> None:
    plan = await planner.decompose("collect requirements and write the report and send it")

    assert plan.objective.startswith("collect requirements")
    assert len(plan.nodes) >= 2
    assert plan.nodes[0].depends_on == []
    assert plan.nodes[1].depends_on == [plan.nodes[0].id]


@pytest.mark.asyncio
async def test_decompose_respects_max_subtasks(planner: Planner) -> None:
    objective = " and ".join(f"step {i}" for i in range(10))
    plan = await planner.decompose(objective, max_subtasks=3)

    assert len(plan.nodes) == 3


@pytest.mark.asyncio
async def test_decompose_single_step_objective_yields_one_node(planner: Planner) -> None:
    plan = await planner.decompose("write a single function")
    assert len(plan.nodes) == 1
    assert plan.nodes[0].depends_on == []


def test_topological_order_resolves_linear_chain() -> None:
    plan = Plan(
        objective="x",
        nodes=[
            PlanNode(id="A", description="a"),
            PlanNode(id="B", description="b", depends_on=["A"]),
            PlanNode(id="C", description="c", depends_on=["B"]),
        ],
    )
    assert plan.topological_order() == ["A", "B", "C"]


def test_topological_order_detects_circular_dependency() -> None:
    plan = Plan(
        objective="x",
        nodes=[
            PlanNode(id="A", description="a", depends_on=["B"]),
            PlanNode(id="B", description="b", depends_on=["A"]),
        ],
    )
    with pytest.raises(ValueError):
        plan.topological_order()


def test_add_fallback_sets_fallback_on_node(planner: Planner) -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="a")])
    planner.add_fallback(plan, "A", "use cached result")
    assert plan.node("A").fallback == "use cached result"


def test_plan_node_lookup_raises_for_unknown_id() -> None:
    plan = Plan(objective="x", nodes=[PlanNode(id="A", description="a")])
    with pytest.raises(KeyError):
        plan.node("Z")


@pytest.mark.asyncio
async def test_replan_replaces_failed_node_with_retry_node(planner: Planner) -> None:
    plan = await planner.decompose("a and b and c")
    replanned = await planner.replan(plan, "T1", "timeout")

    ids = [n.id for n in replanned.nodes]
    assert "T1" not in ids
    assert "T1-r" in ids


@pytest.mark.asyncio
async def test_replan_rewires_dependents_to_retry_node(planner: Planner) -> None:
    plan = await planner.decompose("a and b and c")
    replanned = await planner.replan(plan, "T1", "timeout")

    dependent = replanned.node("T2")
    assert "T1-r" in dependent.depends_on


@pytest.mark.asyncio
async def test_replan_preserves_fallback_from_original_node(planner: Planner) -> None:
    plan = await planner.decompose("a and b")
    planner.add_fallback(plan, "T1", "use default value")

    replanned = await planner.replan(plan, "T1", "service unavailable")
    assert replanned.node("T1-r").fallback == "use default value"