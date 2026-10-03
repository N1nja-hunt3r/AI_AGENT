"""
test_agents.py
Tests for the agent registry: manager, coder, research, memory,
planner, and reviewer agents.
"""

from __future__ import annotations

import pytest

from conftest import MockAgentRegistry


def test_registry_lists_all_expected_agents(mock_agents: MockAgentRegistry) -> None:
    names = mock_agents.list()
    for expected in ("manager", "coder", "research", "memory", "planner", "reviewer"):
        assert expected in names


def test_get_unknown_agent_raises_key_error(mock_agents: MockAgentRegistry) -> None:
    with pytest.raises(KeyError):
        mock_agents.get("nonexistent")


@pytest.mark.asyncio
async def test_manager_agent_completes_delegated_task(mock_agents: MockAgentRegistry) -> None:
    manager = mock_agents.get("manager")
    result = await manager.run({"id": "T1", "description": "delegate the report task"})

    assert result["agent"] == "manager"
    assert result["status"] == "completed"
    assert manager.status == "idle"
    assert manager.history[-1] == result


@pytest.mark.asyncio
async def test_coder_agent_completes_code_task(mock_agents: MockAgentRegistry) -> None:
    coder = mock_agents.get("coder")
    result = await coder.run({"id": "T2", "description": "implement the parser"})

    assert result["agent"] == "coder"
    assert "implement the parser" in result["output"]


@pytest.mark.asyncio
async def test_research_agent_completes_research_task(mock_agents: MockAgentRegistry) -> None:
    research = mock_agents.get("research")
    result = await research.run({"id": "T3", "description": "investigate competitor pricing"})

    assert result["agent"] == "research"
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_memory_agent_completes_memory_task(mock_agents: MockAgentRegistry) -> None:
    memory_agent = mock_agents.get("memory")
    result = await memory_agent.run({"id": "T4", "description": "compress old session memory"})

    assert result["agent"] == "memory"
    assert result["task_id"] == "T4"


@pytest.mark.asyncio
async def test_planner_agent_completes_planning_task(mock_agents: MockAgentRegistry) -> None:
    planner_agent = mock_agents.get("planner")
    result = await planner_agent.run({"id": "T5", "description": "decompose the migration objective"})

    assert result["agent"] == "planner"
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_reviewer_agent_completes_review_task(mock_agents: MockAgentRegistry) -> None:
    reviewer = mock_agents.get("reviewer")
    result = await reviewer.run({"id": "T6", "description": "review the generated code"})

    assert result["agent"] == "reviewer"
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_status_transitions_back_to_idle_after_run(mock_agents: MockAgentRegistry) -> None:
    coder = mock_agents.get("coder")
    assert coder.status == "idle"
    await coder.run({"id": "T7", "description": "write a test"})
    assert coder.status == "idle"


@pytest.mark.asyncio
async def test_agent_history_accumulates_across_multiple_runs(mock_agents: MockAgentRegistry) -> None:
    reviewer = mock_agents.get("reviewer")
    await reviewer.run({"id": "T8", "description": "first review"})
    await reviewer.run({"id": "T9", "description": "second review"})

    assert len(reviewer.history) == 2