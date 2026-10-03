"""
test_services.py
Tests for the service layer wrapping core capabilities: LLM, memory,
tool, RAG, and computer-use services.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from conftest import (
    MockLLM,
    MockMemoryStore,
    MockRetriever,
    MockSecurity,
    MockToolRegistry,
    MockVectorDB,
)


class MockComputerService:
    """Minimal service wrapper around sandboxed on-screen actions."""

    def __init__(self, security: MockSecurity) -> None:
        self.security = security
        self.action_log: List[Dict[str, Any]] = []

    async def perform(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        result = self.security.run_in_sandbox(f"{action}:{kwargs}")
        self.action_log.append({"action": action, "kwargs": kwargs, "result": result})
        return result


@pytest.fixture
def computer_service(mock_security: MockSecurity) -> MockComputerService:
    return MockComputerService(mock_security)


@pytest.mark.asyncio
async def test_llm_service_generates_response(mock_llm: MockLLM) -> None:
    response = await mock_llm.generate("summarize this conversation")
    assert response == mock_llm.default_response
    assert mock_llm.calls[-1]["prompt"] == "summarize this conversation"


@pytest.mark.asyncio
async def test_llm_service_streams_tokens(mock_llm: MockLLM) -> None:
    tokens = [token async for token in mock_llm.stream("hello there")]
    assert "".join(tokens).strip() == mock_llm.default_response


@pytest.mark.asyncio
async def test_llm_service_propagates_failure(mock_llm: MockLLM) -> None:
    mock_llm.fail_next = True
    with pytest.raises(RuntimeError):
        await mock_llm.generate("trigger failure")


def test_memory_service_short_and_long_term_isolation(mock_memory: MockMemoryStore) -> None:
    mock_memory.add_short("transient context")
    mock_memory.add_long("durable preference")

    assert len(mock_memory.short_term) == 1
    assert len(mock_memory.long_term) == 1


@pytest.mark.asyncio
async def test_tool_service_executes_registered_tool(mock_tools: MockToolRegistry) -> None:
    result = await mock_tools.call("calculator", expression="10 / 2")
    assert result == 5


@pytest.mark.asyncio
async def test_rag_service_retrieves_relevant_chunks(mock_retriever: MockRetriever, mock_vector_db: MockVectorDB) -> None:
    await mock_vector_db.upsert("doc-a", [0.5, 0.5], {"text": "service level agreement terms"})
    results = await mock_retriever.retrieve("service level agreement")

    assert results
    assert results[0]["metadata"]["text"] == "service level agreement terms"


@pytest.mark.asyncio
async def test_computer_service_runs_action_in_sandbox(computer_service: MockComputerService) -> None:
    result = await computer_service.perform("click", element="submit_button")

    assert result["isolated"] is True
    assert computer_service.action_log[-1]["action"] == "click"


@pytest.mark.asyncio
async def test_computer_service_logs_every_action(computer_service: MockComputerService) -> None:
    await computer_service.perform("type", text="hello")
    await computer_service.perform("scroll", direction="down")

    assert len(computer_service.action_log) == 2


@pytest.mark.asyncio
async def test_services_compose_for_a_combined_workflow(
    mock_llm: MockLLM, mock_memory: MockMemoryStore, mock_tools: MockToolRegistry
) -> None:
    mock_memory.add_long("user prefers metric units")
    calculation = await mock_tools.call("calculator", expression="100 * 1.1")
    summary = await mock_llm.generate(f"explain result {calculation}")

    assert calculation == pytest.approx(110.0)
    assert summary == mock_llm.default_response
    assert mock_memory.retrieve("metric")