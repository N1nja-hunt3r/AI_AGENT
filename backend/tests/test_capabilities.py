"""
test_capabilities.py
Tests for the capability registry: memory, tool, rag, computer, and
web capabilities.
"""

from __future__ import annotations

import pytest

from conftest import MockCapabilityRegistry


def test_registry_lists_all_expected_capabilities(mock_capabilities: MockCapabilityRegistry) -> None:
    names = mock_capabilities.list()
    for expected in ("memory", "tool", "rag", "computer", "web"):
        assert expected in names


def test_get_unknown_capability_raises_key_error(mock_capabilities: MockCapabilityRegistry) -> None:
    with pytest.raises(KeyError):
        mock_capabilities.get("nonexistent")


@pytest.mark.asyncio
async def test_memory_capability_invokes_handler(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("memory")
    result = await capability.invoke(query="what do you remember about me")

    assert result["status"] == "ok"
    assert result["source"] == "memory"
    assert capability.invocations == 1


@pytest.mark.asyncio
async def test_tool_capability_invokes_handler(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("tool")
    result = await capability.invoke(action="calculate")

    assert result["source"] == "tool"


@pytest.mark.asyncio
async def test_rag_capability_invokes_handler(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("rag")
    result = await capability.invoke(query="what does the contract say")

    assert result["source"] == "rag"


@pytest.mark.asyncio
async def test_computer_capability_invokes_handler(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("computer")
    result = await capability.invoke(action="click_submit")

    assert result["source"] == "computer"


@pytest.mark.asyncio
async def test_web_capability_invokes_handler(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("web")
    result = await capability.invoke(query="latest stock price")

    assert result["source"] == "web"


@pytest.mark.asyncio
async def test_capability_invocation_count_increments_per_call(mock_capabilities: MockCapabilityRegistry) -> None:
    capability = mock_capabilities.get("web")
    await capability.invoke(query="first call")
    await capability.invoke(query="second call")

    assert capability.invocations == 2


@pytest.mark.asyncio
async def test_custom_capability_can_be_registered_and_invoked(mock_capabilities: MockCapabilityRegistry) -> None:
    mock_capabilities.register("custom", lambda **kw: {"status": "ok", "source": "custom"})
    capability = mock_capabilities.get("custom")
    result = await capability.invoke()

    assert result["source"] == "custom"
    assert "custom" in mock_capabilities.list()