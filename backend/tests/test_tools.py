"""
test_tools.py
Tests for the tool registry: calculator, weather, terminal, and search tools.
"""

from __future__ import annotations

import pytest

from conftest import MockToolRegistry


@pytest.mark.asyncio
async def test_calculator_tool_evaluates_expression(mock_tools: MockToolRegistry) -> None:
    result = await mock_tools.call("calculator", expression="2 + 3 * 4")
    assert result == 14


@pytest.mark.asyncio
async def test_calculator_tool_rejects_unsafe_input(mock_tools: MockToolRegistry) -> None:
    with pytest.raises(ValueError):
        await mock_tools.call("calculator", expression="__import__('os')")


@pytest.mark.asyncio
async def test_calculator_tool_rejects_empty_expression(mock_tools: MockToolRegistry) -> None:
    with pytest.raises(ValueError):
        await mock_tools.call("calculator", expression="")


@pytest.mark.asyncio
async def test_weather_tool_returns_location_data(mock_tools: MockToolRegistry) -> None:
    result = await mock_tools.call("weather", location="Lisbon")
    assert result["location"] == "Lisbon"
    assert "temperature_c" in result
    assert "condition" in result


@pytest.mark.asyncio
async def test_terminal_tool_executes_safe_command(mock_tools: MockToolRegistry) -> None:
    result = await mock_tools.call("terminal", command="echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_terminal_tool_blocks_destructive_command(mock_tools: MockToolRegistry) -> None:
    with pytest.raises(PermissionError):
        await mock_tools.call("terminal", command="rm -rf /data")


@pytest.mark.asyncio
async def test_search_tool_returns_results(mock_tools: MockToolRegistry) -> None:
    results = await mock_tools.call("search", query="quarterly earnings")
    assert results
    assert results[0]["title"].startswith("Result for")
    assert "url" in results[0]


@pytest.mark.asyncio
async def test_call_unknown_tool_raises_key_error(mock_tools: MockToolRegistry) -> None:
    with pytest.raises(KeyError):
        await mock_tools.call("unknown_tool")


@pytest.mark.asyncio
async def test_tool_calls_are_logged(mock_tools: MockToolRegistry) -> None:
    await mock_tools.call("weather", location="Oslo")
    assert mock_tools.call_log[-1]["tool"] == "weather"
    assert mock_tools.call_log[-1]["kwargs"]["location"] == "Oslo"


@pytest.mark.asyncio
async def test_multiple_tool_calls_accumulate_in_log(mock_tools: MockToolRegistry) -> None:
    await mock_tools.call("calculator", expression="1 + 1")
    await mock_tools.call("search", query="test")
    assert len(mock_tools.call_log) == 2