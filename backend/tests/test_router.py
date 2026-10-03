"""
test_router.py
Tests for the Router: capability path selection across memory, RAG,
tools, web, and computer-use queries.
"""

from __future__ import annotations

import pytest

from conftest import Router


@pytest.mark.parametrize(
    "query,expected",
    [
        ("remember what I told you earlier", "memory"),
        ("according to the file we uploaded", "rag"),
        ("please calculate 12 * 7", "tool"),
        ("what's the latest news today", "web"),
        ("take a screenshot and click submit", "computer"),
        ("what is the capital of France", "direct_reasoning"),
    ],
)
def test_route_selects_expected_path(router: Router, query: str, expected: str) -> None:
    assert router.route(query) == expected


def test_route_memory_matches_preference_keyword(router: Router) -> None:
    assert router.route("this is my preference going forward") == "memory"


def test_route_rag_matches_document_reference(router: Router) -> None:
    assert router.route("can you summarize the document I shared") == "rag"


def test_route_tool_matches_compute_keyword(router: Router) -> None:
    assert router.route("compute the average for me") == "tool"


def test_route_web_matches_current_events_keyword(router: Router) -> None:
    assert router.route("what is the current exchange rate") == "web"


def test_route_computer_matches_browser_action(router: Router) -> None:
    assert router.route("open the browser and log in") == "computer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("remember my earlier preference", "memory"),
        ("open the browser and type into the field", "computer"),
        ("search the document for the clause", "rag"),
    ],
)
async def test_route_async_matches_sync_route(router: Router, query: str, expected: str) -> None:
    assert await router.route_async(query) == expected


@pytest.mark.asyncio
async def test_route_async_falls_back_to_direct_reasoning(router: Router) -> None:
    assert await router.route_async("explain how gravity works") == "direct_reasoning"