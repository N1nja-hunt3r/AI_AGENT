"""
test_chat_api.py
Tests for the chat HTTP API: POST /chat, POST /stream, session
continuity, and error handling.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from conftest import MockLLM


def test_post_chat_returns_reply_and_session_id(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert "reply" in body


def test_post_chat_reuses_provided_session_id(client: TestClient) -> None:
    first = client.post("/chat", json={"message": "first message", "session_id": "session-123"})
    second = client.post("/chat", json={"message": "second message", "session_id": "session-123"})

    assert first.json()["session_id"] == "session-123"
    assert second.json()["session_id"] == "session-123"


def test_post_chat_generates_new_session_id_when_omitted(client: TestClient) -> None:
    first = client.post("/chat", json={"message": "no session provided"})
    second = client.post("/chat", json={"message": "also no session"})

    assert first.json()["session_id"] != second.json()["session_id"]


def test_post_chat_rejects_invalid_input(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "<script>alert(1)</script>"})
    assert response.status_code == 400


def test_post_chat_rejects_malformed_payload(client: TestClient) -> None:
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_post_stream_returns_text_response(client: TestClient) -> None:
    response = client.post("/stream", json={"message": "stream this please"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text.strip() != ""


def test_post_stream_rejects_invalid_input(client: TestClient) -> None:
    response = client.post("/stream", json={"message": "'; DROP TABLE users; --"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_post_chat_async_client_round_trip(async_client: AsyncClient) -> None:
    response = await async_client.post("/chat", json={"message": "async hello"})

    assert response.status_code == 200
    assert response.json()["reply"]


@pytest.mark.asyncio
async def test_post_chat_reflects_llm_failure_as_server_error(async_client: AsyncClient, mock_llm: MockLLM) -> None:
    mock_llm.fail_next = True
    response = await async_client.post("/chat", json={"message": "trigger failure"})

    assert response.status_code == 500