"""
test_websocket.py
Tests for the /ws/chat WebSocket endpoint: streamed chat events,
validation errors, and disconnect handling.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import MockLLM


def test_websocket_returns_message_event_for_valid_input(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("hello over websocket")
        payload = websocket.receive_json()

    assert payload["event"] == "message"
    assert "reply" in payload


def test_websocket_returns_error_event_for_invalid_input(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("<script>alert(1)</script>")
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["detail"] == "invalid input"


def test_websocket_supports_multiple_messages_in_one_session(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("first message")
        first = websocket.receive_json()
        websocket.send_text("second message")
        second = websocket.receive_json()

    assert first["event"] == "message"
    assert second["event"] == "message"


def test_websocket_uses_llm_response_in_message_event(client: TestClient, mock_llm: MockLLM) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("what is the weather like")
        payload = websocket.receive_json()

    assert payload["reply"] == mock_llm.default_response


def test_websocket_closes_cleanly_on_disconnect(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("one last message before disconnect")
        websocket.receive_json()
        websocket.close()

    # Reconnecting after a clean disconnect should succeed without error.
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("reconnected successfully")
        payload = websocket.receive_json()

    assert payload["event"] == "message"