"""
test_integration.py
End-to-end integration tests for the application bootstrap: startup and
shutdown lifecycle, dependency container wiring, health checks, and
router endpoint behavior.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import MockDatabase, MockLLM, MockSecurity


def test_app_bootstraps_dependency_container(app: FastAPI, mock_llm: MockLLM, mock_database: MockDatabase) -> None:
    container = app.state.container

    assert container["llm"] is mock_llm
    assert container["database"] is mock_database
    assert "security" in container


def test_startup_and_shutdown_events_fire_within_test_client_context(app: FastAPI) -> None:
    assert app.state.startup_events == []
    assert app.state.shutdown_events == []

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert app.state.startup_events == ["startup"]

    assert app.state.shutdown_events == ["shutdown"]


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_route_endpoint_resolves_memory_path(client: TestClient) -> None:
    response = client.get("/route", params={"query": "remember my earlier preference"})
    assert response.status_code == 200
    assert response.json() == {"path": "memory"}


def test_route_endpoint_resolves_web_path(client: TestClient) -> None:
    response = client.get("/route", params={"query": "what's the latest news today"})
    assert response.json()["path"] == "web"


def test_route_endpoint_falls_back_to_direct_reasoning(client: TestClient) -> None:
    response = client.get("/route", params={"query": "what is the speed of light"})
    assert response.json()["path"] == "direct_reasoning"


def test_full_request_lifecycle_through_chat_and_health(client: TestClient) -> None:
    health_response = client.get("/health")
    chat_response = client.post("/chat", json={"message": "integration test message"})

    assert health_response.status_code == 200
    assert chat_response.status_code == 200
    assert chat_response.json()["reply"]


def test_dependency_container_security_component_is_usable(app: FastAPI) -> None:
    security: MockSecurity = app.state.container["security"]
    approval_id = security.request_approval("integration test action")
    assert security.is_approved(approval_id) is False