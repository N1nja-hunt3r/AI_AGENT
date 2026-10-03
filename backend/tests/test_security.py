"""
test_security.py
Tests for the security layer: approval workflow, rate limiting, sandbox
execution, JWT issuance/validation, and input validation.
"""

from __future__ import annotations

import pytest

from conftest import MockSecurity


def test_request_and_approve_action(mock_security: MockSecurity) -> None:
    approval_id = mock_security.request_approval("transfer funds")
    assert mock_security.is_approved(approval_id) is False

    assert mock_security.approve(approval_id) is True
    assert mock_security.is_approved(approval_id) is True


def test_approve_unknown_id_returns_false(mock_security: MockSecurity) -> None:
    assert mock_security.approve("missing") is False


def test_is_approved_for_unknown_id_returns_false(mock_security: MockSecurity) -> None:
    assert mock_security.is_approved("missing") is False


def test_rate_limit_allows_within_window(mock_security: MockSecurity) -> None:
    for _ in range(3):
        assert mock_security.check_rate_limit("user-1", max_calls=3, window_seconds=60) is True


def test_rate_limit_blocks_when_exceeded(mock_security: MockSecurity) -> None:
    for _ in range(3):
        mock_security.check_rate_limit("user-2", max_calls=3, window_seconds=60)
    assert mock_security.check_rate_limit("user-2", max_calls=3, window_seconds=60) is False


def test_rate_limit_tracks_keys_independently(mock_security: MockSecurity) -> None:
    for _ in range(3):
        mock_security.check_rate_limit("user-3", max_calls=3, window_seconds=60)
    assert mock_security.check_rate_limit("user-4", max_calls=3, window_seconds=60) is True


def test_sandbox_executes_and_logs_command(mock_security: MockSecurity) -> None:
    result = mock_security.run_in_sandbox("pip install requests")
    assert result["isolated"] is True
    assert result["exit_code"] == 0
    assert "pip install requests" in mock_security.sandbox_log


def test_issue_and_validate_jwt(mock_security: MockSecurity) -> None:
    token = mock_security.issue_jwt("user-123")
    assert mock_security.validate_jwt(token) is True
    assert mock_security.validate_jwt("not-a-real-token") is False


def test_jwt_tokens_are_unique_per_subject(mock_security: MockSecurity) -> None:
    token_a = mock_security.issue_jwt("user-a")
    token_b = mock_security.issue_jwt("user-b")
    assert token_a != token_b


def test_validator_accepts_clean_input(mock_security: MockSecurity) -> None:
    assert mock_security.validate_input("a normal user message") is True


def test_validator_rejects_oversized_input(mock_security: MockSecurity) -> None:
    assert mock_security.validate_input("x" * 20_000, max_length=1000) is False


def test_validator_rejects_known_attack_patterns(mock_security: MockSecurity) -> None:
    assert mock_security.validate_input("<script>alert(1)</script>") is False
    assert mock_security.validate_input("'; DROP TABLE users; --") is False
    assert mock_security.validate_input("../../etc/passwd") is False


@pytest.mark.asyncio
async def test_security_checks_compose_in_async_flow(mock_security: MockSecurity) -> None:
    async def guarded_action(payload: str) -> str:
        if not mock_security.validate_input(payload):
            raise ValueError("blocked by validator")
        if not mock_security.check_rate_limit("flow", max_calls=5, window_seconds=60):
            raise PermissionError("rate limited")
        return "ok"

    assert await guarded_action("safe input") == "ok"

    with pytest.raises(ValueError):
        await guarded_action("<script>alert(1)</script>")