"""
test_database.py
Tests for the database layer: CRUD operations, session lifecycle,
backup/restore, and caching.
"""

from __future__ import annotations

import pytest

from conftest import MockDatabase


@pytest.mark.asyncio
async def test_create_and_read_record(mock_database: MockDatabase) -> None:
    record_id = await mock_database.create("users", {"name": "Asha"})
    record = await mock_database.read("users", record_id)
    assert record is not None
    assert record["name"] == "Asha"


@pytest.mark.asyncio
async def test_read_missing_record_returns_none(mock_database: MockDatabase) -> None:
    assert await mock_database.read("users", "missing") is None


@pytest.mark.asyncio
async def test_update_record(mock_database: MockDatabase) -> None:
    record_id = await mock_database.create("users", {"name": "Asha"})
    updated = await mock_database.update("users", record_id, {"name": "Asha K"})
    record = await mock_database.read("users", record_id)

    assert updated is True
    assert record["name"] == "Asha K"


@pytest.mark.asyncio
async def test_update_missing_record_returns_false(mock_database: MockDatabase) -> None:
    assert await mock_database.update("users", "missing", {"name": "x"}) is False


@pytest.mark.asyncio
async def test_delete_record(mock_database: MockDatabase) -> None:
    record_id = await mock_database.create("users", {"name": "Asha"})
    assert await mock_database.delete("users", record_id) is True
    assert await mock_database.read("users", record_id) is None


@pytest.mark.asyncio
async def test_delete_missing_record_returns_false(mock_database: MockDatabase) -> None:
    assert await mock_database.delete("users", "missing") is False


@pytest.mark.asyncio
async def test_open_and_close_session(mock_database: MockDatabase) -> None:
    session_id = await mock_database.open_session("user-1")
    assert mock_database.sessions[session_id]["active"] is True

    closed = await mock_database.close_session(session_id)
    assert closed is True
    assert mock_database.sessions[session_id]["active"] is False


@pytest.mark.asyncio
async def test_close_unknown_session_returns_false(mock_database: MockDatabase) -> None:
    assert await mock_database.close_session("missing") is False


@pytest.mark.asyncio
async def test_backup_and_restore_round_trip(mock_database: MockDatabase) -> None:
    record_id = await mock_database.create("users", {"name": "Asha"})
    backup_id = await mock_database.backup()

    await mock_database.delete("users", record_id)
    assert await mock_database.read("users", record_id) is None

    restored = await mock_database.restore(backup_id)
    assert restored is True
    record = await mock_database.read("users", record_id)
    assert record is not None
    assert record["name"] == "Asha"


@pytest.mark.asyncio
async def test_restore_unknown_backup_returns_false(mock_database: MockDatabase) -> None:
    assert await mock_database.restore("missing-backup") is False


@pytest.mark.asyncio
async def test_cache_set_and_get(mock_database: MockDatabase) -> None:
    await mock_database.cache_set("key1", {"value": 42})
    cached = await mock_database.cache_get("key1")
    assert cached == {"value": 42}


@pytest.mark.asyncio
async def test_cache_get_missing_key_returns_none(mock_database: MockDatabase) -> None:
    assert await mock_database.cache_get("missing-key") is None


@pytest.mark.asyncio
async def test_cache_expires_after_ttl(mock_database: MockDatabase) -> None:
    await mock_database.cache_set("key2", "value", ttl=-1)
    assert await mock_database.cache_get("key2") is None