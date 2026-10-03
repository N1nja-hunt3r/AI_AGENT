"""
websocket.py

FastAPI WebSocket router providing realtime chat streaming, agent
lifecycle events, and automation task event broadcasting, with a
lightweight connection manager.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)

PREFIX = ""

router = APIRouter(tags=["websocket"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConnectionManager:
    """Tracks active WebSocket connections grouped by channel name."""

    def __init__(self) -> None:
        self._channels: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._channels.setdefault(channel, set()).add(websocket)

    async def disconnect(self, channel: str, websocket: WebSocket) -> None:
        async with self._lock:
            connections = self._channels.get(channel)
            if connections and websocket in connections:
                connections.discard(websocket)
            if connections is not None and not connections:
                self._channels.pop(channel, None)

    async def broadcast(self, channel: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._channels.get(channel, set()))
        payload = json.dumps(message, default=str)
        stale: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_text(payload)
            except Exception:  # noqa: BLE001
                stale.append(connection)
        if stale:
            async with self._lock:
                for connection in stale:
                    self._channels.get(channel, set()).discard(connection)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(message, default=str))

    def connection_count(self, channel: str) -> int:
        return len(self._channels.get(channel, set()))


manager = ConnectionManager()


async def _authenticate_ws(websocket: WebSocket, token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Verify a JWT passed as a query parameter for WebSocket auth."""
    if not token:
        return None
    auth_service = getattr(websocket.app.state, "auth_service", None)
    if auth_service is None:
        return None
    try:
        return await auth_service.verify_access_token(token)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# WS /ws/chat/{session_id}
# ---------------------------------------------------------------------------
@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(default=None),
) -> None:
    """Realtime bidirectional chat channel for a session, streaming tokens as they arrive."""
    user = await _authenticate_ws(websocket, token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    channel = f"chat:{session_id}"
    await manager.connect(channel, websocket)
    controller = getattr(websocket.app.state, "controller", None)
    memory_capability = getattr(websocket.app.state, "memory_capability", None)

    try:
        await manager.send_personal(websocket, {"event": "connected", "session_id": session_id, "ts": _now_iso()})
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {"event": "error", "message": "Invalid JSON payload"})
                continue

            message_text = data.get("message", "")
            if not message_text:
                await manager.send_personal(websocket, {"event": "error", "message": "Missing 'message' field"})
                continue

            if controller is None:
                await manager.send_personal(websocket, {"event": "error", "message": "Controller unavailable"})
                continue

            memory_context = None
            if memory_capability is not None:
                try:
                    memory_context = await memory_capability.recall(
                        session_id=session_id, user_id=user.get("id"), query=message_text
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("memory recall failed for session=%s", session_id)

            await manager.send_personal(websocket, {"event": "start", "ts": _now_iso()})
            full_parts = []
            try:
                async for chunk in controller.stream_chat(
                    session_id=session_id,
                    user_id=user.get("id"),
                    messages=[{"role": "user", "content": message_text}],
                    model=data.get("model"),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens"),
                    memory_context=memory_context,
                    metadata=data.get("metadata"),
                ):
                    full_parts.append(chunk.get("delta", ""))
                    await manager.send_personal(websocket, {"event": "token", "data": chunk})
            except Exception as exc:  # noqa: BLE001
                logger.exception("stream_chat failed for session=%s", session_id)
                await manager.send_personal(websocket, {"event": "error", "message": str(exc)})
                continue

            full_content = "".join(full_parts)
            if memory_capability is not None:
                try:
                    await memory_capability.store(
                        session_id=session_id, user_id=user.get("id"), role="user", content=message_text
                    )
                    await memory_capability.store(
                        session_id=session_id, user_id=user.get("id"), role="assistant", content=full_content
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("memory store failed for session=%s", session_id)

            await manager.send_personal(websocket, {"event": "done", "ts": _now_iso()})
    except WebSocketDisconnect:
        logger.info("Chat websocket disconnected for session=%s", session_id)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in chat websocket for session=%s", session_id)
    finally:
        await manager.disconnect(channel, websocket)


# ---------------------------------------------------------------------------
# WS /ws/agents/{agent_id}
# ---------------------------------------------------------------------------
@router.websocket("/ws/agents/{agent_id}")
async def agent_events_websocket(
    websocket: WebSocket,
    agent_id: str,
    token: Optional[str] = Query(default=None),
) -> None:
    """Subscribe to realtime lifecycle and status events for a single agent."""
    user = await _authenticate_ws(websocket, token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    channel = f"agent:{agent_id}"
    await manager.connect(channel, websocket)
    try:
        await manager.send_personal(websocket, {"event": "subscribed", "agent_id": agent_id, "ts": _now_iso()})
        while True:
            # Agent channels are server-push only; keep the connection alive
            # by waiting for client pings/heartbeats.
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Agent events websocket disconnected for agent_id=%s", agent_id)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in agent events websocket for agent_id=%s", agent_id)
    finally:
        await manager.disconnect(channel, websocket)


# ---------------------------------------------------------------------------
# WS /ws/tasks/{task_id}
# ---------------------------------------------------------------------------
@router.websocket("/ws/tasks/{task_id}")
async def task_events_websocket(
    websocket: WebSocket,
    task_id: str,
    token: Optional[str] = Query(default=None),
) -> None:
    """Subscribe to realtime status events for an automation task execution."""
    user = await _authenticate_ws(websocket, token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    channel = f"task:{task_id}"
    await manager.connect(channel, websocket)
    try:
        await manager.send_personal(websocket, {"event": "subscribed", "task_id": task_id, "ts": _now_iso()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Task events websocket disconnected for task_id=%s", task_id)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in task events websocket for task_id=%s", task_id)
    finally:
        await manager.disconnect(channel, websocket)


# ---------------------------------------------------------------------------
# Server-side broadcast helpers (called by other services, not HTTP-exposed)
# ---------------------------------------------------------------------------
async def broadcast_agent_event(agent_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Broadcast an agent lifecycle/status event to all subscribers."""
    await manager.broadcast(
        f"agent:{agent_id}",
        {"event": event_type, "agent_id": agent_id, "data": payload, "ts": _now_iso()},
    )


async def broadcast_task_event(task_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Broadcast an automation task status event to all subscribers."""
    await manager.broadcast(
        f"task:{task_id}",
        {"event": event_type, "task_id": task_id, "data": payload, "ts": _now_iso()},
    )


async def broadcast_chat_event(session_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Broadcast a chat-channel event (e.g. system notice) to all subscribers."""
    await manager.broadcast(
        f"chat:{session_id}",
        {"event": event_type, "session_id": session_id, "data": payload, "ts": _now_iso()},
    )
