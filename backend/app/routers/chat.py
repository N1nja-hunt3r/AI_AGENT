from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Fast-lane detection for simple chat (skip memory, tools, RAG)
_SIMPLE_CHAT_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|"
    r"what's up|sup|yo|hi there|hello there|hey there)"
    r"[\.!?\s]*$",
    re.IGNORECASE,
)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# In-memory conversation / message store
# ---------------------------------------------------------------------------
_conversations: Dict[str, Dict[str, Any]] = {}
_messages_store: Dict[str, List[Dict[str, Any]]] = {}


def _create_conversation(conversation_id: str, title: str = "New Conversation") -> Dict[str, Any]:
    now = _now_iso()
    conv: Dict[str, Any] = {
        "id": conversation_id,
        "title": title,
        "createdAt": now,
        "updatedAt": now,
    }
    _conversations[conversation_id] = conv
    _messages_store[conversation_id] = []
    return conv


def _get_or_create_conversation(conversation_id: Optional[str]) -> str:
    if conversation_id and conversation_id in _conversations:
        return conversation_id
    cid = conversation_id or str(uuid.uuid4())
    if cid not in _conversations:
        _create_conversation(cid)
    return cid


def _add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tokens: Optional[int] = None,
) -> Dict[str, Any]:
    now = _now_iso()
    msg: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "status": "complete",
        "createdAt": now,
        "tokens": tokens,
        "metadata": metadata or {},
    }
    if conversation_id in _messages_store:
        _messages_store[conversation_id].append(msg)
    if conversation_id in _conversations:
        _conversations[conversation_id]["updatedAt"] = now
    return msg


def _get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    return _messages_store.get(conversation_id, [])


def _clear_messages(conversation_id: str) -> None:
    if conversation_id in _messages_store:
        _messages_store[conversation_id] = []
    if conversation_id in _conversations:
        _conversations[conversation_id]["updatedAt"] = _now_iso()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str = Field(..., min_length=1, max_length=1_000_000)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    conversationId: Optional[str] = None
    content: Optional[str] = None
    role: Optional[str] = "user"
    systemPrompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=200_000)
    use_memory: bool = True
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_format(self) -> ChatRequest:
        has_legacy = self.messages is not None
        has_simple = self.content is not None
        if not has_legacy and not has_simple:
            raise ValueError("Either 'messages' (legacy) or 'content' (simplified) must be provided")
        if has_legacy and has_simple:
            raise ValueError("Provide either 'messages' (legacy) or 'content' (simplified), not both")
        return self


class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage
    usage: Dict[str, int] = Field(default_factory=dict)
    created_at: str


class PlanRequest(BaseModel):
    session_id: Optional[str] = None
    goal: str = Field(..., min_length=1, max_length=50_000)
    context: Optional[Dict[str, Any]] = None
    max_steps: int = Field(default=10, ge=1, le=100)


class PlanStep(BaseModel):
    step_id: str
    description: str
    tool: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    session_id: str
    plan_id: str
    steps: List[PlanStep]
    created_at: str


class ExecuteRequest(BaseModel):
    session_id: Optional[str] = None
    plan_id: Optional[str] = None
    steps: Optional[List[PlanStep]] = None
    dry_run: bool = False


class ExecuteResult(BaseModel):
    step_id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


class ExecuteResponse(BaseModel):
    session_id: str
    plan_id: Optional[str] = None
    results: List[ExecuteResult]
    completed_at: str


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    checked_at: str


class MessagePayload(BaseModel):
    conversationId: Optional[str] = None
    content: str = Field(..., min_length=1, max_length=1_000_000)
    role: str = "user"
    systemPrompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageInfo(BaseModel):
    promptTokens: int = 0
    completionTokens: int = 0
    totalTokens: int = 0


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    status: str = "complete"
    createdAt: str
    tokens: Optional[int] = None


class MessageResponse(BaseModel):
    message: MessageOut
    conversationId: str
    usage: UsageInfo


class ConversationOut(BaseModel):
    id: str
    title: str
    messages: List[MessageOut]
    createdAt: str
    updatedAt: str


class ConversationResponse(BaseModel):
    conversation: ConversationOut


class ClearConversationResponse(BaseModel):
    success: bool
    conversationId: str
    clearedAt: str


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        user = {"id": "guest", "name": "Guest"}
    return user


# ---------------------------------------------------------------------------
# Dependency accessors (resolved from app.state, wired in main app)
# ---------------------------------------------------------------------------
def get_controller(request: Request) -> Any:
    controller = getattr(request.app.state, "controller", None)
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller service unavailable")
    return controller


def get_planner(request: Request) -> Any:
    planner = getattr(request.app.state, "planner", None)
    if planner is None:
        raise HTTPException(status_code=503, detail="Planner service unavailable")
    return planner


def get_executor(request: Request) -> Any:
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=503, detail="Executor service unavailable")
    return executor


def get_memory_capability(request: Request) -> Optional[Any]:
    return getattr(request.app.state, "memory_capability", None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    return str(uuid.uuid4())


def _build_messages_from_payload(payload: ChatRequest) -> List[ChatMessage]:
    if payload.messages is not None:
        return payload.messages
    messages: List[ChatMessage] = []
    if payload.systemPrompt:
        messages.append(ChatMessage(role="system", content=payload.systemPrompt))
    content = (payload.content or "").strip()
    if not content:
        content = "Hello"
    messages.append(ChatMessage(role=payload.role or "user", content=content))
    return messages


# ---------------------------------------------------------------------------
# POST /chat — legacy endpoint (dual-format)
# ---------------------------------------------------------------------------
@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    controller: Any = Depends(get_controller),
    memory_capability: Optional[Any] = Depends(get_memory_capability),
) -> ChatResponse:
    try:
        session_id = _ensure_session_id(payload.session_id or payload.conversationId)
        messages = _build_messages_from_payload(payload)
        last_content = messages[-1].content.lower().strip()
        is_simple_chat = bool(_SIMPLE_CHAT_RE.match(last_content))

        memory_context: Optional[List[Dict[str, Any]]] = None
        if not is_simple_chat and payload.use_memory and memory_capability is not None:
            memory_context = await memory_capability.recall(
                session_id=session_id,
                user_id=user.get("id"),
                query=messages[-1].content,
            )

        result = await controller.handle_chat(
            session_id=session_id,
            user_id=user.get("id"),
            messages=[m.model_dump() for m in messages],
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            memory_context=memory_context,
            metadata=payload.metadata,
        )

        if not is_simple_chat and payload.use_memory and memory_capability is not None:
            await memory_capability.store(
                session_id=session_id,
                user_id=user.get("id"),
                role="user",
                content=messages[-1].content,
            )
            await memory_capability.store(
                session_id=session_id,
                user_id=user.get("id"),
                role="assistant",
                content=result["content"],
            )

        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content=result["content"]),
            usage=result.get("usage", {}),
            created_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat() failed for session=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /chat/message — new simplified single-message endpoint
# ---------------------------------------------------------------------------
@router.post("/message", response_model=MessageResponse)
async def send_message(
    payload: MessagePayload,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    controller: Any = Depends(get_controller),
    memory_capability: Optional[Any] = Depends(get_memory_capability),
) -> MessageResponse:
    try:
        conversation_id = _get_or_create_conversation(payload.conversationId)

        _add_message(conversation_id, payload.role, payload.content, payload.metadata)

        messages: List[Dict[str, Any]] = []
        if payload.systemPrompt:
            messages.append({"role": "system", "content": payload.systemPrompt})
        messages.append({"role": payload.role, "content": payload.content})

        is_simple_chat = bool(_SIMPLE_CHAT_RE.match(payload.content.lower().strip()))
        memory_context: Optional[List[Dict[str, Any]]] = None
        if not is_simple_chat and payload.metadata and payload.metadata.get("use_memory", True) and memory_capability is not None:
            memory_context = await memory_capability.recall(
                session_id=conversation_id,
                user_id=user.get("id"),
                query=payload.content,
            )

        result = await controller.handle_chat(
            session_id=conversation_id,
            user_id=user.get("id"),
            messages=messages,
            model=None,
            temperature=0.7,
            max_tokens=None,
            memory_context=memory_context,
            metadata=payload.metadata,
        )

        raw_usage = result.get("usage", {})
        usage = UsageInfo(
            promptTokens=raw_usage.get("prompt_tokens", raw_usage.get("promptTokens", 0)),
            completionTokens=raw_usage.get("completion_tokens", raw_usage.get("completionTokens", 0)),
            totalTokens=raw_usage.get("total_tokens", raw_usage.get("totalTokens", 0)),
        )

        assistant_msg = _add_message(
            conversation_id,
            "assistant",
            result["content"],
            tokens=usage.totalTokens or None,
        )

        if not is_simple_chat and memory_capability is not None:
            await memory_capability.store(
                session_id=conversation_id,
                user_id=user.get("id"),
                role="user",
                content=payload.content,
            )
            await memory_capability.store(
                session_id=conversation_id,
                user_id=user.get("id"),
                role="assistant",
                content=result["content"],
            )

        return MessageResponse(
            message=MessageOut(
                id=assistant_msg["id"],
                role=assistant_msg["role"],
                content=assistant_msg["content"],
                status=assistant_msg["status"],
                createdAt=assistant_msg["createdAt"],
                tokens=assistant_msg.get("tokens"),
            ),
            conversationId=conversation_id,
            usage=usage,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("send_message() failed for conversation=%s", conversation_id)
        raise HTTPException(status_code=500, detail=f"Message processing failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE streaming (dual-format)
# ---------------------------------------------------------------------------
async def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_chat_response(
    payload: ChatRequest,
    user: Dict[str, Any],
    controller: Any,
    memory_capability: Optional[Any],
    session_id: str,
) -> AsyncIterator[str]:
    import time
    t0 = time.monotonic()
    full_content_parts: List[str] = []
    first_token_yielded = False
    is_simple_chat = False
    stage = "init"

    try:
        messages = _build_messages_from_payload(payload)
        last_msg = messages[-1].content.lower().strip() if messages else ""
        is_simple_chat = bool(_SIMPLE_CHAT_RE.match(last_msg))
        stage = "setup"

        yield await _sse_event("start", {"session_id": session_id, "conversationId": session_id})
        logger.info("stream_chat_route session=%s stage=start_event_sent setup_ms=%.1f simple=%s", session_id, (time.monotonic() - t0) * 1000, is_simple_chat)

        memory_context: Optional[List[Dict[str, Any]]] = None
        if not is_simple_chat and payload.use_memory and memory_capability is not None:
            stage = "memory_recall"
            t_mem = time.monotonic()
            try:
                memory_context = await memory_capability.recall(
                    session_id=session_id,
                    user_id=user.get("id"),
                    query=messages[-1].content,
                )
                logger.info("stream_chat_route session=%s stage=memory_recall duration_ms=%.1f results=%d", session_id, (time.monotonic() - t_mem) * 1000, len(memory_context))
            except Exception as mem_exc:
                logger.warning("stream_chat_route session=%s stage=memory_recall error: %s (continuing without memory)", session_id, mem_exc)
                memory_context = None

        stage = "provider_stream"
        t_stream = time.monotonic()
        async for chunk in controller.stream_chat(
            session_id=session_id,
            user_id=user.get("id"),
            messages=[m.model_dump() for m in messages],
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            memory_context=memory_context,
            metadata=payload.metadata,
        ):
            delta = chunk.get("delta", "")
            if delta and not first_token_yielded:
                first_token_yielded = True
                ttft = time.monotonic() - t0
                logger.info("stream_chat_route session=%s stage=first_token_sse ttft_s=%.2f", session_id, ttft)
            full_content_parts.append(delta)
            yield await _sse_event(
                "token",
                {
                    "id": chunk.get("id", str(uuid.uuid4())),
                    "conversationId": session_id,
                    "delta": delta,
                    "done": chunk.get("done", False),
                },
            )

        stage = "memory_store"
        full_content = "".join(full_content_parts)
        if not is_simple_chat and payload.use_memory and memory_capability is not None:
            try:
                await memory_capability.store(
                    session_id=session_id,
                    user_id=user.get("id"),
                    role="user",
                    content=messages[-1].content,
                )
                await memory_capability.store(
                    session_id=session_id,
                    user_id=user.get("id"),
                    role="assistant",
                    content=full_content,
                )
            except Exception as mem_exc:
                logger.warning("stream_chat_route session=%s stage=memory_store error: %s (non-fatal)", session_id, mem_exc)

        stage = "done"
        raw_usage = getattr(controller, "last_usage", {}) or {}
        yield await _sse_event(
            "done",
            {
                "session_id": session_id,
                "created_at": _now_iso(),
                "usage": {
                    "promptTokens": raw_usage.get("prompt_tokens", 0),
                    "completionTokens": raw_usage.get("completion_tokens", 0),
                    "totalTokens": raw_usage.get("total_tokens", 0),
                },
            },
        )
        logger.info("stream_chat_route session=%s stage=done total_s=%.2f token_count=%d", session_id, time.monotonic() - t0, len(full_content_parts))

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.exception("stream_chat_route session=%s stage=%s after_s=%.2f first_token=%s", session_id, stage, elapsed, first_token_yielded)
        try:
            yield await _sse_event("error", {
                "error": str(exc),
                "type": type(exc).__name__,
                "provider": payload.model or "default",
                "stage": stage,
                "session": session_id,
                "message": f"Stream failed at stage '{stage}': {exc}",
            })
        except Exception:
            logger.exception("stream_chat_route session=%s stage=error_emit also failed", session_id)


@router.post("/stream")
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    controller: Any = Depends(get_controller),
    memory_capability: Optional[Any] = Depends(get_memory_capability),
) -> StreamingResponse:
    session_id = _ensure_session_id(payload.session_id or payload.conversationId)
    logger.info("stream_chat_route session=%s model=%s user=%s", session_id, payload.model, user.get("id"))
    return StreamingResponse(
        _stream_chat_response(payload, user, controller, memory_capability, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# GET /chat/conversations/{conversation_id}
# ---------------------------------------------------------------------------
@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> ConversationResponse:
    conv = _conversations.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    raw_messages = _get_messages(conversation_id)
    messages_out = [
        MessageOut(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            status=m.get("status", "complete"),
            createdAt=m["createdAt"],
            tokens=m.get("tokens"),
        )
        for m in raw_messages
    ]

    return ConversationResponse(
        conversation=ConversationOut(
            id=conv["id"],
            title=conv["title"],
            messages=messages_out,
            createdAt=conv["createdAt"],
            updatedAt=conv["updatedAt"],
        )
    )


# ---------------------------------------------------------------------------
# DELETE /chat/conversations/{conversation_id}/clear
# ---------------------------------------------------------------------------
@router.delete("/conversations/{conversation_id}/clear", response_model=ClearConversationResponse)
async def clear_conversation(
    conversation_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> ClearConversationResponse:
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    _clear_messages(conversation_id)
    return ClearConversationResponse(
        success=True,
        conversationId=conversation_id,
        clearedAt=_now_iso(),
    )


# ---------------------------------------------------------------------------
# POST /chat/plan
# ---------------------------------------------------------------------------
@router.post("/plan", response_model=PlanResponse)
async def create_plan(
    payload: PlanRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    planner: Any = Depends(get_planner),
) -> PlanResponse:
    session_id = _ensure_session_id(payload.session_id)
    try:
        plan = await planner.create_plan(
            session_id=session_id,
            user_id=user.get("id"),
            goal=payload.goal,
            context=payload.context,
            max_steps=payload.max_steps,
        )
        steps = [
            PlanStep(
                step_id=step["step_id"],
                description=step["description"],
                tool=step.get("tool"),
                depends_on=step.get("depends_on", []),
            )
            for step in plan["steps"]
        ]
        return PlanResponse(
            session_id=session_id,
            plan_id=plan["plan_id"],
            steps=steps,
            created_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("create_plan failed for session=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Planning failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /chat/execute
# ---------------------------------------------------------------------------
@router.post("/execute", response_model=ExecuteResponse)
async def execute_plan(
    payload: ExecuteRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    executor: Any = Depends(get_executor),
    planner: Any = Depends(get_planner),
) -> ExecuteResponse:
    session_id = _ensure_session_id(payload.session_id)

    if not payload.plan_id and not payload.steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either plan_id or steps must be provided",
        )

    try:
        steps = payload.steps
        if payload.plan_id and not steps:
            plan = await planner.get_plan(payload.plan_id)
            if plan is None:
                raise HTTPException(status_code=404, detail=f"Plan not found: {payload.plan_id}")
            steps = [PlanStep(**s) for s in plan["steps"]]

        raw_results = await executor.execute_steps(
            session_id=session_id,
            user_id=user.get("id"),
            steps=[s.model_dump() for s in (steps or [])],
            dry_run=payload.dry_run,
        )

        results = [
            ExecuteResult(
                step_id=r["step_id"],
                status=r["status"],
                output=r.get("output"),
                error=r.get("error"),
            )
            for r in raw_results
        ]

        return ExecuteResponse(
            session_id=session_id,
            plan_id=payload.plan_id,
            results=results,
            completed_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("execute_plan failed for session=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Execution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /chat/health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    components: Dict[str, str] = {}

    for name, attr in (
        ("controller", "controller"),
        ("planner", "planner"),
        ("executor", "executor"),
        ("memory_capability", "memory_capability"),
    ):
        service = getattr(request.app.state, attr, None)
        if service is None:
            components[name] = "unavailable"
            continue
        try:
            if hasattr(service, "health_check"):
                healthy = await service.health_check()
                components[name] = "healthy" if healthy else "unhealthy"
            else:
                components[name] = "unknown"
        except Exception:
            components[name] = "unhealthy"

    overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components, checked_at=_now_iso())
