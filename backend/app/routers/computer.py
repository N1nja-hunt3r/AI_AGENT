"""
computer.py

FastAPI router exposing computer-use primitives: mouse, keyboard,
browser control, screenshots, terminal command execution, and an
approval gate for sensitive actions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["computer"])

SENSITIVE_ACTIONS = {"terminal_exec", "keyboard_type", "mouse_click", "browser_navigate"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MouseClickRequest(BaseModel):
    session_id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    button: Literal["left", "right", "middle"] = "left"
    double_click: bool = False


class MouseMoveRequest(BaseModel):
    session_id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class KeyboardTypeRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, max_length=10_000)


class KeyboardKeyRequest(BaseModel):
    session_id: str
    keys: List[str] = Field(..., min_length=1)


class BrowserNavigateRequest(BaseModel):
    session_id: str
    url: str = Field(..., min_length=1)


class BrowserActionRequest(BaseModel):
    session_id: str
    action: Literal["back", "forward", "reload", "close"]


class TerminalExecRequest(BaseModel):
    session_id: str
    command: str = Field(..., min_length=1, max_length=10_000)
    working_dir: Optional[str] = None
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)


class ActionResponse(BaseModel):
    session_id: str
    action: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    executed_at: str


class ScreenshotResponse(BaseModel):
    session_id: str
    image_base64: str
    width: int
    height: int
    captured_at: str


class TerminalExecResponse(BaseModel):
    session_id: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    executed_at: str


class ApprovalRequiredResponse(BaseModel):
    requires_approval: bool
    action: str
    reason: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_computer_service(request: Request) -> Any:
    service = getattr(request.app.state, "computer_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Computer-use service unavailable")
    return service


def get_security_service(request: Request) -> Any:
    return getattr(request.app.state, "security_service", None)


async def _require_approval(
    action: str,
    session_id: str,
    user: Dict[str, Any],
    security_service: Any,
) -> None:
    if action not in SENSITIVE_ACTIONS:
        return
    if security_service is None:
        return
    approved = await security_service.is_approved(
        user_id=user.get("id"), action=f"computer:{action}", session_id=session_id
    )
    if not approved:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Action '{action}' requires explicit approval before execution",
        )


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------
@router.post("/mouse/click", response_model=ActionResponse)
async def mouse_click(
    payload: MouseClickRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
    security_service: Any = Depends(get_security_service),
) -> ActionResponse:
    """Perform a mouse click at the given coordinates."""
    await _require_approval("mouse_click", payload.session_id, user, security_service)
    try:
        result = await computer_service.mouse_click(
            session_id=payload.session_id,
            x=payload.x,
            y=payload.y,
            button=payload.button,
            double_click=payload.double_click,
        )
        return ActionResponse(
            session_id=payload.session_id, action="mouse_click", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("mouse_click failed")
        raise HTTPException(status_code=500, detail=f"Mouse click failed: {exc}") from exc


@router.post("/mouse/move", response_model=ActionResponse)
async def mouse_move(
    payload: MouseMoveRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
) -> ActionResponse:
    """Move the mouse cursor to the given coordinates."""
    try:
        result = await computer_service.mouse_move(session_id=payload.session_id, x=payload.x, y=payload.y)
        return ActionResponse(
            session_id=payload.session_id, action="mouse_move", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("mouse_move failed")
        raise HTTPException(status_code=500, detail=f"Mouse move failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
@router.post("/keyboard/type", response_model=ActionResponse)
async def keyboard_type(
    payload: KeyboardTypeRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
    security_service: Any = Depends(get_security_service),
) -> ActionResponse:
    """Type the given text using the virtual keyboard."""
    await _require_approval("keyboard_type", payload.session_id, user, security_service)
    try:
        result = await computer_service.keyboard_type(session_id=payload.session_id, text=payload.text)
        return ActionResponse(
            session_id=payload.session_id, action="keyboard_type", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("keyboard_type failed")
        raise HTTPException(status_code=500, detail=f"Keyboard type failed: {exc}") from exc


@router.post("/keyboard/key", response_model=ActionResponse)
async def keyboard_key(
    payload: KeyboardKeyRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
) -> ActionResponse:
    """Send a key combination (e.g. ['ctrl', 'c'])."""
    try:
        result = await computer_service.keyboard_key(session_id=payload.session_id, keys=payload.keys)
        return ActionResponse(
            session_id=payload.session_id, action="keyboard_key", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("keyboard_key failed")
        raise HTTPException(status_code=500, detail=f"Keyboard key action failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
@router.post("/browser/navigate", response_model=ActionResponse)
async def browser_navigate(
    payload: BrowserNavigateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
    security_service: Any = Depends(get_security_service),
) -> ActionResponse:
    """Navigate the browser to the given URL."""
    await _require_approval("browser_navigate", payload.session_id, user, security_service)
    try:
        result = await computer_service.browser_navigate(session_id=payload.session_id, url=payload.url)
        return ActionResponse(
            session_id=payload.session_id, action="browser_navigate", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("browser_navigate failed")
        raise HTTPException(status_code=500, detail=f"Browser navigation failed: {exc}") from exc


@router.post("/browser/action", response_model=ActionResponse)
async def browser_action(
    payload: BrowserActionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
) -> ActionResponse:
    """Perform a basic browser navigation action (back/forward/reload/close)."""
    try:
        result = await computer_service.browser_action(session_id=payload.session_id, action=payload.action)
        return ActionResponse(
            session_id=payload.session_id, action=f"browser_{payload.action}", status="succeeded",
            result=result, executed_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("browser_action failed")
        raise HTTPException(status_code=500, detail=f"Browser action failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------
@router.get("/screenshot/{session_id}", response_model=ScreenshotResponse)
async def take_screenshot(
    session_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
) -> ScreenshotResponse:
    """Capture a screenshot of the current computer-use session."""
    try:
        result = await computer_service.screenshot(session_id=session_id)
        return ScreenshotResponse(
            session_id=session_id,
            image_base64=result["image_base64"],
            width=result.get("width", 0),
            height=result.get("height", 0),
            captured_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("take_screenshot failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Screenshot capture failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
@router.post("/terminal/exec", response_model=TerminalExecResponse)
async def terminal_exec(
    payload: TerminalExecRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    computer_service: Any = Depends(get_computer_service),
    security_service: Any = Depends(get_security_service),
) -> TerminalExecResponse:
    """Execute a shell command within the session's sandboxed terminal."""
    await _require_approval("terminal_exec", payload.session_id, user, security_service)
    import time

    start = time.monotonic()
    try:
        result = await computer_service.terminal_exec(
            session_id=payload.session_id,
            command=payload.command,
            working_dir=payload.working_dir,
            timeout_seconds=payload.timeout_seconds,
        )
        duration_ms = (time.monotonic() - start) * 1000
        return TerminalExecResponse(
            session_id=payload.session_id,
            command=payload.command,
            exit_code=result.get("exit_code", -1),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            duration_ms=round(duration_ms, 2),
            executed_at=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("terminal_exec failed")
        raise HTTPException(status_code=500, detail=f"Terminal execution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------
@router.get("/approvals/check", response_model=ApprovalRequiredResponse)
async def check_approval_requirement(
    action: str,
    session_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    security_service: Any = Depends(get_security_service),
) -> ApprovalRequiredResponse:
    """Check whether a given computer-use action requires explicit approval."""
    if action not in SENSITIVE_ACTIONS:
        return ApprovalRequiredResponse(
            requires_approval=False, action=action, reason="Action is not classified as sensitive"
        )
    if security_service is None:
        return ApprovalRequiredResponse(
            requires_approval=True, action=action, reason="Security service unavailable; defaulting to required"
        )
    approved = await security_service.is_approved(
        user_id=user.get("id"), action=f"computer:{action}", session_id=session_id
    )
    return ApprovalRequiredResponse(
        requires_approval=not approved,
        action=action,
        reason="Already approved" if approved else "Sensitive action requires user approval",
    )
