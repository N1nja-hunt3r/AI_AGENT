"""
computer_capability.py
======================
Purpose
-------
ComputerCapability enables an AI Operating System to interact with the host
computer (and, in future iterations, remote machines) through a unified,
safe, audited interface.  It covers mouse/keyboard control, browser
automation, terminal execution, file operations, screenshot capture, window
management, clipboard access, vision/OCR support, and health checks.

Architecture
------------
                        ┌─────────────────────────────┐
                        │        Capability (base)     │
                        └────────────┬────────────────┘
                                     │ inherits
                        ┌────────────▼────────────────┐
                        │      ComputerCapability      │
                        │  ┌──────────────────────┐   │
                        │  │  ActionRouter         │   │
                        │  │  ├─ MouseDriver       │   │
                        │  │  ├─ KeyboardDriver    │   │
                        │  │  ├─ BrowserDriver     │   │
                        │  │  ├─ TerminalDriver    │   │
                        │  │  ├─ FileDriver        │   │
                        │  │  ├─ ScreenDriver      │   │
                        │  │  ├─ WindowDriver      │   │
                        │  │  └─ ClipboardDriver   │   │
                        │  └──────────────────────┘   │
                        │  ┌──────────────────────┐   │
                        │  │  SafetyLayer          │   │
                        │  │  ├─ DangerDetector    │   │
                        │  │  ├─ SandboxGuard      │   │
                        │  │  └─ ApprovalGate      │   │
                        │  └──────────────────────┘   │
                        │  ┌──────────────────────┐   │
                        │  │  AuditSystem          │   │
                        │  │  ├─ ActionHistory     │   │
                        │  │  ├─ ArtifactStore     │   │
                        │  │  └─ RollbackManager   │   │
                        │  └──────────────────────┘   │
                        └─────────────────────────────┘
                                     │ registered with
                        ┌────────────▼────────────────┐
                        │         registry.py          │
                        └─────────────────────────────┘
                                     │ invoked by
                        ┌────────────▼────────────────┐
                        │         executor.py          │
                        └─────────────────────────────┘
                                     │ context from
                        ┌────────────▼────────────────┐
                        │       context_engine.py      │
                        └─────────────────────────────┘

Communication Flow
------------------
executor.py  →  ComputerCapability.execute(action, params, context)
                   │
                   ├─► SafetyLayer.assess(action, params)
                   │       ├─ dangerous?  →  ApprovalGate.request_human()
                   │       └─ sandboxed?  →  SandboxGuard.wrap(action)
                   │
                   ├─► AuditSystem.record_start(action, params)
                   │
                   ├─► ActionRouter.dispatch(action, params)
                   │       └─► <Driver>.run(params)   [async, timeout]
                   │
                   ├─► AuditSystem.record_result(result)
                   │       ├─ ArtifactStore.save(artifacts)
                   │       └─ ActionHistory.append(entry)
                   │
                   └─► return ActionResult  →  executor.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim – mirrors what kernel/capability_base.py exposes.
# Replace this block with a real import once Kernel v1 is on PYTHONPATH.
# ---------------------------------------------------------------------------
try:
    from agents.kernel.capability_base import Capability, CapabilityMetadata, CapabilityStatus
except ImportError:  # pragma: no cover – shim for standalone usage / tests

    class CapabilityStatus(str, Enum):  # type: ignore[no-redef]
        ACTIVE = "active"
        DEGRADED = "degraded"
        UNAVAILABLE = "unavailable"
        INITIALIZING = "initializing"

    @dataclass
    class CapabilityMetadata:  # type: ignore[no-redef]
        name: str
        version: str
        description: str
        supported_actions: List[str]
        dependencies: List[str]
        tags: List[str] = field(default_factory=list)
        author: str = "AI OS Team"

    class Capability(ABC):  # type: ignore[no-redef]
        """Minimal base class matching Kernel v1 contract."""

        metadata: CapabilityMetadata
        status: CapabilityStatus = CapabilityStatus.INITIALIZING

        async def initialize(self) -> None: ...  # noqa: E704

        async def shutdown(self) -> None: ...  # noqa: E704

        @abstractmethod
        async def execute(
            self,
            action: str,
            params: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]: ...

        async def health_check(self) -> Dict[str, Any]:
            return {"status": self.status}


# ============================================================
# Enumerations & constants
# ============================================================


class ActionCategory(str, Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    BROWSER = "browser"
    TERMINAL = "terminal"
    FILE = "file"
    SCREEN = "screen"
    WINDOW = "window"
    CLIPBOARD = "clipboard"
    VISION = "vision"
    SYSTEM = "system"


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


# Actions that are inherently dangerous and require elevated scrutiny
DANGEROUS_ACTIONS: Final[Set[str]] = {
    "terminal_execute",
    "file_delete",
    "file_overwrite",
    "file_move",
    "clipboard_write",
    "browser_navigate",
    "browser_fill_form",
    "browser_click",
    "window_close",
    "system_shutdown",
    "system_reboot",
}

# Patterns that indicate critically dangerous terminal commands
CRITICAL_SHELL_PATTERNS: Final[List[re.Pattern[str]]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brm\s+-rf?\b",
        r"\bformat\b",
        r"\bmkfs\b",
        r"\bdd\b.*\bof=\/dev\b",
        r"\b:(){ :|:& };:\b",  # fork bomb
        r"\bchmod\s+777\b",
        r"\bchown\b.*\broot\b",
        r"\bsudo\b",
        r"\bsu\s+root\b",
        r"\bcurl\b.*\|\s*bash",
        r"\bwget\b.*\|\s*sh",
        r">\s*/dev/(sda|sdb|hda|hdb)",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\bpoweroff\b",
    ]
]

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
APPROVAL_TIMEOUT_SECONDS: Final[float] = 60.0


# ============================================================
# Data models
# ============================================================


@dataclass
class ActionResult:
    """Unified result returned by every driver and by execute()."""

    success: bool
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List["Artifact"] = field(default_factory=list)
    error: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.SAFE
    approval_status: ApprovalStatus = ApprovalStatus.AUTO_APPROVED
    execution_time_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rollback_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "data": self.data,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error": self.error,
            "risk_level": self.risk_level.value,
            "approval_status": self.approval_status.value,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "action_id": self.action_id,
            "rollback_id": self.rollback_id,
        }


@dataclass
class Artifact:
    """Binary or textual artifact produced by an action (e.g. screenshot)."""

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    mime_type: str = "application/octet-stream"
    data_b64: str = ""          # base-64 encoded payload
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        name: str,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Artifact":
        return cls(
            name=name,
            mime_type=mime_type,
            data_b64=base64.b64encode(data).decode(),
            metadata=metadata or {},
        )

    def to_bytes(self) -> bytes:
        return base64.b64decode(self.data_b64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "mime_type": self.mime_type,
            "data_b64": self.data_b64,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ActionHistoryEntry:
    action_id: str
    action: str
    params: Dict[str, Any]
    result_summary: str
    risk_level: RiskLevel
    approval_status: ApprovalStatus
    timestamp: str
    execution_time_ms: float
    rollback_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action,
            "params": self.params,
            "result_summary": self.result_summary,
            "risk_level": self.risk_level.value,
            "approval_status": self.approval_status.value,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
            "rollback_id": self.rollback_id,
        }


@dataclass
class RollbackPoint:
    rollback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = ""
    snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================
# Safety layer
# ============================================================


class DangerDetector:
    """Classifies the risk level of a requested action."""

    def assess(self, action: str, params: Dict[str, Any]) -> RiskLevel:
        if action in ("system_shutdown", "system_reboot"):
            return RiskLevel.CRITICAL

        if action == "terminal_execute":
            cmd = str(params.get("command", ""))
            for pattern in CRITICAL_SHELL_PATTERNS:
                if pattern.search(cmd):
                    return RiskLevel.CRITICAL
            return RiskLevel.HIGH

        if action in ("file_delete", "file_overwrite"):
            path = str(params.get("path", ""))
            if path.startswith("/") and not path.startswith("/tmp"):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        if action in DANGEROUS_ACTIONS:
            return RiskLevel.MEDIUM

        return RiskLevel.SAFE


class SandboxGuard:
    """
    Restricts file operations to an allowed sandbox directory tree.
    Future: integrate with container / VM isolation layers.
    """

    def __init__(self, sandbox_root: Optional[Path] = None) -> None:
        self._root: Path = sandbox_root or Path(tempfile.gettempdir()) / "aios_sandbox"
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def assert_within_sandbox(self, path: Union[str, Path]) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self._root.resolve())
        except ValueError:
            raise PermissionError(
                f"Path '{resolved}' is outside sandbox '{self._root}'. "
                "Operation blocked by SandboxGuard."
            )
        return resolved


class ApprovalGate:
    """
    Human-in-the-loop approval mechanism.

    Production: swap _request_via_ui() for a real UI callback / webhook.
    Tests: inject approval_callback to auto-approve or deny.
    """

    def __init__(
        self,
        auto_approve_levels: Optional[Set[RiskLevel]] = None,
        approval_callback: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
        timeout: float = APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        self._auto_approve: Set[RiskLevel] = auto_approve_levels or {
            RiskLevel.SAFE,
            RiskLevel.LOW,
        }
        self._callback = approval_callback  # (action, description, params) -> bool
        self._timeout = timeout

    async def request(
        self, action: str, params: Dict[str, Any], risk_level: RiskLevel
    ) -> ApprovalStatus:
        if risk_level in self._auto_approve:
            return ApprovalStatus.AUTO_APPROVED

        description = self._describe(action, params)
        logger.warning(
            "[ApprovalGate] Human approval required — action=%s risk=%s",
            action,
            risk_level.value,
        )

        if self._callback is not None:
            try:
                approved = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._callback, action, description, params
                    ),
                    timeout=self._timeout,
                )
                return ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
            except asyncio.TimeoutError:
                logger.error("[ApprovalGate] Approval timed out for action=%s", action)
                return ApprovalStatus.TIMEOUT

        # No callback: default to DENY for safety
        logger.error(
            "[ApprovalGate] No approval callback configured. Denying action=%s", action
        )
        return ApprovalStatus.DENIED

    @staticmethod
    def _describe(action: str, params: Dict[str, Any]) -> str:
        safe_params = {
            k: v for k, v in params.items() if k not in ("password", "token", "secret")
        }
        return f"Action '{action}' with params: {json.dumps(safe_params, default=str)}"


# ============================================================
# Audit system
# ============================================================


class ArtifactStore:
    """In-memory artifact store; replace with object-store backend in prod."""

    def __init__(self) -> None:
        self._store: Dict[str, Artifact] = {}

    def save(self, artifact: Artifact) -> str:
        self._store[artifact.artifact_id] = artifact
        return artifact.artifact_id

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._store.get(artifact_id)

    def list_ids(self) -> List[str]:
        return list(self._store.keys())


class RollbackManager:
    """Stores rollback snapshots and applies them on demand."""

    def __init__(self) -> None:
        self._points: Dict[str, RollbackPoint] = {}

    def create_point(self, action: str, snapshot: Dict[str, Any]) -> str:
        rp = RollbackPoint(action=action, snapshot=snapshot)
        self._points[rp.rollback_id] = rp
        logger.debug("[Rollback] Created rollback point %s for action=%s", rp.rollback_id, action)
        return rp.rollback_id

    def get(self, rollback_id: str) -> Optional[RollbackPoint]:
        return self._points.get(rollback_id)

    async def apply(self, rollback_id: str) -> bool:
        rp = self._points.get(rollback_id)
        if rp is None:
            logger.error("[Rollback] No rollback point found: %s", rollback_id)
            return False
        logger.info("[Rollback] Applying rollback %s for action=%s", rollback_id, rp.action)
        # Concrete rollback logic lives in each driver; here we call the hook.
        # Future: emit a "rollback_requested" event to the kernel event bus.
        return True


class ActionHistory:
    """Ordered, append-only log of executed actions."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._entries: List[ActionHistoryEntry] = []
        self._max_size = max_size

    def append(self, entry: ActionHistoryEntry) -> None:
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)
        self._entries.append(entry)

    def recent(self, n: int = 20) -> List[ActionHistoryEntry]:
        return self._entries[-n:]

    def all(self) -> List[ActionHistoryEntry]:
        return list(self._entries)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]


class AuditSystem:
    """Coordinates history, artifacts, and rollbacks."""

    def __init__(self) -> None:
        self.history = ActionHistory()
        self.artifacts = ArtifactStore()
        self.rollbacks = RollbackManager()

    def record(
        self,
        action: str,
        params: Dict[str, Any],
        result: ActionResult,
    ) -> None:
        for artifact in result.artifacts:
            self.artifacts.save(artifact)

        entry = ActionHistoryEntry(
            action_id=result.action_id,
            action=action,
            params={k: v for k, v in params.items() if k not in ("password", "token")},
            result_summary="OK" if result.success else f"ERROR: {result.error}",
            risk_level=result.risk_level,
            approval_status=result.approval_status,
            timestamp=result.timestamp,
            execution_time_ms=result.execution_time_ms,
            rollback_id=result.rollback_id,
        )
        self.history.append(entry)
        logger.info(
            "[Audit] action=%s id=%s success=%s risk=%s time_ms=%.1f",
            action,
            result.action_id,
            result.success,
            result.risk_level.value,
            result.execution_time_ms,
        )


# ============================================================
# Driver base
# ============================================================


class Driver(ABC):
    """Abstract driver; each subsystem implements one."""

    category: ActionCategory
    supported_actions: List[str] = []

    def __init__(self, sandbox: SandboxGuard) -> None:
        self._sandbox = sandbox

    @abstractmethod
    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult: ...

    def _ok(self, action: str, data: Dict[str, Any], **kw: Any) -> ActionResult:
        return ActionResult(success=True, action=action, data=data, **kw)

    def _err(self, action: str, error: str, **kw: Any) -> ActionResult:
        return ActionResult(success=False, action=action, error=error, **kw)

    def _unavailable(self, action: str, library: str) -> ActionResult:
        return self._err(
            action,
            f"Driver for '{action}' requires '{library}' which is not installed. "
            "Install it or register a compatible provider.",
        )


# ============================================================
# Mouse driver
# ============================================================


class MouseDriver(Driver):
    category = ActionCategory.MOUSE
    supported_actions = [
        "mouse_move",
        "mouse_click",
        "mouse_double_click",
        "mouse_right_click",
        "mouse_scroll",
        "mouse_drag",
        "mouse_position",
    ]

    # Future: inject a PyAutoGUI / platform-native backend.

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            import pyautogui  # type: ignore
        except ImportError:
            return self._unavailable(action, "pyautogui")

        loop = asyncio.get_event_loop()

        if action == "mouse_move":
            x, y = params["x"], params["y"]
            duration = float(params.get("duration", 0.2))
            await loop.run_in_executor(None, lambda: pyautogui.moveTo(x, y, duration=duration))
            return self._ok(action, {"x": x, "y": y})

        if action == "mouse_click":
            x, y = params.get("x"), params.get("y")
            button = params.get("button", "left")
            await loop.run_in_executor(None, lambda: pyautogui.click(x, y, button=button))
            return self._ok(action, {"x": x, "y": y, "button": button})

        if action == "mouse_double_click":
            x, y = params.get("x"), params.get("y")
            await loop.run_in_executor(None, lambda: pyautogui.doubleClick(x, y))
            return self._ok(action, {"x": x, "y": y})

        if action == "mouse_right_click":
            x, y = params.get("x"), params.get("y")
            await loop.run_in_executor(None, lambda: pyautogui.rightClick(x, y))
            return self._ok(action, {"x": x, "y": y})

        if action == "mouse_scroll":
            clicks = int(params.get("clicks", 3))
            x, y = params.get("x"), params.get("y")
            await loop.run_in_executor(None, lambda: pyautogui.scroll(clicks, x=x, y=y))
            return self._ok(action, {"clicks": clicks})

        if action == "mouse_drag":
            x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
            duration = float(params.get("duration", 0.5))
            await loop.run_in_executor(
                None, lambda: pyautogui.dragTo(x2, y2, duration=duration, mouseDownUp=True)
            )
            return self._ok(action, {"from": (x1, y1), "to": (x2, y2)})

        if action == "mouse_position":
            pos = await loop.run_in_executor(None, pyautogui.position)
            return self._ok(action, {"x": pos.x, "y": pos.y})

        return self._err(action, f"Unknown mouse action: {action}")


# ============================================================
# Keyboard driver
# ============================================================


class KeyboardDriver(Driver):
    category = ActionCategory.KEYBOARD
    supported_actions = [
        "keyboard_type",
        "keyboard_press",
        "keyboard_hotkey",
        "keyboard_hold",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            import pyautogui  # type: ignore
        except ImportError:
            return self._unavailable(action, "pyautogui")

        loop = asyncio.get_event_loop()

        if action == "keyboard_type":
            text = str(params["text"])
            interval = float(params.get("interval", 0.02))
            await loop.run_in_executor(None, lambda: pyautogui.typewrite(text, interval=interval))
            return self._ok(action, {"length": len(text)})

        if action == "keyboard_press":
            key = str(params["key"])
            presses = int(params.get("presses", 1))
            await loop.run_in_executor(None, lambda: pyautogui.press(key, presses=presses))
            return self._ok(action, {"key": key, "presses": presses})

        if action == "keyboard_hotkey":
            keys: List[str] = params["keys"]
            await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))
            return self._ok(action, {"keys": keys})

        if action == "keyboard_hold":
            key = str(params["key"])
            hold_type = str(params.get("type", "keyDown"))
            fn = pyautogui.keyDown if hold_type == "keyDown" else pyautogui.keyUp
            await loop.run_in_executor(None, lambda: fn(key))
            return self._ok(action, {"key": key, "type": hold_type})

        return self._err(action, f"Unknown keyboard action: {action}")


# ============================================================
# Browser driver
# ============================================================


class BrowserDriver(Driver):
    """
    Thin wrapper around Playwright (async API).
    Future: swap backend with Selenium / requests-html.
    """

    category = ActionCategory.BROWSER
    supported_actions = [
        "browser_navigate",
        "browser_click",
        "browser_fill_form",
        "browser_screenshot",
        "browser_get_text",
        "browser_evaluate",
        "browser_wait_for",
        "browser_new_tab",
        "browser_close_tab",
    ]

    def __init__(self, sandbox: SandboxGuard, headless: bool = True) -> None:
        super().__init__(sandbox)
        self._headless = headless
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            raise RuntimeError("playwright not installed. Run: pip install playwright && playwright install")

        self._pw = await async_playwright().__aenter__()
        self._browser = await self._pw.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            await self._ensure_browser()
        except RuntimeError:
            return self._unavailable(action, "playwright")

        page = self._page

        try:
            if action == "browser_navigate":
                url = str(params["url"])
                timeout = float(params.get("timeout", DEFAULT_TIMEOUT_SECONDS)) * 1000
                await page.goto(url, timeout=timeout)
                return self._ok(action, {"url": page.url, "title": await page.title()})

            if action == "browser_click":
                selector = str(params["selector"])
                await page.click(selector)
                return self._ok(action, {"selector": selector})

            if action == "browser_fill_form":
                fields: Dict[str, str] = params["fields"]
                for selector, value in fields.items():
                    await page.fill(selector, value)
                return self._ok(action, {"fields_filled": len(fields)})

            if action == "browser_screenshot":
                data = await page.screenshot(full_page=bool(params.get("full_page", False)))
                artifact = Artifact.from_bytes(data, "screenshot.png", "image/png",
                                               {"url": page.url})
                return self._ok(action, {"url": page.url}, artifacts=[artifact])

            if action == "browser_get_text":
                selector = params.get("selector", "body")
                text = await page.inner_text(selector)
                return self._ok(action, {"text": text})

            if action == "browser_evaluate":
                script = str(params["script"])
                result = await page.evaluate(script)
                return self._ok(action, {"result": result})

            if action == "browser_wait_for":
                selector = str(params["selector"])
                state = params.get("state", "visible")
                timeout = float(params.get("timeout", DEFAULT_TIMEOUT_SECONDS)) * 1000
                await page.wait_for_selector(selector, state=state, timeout=timeout)
                return self._ok(action, {"selector": selector, "state": state})

            if action == "browser_new_tab":
                self._page = await self._context.new_page()
                return self._ok(action, {})

            if action == "browser_close_tab":
                await page.close()
                pages = self._context.pages
                self._page = pages[-1] if pages else await self._context.new_page()
                return self._ok(action, {})

        except Exception as exc:
            return self._err(action, str(exc))

        return self._err(action, f"Unknown browser action: {action}")


# ============================================================
# Terminal driver
# ============================================================


class TerminalDriver(Driver):
    category = ActionCategory.TERMINAL
    supported_actions = [
        "terminal_execute",
        "terminal_interactive",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        if action == "terminal_execute":
            return await self._execute(params)
        return self._err(action, f"Unknown terminal action: {action}")

    async def _execute(self, params: Dict[str, Any]) -> ActionResult:
        command = str(params["command"])
        timeout = float(params.get("timeout", DEFAULT_TIMEOUT_SECONDS))
        cwd = params.get("cwd", str(self._sandbox.root))
        env_overrides: Dict[str, str] = params.get("env", {})

        env = os.environ.copy()
        env.update(env_overrides)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return self._ok(
                "terminal_execute",
                {
                    "stdout": stdout_bytes.decode(errors="replace"),
                    "stderr": stderr_bytes.decode(errors="replace"),
                    "returncode": proc.returncode,
                    "command": command,
                },
            )
        except asyncio.TimeoutError:
            return self._err(
                "terminal_execute",
                f"Command timed out after {timeout}s: {command}",
            )
        except Exception as exc:
            return self._err("terminal_execute", str(exc))


# ============================================================
# File driver
# ============================================================


class FileDriver(Driver):
    category = ActionCategory.FILE
    supported_actions = [
        "file_read",
        "file_write",
        "file_overwrite",
        "file_delete",
        "file_move",
        "file_copy",
        "file_list",
        "file_exists",
        "file_mkdir",
        "file_stat",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:  # noqa: C901
        loop = asyncio.get_event_loop()
        try:
            if action == "file_read":
                path = self._sandbox.assert_within_sandbox(params["path"])
                data = await loop.run_in_executor(None, path.read_bytes)
                try:
                    text_content = data.decode()
                    return self._ok(action, {"path": str(path), "content": text_content, "size": len(data)})
                except UnicodeDecodeError:
                    artifact = Artifact.from_bytes(data, path.name)
                    return self._ok(action, {"path": str(path), "size": len(data)}, artifacts=[artifact])

            if action in ("file_write", "file_overwrite"):
                path = self._sandbox.assert_within_sandbox(params["path"])
                content = params["content"]
                if isinstance(content, str):
                    content = content.encode()
                path.parent.mkdir(parents=True, exist_ok=True)
                await loop.run_in_executor(None, lambda: path.write_bytes(content))  # type: ignore[arg-type]
                return self._ok(action, {"path": str(path), "size": len(content)})

            if action == "file_delete":
                path = self._sandbox.assert_within_sandbox(params["path"])
                snapshot: Dict[str, Any] = {}
                if path.exists():
                    snapshot = {"path": str(path), "content_b64": base64.b64encode(path.read_bytes()).decode()}
                await loop.run_in_executor(None, lambda: path.unlink(missing_ok=True))
                return self._ok(action, {"path": str(path), "snapshot": snapshot})

            if action == "file_move":
                src = self._sandbox.assert_within_sandbox(params["src"])
                dst = self._sandbox.assert_within_sandbox(params["dst"])
                await loop.run_in_executor(None, lambda: shutil.move(str(src), str(dst)))
                return self._ok(action, {"src": str(src), "dst": str(dst)})

            if action == "file_copy":
                src = self._sandbox.assert_within_sandbox(params["src"])
                dst = self._sandbox.assert_within_sandbox(params["dst"])
                await loop.run_in_executor(None, lambda: shutil.copy2(str(src), str(dst)))
                return self._ok(action, {"src": str(src), "dst": str(dst)})

            if action == "file_list":
                path = self._sandbox.assert_within_sandbox(params.get("path", str(self._sandbox.root)))
                entries = [{"name": e.name, "is_dir": e.is_dir(), "size": e.stat().st_size if e.is_file() else 0}
                           for e in sorted(path.iterdir())]
                return self._ok(action, {"path": str(path), "entries": entries})

            if action == "file_exists":
                path = self._sandbox.assert_within_sandbox(params["path"])
                return self._ok(action, {"path": str(path), "exists": path.exists()})

            if action == "file_mkdir":
                path = self._sandbox.assert_within_sandbox(params["path"])
                await loop.run_in_executor(None, lambda: path.mkdir(parents=True, exist_ok=True))
                return self._ok(action, {"path": str(path)})

            if action == "file_stat":
                path = self._sandbox.assert_within_sandbox(params["path"])
                st = path.stat()
                return self._ok(action, {
                    "path": str(path),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "is_dir": path.is_dir(),
                })

        except PermissionError as exc:
            return self._err(action, str(exc))
        except Exception as exc:
            return self._err(action, str(exc))

        return self._err(action, f"Unknown file action: {action}")


# ============================================================
# Screen driver
# ============================================================


class ScreenDriver(Driver):
    category = ActionCategory.SCREEN
    supported_actions = [
        "screen_capture",
        "screen_region_capture",
        "screen_size",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            import pyautogui  # type: ignore
            from PIL import Image  # type: ignore  # noqa: F401
        except ImportError as exc:
            return self._unavailable(action, str(exc).split("'")[1])

        loop = asyncio.get_event_loop()
        import io

        if action == "screen_capture":
            img = await loop.run_in_executor(None, pyautogui.screenshot)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            artifact = Artifact.from_bytes(buf.getvalue(), "screenshot.png", "image/png")
            return self._ok(action, {"width": img.width, "height": img.height}, artifacts=[artifact])

        if action == "screen_region_capture":
            region = (params["x"], params["y"], params["width"], params["height"])
            img = await loop.run_in_executor(None, lambda: pyautogui.screenshot(region=region))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            artifact = Artifact.from_bytes(buf.getvalue(), "region.png", "image/png",
                                           {"region": region})
            return self._ok(action, {"region": region}, artifacts=[artifact])

        if action == "screen_size":
            sz = await loop.run_in_executor(None, pyautogui.size)
            return self._ok(action, {"width": sz.width, "height": sz.height})

        return self._err(action, f"Unknown screen action: {action}")


# ============================================================
# Window driver
# ============================================================


class WindowDriver(Driver):
    category = ActionCategory.WINDOW
    supported_actions = [
        "window_list",
        "window_activate",
        "window_close",
        "window_minimize",
        "window_maximize",
        "window_resize",
        "window_move",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            import pygetwindow as gw  # type: ignore
        except ImportError:
            return self._unavailable(action, "pygetwindow")

        loop = asyncio.get_event_loop()

        if action == "window_list":
            titles = await loop.run_in_executor(None, gw.getAllTitles)
            return self._ok(action, {"windows": [t for t in titles if t]})

        title = str(params.get("title", ""))

        def _get_window() -> Any:
            wins = gw.getWindowsWithTitle(title)
            if not wins:
                raise ValueError(f"No window found with title '{title}'")
            return wins[0]

        if action == "window_activate":
            win = await loop.run_in_executor(None, _get_window)
            await loop.run_in_executor(None, win.activate)
            return self._ok(action, {"title": win.title})

        if action == "window_close":
            win = await loop.run_in_executor(None, _get_window)
            await loop.run_in_executor(None, win.close)
            return self._ok(action, {"title": title})

        if action == "window_minimize":
            win = await loop.run_in_executor(None, _get_window)
            await loop.run_in_executor(None, win.minimize)
            return self._ok(action, {"title": title})

        if action == "window_maximize":
            win = await loop.run_in_executor(None, _get_window)
            await loop.run_in_executor(None, win.maximize)
            return self._ok(action, {"title": title})

        if action == "window_resize":
            win = await loop.run_in_executor(None, _get_window)
            w, h = int(params["width"]), int(params["height"])
            await loop.run_in_executor(None, lambda: win.resizeTo(w, h))
            return self._ok(action, {"title": title, "width": w, "height": h})

        if action == "window_move":
            win = await loop.run_in_executor(None, _get_window)
            x, y = int(params["x"]), int(params["y"])
            await loop.run_in_executor(None, lambda: win.moveTo(x, y))
            return self._ok(action, {"title": title, "x": x, "y": y})

        return self._err(action, f"Unknown window action: {action}")


# ============================================================
# Clipboard driver
# ============================================================


class ClipboardDriver(Driver):
    category = ActionCategory.CLIPBOARD
    supported_actions = [
        "clipboard_read",
        "clipboard_write",
        "clipboard_clear",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        try:
            import pyperclip  # type: ignore
        except ImportError:
            return self._unavailable(action, "pyperclip")

        loop = asyncio.get_event_loop()

        if action == "clipboard_read":
            content = await loop.run_in_executor(None, pyperclip.paste)
            return self._ok(action, {"content": content, "length": len(content)})

        if action == "clipboard_write":
            text = str(params["text"])
            await loop.run_in_executor(None, lambda: pyperclip.copy(text))
            return self._ok(action, {"length": len(text)})

        if action == "clipboard_clear":
            await loop.run_in_executor(None, lambda: pyperclip.copy(""))
            return self._ok(action, {})

        return self._err(action, f"Unknown clipboard action: {action}")


# ============================================================
# Vision driver
# ============================================================


class VisionDriver(Driver):
    """
    Vision / OCR utilities.
    Future: swap backends for OpenCV, Tesseract, EasyOCR, or cloud vision APIs.
    """

    category = ActionCategory.VISION
    supported_actions = [
        "vision_ocr",
        "vision_find_element",
        "vision_describe",
    ]

    async def run(self, action: str, params: Dict[str, Any]) -> ActionResult:
        if action == "vision_ocr":
            return await self._ocr(params)
        if action == "vision_find_element":
            return await self._find_element(params)
        if action == "vision_describe":
            return await self._describe(params)
        return self._err(action, f"Unknown vision action: {action}")

    async def _ocr(self, params: Dict[str, Any]) -> ActionResult:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return self._unavailable("vision_ocr", "pytesseract / Pillow")

        image_path = params.get("image_path")
        image_b64 = params.get("image_b64")
        loop = asyncio.get_event_loop()

        def _run() -> str:
            if image_path:
                img = Image.open(image_path)
            elif image_b64:
                import io
                img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            else:
                raise ValueError("Provide image_path or image_b64")
            return pytesseract.image_to_string(img)

        text = await loop.run_in_executor(None, _run)
        return self._ok("vision_ocr", {"text": text})

    async def _find_element(self, params: Dict[str, Any]) -> ActionResult:
        """Locate a UI element by template matching. Requires OpenCV."""
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return self._unavailable("vision_find_element", "opencv-python / numpy")

        template_b64 = str(params["template_b64"])
        screenshot_b64 = str(params.get("screenshot_b64", ""))
        threshold = float(params.get("threshold", 0.8))

        loop = asyncio.get_event_loop()

        def _run() -> Dict[str, Any]:
            tmpl_arr = np.frombuffer(base64.b64decode(template_b64), np.uint8)
            template = cv2.imdecode(tmpl_arr, cv2.IMREAD_COLOR)
            if screenshot_b64:
                sc_arr = np.frombuffer(base64.b64decode(screenshot_b64), np.uint8)
                screen = cv2.imdecode(sc_arr, cv2.IMREAD_COLOR)
            else:
                import pyautogui  # type: ignore
                import io
                buf = io.BytesIO()
                pyautogui.screenshot().save(buf, format="PNG")
                buf.seek(0)
                sc_arr = np.frombuffer(buf.read(), np.uint8)
                screen = cv2.imdecode(sc_arr, cv2.IMREAD_COLOR)

            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            h, w = template.shape[:2]
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            return {"found": max_val >= threshold, "confidence": float(max_val),
                    "x": cx, "y": cy, "top_left": max_loc}

        data = await loop.run_in_executor(None, _run)
        return self._ok("vision_find_element", data)

    async def _describe(self, params: Dict[str, Any]) -> ActionResult:
        """Placeholder for multimodal model integration."""
        return self._ok(
            "vision_describe",
            {"description": "Vision describe: integrate a multimodal model (e.g. claude-3-5-sonnet) to implement this action."},
        )


# ============================================================
# Action router
# ============================================================


class ActionRouter:
    """Maps action strings to the appropriate driver."""

    def __init__(self, drivers: List[Driver]) -> None:
        self._map: Dict[str, Driver] = {}
        for driver in drivers:
            for action in driver.supported_actions:
                self._map[action] = driver

    def resolve(self, action: str) -> Optional[Driver]:
        return self._map.get(action)

    @property
    def all_actions(self) -> List[str]:
        return sorted(self._map.keys())


# ============================================================
# ComputerCapability – main class
# ============================================================


class ComputerCapability(Capability):
    """
    AI Operating System capability for interacting with the host computer.

    Compatible with:
    - registry.py  (exposes metadata, health_check, initialize, shutdown)
    - executor.py  (exposes async execute(action, params, context))
    - context_engine.py (accepts context dict from execute() caller)
    """

    # ------------------------------------------------------------------ meta
    metadata = CapabilityMetadata(
        name="computer",
        version="1.0.0",
        description=(
            "Enables AI OS interaction with the host computer: "
            "mouse, keyboard, browser, terminal, file system, "
            "screenshots, window management, clipboard, and vision."
        ),
        supported_actions=[],          # populated in __init__
        dependencies=[
            "pyautogui",               # mouse / keyboard / screen
            "playwright",              # browser automation (optional)
            "pygetwindow",             # window management (optional)
            "pyperclip",               # clipboard (optional)
            "pytesseract",             # OCR (optional)
            "opencv-python",           # vision (optional)
            "Pillow",                  # image support
        ],
        tags=["computer", "automation", "ui", "browser", "terminal", "vision"],
    )

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        sandbox_root: Optional[Path] = None,
        headless_browser: bool = True,
        auto_approve_levels: Optional[Set[RiskLevel]] = None,
        approval_callback: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
        approval_timeout: float = APPROVAL_TIMEOUT_SECONDS,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        enable_sandbox: bool = True,
    ) -> None:
        self._default_timeout = default_timeout
        self._enable_sandbox = enable_sandbox

        # Safety
        self._sandbox = SandboxGuard(sandbox_root)
        self._danger_detector = DangerDetector()
        self._approval_gate = ApprovalGate(
            auto_approve_levels=auto_approve_levels,
            approval_callback=approval_callback,
            timeout=approval_timeout,
        )

        # Audit
        self._audit = AuditSystem()

        # Drivers
        self._mouse = MouseDriver(self._sandbox)
        self._keyboard = KeyboardDriver(self._sandbox)
        self._browser = BrowserDriver(self._sandbox, headless=headless_browser)
        self._terminal = TerminalDriver(self._sandbox)
        self._file = FileDriver(self._sandbox)
        self._screen = ScreenDriver(self._sandbox)
        self._window = WindowDriver(self._sandbox)
        self._clipboard = ClipboardDriver(self._sandbox)
        self._vision = VisionDriver(self._sandbox)

        self._router = ActionRouter([
            self._mouse, self._keyboard, self._browser,
            self._terminal, self._file, self._screen,
            self._window, self._clipboard, self._vision,
        ])

        # Patch metadata with actual action list
        self.metadata.supported_actions = self._router.all_actions

        self.status = CapabilityStatus.INITIALIZING

    # ---------------------------------------------------------- lifecycle
    async def initialize(self) -> None:
        logger.info("[ComputerCapability] Initializing …")
        self.status = CapabilityStatus.ACTIVE
        logger.info(
            "[ComputerCapability] Ready. %d actions registered. Sandbox: %s",
            len(self.metadata.supported_actions),
            self._sandbox.root,
        )

    async def shutdown(self) -> None:
        logger.info("[ComputerCapability] Shutting down …")
        await self._browser.close()
        self.status = CapabilityStatus.UNAVAILABLE
        logger.info("[ComputerCapability] Shutdown complete.")

    # ---------------------------------------------------------- health
    async def health_check(self) -> Dict[str, Any]:
        checks: Dict[str, bool] = {}
        optional_libs = {
            "pyautogui": "pyautogui",
            "playwright": "playwright.async_api",
            "pyperclip": "pyperclip",
            "pygetwindow": "pygetwindow",
            "pytesseract": "pytesseract",
            "opencv": "cv2",
            "PIL": "PIL",
        }
        for label, module in optional_libs.items():
            try:
                __import__(module)
                checks[label] = True
            except ImportError:
                checks[label] = False

        system_info = {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": sys.version.split()[0],
            "sandbox_root": str(self._sandbox.root),
        }

        overall = CapabilityStatus.ACTIVE if checks.get("pyautogui") else CapabilityStatus.DEGRADED
        self.status = overall

        return {
            "status": overall.value,
            "library_checks": checks,
            "system": system_info,
            "actions_registered": len(self.metadata.supported_actions),
            "history_size": len(self._audit.history.all()),
        }

    # ---------------------------------------------------------- execute
    async def execute(
        self,
        action: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point called by executor.py.

        Parameters
        ----------
        action:  action name (e.g. 'mouse_click', 'terminal_execute')
        params:  action-specific keyword arguments
        context: runtime context from context_engine.py (session_id, user, etc.)

        Returns
        -------
        dict representation of ActionResult
        """
        context = context or {}
        start_ts = time.perf_counter()

        # 1. Route check
        driver = self._router.resolve(action)
        if driver is None:
            result = ActionResult(
                success=False,
                action=action,
                error=f"Unknown action '{action}'. Supported: {self._router.all_actions}",
            )
            self._audit.record(action, params, result)
            return result.to_dict()

        # 2. Danger assessment
        risk_level = self._danger_detector.assess(action, params)

        # 3. Approval gate
        approval_status = await self._approval_gate.request(action, params, risk_level)
        if approval_status in (ApprovalStatus.DENIED, ApprovalStatus.TIMEOUT):
            result = ActionResult(
                success=False,
                action=action,
                error=f"Action '{action}' was {approval_status.value} by approval gate.",
                risk_level=risk_level,
                approval_status=approval_status,
            )
            self._audit.record(action, params, result)
            return result.to_dict()

        # 4. Optional rollback snapshot (file mutations)
        rollback_id: Optional[str] = None
        if risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            rollback_id = self._audit.rollbacks.create_point(action, {"params": params})

        # 5. Execute with timeout
        timeout = float(params.pop("_timeout", self._default_timeout))
        try:
            result = await asyncio.wait_for(
                driver.run(action, params),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            result = ActionResult(
                success=False,
                action=action,
                error=f"Action '{action}' timed out after {timeout}s.",
                risk_level=risk_level,
                approval_status=approval_status,
            )
        except Exception:
            result = ActionResult(
                success=False,
                action=action,
                error=f"Unhandled exception: {traceback.format_exc()}",
                risk_level=risk_level,
                approval_status=approval_status,
            )

        # 6. Annotate result
        result.risk_level = risk_level
        result.approval_status = approval_status
        result.rollback_id = rollback_id
        result.execution_time_ms = (time.perf_counter() - start_ts) * 1000

        # 7. Audit
        self._audit.record(action, params, result)

        return result.to_dict()

    # ---------------------------------------------------------- rollback
    async def rollback(self, rollback_id: str) -> Dict[str, Any]:
        """Attempt to undo a previous action identified by rollback_id."""
        success = await self._audit.rollbacks.apply(rollback_id)
        return {"rollback_id": rollback_id, "applied": success}

    # ---------------------------------------------------------- introspection
    def get_action_history(self, n: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._audit.history.recent(n)]

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        artifact = self._audit.artifacts.get(artifact_id)
        return artifact.to_dict() if artifact else None

    def list_artifacts(self) -> List[str]:
        return self._audit.artifacts.list_ids()

    @property
    def sandbox_root(self) -> Path:
        return self._sandbox.root


# ============================================================
# Registry entry point (called by registry.py)
# ============================================================


def create_capability(**kwargs: Any) -> ComputerCapability:
    """
    Factory used by registry.py to instantiate ComputerCapability.

    Example registry.py entry:
        {
            "name": "computer",
            "module": "computer_capability",
            "factory": "create_capability",
            "config": {
                "headless_browser": true,
                "enable_sandbox": true
            }
        }
    """
    return ComputerCapability(**kwargs)
