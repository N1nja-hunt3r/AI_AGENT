from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.computer_use.base_computer import (
    CapabilityMetadata,
    CapabilityPriority,
    ComputerCapability,
    ExecutionRequest,
    HealthCheckResult,
    HealthStatus,
)


@dataclass(frozen=True)
class KeyboardActionResult:
    action: str
    detail: str
    duration_seconds: float


class KeyboardError(Exception):
    pass


class KeyboardBackend:
    async def type_text(self, text: str, interval_seconds: float) -> None:
        raise NotImplementedError

    async def key_down(self, key: str) -> None:
        raise NotImplementedError

    async def key_up(self, key: str) -> None:
        raise NotImplementedError

    async def press_key(self, key: str) -> None:
        raise NotImplementedError

    async def hotkey(self, keys: tuple[str, ...]) -> None:
        raise NotImplementedError

    async def get_clipboard(self) -> str:
        raise NotImplementedError

    async def set_clipboard(self, text: str) -> None:
        raise NotImplementedError


class PyAutoGuiKeyboardBackend(KeyboardBackend):
    def __init__(self) -> None:
        self._pyautogui: Any = None
        self._pyperclip: Any = None

    def _load_pyautogui(self) -> Any:
        if self._pyautogui is None:
            import pyautogui

            pyautogui.FAILSAFE = False
            self._pyautogui = pyautogui
        return self._pyautogui

    def _load_pyperclip(self) -> Any:
        if self._pyperclip is None:
            import pyperclip

            self._pyperclip = pyperclip
        return self._pyperclip

    async def type_text(self, text: str, interval_seconds: float) -> None:
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.typewrite, text, interval=interval_seconds)

    async def key_down(self, key: str) -> None:
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.keyDown, key)

    async def key_up(self, key: str) -> None:
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.keyUp, key)

    async def press_key(self, key: str) -> None:
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.press, key)

    async def hotkey(self, keys: tuple[str, ...]) -> None:
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.hotkey, *keys)

    async def get_clipboard(self) -> str:
        pyperclip = self._load_pyperclip()
        return await asyncio.to_thread(pyperclip.paste)

    async def set_clipboard(self, text: str) -> None:
        pyperclip = self._load_pyperclip()
        await asyncio.to_thread(pyperclip.copy, text)


class KeyboardCapability(ComputerCapability):
    def __init__(
        self,
        backend: Optional[KeyboardBackend] = None,
        metadata: Optional[CapabilityMetadata] = None,
        default_type_interval_seconds: float = 0.01,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="keyboard",
                version="1.0.0",
                description="Controls keyboard typing, hotkeys, and clipboard text",
                priority=CapabilityPriority.HIGH,
                timeout_seconds=15.0,
                approval_required=False,
            )
        )
        self._backend = backend or PyAutoGuiKeyboardBackend()
        self._default_type_interval_seconds = default_type_interval_seconds
        self._held_keys: set[str] = set()

    async def _on_initialize(self) -> None:
        self._held_keys = set()

    async def _on_shutdown(self) -> None:
        for key in list(self._held_keys):
            try:
                await self._backend.key_up(key)
            except Exception:
                pass
        self._held_keys.clear()

    async def _on_health_check(self) -> HealthCheckResult:
        try:
            await self._backend.get_clipboard()
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.HEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
            )
        except Exception as exc:
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.DEGRADED,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=str(exc),
            )

    async def _on_execute(self, request: ExecutionRequest) -> Any:
        action = request.action
        params = request.parameters

        if action == "type_text":
            return await self.type_text(
                str(params["text"]),
                float(params.get("interval_seconds", self._default_type_interval_seconds)),
            )
        if action == "hotkey":
            return await self.hotkey(tuple(params["keys"]))
        if action == "press":
            return await self.press(str(params["key"]))
        if action == "release":
            return await self.release(str(params["key"]))
        if action == "copy":
            return await self.copy(params.get("text"))
        if action == "paste":
            return await self.paste()

        raise ValueError(f"unknown keyboard action: {action}")

    async def type_text(
        self, text: str, interval_seconds: Optional[float] = None
    ) -> KeyboardActionResult:
        start = time.monotonic()
        interval = interval_seconds if interval_seconds is not None else self._default_type_interval_seconds
        await self._backend.type_text(text, interval)
        return KeyboardActionResult(
            action="type_text",
            detail=f"length={len(text)}",
            duration_seconds=time.monotonic() - start,
        )

    async def hotkey(self, keys: tuple[str, ...]) -> KeyboardActionResult:
        if not keys:
            raise KeyboardError("hotkey requires at least one key")
        start = time.monotonic()
        await self._backend.hotkey(keys)
        return KeyboardActionResult(
            action="hotkey",
            detail="+".join(keys),
            duration_seconds=time.monotonic() - start,
        )

    async def press(self, key: str) -> KeyboardActionResult:
        start = time.monotonic()
        await self._backend.key_down(key)
        self._held_keys.add(key)
        return KeyboardActionResult(
            action="press",
            detail=key,
            duration_seconds=time.monotonic() - start,
        )

    async def release(self, key: str) -> KeyboardActionResult:
        start = time.monotonic()
        await self._backend.key_up(key)
        self._held_keys.discard(key)
        return KeyboardActionResult(
            action="release",
            detail=key,
            duration_seconds=time.monotonic() - start,
        )

    async def copy(self, text: Optional[str] = None) -> KeyboardActionResult:
        start = time.monotonic()
        if text is not None:
            await self._backend.set_clipboard(text)
            detail = f"set_clipboard length={len(text)}"
        else:
            await self._backend.hotkey(("ctrl", "c"))
            detail = "copy_selection"
        return KeyboardActionResult(
            action="copy",
            detail=detail,
            duration_seconds=time.monotonic() - start,
        )

    async def paste(self) -> str:
        await self._backend.hotkey(("ctrl", "v"))
        return await self._backend.get_clipboard()
