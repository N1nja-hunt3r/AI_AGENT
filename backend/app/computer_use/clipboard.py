from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Optional

from app.computer_use.base_computer import (
    CapabilityMetadata,
    CapabilityPriority,
    ComputerCapability,
    ExecutionRequest,
    HealthCheckResult,
    HealthStatus,
)


class ClipboardContentType(Enum):
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClipboardEntry:
    content: str
    content_type: ClipboardContentType
    captured_at_epoch: float
    source: str = "clipboard"


class ClipboardError(Exception):
    pass


class ClipboardBackend:
    async def get_text(self) -> str:
        raise NotImplementedError

    async def set_text(self, text: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError


class PyperclipClipboardBackend(ClipboardBackend):
    def __init__(self) -> None:
        self._pyperclip: Any = None

    def _load(self) -> Any:
        if self._pyperclip is None:
            import pyperclip

            self._pyperclip = pyperclip
        return self._pyperclip

    async def get_text(self) -> str:
        pyperclip = self._load()
        return await asyncio.to_thread(pyperclip.paste)

    async def set_text(self, text: str) -> None:
        pyperclip = self._load()
        await asyncio.to_thread(pyperclip.copy, text)

    async def clear(self) -> None:
        pyperclip = self._load()
        await asyncio.to_thread(pyperclip.copy, "")


class ClipboardCapability(ComputerCapability):
    def __init__(
        self,
        backend: Optional[ClipboardBackend] = None,
        history_max_entries: int = 100,
        metadata: Optional[CapabilityMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="clipboard",
                version="1.0.0",
                description="Reads, writes, and tracks history of system clipboard content",
                priority=CapabilityPriority.NORMAL,
                timeout_seconds=5.0,
                approval_required=False,
            )
        )
        self._backend = backend or PyperclipClipboardBackend()
        self._history: Deque[ClipboardEntry] = deque(maxlen=history_max_entries)
        self._history_lock = asyncio.Lock()

    async def _on_initialize(self) -> None:
        pass

    async def _on_shutdown(self) -> None:
        pass

    async def _on_health_check(self) -> HealthCheckResult:
        try:
            await self._backend.get_text()
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.HEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
            )
        except Exception as exc:
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=str(exc),
            )

    async def _on_execute(self, request: ExecutionRequest) -> Any:
        action = request.action
        params = request.parameters

        if action == "copy":
            return await self.copy(str(params["text"]))
        if action == "paste":
            return await self.paste()
        if action == "clear":
            return await self.clear()
        if action == "history":
            return await self.history(int(params.get("limit", 50)))

        raise ValueError(f"unknown clipboard action: {action}")

    async def copy(self, text: str) -> ClipboardEntry:
        await self._backend.set_text(text)
        entry = ClipboardEntry(
            content=text,
            content_type=ClipboardContentType.TEXT,
            captured_at_epoch=time.time(),
            source="copy",
        )
        await self._record(entry)
        return entry

    async def paste(self) -> str:
        text = await self._backend.get_text()
        entry = ClipboardEntry(
            content=text,
            content_type=ClipboardContentType.TEXT,
            captured_at_epoch=time.time(),
            source="paste",
        )
        await self._record(entry)
        return text

    async def clear(self) -> None:
        await self._backend.clear()
        entry = ClipboardEntry(
            content="",
            content_type=ClipboardContentType.TEXT,
            captured_at_epoch=time.time(),
            source="clear",
        )
        await self._record(entry)

    async def history(self, limit: int = 50) -> list[ClipboardEntry]:
        async with self._history_lock:
            items = list(self._history)
        return items[-limit:]

    async def _record(self, entry: ClipboardEntry) -> None:
        async with self._history_lock:
            self._history.append(entry)
