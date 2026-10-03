from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from app.computer_use.base_computer import (
    CapabilityMetadata,
    CapabilityPriority,
    ComputerCapability,
    ExecutionRequest,
    HealthCheckResult,
    HealthStatus,
)


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class EasingFunction(Enum):
    LINEAR = "linear"
    EASE_IN_OUT = "ease_in_out"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class MouseActionResult:
    action: str
    position: Point
    duration_seconds: float
    detail: str = ""


class MouseBackend:
    async def move_to(self, x: int, y: int) -> None:
        raise NotImplementedError

    async def click(self, button: MouseButton) -> None:
        raise NotImplementedError

    async def double_click(self, button: MouseButton) -> None:
        raise NotImplementedError

    async def mouse_down(self, button: MouseButton) -> None:
        raise NotImplementedError

    async def mouse_up(self, button: MouseButton) -> None:
        raise NotImplementedError

    async def scroll(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    async def get_position(self) -> Point:
        raise NotImplementedError

    async def get_screen_bounds(self) -> tuple[int, int]:
        raise NotImplementedError


class PyAutoGuiMouseBackend(MouseBackend):
    def __init__(self) -> None:
        self._pyautogui: Any = None

    def _load(self) -> Any:
        if self._pyautogui is None:
            import pyautogui

            pyautogui.FAILSAFE = False
            self._pyautogui = pyautogui
        return self._pyautogui

    async def move_to(self, x: int, y: int) -> None:
        pyautogui = self._load()
        await asyncio.to_thread(pyautogui.moveTo, x, y)

    async def click(self, button: MouseButton) -> None:
        pyautogui = self._load()
        await asyncio.to_thread(pyautogui.click, button=button.value)

    async def double_click(self, button: MouseButton) -> None:
        pyautogui = self._load()
        await asyncio.to_thread(pyautogui.doubleClick, button=button.value)

    async def mouse_down(self, button: MouseButton) -> None:
        pyautogui = self._load()
        await asyncio.to_thread(pyautogui.mouseDown, button=button.value)

    async def mouse_up(self, button: MouseButton) -> None:
        pyautogui = self._load()
        await asyncio.to_thread(pyautogui.mouseUp, button=button.value)

    async def scroll(self, dx: int, dy: int) -> None:
        pyautogui = self._load()
        if dy:
            await asyncio.to_thread(pyautogui.vscroll, dy)
        if dx:
            await asyncio.to_thread(pyautogui.hscroll, dx)

    async def get_position(self) -> Point:
        pyautogui = self._load()
        pos = await asyncio.to_thread(pyautogui.position)
        return Point(x=int(pos.x), y=int(pos.y))

    async def get_screen_bounds(self) -> tuple[int, int]:
        pyautogui = self._load()
        size = await asyncio.to_thread(pyautogui.size)
        return int(size.width), int(size.height)


class MouseError(Exception):
    pass


class MouseOutOfBoundsError(MouseError):
    pass


class MouseCapability(ComputerCapability):
    def __init__(
        self,
        backend: Optional[MouseBackend] = None,
        metadata: Optional[CapabilityMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="mouse",
                version="1.0.0",
                description="Controls mouse movement, clicks, drags, and scrolling",
                priority=CapabilityPriority.HIGH,
                timeout_seconds=10.0,
                approval_required=False,
            )
        )
        self._backend = backend or PyAutoGuiMouseBackend()
        self._bounds: Optional[tuple[int, int]] = None

    async def _on_initialize(self) -> None:
        self._bounds = await self._backend.get_screen_bounds()

    async def _on_shutdown(self) -> None:
        self._bounds = None

    async def _on_health_check(self) -> HealthCheckResult:
        try:
            position = await self._backend.get_position()
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.HEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=f"position={position.x},{position.y}",
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

        if action == "move":
            return await self.move(int(params["x"]), int(params["y"]))
        if action == "click":
            return await self.click(
                int(params["x"]),
                int(params["y"]),
                MouseButton(params.get("button", "left")),
            )
        if action == "double_click":
            return await self.double_click(
                int(params["x"]),
                int(params["y"]),
                MouseButton(params.get("button", "left")),
            )
        if action == "drag":
            return await self.drag(
                int(params["start_x"]),
                int(params["start_y"]),
                int(params["end_x"]),
                int(params["end_y"]),
                MouseButton(params.get("button", "left")),
                float(params.get("duration_seconds", 0.3)),
            )
        if action == "scroll":
            return await self.scroll(int(params.get("dx", 0)), int(params.get("dy", 0)))
        if action == "position":
            return await self.position()

        raise ValueError(f"unknown mouse action: {action}")

    def _validate_bounds(self, x: int, y: int) -> None:
        if self._bounds is None:
            return
        max_x, max_y = self._bounds
        if x < 0 or y < 0 or x > max_x or y > max_y:
            raise MouseOutOfBoundsError(
                f"coordinates ({x}, {y}) are outside screen bounds {self._bounds}"
            )

    async def move(
        self, x: int, y: int, duration_seconds: float = 0.0
    ) -> MouseActionResult:
        self._validate_bounds(x, y)
        start = time.monotonic()
        if duration_seconds > 0:
            await self._move_smoothly(x, y, duration_seconds)
        else:
            await self._backend.move_to(x, y)
        return MouseActionResult(
            action="move",
            position=Point(x=x, y=y),
            duration_seconds=time.monotonic() - start,
        )

    async def _move_smoothly(self, x: int, y: int, duration_seconds: float) -> None:
        start_pos = await self._backend.get_position()
        steps = max(1, int(duration_seconds * 60))
        for step in range(1, steps + 1):
            progress = step / steps
            interp_x = int(start_pos.x + (x - start_pos.x) * progress)
            interp_y = int(start_pos.y + (y - start_pos.y) * progress)
            await self._backend.move_to(interp_x, interp_y)
            await asyncio.sleep(duration_seconds / steps)

    async def click(
        self, x: int, y: int, button: MouseButton = MouseButton.LEFT
    ) -> MouseActionResult:
        self._validate_bounds(x, y)
        start = time.monotonic()
        await self._backend.move_to(x, y)
        await self._backend.click(button)
        return MouseActionResult(
            action="click",
            position=Point(x=x, y=y),
            duration_seconds=time.monotonic() - start,
            detail=button.value,
        )

    async def double_click(
        self, x: int, y: int, button: MouseButton = MouseButton.LEFT
    ) -> MouseActionResult:
        self._validate_bounds(x, y)
        start = time.monotonic()
        await self._backend.move_to(x, y)
        await self._backend.double_click(button)
        return MouseActionResult(
            action="double_click",
            position=Point(x=x, y=y),
            duration_seconds=time.monotonic() - start,
            detail=button.value,
        )

    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: MouseButton = MouseButton.LEFT,
        duration_seconds: float = 0.3,
    ) -> MouseActionResult:
        self._validate_bounds(start_x, start_y)
        self._validate_bounds(end_x, end_y)
        start = time.monotonic()
        await self._backend.move_to(start_x, start_y)
        await self._backend.mouse_down(button)
        try:
            await self._move_smoothly(end_x, end_y, duration_seconds)
        finally:
            await self._backend.mouse_up(button)
        return MouseActionResult(
            action="drag",
            position=Point(x=end_x, y=end_y),
            duration_seconds=time.monotonic() - start,
            detail=f"from=({start_x},{start_y}) button={button.value}",
        )

    async def scroll(self, dx: int = 0, dy: int = 0) -> MouseActionResult:
        start = time.monotonic()
        await self._backend.scroll(dx, dy)
        position = await self._backend.get_position()
        return MouseActionResult(
            action="scroll",
            position=position,
            duration_seconds=time.monotonic() - start,
            detail=f"dx={dx} dy={dy}",
        )

    async def position(self) -> Point:
        return await self._backend.get_position()
