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
class Region:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class MonitorInfo:
    monitor_id: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool


@dataclass(frozen=True)
class CaptureResult:
    image_bytes: bytes
    width: int
    height: int
    format: str
    captured_at_epoch: float
    monitor_id: Optional[int] = None
    region: Optional[Region] = None


class ScreenError(Exception):
    pass


class MonitorNotFoundError(ScreenError):
    pass


class ScreenBackend:
    async def list_monitors(self) -> list[MonitorInfo]:
        raise NotImplementedError

    async def capture_full(self, monitor_id: Optional[int]) -> tuple[bytes, int, int]:
        raise NotImplementedError

    async def capture_region(self, region: Region) -> tuple[bytes, int, int]:
        raise NotImplementedError


class MssScreenBackend(ScreenBackend):
    def __init__(self) -> None:
        self._mss_module: Any = None

    def _load(self) -> Any:
        if self._mss_module is None:
            import mss

            self._mss_module = mss
        return self._mss_module

    def _list_monitors_sync(self) -> list[MonitorInfo]:
        mss_module = self._load()
        monitors: list[MonitorInfo] = []
        with mss_module.mss() as sct:
            for index, mon in enumerate(sct.monitors):
                if index == 0:
                    continue
                monitors.append(
                    MonitorInfo(
                        monitor_id=index,
                        x=mon["left"],
                        y=mon["top"],
                        width=mon["width"],
                        height=mon["height"],
                        is_primary=(index == 1),
                    )
                )
        return monitors

    async def list_monitors(self) -> list[MonitorInfo]:
        return await asyncio.to_thread(self._list_monitors_sync)

    def _capture_full_sync(self, monitor_id: Optional[int]) -> tuple[bytes, int, int]:
        mss_module = self._load()
        with mss_module.mss() as sct:
            target_index = monitor_id if monitor_id is not None else 1
            if target_index >= len(sct.monitors):
                raise MonitorNotFoundError(f"monitor {target_index} not found")
            shot = sct.grab(sct.monitors[target_index])
            png_bytes = mss_module.tools.to_png(shot.rgb, shot.size)
            return png_bytes, shot.size[0], shot.size[1]

    async def capture_full(self, monitor_id: Optional[int]) -> tuple[bytes, int, int]:
        return await asyncio.to_thread(self._capture_full_sync, monitor_id)

    def _capture_region_sync(self, region: Region) -> tuple[bytes, int, int]:
        mss_module = self._load()
        bbox = {
            "left": region.x,
            "top": region.y,
            "width": region.width,
            "height": region.height,
        }
        with mss_module.mss() as sct:
            shot = sct.grab(bbox)
            png_bytes = mss_module.tools.to_png(shot.rgb, shot.size)
            return png_bytes, shot.size[0], shot.size[1]

    async def capture_region(self, region: Region) -> tuple[bytes, int, int]:
        return await asyncio.to_thread(self._capture_region_sync, region)


class ScreenCapability(ComputerCapability):
    def __init__(
        self,
        backend: Optional[ScreenBackend] = None,
        metadata: Optional[CapabilityMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="screen",
                version="1.0.0",
                description="Captures screenshots, regions, and monitor info",
                priority=CapabilityPriority.HIGH,
                timeout_seconds=10.0,
                approval_required=False,
            )
        )
        self._backend = backend or MssScreenBackend()
        self._monitors_cache: list[MonitorInfo] = []

    async def _on_initialize(self) -> None:
        self._monitors_cache = await self._backend.list_monitors()

    async def _on_shutdown(self) -> None:
        self._monitors_cache = []

    async def _on_health_check(self) -> HealthCheckResult:
        try:
            monitors = await self._backend.list_monitors()
            if not monitors:
                return HealthCheckResult(
                    capability_name=self.name,
                    status=HealthStatus.DEGRADED,
                    latency_seconds=None,
                    checked_at_epoch=time.time(),
                    detail="no monitors detected",
                )
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.HEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=f"monitors={len(monitors)}",
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

        if action == "screenshot":
            return await self.screenshot(params.get("monitor_id"))
        if action == "capture_region":
            return await self.capture_region(
                Region(
                    x=int(params["x"]),
                    y=int(params["y"]),
                    width=int(params["width"]),
                    height=int(params["height"]),
                )
            )
        if action == "monitor_selection":
            return await self.monitor_selection()
        if action == "save_image":
            return await self.save_image(bytes(params["image_bytes"]), str(params["path"]))

        raise ValueError(f"unknown screen action: {action}")

    async def screenshot(self, monitor_id: Optional[int] = None) -> CaptureResult:
        image_bytes, width, height = await self._backend.capture_full(monitor_id)
        return CaptureResult(
            image_bytes=image_bytes,
            width=width,
            height=height,
            format="png",
            captured_at_epoch=time.time(),
            monitor_id=monitor_id,
        )

    async def capture_region(self, region: Region) -> CaptureResult:
        if region.width <= 0 or region.height <= 0:
            raise ScreenError("region width and height must be positive")
        image_bytes, width, height = await self._backend.capture_region(region)
        return CaptureResult(
            image_bytes=image_bytes,
            width=width,
            height=height,
            format="png",
            captured_at_epoch=time.time(),
            region=region,
        )

    async def monitor_selection(self) -> list[MonitorInfo]:
        self._monitors_cache = await self._backend.list_monitors()
        return list(self._monitors_cache)

    async def save_image(self, image_bytes: bytes, path: str) -> str:
        def write() -> None:
            with open(path, "wb") as handle:
                handle.write(image_bytes)

        await asyncio.to_thread(write)
        return path
