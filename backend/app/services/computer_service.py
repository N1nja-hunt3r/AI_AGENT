from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()  # "Linux" | "Darwin" | "Windows"


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class KeyModifier(str, Enum):
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    META = "meta"
    WIN = "win"
    CMD = "command"


class BrowserAction(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    WAIT = "wait"
    EVALUATE = "evaluate"
    BACK = "back"
    FORWARD = "forward"
    RELOAD = "reload"


class TerminalMode(str, Enum):
    SHELL = "shell"
    PYTHON = "python"
    NODE = "node"


@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class MousePosition:
    x: int
    y: int


@dataclass
class TerminalResult:
    stdout: str
    stderr: str
    return_code: int
    duration_ms: float
    command: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "timed_out": self.timed_out,
        }


@dataclass
class BrowserResult:
    action: BrowserAction
    success: bool
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    screenshot_b64: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "success": self.success,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "has_screenshot": self.screenshot_b64 is not None,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class VisionResult:
    description: str
    objects: List[Dict[str, Any]] = field(default_factory=list)
    text_content: str = ""
    screenshot_b64: Optional[str] = None
    region: Optional[ScreenRegion] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "objects": self.objects,
            "text_content": self.text_content,
            "has_screenshot": self.screenshot_b64 is not None,
        }


@dataclass
class WindowInfo:
    title: str
    pid: int
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "pid": self.pid,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "is_active": self.is_active,
        }


@dataclass
class ComputerServiceConfig:
    screenshot_format: str = "png"
    screenshot_quality: int = 85
    terminal_timeout: float = 30.0
    terminal_shell: str = "/bin/bash" if _SYSTEM != "Windows" else "cmd"
    browser_headless: bool = True
    browser_timeout: float = 30.0
    browser_viewport_width: int = 1280
    browser_viewport_height: int = 800
    mouse_move_duration: float = 0.1
    typing_interval: float = 0.02
    vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    vision_api_key: Optional[str] = None
    clipboard_timeout: float = 5.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ComputerServiceError(Exception):
    pass

class MouseError(ComputerServiceError):
    pass

class KeyboardError(ComputerServiceError):
    pass

class BrowserError(ComputerServiceError):
    pass

class TerminalError(ComputerServiceError):
    pass

class VisionError(ComputerServiceError):
    pass

class WindowError(ComputerServiceError):
    pass


# ---------------------------------------------------------------------------
# Mouse controller
# ---------------------------------------------------------------------------

class MouseController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config

    def _get_pyautogui(self) -> Any:
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.01
            return pyautogui
        except ImportError as exc:
            raise MouseError("pip install pyautogui") from exc

    async def move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        pag = self._get_pyautogui()
        dur = duration if duration is not None else self._config.mouse_move_duration
        await asyncio.to_thread(pag.moveTo, x, y, duration=dur)

    async def click(
        self,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.click, x, y, button=button.value, clicks=clicks, interval=interval)

    async def double_click(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> None:
        await self.click(x, y, button, clicks=2)

    async def right_click(self, x: int, y: int) -> None:
        await self.click(x, y, MouseButton.RIGHT)

    async def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.dragTo, x2, y2, duration=duration, startX=x1, startY=y1)

    async def scroll(self, x: int, y: int, dx: int = 0, dy: int = 3) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.moveTo, x, y)
        await asyncio.to_thread(pag.scroll, dy, x=x, y=y)

    async def position(self) -> MousePosition:
        pag = self._get_pyautogui()
        pos = await asyncio.to_thread(pag.position)
        return MousePosition(x=pos.x, y=pos.y)

    async def screen_size(self) -> Tuple[int, int]:
        pag = self._get_pyautogui()
        size = await asyncio.to_thread(pag.size)
        return size.width, size.height


# ---------------------------------------------------------------------------
# Keyboard controller
# ---------------------------------------------------------------------------

class KeyboardController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config

    def _get_pyautogui(self) -> Any:
        try:
            import pyautogui
            return pyautogui
        except ImportError as exc:
            raise KeyboardError("pip install pyautogui") from exc

    async def type_text(self, text: str, interval: Optional[float] = None) -> None:
        pag = self._get_pyautogui()
        ivl = interval if interval is not None else self._config.typing_interval
        await asyncio.to_thread(pag.typewrite, text, interval=ivl)

    async def press(self, key: str) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.press, key)

    async def hotkey(self, *keys: str) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.hotkey, *keys)

    async def key_down(self, key: str) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.keyDown, key)

    async def key_up(self, key: str) -> None:
        pag = self._get_pyautogui()
        await asyncio.to_thread(pag.keyUp, key)

    async def copy(self) -> None:
        mod = "command" if _SYSTEM == "Darwin" else "ctrl"
        await self.hotkey(mod, "c")

    async def paste(self) -> None:
        mod = "command" if _SYSTEM == "Darwin" else "ctrl"
        await self.hotkey(mod, "v")

    async def select_all(self) -> None:
        mod = "command" if _SYSTEM == "Darwin" else "ctrl"
        await self.hotkey(mod, "a")

    async def undo(self) -> None:
        mod = "command" if _SYSTEM == "Darwin" else "ctrl"
        await self.hotkey(mod, "z")


# ---------------------------------------------------------------------------
# Browser controller (Playwright)
# ---------------------------------------------------------------------------

class BrowserController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config
        self._browser: Any = None
        self._page: Any = None
        self._playwright: Any = None

    async def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise BrowserError("pip install playwright && playwright install chromium") from exc
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._config.browser_headless
        )
        context = await self._browser.new_context(
            viewport={
                "width": self._config.browser_viewport_width,
                "height": self._config.browser_viewport_height,
            }
        )
        self._page = await context.new_page()

    async def navigate(self, url: str) -> BrowserResult:
        t0 = time.monotonic()
        await self._ensure_browser()
        try:
            await self._page.goto(url, timeout=self._config.browser_timeout * 1000)
            title = await self._page.title()
            return BrowserResult(
                action=BrowserAction.NAVIGATE,
                success=True,
                url=self._page.url,
                title=title,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BrowserResult(
                action=BrowserAction.NAVIGATE,
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def click(self, selector: str) -> BrowserResult:
        t0 = time.monotonic()
        await self._ensure_browser()
        try:
            await self._page.click(selector, timeout=self._config.browser_timeout * 1000)
            return BrowserResult(
                action=BrowserAction.CLICK,
                success=True,
                url=self._page.url,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BrowserResult(action=BrowserAction.CLICK, success=False, error=str(exc))

    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> BrowserResult:
        t0 = time.monotonic()
        await self._ensure_browser()
        try:
            if clear_first:
                await self._page.fill(selector, "")
            await self._page.type(selector, text)
            return BrowserResult(
                action=BrowserAction.TYPE,
                success=True,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BrowserResult(action=BrowserAction.TYPE, success=False, error=str(exc))

    async def screenshot(self, region: Optional[ScreenRegion] = None) -> BrowserResult:
        t0 = time.monotonic()
        await self._ensure_browser()
        try:
            clip = None
            if region:
                clip = {"x": region.x, "y": region.y, "width": region.width, "height": region.height}
            data = await self._page.screenshot(type="png", clip=clip)
            b64 = base64.b64encode(data).decode()
            return BrowserResult(
                action=BrowserAction.SCREENSHOT,
                success=True,
                url=self._page.url,
                screenshot_b64=b64,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BrowserResult(action=BrowserAction.SCREENSHOT, success=False, error=str(exc))

    async def get_content(self) -> BrowserResult:
        await self._ensure_browser()
        try:
            content = await self._page.inner_text("body")
            title = await self._page.title()
            return BrowserResult(
                action=BrowserAction.SCREENSHOT,
                success=True,
                url=self._page.url,
                title=title,
                content=content,
            )
        except Exception as exc:
            return BrowserResult(action=BrowserAction.SCREENSHOT, success=False, error=str(exc))

    async def evaluate(self, script: str) -> BrowserResult:
        await self._ensure_browser()
        try:
            result = await self._page.evaluate(script)
            return BrowserResult(
                action=BrowserAction.EVALUATE,
                success=True,
                content=str(result),
            )
        except Exception as exc:
            return BrowserResult(action=BrowserAction.EVALUATE, success=False, error=str(exc))

    async def scroll(self, x: int = 0, y: int = 500) -> BrowserResult:
        await self._ensure_browser()
        try:
            await self._page.evaluate(f"window.scrollBy({x}, {y})")
            return BrowserResult(action=BrowserAction.SCROLL, success=True)
        except Exception as exc:
            return BrowserResult(action=BrowserAction.SCROLL, success=False, error=str(exc))

    async def back(self) -> BrowserResult:
        await self._ensure_browser()
        try:
            await self._page.go_back()
            return BrowserResult(action=BrowserAction.BACK, success=True, url=self._page.url)
        except Exception as exc:
            return BrowserResult(action=BrowserAction.BACK, success=False, error=str(exc))

    async def wait_for(self, selector: str, timeout: Optional[float] = None) -> BrowserResult:
        await self._ensure_browser()
        try:
            await self._page.wait_for_selector(
                selector, timeout=(timeout or self._config.browser_timeout) * 1000
            )
            return BrowserResult(action=BrowserAction.WAIT, success=True)
        except Exception as exc:
            return BrowserResult(action=BrowserAction.WAIT, success=False, error=str(exc))

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


# ---------------------------------------------------------------------------
# Terminal controller
# ---------------------------------------------------------------------------

class TerminalController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config
        self._env: Dict[str, str] = dict(os.environ)
        self._cwd: str = os.path.expanduser("~")

    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalResult:
        t0 = time.monotonic()
        effective_timeout = timeout or self._config.terminal_timeout
        effective_cwd = cwd or self._cwd
        effective_env = {**self._env, **(env or {})}

        async def _run() -> TerminalResult:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
                env=effective_env,
                executable=self._config.terminal_shell if _SYSTEM != "Windows" else None,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
                return TerminalResult(
                    stdout=stdout.decode(errors="replace"),
                    stderr=stderr.decode(errors="replace"),
                    return_code=proc.returncode or 0,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    command=command,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return TerminalResult(
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    return_code=-1,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    command=command,
                    timed_out=True,
                )

        return await _run()

    async def execute_batch(
        self,
        commands: List[str],
        stop_on_error: bool = True,
        **kwargs: Any,
    ) -> List[TerminalResult]:
        results: List[TerminalResult] = []
        for cmd in commands:
            r = await self.execute(cmd, **kwargs)
            results.append(r)
            if stop_on_error and not r.success:
                break
        return results

    def set_cwd(self, path: str) -> None:
        if not os.path.isdir(path):
            raise TerminalError(f"Directory not found: {path}")
        self._cwd = path

    def set_env(self, key: str, value: str) -> None:
        self._env[key] = value

    def get_cwd(self) -> str:
        return self._cwd


# ---------------------------------------------------------------------------
# Vision controller
# ---------------------------------------------------------------------------

class VisionController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config

    async def screenshot(self, region: Optional[ScreenRegion] = None) -> str:
        try:
            import pyautogui  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise VisionError("pip install pyautogui pillow") from exc

        def _take() -> str:
            if region:
                img = pyautogui.screenshot(region=(region.x, region.y, region.width, region.height))
            else:
                img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format=self._config.screenshot_format.upper(),
                     quality=self._config.screenshot_quality)
            return base64.b64encode(buf.getvalue()).decode()

        return await asyncio.to_thread(_take)

    async def describe_screen(self, region: Optional[ScreenRegion] = None) -> VisionResult:
        b64 = await self.screenshot(region)
        description = await self._analyze_with_vision(b64, "Describe what you see on the screen.")
        return VisionResult(
            description=description,
            screenshot_b64=b64,
            region=region,
        )

    async def find_text_on_screen(self, target_text: str) -> VisionResult:
        b64 = await self.screenshot()
        prompt = f"Find all occurrences of '{target_text}' on the screen. Report their approximate pixel coordinates."
        result = await self._analyze_with_vision(b64, prompt)
        return VisionResult(description=result, text_content=target_text, screenshot_b64=b64)

    async def ocr(self, region: Optional[ScreenRegion] = None) -> str:
        b64 = await self.screenshot(region)
        return await self._analyze_with_vision(b64, "Extract all visible text from this image. Return only the text content.")

    async def _analyze_with_vision(self, image_b64: str, prompt: str) -> str:
        try:
            import os
            import base64
            from openai import AsyncOpenAI
            base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            api_key = self._config.vision_api_key or os.environ.get("NVIDIA_VISION_API_KEY", "")
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
            data_url = f"data:image/png;base64,{image_b64}"
            resp = await client.chat.completions.create(
                model=self._config.vision_model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Vision analysis failed: %s", exc)
            return f"[Vision error: {exc}]"


# ---------------------------------------------------------------------------
# Clipboard controller
# ---------------------------------------------------------------------------

class ClipboardController:
    def __init__(self, config: ComputerServiceConfig) -> None:
        self._config = config

    def _get_pyperclip(self) -> Any:
        try:
            import pyperclip
            return pyperclip
        except ImportError as exc:
            raise ComputerServiceError("pip install pyperclip") from exc

    async def get(self) -> str:
        pc = self._get_pyperclip()
        return await asyncio.to_thread(pc.paste)

    async def set(self, text: str) -> None:
        pc = self._get_pyperclip()
        await asyncio.to_thread(pc.copy, text)

    async def clear(self) -> None:
        await self.set("")


# ---------------------------------------------------------------------------
# Window manager
# ---------------------------------------------------------------------------

class WindowManager:
    def __init__(self) -> None:
        pass

    async def list_windows(self) -> List[WindowInfo]:
        if _SYSTEM == "Darwin":
            return await self._list_macos()
        if _SYSTEM == "Linux":
            return await self._list_linux()
        if _SYSTEM == "Windows":
            return await self._list_windows()
        return []

    async def get_active_window(self) -> Optional[WindowInfo]:
        windows = await self.list_windows()
        for w in windows:
            if w.is_active:
                return w
        return None

    async def focus_window(self, title: str) -> bool:
        if _SYSTEM == "Darwin":
            script = f'tell application "{title}" to activate'
            result = await asyncio.to_thread(
                subprocess.run, ["osascript", "-e", script],
                capture_output=True
            )
            return result.returncode == 0
        if _SYSTEM == "Linux":
            result = await asyncio.to_thread(
                subprocess.run, ["wmctrl", "-a", title],
                capture_output=True
            )
            return result.returncode == 0
        logger.warning("focus_window not implemented for %s", _SYSTEM)
        return False

    async def close_window(self, title: str) -> bool:
        if _SYSTEM == "Linux":
            result = await asyncio.to_thread(
                subprocess.run, ["wmctrl", "-c", title],
                capture_output=True
            )
            return result.returncode == 0
        logger.warning("close_window not implemented for %s", _SYSTEM)
        return False

    async def move_window(self, title: str, x: int, y: int, width: int, height: int) -> bool:
        if _SYSTEM == "Linux":
            result = await asyncio.to_thread(
                subprocess.run,
                ["wmctrl", "-r", title, "-e", f"0,{x},{y},{width},{height}"],
                capture_output=True,
            )
            return result.returncode == 0
        logger.warning("move_window not implemented for %s", _SYSTEM)
        return False

    async def _list_macos(self) -> List[WindowInfo]:
        script = (
            'tell application "System Events" to get {name, unix id} '
            'of every process whose background only is false'
        )
        try:
            result = await asyncio.to_thread(
                subprocess.run, ["osascript", "-e", script],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().split(", ")
            windows: List[WindowInfo] = []
            half = len(lines) // 2
            for i in range(half):
                try:
                    pid = int(lines[half + i])
                    windows.append(WindowInfo(title=lines[i], pid=pid))
                except (ValueError, IndexError):
                    pass
            return windows
        except Exception:
            return []

    async def _list_linux(self) -> List[WindowInfo]:
        try:
            result = await asyncio.to_thread(
                subprocess.run, ["wmctrl", "-l", "-p"],
                capture_output=True, text=True
            )
            windows: List[WindowInfo] = []
            for line in result.stdout.strip().splitlines():
                parts = line.split(None, 4)
                if len(parts) >= 5:
                    try:
                        windows.append(WindowInfo(title=parts[4], pid=int(parts[2])))
                    except ValueError:
                        pass
            return windows
        except Exception:
            return []

    async def _list_windows(self) -> List[WindowInfo]:
        try:
            import ctypes
            windows: List[WindowInfo] = []

            def callback(hwnd: int, _: Any) -> bool:
                import ctypes
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                        pid = ctypes.c_ulong()
                        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        windows.append(WindowInfo(title=buf.value, pid=pid.value))
                return True

            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            ctypes.windll.user32.EnumWindows(EnumWindowsProc(callback), 0)
            return windows
        except Exception:
            return []


# ---------------------------------------------------------------------------
# ComputerService
# ---------------------------------------------------------------------------

class ComputerService:
    """
    Unified computer automation service covering mouse, keyboard, browser,
    terminal, vision, clipboard, and window management.
    """

    def __init__(self, config: Optional[ComputerServiceConfig] = None) -> None:
        self._config = config or ComputerServiceConfig()
        self.mouse = MouseController(self._config)
        self.keyboard = KeyboardController(self._config)
        self.browser = BrowserController(self._config)
        self.terminal = TerminalController(self._config)
        self.vision = VisionController(self._config)
        self.clipboard = ClipboardController(self._config)
        self.windows = WindowManager()
        self._action_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Composite helpers
    # ------------------------------------------------------------------

    async def click_and_type(self, x: int, y: int, text: str) -> None:
        await self.mouse.click(x, y)
        await asyncio.sleep(0.1)
        await self.keyboard.type_text(text)
        self._log("click_and_type", {"x": x, "y": y, "text": text[:30]})

    async def find_and_click(self, target_text: str) -> bool:
        result = await self.vision.find_text_on_screen(target_text)
        self._log("find_and_click", {"target": target_text, "found": bool(result.description)})
        return bool(result.description)

    async def run_command(self, command: str, **kwargs: Any) -> TerminalResult:
        result = await self.terminal.execute(command, **kwargs)
        self._log("run_command", {"command": command, "success": result.success})
        return result

    async def browse_and_extract(self, url: str) -> Dict[str, Any]:
        nav = await self.browser.navigate(url)
        if not nav.success:
            return {"success": False, "error": nav.error}
        content = await self.browser.get_content()
        shot = await self.browser.screenshot()
        self._log("browse_and_extract", {"url": url})
        return {
            "success": True,
            "url": nav.url,
            "title": content.title,
            "content": content.content,
            "screenshot_b64": shot.screenshot_b64,
        }

    async def copy_to_clipboard(self) -> str:
        await self.keyboard.copy()
        await asyncio.sleep(0.2)
        text = await self.clipboard.get()
        self._log("copy_to_clipboard", {"chars": len(text)})
        return text

    async def paste_from_clipboard(self, text: Optional[str] = None) -> None:
        if text is not None:
            await self.clipboard.set(text)
            await asyncio.sleep(0.1)
        await self.keyboard.paste()
        self._log("paste_from_clipboard", {})

    async def take_screenshot(self, region: Optional[ScreenRegion] = None) -> str:
        b64 = await self.vision.screenshot(region)
        self._log("take_screenshot", {"region": region})
        return b64

    async def describe_screen(self) -> str:
        result = await self.vision.describe_screen()
        return result.description

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self.browser.close()

    async def __aenter__(self) -> ComputerService:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def get_action_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._action_log[-limit:]

    def clear_log(self) -> None:
        self._action_log.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_actions": len(self._action_log),
            "system": _SYSTEM,
            "browser_headless": self._config.browser_headless,
            "terminal_shell": self._config.terminal_shell,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, action: str, details: Dict[str, Any]) -> None:
        self._action_log.append({
            "action": action,
            "details": details,
            "timestamp": time.time(),
        })
        if len(self._action_log) > 2000:
            self._action_log = self._action_log[-2000:]
