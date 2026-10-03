"""
computer_connector.py - Production-grade computer control connector.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_SYSTEM = platform.system().lower()

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False
    pyautogui = None  # type: ignore

try:
    import pyperclip
    _PYPERCLIP_AVAILABLE = True
except ImportError:
    _PYPERCLIP_AVAILABLE = False
    pyperclip = None  # type: ignore

try:
    from PIL import Image, ImageGrab
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    Image = None  # type: ignore
    ImageGrab = None  # type: ignore

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False
    webdriver = None  # type: ignore

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore
    np = None  # type: ignore

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"

class KeyModifier(str, Enum):
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    WIN = "win"
    CMD = "command"

class BrowserType(str, Enum):
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"

class WindowAction(str, Enum):
    FOCUS = "focus"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CLOSE = "close"
    RESIZE = "resize"
    MOVE = "move"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass
class ScreenInfo:
    width: int
    height: int
    dpi: float = 96.0
    scale_factor: float = 1.0


@dataclass
class WindowInfo:
    id: Any
    title: str
    x: int
    y: int
    width: int
    height: int
    is_active: bool = False
    is_minimized: bool = False


@dataclass
class BrowserTab:
    index: int
    url: str
    title: str
    active: bool = False


@dataclass
class ComputerActionResult:
    action: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    screenshot: Optional[str] = None  # base64 PNG
    latency_ms: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ComputerConfig:
    screenshot_on_action: bool = False
    mouse_move_duration: float = 0.3
    type_interval: float = 0.02
    browser_type: BrowserType = BrowserType.CHROME
    browser_headless: bool = False
    browser_timeout: float = 30.0
    browser_implicit_wait: float = 10.0
    vision_confidence: float = 0.8
    safe_mode: bool = True
    allowed_regions: Optional[List[ScreenRegion]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ComputerConnectorError(Exception):
    pass

class MouseError(ComputerConnectorError):
    pass

class KeyboardError(ComputerConnectorError):
    pass

class BrowserError(ComputerConnectorError):
    pass

class VisionError(ComputerConnectorError):
    pass

class ClipboardError(ComputerConnectorError):
    pass

class WindowManagerError(ComputerConnectorError):
    pass

class SafetyError(ComputerConnectorError):
    pass

# ---------------------------------------------------------------------------
# Mouse controller
# ---------------------------------------------------------------------------

class MouseController:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config

    def _check_available(self) -> None:
        if not _PYAUTOGUI_AVAILABLE:
            raise MouseError("pyautogui not installed. pip install pyautogui")

    async def move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        self._check_available()
        d = duration or self.config.mouse_move_duration
        await asyncio.to_thread(pyautogui.moveTo, x, y, duration=d)

    async def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        self._check_available()
        if x is not None and y is not None:
            await self.move(x, y)
        await asyncio.to_thread(
            pyautogui.click,
            x=x, y=y,
            button=button.value,
            clicks=clicks,
            interval=interval,
        )

    async def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        await self.click(x=x, y=y, clicks=2)

    async def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        await self.click(x=x, y=y, button=MouseButton.RIGHT)

    async def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> None:
        self._check_available()
        await asyncio.to_thread(pyautogui.moveTo, x1, y1)
        await asyncio.to_thread(pyautogui.dragTo, x2, y2, duration=duration, button="left")

    async def scroll(self, x: int, y: int, amount: int = 3, direction: str = "down") -> None:
        self._check_available()
        clicks = -amount if direction == "down" else amount
        await asyncio.to_thread(pyautogui.scroll, clicks, x=x, y=y)

    async def get_position(self) -> Tuple[int, int]:
        self._check_available()
        pos = await asyncio.to_thread(pyautogui.position)
        return pos.x, pos.y

    async def health_check(self) -> bool:
        return _PYAUTOGUI_AVAILABLE


class KeyboardController:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config

    def _check_available(self) -> None:
        if not _PYAUTOGUI_AVAILABLE:
            raise KeyboardError("pyautogui not installed. pip install pyautogui")

    async def type_text(self, text: str, interval: Optional[float] = None) -> None:
        self._check_available()
        iv = interval or self.config.type_interval
        await asyncio.to_thread(pyautogui.typewrite, text, interval=iv)

    async def type_string(self, text: str) -> None:
        """Type arbitrary unicode strings including special characters."""
        self._check_available()
        await asyncio.to_thread(pyautogui.write, text, interval=self.config.type_interval)

    async def press_key(self, key: str) -> None:
        self._check_available()
        await asyncio.to_thread(pyautogui.press, key)

    async def hold_key(self, key: str, duration: float = 0.1) -> None:
        self._check_available()
        await asyncio.to_thread(pyautogui.keyDown, key)
        await asyncio.sleep(duration)
        await asyncio.to_thread(pyautogui.keyUp, key)

    async def hotkey(self, *keys: str) -> None:
        self._check_available()
        await asyncio.to_thread(pyautogui.hotkey, *keys)

    async def key_combination(self, modifier: KeyModifier, key: str) -> None:
        mod = modifier.value
        if _SYSTEM == "darwin" and modifier == KeyModifier.CTRL:
            mod = "command"
        await self.hotkey(mod, key)

    async def copy(self) -> None:
        mod = "command" if _SYSTEM == "darwin" else "ctrl"
        await self.hotkey(mod, "c")

    async def paste(self) -> None:
        mod = "command" if _SYSTEM == "darwin" else "ctrl"
        await self.hotkey(mod, "v")

    async def select_all(self) -> None:
        mod = "command" if _SYSTEM == "darwin" else "ctrl"
        await self.hotkey(mod, "a")

    async def health_check(self) -> bool:
        return _PYAUTOGUI_AVAILABLE


# ---------------------------------------------------------------------------
# Vision controller
# ---------------------------------------------------------------------------

class VisionController:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config

    async def screenshot(
        self,
        region: Optional[ScreenRegion] = None,
        as_base64: bool = True,
    ) -> Union[str, Any]:
        if not _PYAUTOGUI_AVAILABLE and not _PIL_AVAILABLE:
            raise VisionError("pyautogui or pillow not installed")
        bbox = region.bbox if region else None
        if _PYAUTOGUI_AVAILABLE:
            img = await asyncio.to_thread(pyautogui.screenshot, region=bbox)
        elif _PIL_AVAILABLE:
            img = await asyncio.to_thread(ImageGrab.grab, bbox=bbox)
        else:
            raise VisionError("No screenshot backend available")
        if as_base64:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        return img

    async def get_screen_size(self) -> ScreenInfo:
        if _PYAUTOGUI_AVAILABLE:
            size = await asyncio.to_thread(pyautogui.size)
            return ScreenInfo(width=size.width, height=size.height)
        return ScreenInfo(width=1920, height=1080)

    async def find_image_on_screen(
        self,
        template_path: str,
        confidence: Optional[float] = None,
    ) -> Optional[ScreenRegion]:
        if not _PYAUTOGUI_AVAILABLE:
            raise VisionError("pyautogui not installed")
        conf = confidence or self.config.vision_confidence
        try:
            result = await asyncio.to_thread(
                pyautogui.locateOnScreen, template_path, confidence=conf
            )
            if result:
                return ScreenRegion(
                    x=int(result.left),
                    y=int(result.top),
                    width=int(result.width),
                    height=int(result.height),
                )
            return None
        except Exception:
            return None

    async def find_text_on_screen(self, text: str) -> Optional[Tuple[int, int]]:
        """Basic OCR using pytesseract if available."""
        try:
            import pytesseract  # type: ignore
            img_b64 = await self.screenshot(as_base64=False)
            data = await asyncio.to_thread(
                pytesseract.image_to_data, img_b64,
                output_type=pytesseract.Output.DICT
            )
            for i, word in enumerate(data["text"]):
                if text.lower() in word.lower():
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i] + data["height"][i] // 2
                    return x, y
        except ImportError:
            logger.warning("pytesseract not installed for OCR")
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
        return None

    async def pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        if not _PYAUTOGUI_AVAILABLE:
            raise VisionError("pyautogui not installed")
        color = await asyncio.to_thread(pyautogui.pixel, x, y)
        return tuple(color)  # type: ignore[return-value]

    async def health_check(self) -> bool:
        return _PYAUTOGUI_AVAILABLE or _PIL_AVAILABLE


# ---------------------------------------------------------------------------
# Clipboard controller
# ---------------------------------------------------------------------------

class ClipboardController:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config

    async def get(self) -> str:
        if _PYPERCLIP_AVAILABLE:
            return await asyncio.to_thread(pyperclip.paste)
        elif _PYAUTOGUI_AVAILABLE:
            try:
                return await asyncio.to_thread(pyautogui.hotkey, "ctrl", "c")
            except Exception:
                pass
        raise ClipboardError("No clipboard backend available")

    async def set(self, text: str) -> None:
        if _PYPERCLIP_AVAILABLE:
            await asyncio.to_thread(pyperclip.copy, text)
        elif _SYSTEM == "darwin":
            proc = await asyncio.create_subprocess_exec(
                "pbcopy", stdin=asyncio.subprocess.PIPE
            )
            await proc.communicate(input=text.encode())
        elif _SYSTEM == "linux":
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard",
                stdin=asyncio.subprocess.PIPE
            )
            await proc.communicate(input=text.encode())
        else:
            raise ClipboardError("No clipboard backend available")

    async def clear(self) -> None:
        await self.set("")

    async def health_check(self) -> bool:
        return _PYPERCLIP_AVAILABLE


# ---------------------------------------------------------------------------
# Window manager
# ---------------------------------------------------------------------------

class WindowManager:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config
        self._wmctrl_available: Optional[bool] = None

    def _has_wmctrl(self) -> bool:
        if self._wmctrl_available is None:
            self._wmctrl_available = (
                subprocess.run(["which", "wmctrl"], capture_output=True).returncode == 0
            ) if _SYSTEM == "linux" else False
        return self._wmctrl_available  # type: ignore[return-value]

    async def list_windows(self) -> List[WindowInfo]:
        windows: List[WindowInfo] = []
        if _SYSTEM == "linux" and self._has_wmctrl():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "wmctrl", "-lG",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                for line in stdout.decode().splitlines():
                    parts = line.split(None, 7)
                    if len(parts) >= 8:
                        windows.append(WindowInfo(
                            id=parts[0],
                            title=parts[7],
                            x=int(parts[2]),
                            y=int(parts[3]),
                            width=int(parts[4]),
                            height=int(parts[5]),
                        ))
            except Exception as exc:
                logger.warning("wmctrl list failed: %s", exc)
        elif _SYSTEM == "darwin":
            try:
                script = 'tell app "System Events" to get name of every process where background only is false'
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                for idx, name in enumerate(stdout.decode().strip().split(", ")):
                    windows.append(WindowInfo(id=idx, title=name.strip(), x=0, y=0, width=0, height=0))
            except Exception as exc:
                logger.warning("osascript failed: %s", exc)
        return windows

    async def focus_window(self, title: str) -> bool:
        if _SYSTEM == "linux" and self._has_wmctrl():
            proc = await asyncio.create_subprocess_exec("wmctrl", "-a", title)
            await proc.communicate()
            return proc.returncode == 0
        elif _SYSTEM == "darwin":
            script = f'tell app "{title}" to activate'
            proc = await asyncio.create_subprocess_exec("osascript", "-e", script)
            await proc.communicate()
            return proc.returncode == 0
        return False

    async def close_window(self, title: str) -> bool:
        if _SYSTEM == "linux" and self._has_wmctrl():
            proc = await asyncio.create_subprocess_exec("wmctrl", "-c", title)
            await proc.communicate()
            return proc.returncode == 0
        return False

    async def get_active_window_title(self) -> Optional[str]:
        if _SYSTEM == "linux":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "xdotool", "getactivewindow", "getwindowname",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return stdout.decode().strip()
            except Exception:
                pass
        elif _SYSTEM == "darwin":
            try:
                script = 'tell app "System Events" to get name of first process whose frontmost is true'
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return stdout.decode().strip()
            except Exception:
                pass
        return None

    async def health_check(self) -> bool:
        return _SYSTEM in ("linux", "darwin", "windows")


# ---------------------------------------------------------------------------
# Browser controller
# ---------------------------------------------------------------------------

class BrowserController:
    def __init__(self, config: ComputerConfig) -> None:
        self.config = config
        self._driver: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not _SELENIUM_AVAILABLE:
            raise BrowserError("selenium not installed. pip install selenium")

        def _create_driver() -> Any:
            if self.config.browser_type == BrowserType.CHROME:
                opts = ChromeOptions()
                if self.config.browser_headless:
                    opts.add_argument("--headless=new")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
                opts.add_argument("--disable-gpu")
                return webdriver.Chrome(options=opts)
            elif self.config.browser_type == BrowserType.FIREFOX:
                from selenium.webdriver.firefox.options import Options as FFOptions  # type: ignore
                opts = FFOptions()
                if self.config.browser_headless:
                    opts.add_argument("--headless")
                return webdriver.Firefox(options=opts)
            else:
                raise BrowserError(f"Unsupported browser: {self.config.browser_type}")

        self._driver = await asyncio.to_thread(_create_driver)
        self._driver.implicitly_wait(self.config.browser_implicit_wait)
        self._initialized = True
        logger.info("Browser initialized: %s", self.config.browser_type.value)

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def navigate(self, url: str) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.get, url)

    async def get_current_url(self) -> str:
        await self._ensure()
        return await asyncio.to_thread(lambda: self._driver.current_url)

    async def get_title(self) -> str:
        await self._ensure()
        return await asyncio.to_thread(lambda: self._driver.title)

    async def get_page_source(self) -> str:
        await self._ensure()
        return await asyncio.to_thread(lambda: self._driver.page_source)

    async def find_element(self, selector: str, by: str = "css") -> Any:
        await self._ensure()
        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
            "text": By.LINK_TEXT,
        }
        by_type = by_map.get(by, By.CSS_SELECTOR)
        return await asyncio.to_thread(self._driver.find_element, by_type, selector)

    async def click_element(self, selector: str, by: str = "css") -> None:
        elem = await self.find_element(selector, by)
        await asyncio.to_thread(elem.click)

    async def type_into(self, selector: str, text: str, clear_first: bool = True, by: str = "css") -> None:
        elem = await self.find_element(selector, by)
        if clear_first:
            await asyncio.to_thread(elem.clear)
        await asyncio.to_thread(elem.send_keys, text)

    async def execute_script(self, script: str, *args: Any) -> Any:
        await self._ensure()
        return await asyncio.to_thread(self._driver.execute_script, script, *args)

    async def screenshot(self, as_base64: bool = True) -> Union[str, bytes]:
        await self._ensure()
        png = await asyncio.to_thread(self._driver.get_screenshot_as_png)
        if as_base64:
            return base64.b64encode(png).decode("utf-8")
        return png

    async def scroll_to(self, x: int = 0, y: int = 0) -> None:
        await self.execute_script(f"window.scrollTo({x}, {y});")

    async def scroll_to_bottom(self) -> None:
        await self.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    async def new_tab(self, url: Optional[str] = None) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.execute_script, "window.open('');")
        handles = await asyncio.to_thread(lambda: self._driver.window_handles)
        await asyncio.to_thread(self._driver.switch_to.window, handles[-1])
        if url:
            await self.navigate(url)

    async def close_tab(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.close)
        handles = await asyncio.to_thread(lambda: self._driver.window_handles)
        if handles:
            await asyncio.to_thread(self._driver.switch_to.window, handles[-1])

    async def get_tabs(self) -> List[BrowserTab]:
        await self._ensure()
        handles = await asyncio.to_thread(lambda: self._driver.window_handles)
        current = await asyncio.to_thread(lambda: self._driver.current_window_handle)
        tabs = []
        for idx, handle in enumerate(handles):
            await asyncio.to_thread(self._driver.switch_to.window, handle)
            tabs.append(BrowserTab(
                index=idx,
                url=self._driver.current_url,
                title=self._driver.title,
                active=handle == current,
            ))
        await asyncio.to_thread(self._driver.switch_to.window, current)
        return tabs

    async def wait_for_element(self, selector: str, timeout: Optional[float] = None) -> Any:
        await self._ensure()
        t = timeout or self.config.browser_timeout
        wait = WebDriverWait(self._driver, t)
        return await asyncio.to_thread(
            wait.until, EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )

    async def go_back(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.back)

    async def go_forward(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.forward)

    async def refresh(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._driver.refresh)

    async def close(self) -> None:
        if self._driver:
            await asyncio.to_thread(self._driver.quit)
            self._driver = None
            self._initialized = False

    async def health_check(self) -> bool:
        if not _SELENIUM_AVAILABLE:
            return False
        if not self._initialized:
            return True  # Not started yet – report as available
        try:
            _ = await asyncio.to_thread(lambda: self._driver.current_url)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ComputerConnector – Singleton
# ---------------------------------------------------------------------------

class ComputerConnector:
    _instance: Optional["ComputerConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = ComputerConfig()
        self.mouse: Optional[MouseController] = None
        self.keyboard: Optional[KeyboardController] = None
        self.vision: Optional[VisionController] = None
        self.clipboard: Optional[ClipboardController] = None
        self.window_manager: Optional[WindowManager] = None
        self.browser: Optional[BrowserController] = None
        self._initialized = False
        self._action_log: List[ComputerActionResult] = []

    @classmethod
    def get_instance(cls) -> "ComputerConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "ComputerConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def initialize(self, config: Optional[ComputerConfig] = None) -> "ComputerConnector":
        if self._initialized:
            return self
        if config:
            self._config = config
        self.mouse = MouseController(self._config)
        self.keyboard = KeyboardController(self._config)
        self.vision = VisionController(self._config)
        self.clipboard = ClipboardController(self._config)
        self.window_manager = WindowManager(self._config)
        self.browser = BrowserController(self._config)
        self._initialized = True
        logger.info("ComputerConnector initialized on %s", platform.system())
        return self

    def _ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    def _log(self, result: ComputerActionResult) -> None:
        self._action_log.append(result)
        if len(self._action_log) > 1000:
            self._action_log = self._action_log[-500:]

    # ------------------------------------------------------------------
    # High-level actions
    # ------------------------------------------------------------------

    async def click(
        self, x: int, y: int,
        button: MouseButton = MouseButton.LEFT,
        screenshot_after: bool = False,
    ) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            await self.mouse.click(x=x, y=y, button=button)  # type: ignore[union-attr]
            shot = None
            if screenshot_after or self._config.screenshot_on_action:
                shot = await self.vision.screenshot()  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="click",
                success=True,
                output={"x": x, "y": y, "button": button.value},
                screenshot=shot,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="click", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def type_text(self, text: str) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            await self.keyboard.type_text(text)  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="type_text", success=True,
                output={"length": len(text)},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="type_text", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def hotkey(self, *keys: str) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            await self.keyboard.hotkey(*keys)  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="hotkey", success=True,
                output={"keys": keys},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="hotkey", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def screenshot(self, region: Optional[ScreenRegion] = None) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            img = await self.vision.screenshot(region=region)  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="screenshot", success=True,
                output={"size": len(img) if img else 0},
                screenshot=img,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="screenshot", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def open_browser(self, url: str) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            await self.browser.initialize()  # type: ignore[union-attr]
            await self.browser.navigate(url)  # type: ignore[union-attr]
            title = await self.browser.get_title()  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="open_browser", success=True,
                output={"url": url, "title": title},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="open_browser", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def copy_to_clipboard(self, text: str) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            await self.clipboard.set(text)  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="copy_to_clipboard", success=True,
                output={"length": len(text)},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="copy_to_clipboard", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    async def get_clipboard(self) -> ComputerActionResult:
        self._ensure()
        t0 = time.perf_counter()
        try:
            text = await self.clipboard.get()  # type: ignore[union-attr]
            result = ComputerActionResult(
                action="get_clipboard", success=True,
                output={"text": text},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ComputerActionResult(
                action="get_clipboard", success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self._log(result)
        return result

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        self._ensure()
        mouse_ok = await self.mouse.health_check()  # type: ignore[union-attr]
        keyboard_ok = await self.keyboard.health_check()  # type: ignore[union-attr]
        vision_ok = await self.vision.health_check()  # type: ignore[union-attr]
        clipboard_ok = await self.clipboard.health_check()  # type: ignore[union-attr]
        wm_ok = await self.window_manager.health_check()  # type: ignore[union-attr]
        browser_ok = await self.browser.health_check()  # type: ignore[union-attr]
        return {
            "healthy": mouse_ok or vision_ok,
            "platform": platform.system(),
            "mouse": mouse_ok,
            "keyboard": keyboard_ok,
            "vision": vision_ok,
            "clipboard": clipboard_ok,
            "window_manager": wm_ok,
            "browser": browser_ok,
            "pyautogui": _PYAUTOGUI_AVAILABLE,
            "selenium": _SELENIUM_AVAILABLE,
            "pillow": _PIL_AVAILABLE,
            "pyperclip": _PYPERCLIP_AVAILABLE,
            "total_actions": len(self._action_log),
        }

    async def __aenter__(self) -> "ComputerConnector":
        self._ensure()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self.browser:
            await self.browser.close()

    def __repr__(self) -> str:
        return f"ComputerConnector(platform={platform.system()}, actions={len(self._action_log)})"


def get_computer_connector() -> ComputerConnector:
    connector = ComputerConnector.get_instance()
    if not connector._initialized:
        connector.initialize()
    return connector
