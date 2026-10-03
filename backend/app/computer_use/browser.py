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


class BrowserEngine(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"


class BrowserType(Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME = "chrome"


class BrowserError(Exception):
    pass


class BrowserNotOpenError(BrowserError):
    pass


class TabNotFoundError(BrowserError):
    pass


@dataclass(frozen=True)
class Cookie:
    name: str
    value: str
    domain: str
    path: str = "/"
    expires: Optional[float] = None
    secure: bool = False
    http_only: bool = False


@dataclass(frozen=True)
class TabInfo:
    tab_id: str
    url: str
    title: str
    is_active: bool


@dataclass(frozen=True)
class BrowserActionResult:
    action: str
    detail: str
    duration_seconds: float


class BrowserBackend:
    async def open(self, headless: bool, browser_type: BrowserType) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def navigate(self, url: str, tab_id: Optional[str]) -> None:
        raise NotImplementedError

    async def click(self, selector: str, tab_id: Optional[str]) -> None:
        raise NotImplementedError

    async def fill(self, selector: str, value: str, tab_id: Optional[str]) -> None:
        raise NotImplementedError

    async def extract_text(self, selector: Optional[str], tab_id: Optional[str]) -> str:
        raise NotImplementedError

    async def screenshot(self, path: str, tab_id: Optional[str]) -> bytes:
        raise NotImplementedError

    async def get_cookies(self, tab_id: Optional[str]) -> list[Cookie]:
        raise NotImplementedError

    async def set_cookies(self, cookies: list[Cookie], tab_id: Optional[str]) -> None:
        raise NotImplementedError

    async def list_tabs(self) -> list[TabInfo]:
        raise NotImplementedError

    async def new_tab(self, url: Optional[str]) -> str:
        raise NotImplementedError

    async def switch_tab(self, tab_id: str) -> None:
        raise NotImplementedError

    async def close_tab(self, tab_id: str) -> None:
        raise NotImplementedError

    async def is_alive(self) -> bool:
        raise NotImplementedError


class PlaywrightBackend(BrowserBackend):
    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._pages: dict[str, Any] = {}
        self._active_tab_id: Optional[str] = None

    async def open(self, headless: bool, browser_type: BrowserType) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        engine_map = {
            BrowserType.CHROMIUM: self._playwright.chromium,
            BrowserType.FIREFOX: self._playwright.firefox,
            BrowserType.WEBKIT: self._playwright.webkit,
            BrowserType.CHROME: self._playwright.chromium,
        }
        launcher = engine_map.get(browser_type, self._playwright.chromium)
        launch_kwargs: dict[str, Any] = {"headless": headless}
        if browser_type == BrowserType.CHROME:
            launch_kwargs["channel"] = "chrome"
        self._browser = await launcher.launch(**launch_kwargs)
        page = await self._browser.new_page()
        tab_id = f"tab-{id(page)}"
        self._pages[tab_id] = page
        self._active_tab_id = tab_id

    async def close(self) -> None:
        for page in list(self._pages.values()):
            try:
                await page.close()
            except Exception:
                pass
        self._pages.clear()
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    def _resolve_page(self, tab_id: Optional[str]) -> Any:
        resolved_id = tab_id or self._active_tab_id
        if resolved_id is None or resolved_id not in self._pages:
            raise TabNotFoundError(f"tab '{tab_id}' not found")
        return self._pages[resolved_id]

    async def navigate(self, url: str, tab_id: Optional[str]) -> None:
        page = self._resolve_page(tab_id)
        await page.goto(url, wait_until="load")

    async def click(self, selector: str, tab_id: Optional[str]) -> None:
        page = self._resolve_page(tab_id)
        await page.click(selector)

    async def fill(self, selector: str, value: str, tab_id: Optional[str]) -> None:
        page = self._resolve_page(tab_id)
        await page.fill(selector, value)

    async def extract_text(self, selector: Optional[str], tab_id: Optional[str]) -> str:
        page = self._resolve_page(tab_id)
        if selector:
            element = await page.query_selector(selector)
            if element is None:
                raise BrowserError(f"selector '{selector}' not found")
            text = await element.inner_text()
            return str(text)
        body_text = await page.inner_text("body")
        return str(body_text)

    async def screenshot(self, path: str, tab_id: Optional[str]) -> bytes:
        page = self._resolve_page(tab_id)
        return await page.screenshot(path=path)

    async def get_cookies(self, tab_id: Optional[str]) -> list[Cookie]:
        context = self._resolve_page(tab_id).context
        raw_cookies = await context.cookies()
        return [
            Cookie(
                name=c.get("name", ""),
                value=c.get("value", ""),
                domain=c.get("domain", ""),
                path=c.get("path", "/"),
                expires=c.get("expires"),
                secure=bool(c.get("secure", False)),
                http_only=bool(c.get("httpOnly", False)),
            )
            for c in raw_cookies
        ]

    async def set_cookies(self, cookies: list[Cookie], tab_id: Optional[str]) -> None:
        context = self._resolve_page(tab_id).context
        await context.add_cookies(
            [
                {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path,
                    **({"expires": c.expires} if c.expires is not None else {}),
                    "secure": c.secure,
                    "httpOnly": c.http_only,
                }
                for c in cookies
            ]
        )

    async def list_tabs(self) -> list[TabInfo]:
        tabs = []
        for tab_id, page in self._pages.items():
            tabs.append(
                TabInfo(
                    tab_id=tab_id,
                    url=page.url,
                    title=await page.title(),
                    is_active=(tab_id == self._active_tab_id),
                )
            )
        return tabs

    async def new_tab(self, url: Optional[str]) -> str:
        if self._browser is None:
            raise BrowserNotOpenError("browser is not open")
        page = await self._browser.new_page()
        tab_id = f"tab-{id(page)}"
        self._pages[tab_id] = page
        self._active_tab_id = tab_id
        if url:
            await page.goto(url, wait_until="load")
        return tab_id

    async def switch_tab(self, tab_id: str) -> None:
        if tab_id not in self._pages:
            raise TabNotFoundError(f"tab '{tab_id}' not found")
        self._active_tab_id = tab_id
        await self._pages[tab_id].bring_to_front()

    async def close_tab(self, tab_id: str) -> None:
        page = self._pages.pop(tab_id, None)
        if page is None:
            raise TabNotFoundError(f"tab '{tab_id}' not found")
        await page.close()
        if self._active_tab_id == tab_id:
            self._active_tab_id = next(iter(self._pages), None)

    async def is_alive(self) -> bool:
        return self._browser is not None and self._browser.is_connected()


class SeleniumBackend(BrowserBackend):
    def __init__(self) -> None:
        self._driver: Any = None
        self._handles: dict[str, str] = {}

    async def open(self, headless: bool, browser_type: BrowserType) -> None:
        await asyncio.to_thread(self._open_sync, headless, browser_type)

    def _open_sync(self, headless: bool, browser_type: BrowserType) -> None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.firefox.options import Options as FirefoxOptions

        if browser_type == BrowserType.FIREFOX:
            options = FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            self._driver = webdriver.Firefox(options=options)
        else:
            options = ChromeOptions()
            if headless:
                options.add_argument("--headless=new")
            self._driver = webdriver.Chrome(options=options)

        handle = self._driver.current_window_handle
        self._handles[f"tab-{handle}"] = handle

    async def close(self) -> None:
        if self._driver is not None:
            await asyncio.to_thread(self._driver.quit)
            self._driver = None
            self._handles.clear()

    def _require_driver(self) -> Any:
        if self._driver is None:
            raise BrowserNotOpenError("browser is not open")
        return self._driver

    def _switch_to_tab_sync(self, tab_id: Optional[str]) -> None:
        driver = self._require_driver()
        if tab_id is None:
            return
        handle = self._handles.get(tab_id)
        if handle is None:
            raise TabNotFoundError(f"tab '{tab_id}' not found")
        driver.switch_to.window(handle)

    async def navigate(self, url: str, tab_id: Optional[str]) -> None:
        def run() -> None:
            self._switch_to_tab_sync(tab_id)
            self._require_driver().get(url)

        await asyncio.to_thread(run)

    async def click(self, selector: str, tab_id: Optional[str]) -> None:
        def run() -> None:
            from selenium.webdriver.common.by import By

            self._switch_to_tab_sync(tab_id)
            element = self._require_driver().find_element(By.CSS_SELECTOR, selector)
            element.click()

        await asyncio.to_thread(run)

    async def fill(self, selector: str, value: str, tab_id: Optional[str]) -> None:
        def run() -> None:
            from selenium.webdriver.common.by import By

            self._switch_to_tab_sync(tab_id)
            element = self._require_driver().find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(value)

        await asyncio.to_thread(run)

    async def extract_text(self, selector: Optional[str], tab_id: Optional[str]) -> str:
        def run() -> str:
            from selenium.webdriver.common.by import By

            self._switch_to_tab_sync(tab_id)
            driver = self._require_driver()
            if selector:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                return str(element.text)
            return str(driver.find_element(By.TAG_NAME, "body").text)

        return await asyncio.to_thread(run)

    async def screenshot(self, path: str, tab_id: Optional[str]) -> bytes:
        def run() -> bytes:
            self._switch_to_tab_sync(tab_id)
            driver = self._require_driver()
            driver.save_screenshot(path)
            with open(path, "rb") as handle:
                return handle.read()

        return await asyncio.to_thread(run)

    async def get_cookies(self, tab_id: Optional[str]) -> list[Cookie]:
        def run() -> list[Cookie]:
            self._switch_to_tab_sync(tab_id)
            raw_cookies = self._require_driver().get_cookies()
            return [
                Cookie(
                    name=c.get("name", ""),
                    value=c.get("value", ""),
                    domain=c.get("domain", ""),
                    path=c.get("path", "/"),
                    expires=c.get("expiry"),
                    secure=bool(c.get("secure", False)),
                    http_only=bool(c.get("httpOnly", False)),
                )
                for c in raw_cookies
            ]

        return await asyncio.to_thread(run)

    async def set_cookies(self, cookies: list[Cookie], tab_id: Optional[str]) -> None:
        def run() -> None:
            self._switch_to_tab_sync(tab_id)
            driver = self._require_driver()
            for cookie in cookies:
                cookie_dict: dict[str, Any] = {"name": cookie.name, "value": cookie.value, "path": cookie.path}
                if cookie.expires is not None:
                    cookie_dict["expiry"] = cookie.expires
                driver.add_cookie(cookie_dict)

        await asyncio.to_thread(run)

    async def list_tabs(self) -> list[TabInfo]:
        def run() -> list[TabInfo]:
            driver = self._require_driver()
            current = driver.current_window_handle
            tabs = []
            for tab_id, handle in self._handles.items():
                driver.switch_to.window(handle)
                tabs.append(
                    TabInfo(
                        tab_id=tab_id,
                        url=driver.current_url,
                        title=driver.title,
                        is_active=(handle == current),
                    )
                )
            driver.switch_to.window(current)
            return tabs

        return await asyncio.to_thread(run)

    async def new_tab(self, url: Optional[str]) -> str:
        def run() -> str:
            driver = self._require_driver()
            driver.switch_to.new_window("tab")
            handle = driver.current_window_handle
            tab_id = f"tab-{handle}"
            self._handles[tab_id] = handle
            if url:
                driver.get(url)
            return tab_id

        return await asyncio.to_thread(run)

    async def switch_tab(self, tab_id: str) -> None:
        await asyncio.to_thread(self._switch_to_tab_sync, tab_id)

    async def close_tab(self, tab_id: str) -> None:
        def run() -> None:
            handle = self._handles.pop(tab_id, None)
            if handle is None:
                raise TabNotFoundError(f"tab '{tab_id}' not found")
            driver = self._require_driver()
            driver.switch_to.window(handle)
            driver.close()
            if self._handles:
                driver.switch_to.window(next(iter(self._handles.values())))

        await asyncio.to_thread(run)

    async def is_alive(self) -> bool:
        if self._driver is None:
            return False
        try:
            await asyncio.to_thread(lambda: self._driver.current_window_handle)
            return True
        except Exception:
            return False


class BrowserCapability(ComputerCapability):
    def __init__(
        self,
        engine: BrowserEngine = BrowserEngine.PLAYWRIGHT,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        metadata: Optional[CapabilityMetadata] = None,
        backend: Optional[BrowserBackend] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="browser",
                version="1.0.0",
                description="Drives a browser via Playwright or Selenium",
                priority=CapabilityPriority.NORMAL,
                timeout_seconds=30.0,
                approval_required=True,
            )
        )
        self._engine = engine
        self._browser_type = browser_type
        self._headless = headless
        self._backend = backend or (
            PlaywrightBackend() if engine == BrowserEngine.PLAYWRIGHT else SeleniumBackend()
        )
        self._is_open = False

    async def _on_initialize(self) -> None:
        pass

    async def _on_shutdown(self) -> None:
        if self._is_open:
            await self._backend.close()
            self._is_open = False

    async def _on_health_check(self) -> HealthCheckResult:
        if not self._is_open:
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.UNKNOWN,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail="browser is not open",
            )
        alive = await self._backend.is_alive()
        return HealthCheckResult(
            capability_name=self.name,
            status=HealthStatus.HEALTHY if alive else HealthStatus.UNHEALTHY,
            latency_seconds=None,
            checked_at_epoch=time.time(),
        )

    async def _on_execute(self, request: ExecutionRequest) -> Any:
        action = request.action
        params = request.parameters

        if action == "open":
            return await self.open()
        if action == "close":
            return await self.close()
        if action == "navigate":
            return await self.navigate(str(params["url"]), params.get("tab_id"))
        if action == "click":
            return await self.click(str(params["selector"]), params.get("tab_id"))
        if action == "fill":
            return await self.fill(str(params["selector"]), str(params["value"]), params.get("tab_id"))
        if action == "extract_text":
            return await self.extract_text(params.get("selector"), params.get("tab_id"))
        if action == "screenshot":
            return await self.screenshot(str(params.get("path", "screenshot.png")), params.get("tab_id"))
        if action == "cookies":
            return await self.cookies(params.get("tab_id"))
        if action == "tabs":
            return await self.tabs()

        raise ValueError(f"unknown browser action: {action}")

    def _require_open(self) -> None:
        if not self._is_open:
            raise BrowserNotOpenError("browser is not open; call open() first")

    async def open(self) -> BrowserActionResult:
        start = time.monotonic()
        if not self._is_open:
            await self._backend.open(self._headless, self._browser_type)
            self._is_open = True
        return BrowserActionResult(
            action="open",
            detail=f"engine={self._engine.value} type={self._browser_type.value} headless={self._headless}",
            duration_seconds=time.monotonic() - start,
        )

    async def close(self) -> BrowserActionResult:
        start = time.monotonic()
        if self._is_open:
            await self._backend.close()
            self._is_open = False
        return BrowserActionResult(
            action="close",
            detail="",
            duration_seconds=time.monotonic() - start,
        )

    async def navigate(self, url: str, tab_id: Optional[str] = None) -> BrowserActionResult:
        self._require_open()
        start = time.monotonic()
        await self._backend.navigate(url, tab_id)
        return BrowserActionResult(
            action="navigate",
            detail=url,
            duration_seconds=time.monotonic() - start,
        )

    async def click(self, selector: str, tab_id: Optional[str] = None) -> BrowserActionResult:
        self._require_open()
        start = time.monotonic()
        await self._backend.click(selector, tab_id)
        return BrowserActionResult(
            action="click",
            detail=selector,
            duration_seconds=time.monotonic() - start,
        )

    async def fill(
        self, selector: str, value: str, tab_id: Optional[str] = None
    ) -> BrowserActionResult:
        self._require_open()
        start = time.monotonic()
        await self._backend.fill(selector, value, tab_id)
        return BrowserActionResult(
            action="fill",
            detail=f"{selector} length={len(value)}",
            duration_seconds=time.monotonic() - start,
        )

    async def extract_text(
        self, selector: Optional[str] = None, tab_id: Optional[str] = None
    ) -> str:
        self._require_open()
        return await self._backend.extract_text(selector, tab_id)

    async def screenshot(
        self, path: str = "screenshot.png", tab_id: Optional[str] = None
    ) -> bytes:
        self._require_open()
        return await self._backend.screenshot(path, tab_id)

    async def cookies(self, tab_id: Optional[str] = None) -> list[Cookie]:
        self._require_open()
        return await self._backend.get_cookies(tab_id)

    async def set_cookies(self, cookies: list[Cookie], tab_id: Optional[str] = None) -> None:
        self._require_open()
        await self._backend.set_cookies(cookies, tab_id)

    async def tabs(self) -> list[TabInfo]:
        self._require_open()
        return await self._backend.list_tabs()

    async def new_tab(self, url: Optional[str] = None) -> str:
        self._require_open()
        return await self._backend.new_tab(url)

    async def switch_tab(self, tab_id: str) -> None:
        self._require_open()
        await self._backend.switch_tab(tab_id)

    async def close_tab(self, tab_id: str) -> None:
        self._require_open()
        await self._backend.close_tab(tab_id)
