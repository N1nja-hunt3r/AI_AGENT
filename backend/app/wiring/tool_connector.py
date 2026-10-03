"""
tool_connector.py - Production-grade tool connector with registry integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

try:
    import aiofiles
    _AIOFILES_AVAILABLE = True
except ImportError:
    _AIOFILES_AVAILABLE = False
    aiofiles = None  # type: ignore

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolStatus(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"

class ToolCategory(str, Enum):
    MATH = "math"
    WEATHER = "weather"
    TERMINAL = "terminal"
    WEB = "web"
    FILE = "file"
    UTILITY = "utility"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    category: ToolCategory = ToolCategory.UTILITY
    version: str = "1.0.0"
    enabled: bool = True
    timeout: float = 30.0
    requires_confirm: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Dict[str, Any]:
        content = self.output if isinstance(self.output, str) else json.dumps(self.output)
        if not self.success:
            content = f"Error: {self.error}"
        return {"role": "tool", "tool_call_id": self.call_id, "content": content}


@dataclass
class ToolStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls else 0.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolConnectorError(Exception):
    pass

class ToolNotFoundError(ToolConnectorError):
    pass

class ToolExecutionError(ToolConnectorError):
    pass

class ToolTimeoutError(ToolConnectorError):
    pass

class ToolDisabledError(ToolConnectorError):
    pass

# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

class CalculatorTool:
    DEFINITION = ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations. Supports basic arithmetic, math functions.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 2', 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
        category=ToolCategory.MATH,
    )

    async def execute(self, expression: str, **kwargs: Any) -> Any:
        import math
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        safe_expr = expression.strip()
        # Basic safety check
        forbidden = ["import", "exec", "eval", "__", "open", "os", "sys"]
        for token in forbidden:
            if token in safe_expr:
                raise ToolExecutionError(f"Forbidden token in expression: {token}")
        try:
            result = eval(safe_expr, {"__builtins__": {}}, allowed_names)  # noqa: S307
            return {"result": result, "expression": expression}
        except Exception as exc:
            raise ToolExecutionError(f"Calculation error: {exc}") from exc


class WeatherTool:
    DEFINITION = ToolDefinition(
        name="weather",
        description="Get current weather for a location using wttr.in API.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or coordinates"},
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units",
                    "default": "metric",
                },
            },
            "required": ["location"],
        },
        category=ToolCategory.WEATHER,
    )

    async def execute(self, location: str, units: str = "metric", **kwargs: Any) -> Any:
        if not _HTTPX_AVAILABLE:
            raise ToolExecutionError("httpx not installed for weather tool")
        unit_param = "m" if units == "metric" else "u"
        url = f"https://wttr.in/{location}?format=j1&{unit_param}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_kmph = current.get("windspeedKmph", "N/A")
        temp = temp_c if units == "metric" else temp_f
        unit_label = "°C" if units == "metric" else "°F"
        return {
            "location": location,
            "temperature": f"{temp}{unit_label}",
            "description": desc,
            "humidity": f"{humidity}%",
            "wind_speed": f"{wind_kmph} km/h",
        }


class TerminalTool:
    DEFINITION = ToolDefinition(
        name="terminal",
        description="Execute shell commands in a subprocess. Use with caution.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30},
                "working_dir": {"type": "string", "description": "Working directory"},
            },
            "required": ["command"],
        },
        category=ToolCategory.TERMINAL,
        requires_confirm=True,
        timeout=60.0,
    )

    _BLOCKED = [
        "rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero",
        "chmod -R 777 /", "> /dev/sda",
    ]

    async def execute(
        self,
        command: str,
        timeout: float = 30.0,
        working_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        for blocked in self._BLOCKED:
            if blocked in command:
                raise ToolExecutionError(f"Blocked command pattern: {blocked}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise ToolTimeoutError(f"Command timed out after {timeout}s")
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "return_code": proc.returncode,
                "command": command,
            }
        except (ToolTimeoutError, ToolExecutionError):
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Terminal error: {exc}") from exc


class WebSearchTool:
    DEFINITION = ToolDefinition(
        name="web_search",
        description="Search the web using DuckDuckGo instant answers API.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        category=ToolCategory.WEB,
    )

    async def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> Any:
        if not _HTTPX_AVAILABLE:
            raise ToolExecutionError("httpx not installed for web_search tool")
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", ""),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
                "source": "DuckDuckGo Abstract",
            })
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo Related",
                })
        return {"query": query, "results": results[:max_results], "total": len(results)}


class FileReaderTool:
    DEFINITION = ToolDefinition(
        name="file_reader",
        description="Read file contents from the local filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes to read",
                    "default": 1048576,
                },
                "start_line": {"type": "integer", "description": "Start line (1-indexed)"},
                "end_line": {"type": "integer", "description": "End line (inclusive)"},
            },
            "required": ["path"],
        },
        category=ToolCategory.FILE,
    )

    _BLOCKED_PATHS = ["/etc/shadow", "/etc/passwd", "~/.ssh/"]

    async def execute(
        self,
        path: str,
        encoding: str = "utf-8",
        max_bytes: int = 1_048_576,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        import pathlib
        resolved = str(pathlib.Path(path).resolve())
        for blocked in self._BLOCKED_PATHS:
            if blocked in resolved:
                raise ToolExecutionError(f"Access to path blocked: {path}")
        if not os.path.exists(resolved):
            raise ToolExecutionError(f"File not found: {path}")
        if os.path.getsize(resolved) > max_bytes * 2:
            raise ToolExecutionError(f"File too large: {os.path.getsize(resolved)} bytes")
        try:
            with open(resolved, "r", encoding=encoding, errors="replace") as f:
                if start_line or end_line:
                    lines = f.readlines()
                    s = (start_line or 1) - 1
                    e = end_line or len(lines)
                    content = "".join(lines[s:e])
                else:
                    content = f.read(max_bytes)
            return {
                "path": path,
                "content": content,
                "size_bytes": len(content.encode(encoding)),
                "encoding": encoding,
            }
        except Exception as exc:
            raise ToolExecutionError(f"File read error: {exc}") from exc


# ---------------------------------------------------------------------------
# Tool registry entry
# ---------------------------------------------------------------------------

@dataclass
class ToolRegistryEntry:
    definition: ToolDefinition
    handler: Any  # instance with async execute() method or callable
    stats: ToolStats = field(default_factory=ToolStats)
    status: ToolStatus = ToolStatus.READY
    registered_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ToolConnector – Singleton
# ---------------------------------------------------------------------------

class ToolConnector:
    _instance: Optional["ToolConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._registry: Dict[str, ToolRegistryEntry] = {}
        self._initialized = False
        self._pre_execute_hooks: List[Callable] = []
        self._post_execute_hooks: List[Callable] = []

    @classmethod
    def get_instance(cls) -> "ToolConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "ToolConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, register_builtins: bool = True) -> "ToolConnector":
        if self._initialized:
            return self
        if register_builtins:
            self._register_builtins()
        self._initialized = True
        logger.info("ToolConnector initialized with %d tools", len(self._registry))
        return self

    def _register_builtins(self) -> None:
        self.register(CalculatorTool.DEFINITION, CalculatorTool())
        self.register(WeatherTool.DEFINITION, WeatherTool())
        self.register(TerminalTool.DEFINITION, TerminalTool())
        self.register(WebSearchTool.DEFINITION, WebSearchTool())
        self.register(FileReaderTool.DEFINITION, FileReaderTool())

    def _ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        definition: ToolDefinition,
        handler: Any,
        overwrite: bool = False,
    ) -> None:
        if definition.name in self._registry and not overwrite:
            logger.warning("Tool already registered: %s", definition.name)
            return
        entry = ToolRegistryEntry(definition=definition, handler=handler)
        self._registry[definition.name] = entry
        logger.debug("Registered tool: %s v%s", definition.name, definition.version)

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        category: ToolCategory = ToolCategory.CUSTOM,
        **kwargs: Any,
    ) -> None:
        defn = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            **kwargs,
        )

        class _FuncHandler:
            def __init__(self, fn: Callable) -> None:
                self._fn = fn

            async def execute(self, **kw: Any) -> Any:
                if asyncio.iscoroutinefunction(self._fn):
                    return await self._fn(**kw)
                return await asyncio.to_thread(self._fn, **kw)

        self.register(defn, _FuncHandler(func))

    def unregister(self, name: str) -> bool:
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def enable(self, name: str) -> None:
        if name in self._registry:
            self._registry[name].definition.enabled = True
            self._registry[name].status = ToolStatus.READY

    def disable(self, name: str) -> None:
        if name in self._registry:
            self._registry[name].definition.enabled = False
            self._registry[name].status = ToolStatus.DISABLED

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> ToolResult:
        self._ensure()
        cid = call_id or str(uuid.uuid4())
        entry = self._registry.get(name)
        if not entry:
            return ToolResult(
                call_id=cid, tool_name=name, success=False,
                output=None, error=f"Tool not found: {name}"
            )
        if not entry.definition.enabled:
            return ToolResult(
                call_id=cid, tool_name=name, success=False,
                output=None, error=f"Tool disabled: {name}"
            )

        for hook in self._pre_execute_hooks:
            try:
                await hook(name, arguments) if asyncio.iscoroutinefunction(hook) else hook(name, arguments)
            except Exception as exc:
                logger.warning("Pre-execute hook error: %s", exc)

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                entry.handler.execute(**arguments),
                timeout=entry.definition.timeout,
            )
            latency = (time.perf_counter() - t0) * 1000
            entry.stats.total_calls += 1
            entry.stats.successful_calls += 1
            entry.stats.total_latency_ms += latency
            tool_result = ToolResult(
                call_id=cid,
                tool_name=name,
                success=True,
                output=result,
                latency_ms=latency,
            )
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000
            entry.stats.total_calls += 1
            entry.stats.failed_calls += 1
            entry.stats.total_latency_ms += latency
            tool_result = ToolResult(
                call_id=cid, tool_name=name, success=False,
                output=None, error=f"Timeout after {entry.definition.timeout}s",
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            entry.stats.total_calls += 1
            entry.stats.failed_calls += 1
            entry.stats.total_latency_ms += latency
            tool_result = ToolResult(
                call_id=cid, tool_name=name, success=False,
                output=None, error=str(exc), latency_ms=latency,
            )
            logger.warning("Tool execution error [%s]: %s", name, exc)

        for hook in self._post_execute_hooks:
            try:
                await hook(tool_result) if asyncio.iscoroutinefunction(hook) else hook(tool_result)
            except Exception as exc:
                logger.warning("Post-execute hook error: %s", exc)

        return tool_result

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        return await self.execute(
            name=tool_call.name,
            arguments=tool_call.arguments,
            call_id=tool_call.id,
            session_id=tool_call.session_id,
            agent_id=tool_call.agent_id,
        )

    async def execute_batch(self, calls: List[ToolCall]) -> List[ToolResult]:
        return await asyncio.gather(*[self.execute_tool_call(c) for c in calls])

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> List[ToolDefinition]:
        self._ensure()
        results = []
        for entry in self._registry.values():
            if enabled_only and not entry.definition.enabled:
                continue
            if category and entry.definition.category != category:
                continue
            results.append(entry.definition)
        return results

    def get_openai_schemas(
        self,
        names: Optional[List[str]] = None,
        category: Optional[ToolCategory] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure()
        tools = self.list_tools(category=category)
        if names:
            tools = [t for t in tools if t.name in names]
        return [t.to_openai_schema() for t in tools]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        entry = self._registry.get(name)
        return entry.definition if entry else None

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        self._ensure()
        if name:
            entry = self._registry.get(name)
            if not entry:
                return {}
            return {
                "name": name,
                "total_calls": entry.stats.total_calls,
                "successful_calls": entry.stats.successful_calls,
                "failed_calls": entry.stats.failed_calls,
                "success_rate": entry.stats.success_rate,
                "avg_latency_ms": entry.stats.avg_latency_ms,
                "status": entry.status.value,
            }
        return {
            name: {
                "total_calls": e.stats.total_calls,
                "success_rate": e.stats.success_rate,
                "avg_latency_ms": e.stats.avg_latency_ms,
                "status": e.status.value,
            }
            for name, e in self._registry.items()
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def add_pre_execute_hook(self, hook: Callable) -> None:
        self._pre_execute_hooks.append(hook)

    def add_post_execute_hook(self, hook: Callable) -> None:
        self._post_execute_hooks.append(hook)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        self._ensure()
        tool_health = {}
        for name, entry in self._registry.items():
            tool_health[name] = {
                "status": entry.status.value,
                "enabled": entry.definition.enabled,
                "category": entry.definition.category.value,
                "calls": entry.stats.total_calls,
                "success_rate": entry.stats.success_rate,
            }
        return {
            "healthy": True,
            "total_tools": len(self._registry),
            "enabled_tools": sum(1 for e in self._registry.values() if e.definition.enabled),
            "tools": tool_health,
        }

    def __repr__(self) -> str:
        return f"ToolConnector(tools={list(self._registry.keys())})"


def get_tool_connector() -> ToolConnector:
    connector = ToolConnector.get_instance()
    if not connector._initialized:
        connector.initialize()
    return connector


async def execute_tool(name: str, arguments: Dict[str, Any], **kwargs: Any) -> ToolResult:
    return await get_tool_connector().execute(name=name, arguments=arguments, **kwargs)
