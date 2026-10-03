from __future__ import annotations

import asyncio
import inspect
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------

class ToolStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ToolCategory(str, Enum):
    SYSTEM = "system"
    NETWORK = "network"
    FILE = "file"
    CODE = "code"
    MEMORY = "memory"
    SEARCH = "search"
    BROWSER = "browser"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolMetadata:
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = "Any"
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 0.5
    requires_confirmation: bool = False
    is_async: bool = True

    def to_json_schema(self) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in self.parameters:
            prop: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "object",
            "properties": props,
            "required": required,
        }

    def to_llm_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }


@dataclass
class ToolResult:
    tool_name: str
    call_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
        }


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class HealthCheckResult:
    tool_name: str
    status: ToolStatus
    latency_ms: float
    error: Optional[str] = None
    checked_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolServiceError(Exception):
    pass

class ToolNotFoundError(ToolServiceError):
    pass

class ToolExecutionError(ToolServiceError):
    pass

class ToolTimeoutError(ToolServiceError):
    pass

class ToolValidationError(ToolServiceError):
    pass


# ---------------------------------------------------------------------------
# Tool registration entry
# ---------------------------------------------------------------------------

@dataclass
class RegisteredTool:
    metadata: ToolMetadata
    fn: Callable[..., Any]
    health_fn: Optional[Callable[[], Any]] = None
    _call_count: int = 0
    _error_count: int = 0
    _total_latency_ms: float = 0.0
    _status: ToolStatus = ToolStatus.UNKNOWN

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._total_latency_ms / self._call_count

    def record_call(self, latency_ms: float, success: bool) -> None:
        self._call_count += 1
        self._total_latency_ms += latency_ms
        if not success:
            self._error_count += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.metadata.name,
            "category": self.metadata.category.value,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "status": self._status.value,
        }


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(
    fn: Callable[[], Any],
    max_retries: int,
    delay: float,
    tool_name: str,
) -> Tuple[Any, int]:
    last_exc: Exception = ToolExecutionError("No attempts")
    for attempt in range(max_retries + 1):
        try:
            result = await fn()
            return result, attempt
        except (ToolTimeoutError, ToolExecutionError) as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))
                logger.warning("Tool %s retry %d/%d", tool_name, attempt + 1, max_retries)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# ToolService
# ---------------------------------------------------------------------------

class ToolService:
    """
    Central registry and executor for all agent tools.
    Compatible with executor.py, planner.py, llm_service.py tool calling conventions.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredTool] = {}
        self._lock = asyncio.Lock()
        self._execution_history: List[ToolResult] = []
        self._max_history: int = 1000

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        metadata: ToolMetadata,
        fn: Callable[..., Any],
        health_fn: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._registry[metadata.name] = RegisteredTool(
            metadata=metadata, fn=fn, health_fn=health_fn
        )
        logger.info("Registered tool: %s [%s]", metadata.name, metadata.category.value)

    def register_decorator(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            meta = ToolMetadata(
                name=name,
                description=description,
                category=category,
                parameters=parameters or [],
                timeout=timeout,
                max_retries=max_retries,
                is_async=asyncio.iscoroutinefunction(fn),
            )
            self.register(meta, fn)
            return fn
        return decorator

    def unregister(self, name: str) -> bool:
        return self._registry.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> RegisteredTool:
        if name not in self._registry:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return self._registry[name]

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
    ) -> List[ToolMetadata]:
        tools = [t.metadata for t in self._registry.values()]
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_llm_schemas(
        self,
        names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        tools: list[RegisteredTool] = list(self._registry.values())
        if names:
            tools = [t for t in tools if t.metadata.name in names]
        return [t.metadata.to_llm_dict() for t in tools]

    def has_tool(self, name: str) -> bool:
        return name in self._registry

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if call.name not in self._registry:
            return ToolResult(
                tool_name=call.name,
                call_id=call.call_id,
                success=False,
                output=None,
                error=f"Tool '{call.name}' not found",
                error_type="ToolNotFoundError",
            )

        registered = self._registry[call.name]
        meta = registered.metadata

        try:
            self._validate_arguments(meta, call.arguments)
        except ToolValidationError as exc:
            return ToolResult(
                tool_name=call.name,
                call_id=call.call_id,
                success=False,
                output=None,
                error=str(exc),
                error_type="ToolValidationError",
            )

        t0 = time.monotonic()
        retries = 0

        async def _invoke() -> Any:
            args = dict(call.arguments)
            if context:
                sig = inspect.signature(registered.fn)
                if "_context" in sig.parameters:
                    args["_context"] = context
            if asyncio.iscoroutinefunction(registered.fn):
                return await asyncio.wait_for(registered.fn(**args), timeout=meta.timeout)
            else:
                return await asyncio.wait_for(
                    asyncio.to_thread(registered.fn, **args), timeout=meta.timeout
                )

        try:
            output, retries = await _with_retry(
                _invoke, meta.max_retries, meta.retry_delay, call.name
            )
            latency = (time.monotonic() - t0) * 1000
            registered.record_call(latency, True)
            result = ToolResult(
                tool_name=call.name,
                call_id=call.call_id,
                success=True,
                output=output,
                latency_ms=latency,
                retries=retries,
            )
        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            registered.record_call(latency, False)
            result = ToolResult(
                tool_name=call.name,
                call_id=call.call_id,
                success=False,
                output=None,
                error=f"Tool timed out after {meta.timeout}s",
                error_type="ToolTimeoutError",
                latency_ms=latency,
                retries=retries,
            )
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            registered.record_call(latency, False)
            result = ToolResult(
                tool_name=call.name,
                call_id=call.call_id,
                success=False,
                output=None,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=latency,
                retries=retries,
            )
            logger.error("Tool %s failed: %s\n%s", call.name, exc, traceback.format_exc())

        self._record_history(result)
        return result

    async def execute_batch(
        self,
        calls: List[ToolCall],
        parallel: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        if parallel:
            return list(await asyncio.gather(*[self.execute(c, context) for c in calls]))
        results: List[ToolResult] = []
        for call in calls:
            results.append(await self.execute(call, context))
        return results

    async def execute_by_name(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        return await self.execute(ToolCall(name=name, arguments=arguments), context)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check(self, name: str) -> HealthCheckResult:
        if name not in self._registry:
            return HealthCheckResult(
                tool_name=name,
                status=ToolStatus.UNAVAILABLE,
                latency_ms=0.0,
                error="Not registered",
            )
        registered = self._registry[name]
        t0 = time.monotonic()
        if registered.health_fn is None:
            registered._status = ToolStatus.HEALTHY
            return HealthCheckResult(
                tool_name=name,
                status=ToolStatus.HEALTHY,
                latency_ms=0.0,
            )
        try:
            if asyncio.iscoroutinefunction(registered.health_fn):
                await asyncio.wait_for(registered.health_fn(), timeout=5.0)
            else:
                await asyncio.to_thread(registered.health_fn)
            latency = (time.monotonic() - t0) * 1000
            registered._status = ToolStatus.HEALTHY
            return HealthCheckResult(tool_name=name, status=ToolStatus.HEALTHY, latency_ms=latency)
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            registered._status = ToolStatus.DEGRADED
            return HealthCheckResult(
                tool_name=name,
                status=ToolStatus.DEGRADED,
                latency_ms=latency,
                error=str(exc),
            )

    async def health_check_all(self) -> List[HealthCheckResult]:
        return list(await asyncio.gather(*[self.health_check(n) for n in self._registry]))

    # ------------------------------------------------------------------
    # History & stats
    # ------------------------------------------------------------------

    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[ToolResult]:
        hist = self._execution_history
        if tool_name:
            hist = [r for r in hist if r.tool_name == tool_name]
        return hist[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_tools": len(self._registry),
            "total_executions": sum(t.call_count for t in self._registry.values()),
            "total_errors": sum(t.error_count for t in self._registry.values()),
            "tools": [t.stats() for t in self._registry.values()],
        }

    def clear_history(self) -> None:
        self._execution_history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_arguments(self, meta: ToolMetadata, args: Dict[str, Any]) -> None:
        for param in meta.parameters:
            if param.required and param.name not in args:
                raise ToolValidationError(
                    f"Tool '{meta.name}': required parameter '{param.name}' missing"
                )
            if param.enum and param.name in args and args[param.name] not in param.enum:
                raise ToolValidationError(
                    f"Tool '{meta.name}': '{param.name}' must be one of {param.enum}"
                )

    def _record_history(self, result: ToolResult) -> None:
        self._execution_history.append(result)
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history :]
