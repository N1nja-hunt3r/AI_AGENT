"""
capability_connector.py - Production-grade capability connector with registry and DI.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CapabilityStatus(str, Enum):
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class CapabilityType(str, Enum):
    MEMORY = "memory"
    TOOL = "tool"
    WEB = "web"
    RAG = "rag"
    COMPUTER = "computer"
    CODING = "coding"
    VISION = "vision"
    FILES = "files"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Base capability interface
# ---------------------------------------------------------------------------

class BaseCapability:
    capability_type: CapabilityType = CapabilityType.CUSTOM
    capability_name: str = "base"
    version: str = "1.0.0"
    dependencies: List[str] = []

    async def initialize(self, **kwargs: Any) -> None:
        pass

    async def execute(self, action: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": self.capability_name}

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CapabilityRecord:
    name: str
    capability_type: CapabilityType
    instance: BaseCapability
    status: CapabilityStatus = CapabilityStatus.REGISTERED
    registered_at: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    health_check_result: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    capability_name: str
    action: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CapabilityConnectorConfig:
    auto_discover: bool = True
    discovery_paths: List[str] = field(default_factory=lambda: ["capabilities"])
    health_check_interval: float = 60.0
    execution_timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_metrics: bool = True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CapabilityConnectorError(Exception):
    pass

class CapabilityNotFoundError(CapabilityConnectorError):
    pass

class CapabilityInitError(CapabilityConnectorError):
    pass

class CapabilityExecutionError(CapabilityConnectorError):
    pass


# ---------------------------------------------------------------------------
# Built-in stub capabilities
# ---------------------------------------------------------------------------

class MemoryCapabilityStub(BaseCapability):
    capability_type = CapabilityType.MEMORY
    capability_name = "memory"

    def __init__(self, memory_connector: Any = None) -> None:
        self._connector = memory_connector

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if self._connector is None:
            return {"action": action, "status": "no_connector"}
        if action == "store":
            return await self._connector.store(**kwargs)
        elif action == "retrieve":
            return await self._connector.retrieve(**kwargs)
        elif action == "get":
            return await self._connector.get(**kwargs)
        elif action == "delete":
            return await self._connector.delete(**kwargs)
        raise CapabilityExecutionError(f"Unknown memory action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        if self._connector:
            try:
                return await self._connector.health_check()
            except Exception as exc:
                return {"healthy": False, "error": str(exc)}
        return {"healthy": True, "name": "memory", "connector": "none"}


class ToolCapabilityStub(BaseCapability):
    capability_type = CapabilityType.TOOL
    capability_name = "tool"

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    async def execute(self, action: str, **kwargs: Any) -> Any:
        tool = self._tools.get(action)
        if tool is None:
            raise CapabilityExecutionError(f"Tool '{action}' not registered")
        if inspect.iscoroutinefunction(tool):
            return await tool(**kwargs)
        return await asyncio.to_thread(tool, **kwargs)

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": "tool", "tools": list(self._tools.keys())}


class WebCapabilityStub(BaseCapability):
    capability_type = CapabilityType.WEB
    capability_name = "web"

    async def execute(self, action: str, **kwargs: Any) -> Any:
        try:
            import httpx
            if action == "get":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(kwargs.get("url", ""))
                    return {"status_code": resp.status_code, "text": resp.text[:5000]}
            elif action == "post":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        kwargs.get("url", ""),
                        json=kwargs.get("json"),
                        data=kwargs.get("data"),
                    )
                    return {"status_code": resp.status_code, "text": resp.text[:5000]}
            elif action == "search":
                return {"query": kwargs.get("query"), "results": [], "note": "web search stub"}
        except ImportError:
            return {"error": "httpx not installed"}
        raise CapabilityExecutionError(f"Unknown web action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://www.google.com")
                return {"healthy": resp.status_code < 500, "name": "web"}
        except Exception as exc:
            return {"healthy": False, "name": "web", "error": str(exc)}


class RAGCapabilityStub(BaseCapability):
    capability_type = CapabilityType.RAG
    capability_name = "rag"

    def __init__(self, memory_connector: Any = None, llm_connector: Any = None) -> None:
        self._memory = memory_connector
        self._llm = llm_connector

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "retrieve":
            if self._memory:
                return await self._memory.retrieve(
                    query=kwargs.get("query", ""),
                    top_k=kwargs.get("top_k", 5),
                )
            return []
        elif action == "generate":
            query = kwargs.get("query", "")
            documents = kwargs.get("documents", [])
            if self._llm:
                messages = [
                    {"role": "system", "content": "Answer based on provided documents."},
                    {"role": "user", "content": f"Documents: {documents}\n\nQuestion: {query}"},
                ]
                return await self._llm.complete(messages=messages)
            return {"query": query, "answer": "LLM not available"}
        raise CapabilityExecutionError(f"Unknown RAG action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": "rag", "memory": self._memory is not None}


class CodingCapabilityStub(BaseCapability):
    capability_type = CapabilityType.CODING
    capability_name = "coding"

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "generate":
            return {"action": "generate", "language": kwargs.get("language", "python"), "description": kwargs.get("description", "")}
        elif action == "review":
            return {"action": "review", "issues": [], "code": kwargs.get("code", "")}
        elif action == "explain":
            return {"action": "explain", "explanation": f"Explanation for {len(kwargs.get('code', ''))} chars of code"}
        raise CapabilityExecutionError(f"Unknown coding action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": "coding"}


class VisionCapabilityStub(BaseCapability):
    capability_type = CapabilityType.VISION
    capability_name = "vision"

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "describe":
            return {"action": "describe", "description": "Image description placeholder"}
        elif action == "ocr":
            return {"action": "ocr", "text": "OCR text placeholder", "confidence": 0.9}
        elif action == "analyze":
            return {"action": "analyze", "analysis": {"subjects": [], "colors": []}}
        raise CapabilityExecutionError(f"Unknown vision action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": "vision"}


class FileCapabilityStub(BaseCapability):
    capability_type = CapabilityType.FILES
    capability_name = "files"

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "list":
            return {"action": "list", "path": kwargs.get("path", "."), "entries": []}
        elif action == "read":
            return {"action": "read", "path": kwargs.get("path", ""), "content": ""}
        elif action == "write":
            return {"action": "write", "path": kwargs.get("path", ""), "size": 0}
        elif action == "search":
            return {"action": "search", "query": kwargs.get("query", ""), "results": []}
        raise CapabilityExecutionError(f"Unknown file action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "name": "files"}


class ComputerCapabilityStub(BaseCapability):
    capability_type = CapabilityType.COMPUTER
    capability_name = "computer"

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "screenshot":
            try:
                import pyautogui
                screenshot = await asyncio.to_thread(pyautogui.screenshot)
                return {"action": "screenshot", "size": screenshot.size}
            except ImportError:
                return {"action": "screenshot", "error": "pyautogui not installed"}
        elif action == "click":
            try:
                import pyautogui
                await asyncio.to_thread(pyautogui.click, kwargs.get("x", 0), kwargs.get("y", 0))
                return {"action": "click", "x": kwargs.get("x"), "y": kwargs.get("y")}
            except ImportError:
                return {"action": "click", "error": "pyautogui not installed"}
        elif action == "type":
            try:
                import pyautogui
                await asyncio.to_thread(pyautogui.typewrite, kwargs.get("text", ""))
                return {"action": "type", "text": kwargs.get("text")}
            except ImportError:
                return {"action": "type", "error": "pyautogui not installed"}
        raise CapabilityExecutionError(f"Unknown computer action: {action}")

    async def health_check(self) -> Dict[str, Any]:
        import importlib.util
        pyautogui_ok = importlib.util.find_spec("pyautogui") is not None
        return {"healthy": pyautogui_ok, "name": "computer", "backend": "pyautogui" if pyautogui_ok else None}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class _CapabilityRegistry:
    def __init__(self) -> None:
        self._records: Dict[str, CapabilityRecord] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        instance: BaseCapability,
        capability_type: Optional[CapabilityType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CapabilityRecord:
        ctype = capability_type or instance.capability_type
        record = CapabilityRecord(
            name=name,
            capability_type=ctype,
            instance=instance,
            status=CapabilityStatus.REGISTERED,
            metadata=metadata or {},
        )
        async with self._lock:
            self._records[name] = record
        logger.debug("Registered capability '%s' type=%s", name, ctype.value)
        return record

    async def unregister(self, name: str) -> bool:
        async with self._lock:
            return bool(self._records.pop(name, None))

    def get(self, name: str) -> Optional[CapabilityRecord]:
        return self._records.get(name)

    def list_names(self) -> List[str]:
        return list(self._records.keys())

    def list_by_type(self, capability_type: CapabilityType) -> List[CapabilityRecord]:
        return [r for r in self._records.values() if r.capability_type == capability_type]

    def all_records(self) -> Dict[str, CapabilityRecord]:
        return dict(self._records)

    async def update_status(self, name: str, status: CapabilityStatus) -> None:
        async with self._lock:
            if name in self._records:
                self._records[name].status = status


# ---------------------------------------------------------------------------
# CapabilityConnector – Singleton
# ---------------------------------------------------------------------------

class CapabilityConnector:
    _instance: Optional["CapabilityConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = CapabilityConnectorConfig()
        self._registry = _CapabilityRegistry()
        self._initialized = False
        self._health_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._injected: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CapabilityConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "CapabilityConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: Optional[CapabilityConnectorConfig] = None,
        memory_connector: Any = None,
        llm_connector: Any = None,
        **injected: Any,
    ) -> "CapabilityConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        # Store injected dependencies
        self._injected = {
            "memory_connector": memory_connector,
            "llm_connector": llm_connector,
            **injected,
        }

        # Register built-in capabilities
        await self._register_builtin_capabilities()

        if self._config.auto_discover:
            await self.discover()

        if self._config.health_check_interval > 0:
            self._health_task = asyncio.create_task(self._health_loop())

        self._initialized = True
        logger.info("CapabilityConnector initialized with %d capabilities", len(self._registry.list_names()))
        return self

    async def _register_builtin_capabilities(self) -> None:
        memory_conn = self._injected.get("memory_connector")
        llm_conn = self._injected.get("llm_connector")

        builtins: List[tuple[str, BaseCapability]] = [
            ("memory", MemoryCapabilityStub(memory_connector=memory_conn)),
            ("tool", ToolCapabilityStub()),
            ("web", WebCapabilityStub()),
            ("rag", RAGCapabilityStub(memory_connector=memory_conn, llm_connector=llm_conn)),
            ("coding", CodingCapabilityStub()),
            ("vision", VisionCapabilityStub()),
            ("files", FileCapabilityStub()),
            ("computer", ComputerCapabilityStub()),
        ]
        for name, instance in builtins:
            record = await self._registry.register(name, instance)
            await self._init_capability(record)

    async def _init_capability(self, record: CapabilityRecord) -> None:
        await self._registry.update_status(record.name, CapabilityStatus.INITIALIZING)
        try:
            await record.instance.initialize(**self._injected)
            await self._registry.update_status(record.name, CapabilityStatus.READY)
            logger.debug("Capability '%s' initialized", record.name)
        except Exception as exc:
            await self._registry.update_status(record.name, CapabilityStatus.FAILED)
            logger.warning("Capability '%s' init failed: %s", record.name, exc)

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    async def register(
        self,
        name: str,
        instance: BaseCapability,
        capability_type: Optional[CapabilityType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_init: bool = True,
    ) -> CapabilityRecord:
        await self._ensure_initialized()
        record = await self._registry.register(name, instance, capability_type, metadata)
        if auto_init:
            await self._init_capability(record)
        return record

    async def register_class(
        self,
        name: str,
        cls: Type[BaseCapability],
        **init_kwargs: Any,
    ) -> CapabilityRecord:
        instance = cls(**init_kwargs)
        return await self.register(name, instance)

    # ------------------------------------------------------------------
    # Discover
    # ------------------------------------------------------------------

    async def discover(self) -> List[str]:
        discovered: List[str] = []
        for path in self._config.discovery_paths:
            found = await self._discover_in_path(path)
            discovered.extend(found)
        return discovered

    async def _discover_in_path(self, path: str) -> List[str]:
        discovered: List[str] = []
        try:
            import importlib.util
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                return []
            for py_file in p.rglob("*.py"):
                module_name = py_file.stem
                if module_name.startswith("_"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)  # type: ignore[union-attr]
                        for attr_name in dir(mod):
                            attr = getattr(mod, attr_name)
                            if (
                                inspect.isclass(attr)
                                and issubclass(attr, BaseCapability)
                                and attr is not BaseCapability
                                and attr.capability_name not in self._registry.list_names()
                            ):
                                instance = attr()
                                record = await self._registry.register(
                                    attr.capability_name, instance
                                )
                                await self._init_capability(record)
                                discovered.append(attr.capability_name)
                                logger.info("Discovered capability '%s'", attr.capability_name)
                except Exception as exc:
                    logger.debug("Discovery skip %s: %s", py_file, exc)
        except Exception as exc:
            logger.warning("Discovery path '%s' error: %s", path, exc)
        return discovered

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        capability_name: str,
        action: str,
        retries: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        await self._ensure_initialized()
        record = self._registry.get(capability_name)
        if not record:
            raise CapabilityNotFoundError(f"Capability '{capability_name}' not found")

        max_retries = retries if retries is not None else self._config.max_retries
        exec_timeout = timeout or self._config.execution_timeout
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    record.instance.execute(action, **kwargs),
                    timeout=exec_timeout,
                )
                latency = (time.perf_counter() - t0) * 1000
                record.execution_count += 1
                return ExecutionResult(
                    capability_name=capability_name,
                    action=action,
                    success=True,
                    result=result,
                    latency_ms=latency,
                )
            except asyncio.TimeoutError as exc:
                last_exc = exc
                record.error_count += 1
                logger.warning("Capability '%s' action '%s' timed out (attempt %d)", capability_name, action, attempt + 1)
            except Exception as exc:
                last_exc = exc
                record.error_count += 1
                logger.warning("Capability '%s' action '%s' failed: %s (attempt %d)", capability_name, action, exc, attempt + 1)
            if attempt < max_retries - 1:
                await asyncio.sleep(self._config.retry_delay * (2 ** attempt))

        latency = (time.perf_counter() - t0) * 1000  # type: ignore[used-before-assignment]
        return ExecutionResult(
            capability_name=capability_name,
            action=action,
            success=False,
            error=str(last_exc),
            latency_ms=latency,
        )

    async def execute_any(
        self,
        capability_type: CapabilityType,
        action: str,
        **kwargs: Any,
    ) -> ExecutionResult:
        records = self._registry.list_by_type(capability_type)
        ready = [r for r in records if r.status == CapabilityStatus.READY]
        if not ready:
            raise CapabilityNotFoundError(f"No ready capability of type '{capability_type}'")
        return await self.execute(ready[0].name, action, **kwargs)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self, name: Optional[str] = None) -> Dict[str, Any]:
        await self._ensure_initialized()
        names = [name] if name else self._registry.list_names()
        results: Dict[str, Any] = {}
        checks = await asyncio.gather(
            *[self._check_one(n) for n in names],
            return_exceptions=True,
        )
        for n, outcome in zip(names, checks):
            if isinstance(outcome, Exception):
                results[n] = {"healthy": False, "error": str(outcome)}
            else:
                results[n] = outcome
        overall = all(r.get("healthy", False) for r in results.values())
        return {"healthy": overall, "capabilities": results, "total": len(names)}

    async def _check_one(self, name: str) -> Dict[str, Any]:
        record = self._registry.get(name)
        if not record:
            return {"healthy": False, "error": "not found"}
        try:
            result = await asyncio.wait_for(record.instance.health_check(), timeout=10.0)
            record.last_health_check = time.time()
            record.health_check_result = result
            if result.get("healthy", False):
                await self._registry.update_status(name, CapabilityStatus.READY)
            else:
                await self._registry.update_status(name, CapabilityStatus.DEGRADED)
            return result
        except Exception as exc:
            await self._registry.update_status(name, CapabilityStatus.DEGRADED)
            return {"healthy": False, "name": name, "error": str(exc)}

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.health_check_interval)
            try:
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Health loop error: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        for name in self._registry.list_names():
            record = self._registry.get(name)
            if record:
                try:
                    await record.instance.shutdown()
                    await self._registry.update_status(name, CapabilityStatus.SHUTDOWN)
                except Exception as exc:
                    logger.warning("Shutdown error for '%s': %s", name, exc)
        self._initialized = False
        logger.info("CapabilityConnector shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def get_capability(self, name: str) -> Optional[BaseCapability]:
        record = self._registry.get(name)
        return record.instance if record else None

    def list_capabilities(self) -> List[str]:
        return self._registry.list_names()

    def get_record(self, name: str) -> Optional[CapabilityRecord]:
        return self._registry.get(name)

    async def __aenter__(self) -> "CapabilityConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return f"CapabilityConnector(capabilities={self._registry.list_names()}, initialized={self._initialized})"


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def get_capability_connector() -> CapabilityConnector:
    return CapabilityConnector.get_instance()
