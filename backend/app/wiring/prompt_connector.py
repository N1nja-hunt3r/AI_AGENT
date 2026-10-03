"""
prompt_connector.py - Production-grade prompt connector with Jinja2, caching, versioning.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from jinja2 import Environment
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False
    Environment = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_NAMES = [
    "system_prompt",
    "planner_prompt",
    "executor_prompt",
    "memory_prompt",
    "rag_prompt",
    "tool_prompt",
    "web_prompt",
    "computer_prompt",
    "reviewer_prompt",
    "manager_prompt",
]

DEFAULT_PROMPTS: Dict[str, str] = {
    "system_prompt": (
        "You are a helpful, accurate, and capable AI assistant.\n"
        "{% if agent_name %}Your name is {{ agent_name }}.{% endif %}\n"
        "{% if context %}Context:\n{{ context }}{% endif %}"
    ),
    "planner_prompt": (
        "You are a strategic planner. Given the following goal, produce a structured plan.\n"
        "Goal: {{ goal }}\n"
        "{% if constraints %}Constraints: {{ constraints }}{% endif %}\n"
        "{% if tools %}Available tools: {{ tools | join(', ') }}{% endif %}\n"
        "Produce a numbered list of steps."
    ),
    "executor_prompt": (
        "You are an executor agent. Execute the following step precisely.\n"
        "Step: {{ step }}\n"
        "{% if context %}Context:\n{{ context }}{% endif %}\n"
        "{% if tools %}Available tools: {{ tools | join(', ') }}{% endif %}\n"
        "Return the result of the step."
    ),
    "memory_prompt": (
        "You have access to the following memories relevant to the current task.\n"
        "{% for memory in memories %}- {{ memory }}{% endfor %}\n"
        "Use these memories to inform your response to: {{ query }}"
    ),
    "rag_prompt": (
        "Answer the question based on the following retrieved context.\n"
        "Context:\n{% for doc in documents %}{{ loop.index }}. {{ doc }}{% endfor %}\n"
        "Question: {{ question }}\n"
        "If the context does not contain the answer, say so."
    ),
    "tool_prompt": (
        "You have access to the following tools:\n"
        "{% for tool in tools %}{{ loop.index }}. {{ tool.name }}: {{ tool.description }}\n{% endfor %}\n"
        "Task: {{ task }}\n"
        "Select and use the appropriate tool to complete the task."
    ),
    "web_prompt": (
        "You are a web research agent. Search the web to answer the following query.\n"
        "Query: {{ query }}\n"
        "{% if results %}Search results:\n{% for r in results %}- {{ r }}{% endfor %}{% endif %}\n"
        "Summarize the most relevant and accurate information."
    ),
    "computer_prompt": (
        "You are a computer use agent. Perform the following computer task.\n"
        "Task: {{ task }}\n"
        "{% if screen_context %}Current screen: {{ screen_context }}{% endif %}\n"
        "Describe each action you take step by step."
    ),
    "reviewer_prompt": (
        "You are a quality reviewer. Review the following output for accuracy and completeness.\n"
        "Original task: {{ task }}\n"
        "Output to review:\n{{ output }}\n"
        "Provide specific feedback and a score from 1-10."
    ),
    "manager_prompt": (
        "You are a manager agent coordinating multiple sub-agents.\n"
        "Objective: {{ objective }}\n"
        "{% if agents %}Available agents: {{ agents | join(', ') }}{% endif %}\n"
        "{% if progress %}Current progress:\n{{ progress }}{% endif %}\n"
        "Determine the next action or delegation."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptVersion:
    name: str
    content: str
    version: int
    checksum: str
    loaded_at: float
    source: str  # "file" | "default" | "custom"
    path: Optional[str] = None


@dataclass
class RenderResult:
    name: str
    rendered: str
    version: int
    render_time_ms: float
    variables_used: List[str] = field(default_factory=list)


@dataclass
class PromptConnectorConfig:
    prompts_dir: str = "prompts"
    enable_cache: bool = True
    cache_ttl: float = 300.0  # seconds
    auto_reload: bool = False
    reload_interval: float = 60.0
    file_extensions: List[str] = field(default_factory=lambda: [".j2", ".jinja2", ".txt", ".md"])
    encoding: str = "utf-8"
    strict_undefined: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PromptConnectorError(Exception):
    pass

class PromptNotFoundError(PromptConnectorError):
    pass

class PromptRenderError(PromptConnectorError):
    pass


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    result: str
    created_at: float
    ttl: float

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class _RenderCache:
    def __init__(self, ttl: float = 300.0) -> None:
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl

    def _make_key(self, name: str, variables: Dict[str, Any]) -> str:
        raw = f"{name}:{sorted(variables.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, name: str, variables: Dict[str, Any]) -> Optional[str]:
        key = self._make_key(name, variables)
        async with self._lock:
            entry = self._store.get(key)
            if entry and not entry.is_expired():
                return entry.result
            if entry:
                del self._store[key]
        return None

    async def set(self, name: str, variables: Dict[str, Any], result: str) -> None:
        key = self._make_key(name, variables)
        async with self._lock:
            self._store[key] = _CacheEntry(result=result, created_at=time.time(), ttl=self._ttl)

    async def invalidate(self, name: Optional[str] = None) -> None:
        async with self._lock:
            if name is None:
                self._store.clear()
            else:
                keys_to_del = [k for k in self._store if name in k]
                for k in keys_to_del:
                    del self._store[k]

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Jinja2 renderer
# ---------------------------------------------------------------------------

class _TemplateRenderer:
    def __init__(self, strict: bool = False) -> None:
        self._strict = strict
        self._env: Any = None

    def _get_env(self, templates: Dict[str, str]) -> Any:
        if not _JINJA2_AVAILABLE:
            return None
        from jinja2 import DictLoader, StrictUndefined, Undefined
        loader = DictLoader(templates)
        undef = StrictUndefined if self._strict else Undefined
        env = Environment(
            loader=loader,
            autoescape=False,
            undefined=undef,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env

    def render(self, template_str: str, variables: Dict[str, Any], name: str = "") -> str:
        if not _JINJA2_AVAILABLE:
            return self._simple_render(template_str, variables)
        try:
            from jinja2 import Template, StrictUndefined, Undefined
            undef = StrictUndefined if self._strict else Undefined
            tmpl = Template(template_str, undefined=undef)
            tmpl.environment.trim_blocks = True
            tmpl.environment.lstrip_blocks = True
            return tmpl.render(**variables)
        except Exception as exc:
            raise PromptRenderError(f"Render failed for '{name}': {exc}") from exc

    def _simple_render(self, template: str, variables: Dict[str, Any]) -> str:
        result = template
        for k, v in variables.items():
            result = result.replace("{{ " + k + " }}", str(v))
            result = result.replace("{{" + k + "}}", str(v))
        return result

    def extract_variables(self, template_str: str) -> List[str]:
        import re
        return list(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", template_str)))


# ---------------------------------------------------------------------------
# PromptConnector – Singleton
# ---------------------------------------------------------------------------

class PromptConnector:
    _instance: Optional["PromptConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = PromptConnectorConfig()
        self._prompts: Dict[str, PromptVersion] = {}
        self._cache = _RenderCache()
        self._renderer = _TemplateRenderer()
        self._initialized = False
        self._reload_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._version_counter: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "PromptConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "PromptConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    async def load(self, config: Optional[PromptConnectorConfig] = None) -> "PromptConnector":
        if config:
            self._config = config
        self._renderer = _TemplateRenderer(strict=self._config.strict_undefined)
        self._cache = _RenderCache(ttl=self._config.cache_ttl)

        for name in PROMPT_NAMES:
            await self._load_prompt(name)

        if self._config.auto_reload:
            self._reload_task = asyncio.create_task(self._auto_reload_loop())

        self._initialized = True
        logger.info("PromptConnector loaded %d prompts from '%s'", len(self._prompts), self._config.prompts_dir)
        return self

    async def _load_prompt(self, name: str) -> PromptVersion:
        content, source, path = await self._resolve_prompt(name)
        checksum = hashlib.sha256(content.encode()).hexdigest()
        version = self._version_counter.get(name, 0) + 1
        self._version_counter[name] = version
        pv = PromptVersion(
            name=name,
            content=content,
            version=version,
            checksum=checksum,
            loaded_at=time.time(),
            source=source,
            path=path,
        )
        self._prompts[name] = pv
        logger.debug("Loaded prompt '%s' v%d from %s", name, version, source)
        return pv

    async def _resolve_prompt(self, name: str) -> Tuple[str, str, Optional[str]]:
        prompts_dir = Path(self._config.prompts_dir)
        for ext in self._config.file_extensions:
            candidate = prompts_dir / f"{name}{ext}"
            if candidate.exists():
                try:
                    content = await asyncio.to_thread(
                        candidate.read_text, encoding=self._config.encoding
                    )
                    return content, "file", str(candidate)
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", candidate, exc)

        # Check environment variable override
        env_key = f"PROMPT_{name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val, "env", None

        # Fall back to built-in defaults
        default = DEFAULT_PROMPTS.get(name, f"# Prompt: {name}\n{{{{ content }}}}")
        return default, "default", None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    async def render(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> RenderResult:
        await self._ensure_initialized()
        variables = variables or {}

        if self._config.enable_cache and use_cache:
            cached = await self._cache.get(name, variables)
            if cached is not None:
                pv = self._prompts.get(name)
                return RenderResult(
                    name=name,
                    rendered=cached,
                    version=pv.version if pv else 0,
                    render_time_ms=0.0,
                )

        pv = self._prompts.get(name)
        if not pv:
            raise PromptNotFoundError(f"Prompt '{name}' not found")

        t0 = time.perf_counter()
        rendered = self._renderer.render(pv.content, variables, name=name)
        elapsed = (time.perf_counter() - t0) * 1000

        if self._config.enable_cache and use_cache:
            await self._cache.set(name, variables, rendered)

        vars_used = self._renderer.extract_variables(pv.content)
        return RenderResult(
            name=name,
            rendered=rendered,
            version=pv.version,
            render_time_ms=elapsed,
            variables_used=vars_used,
        )

    async def render_system_prompt(self, **variables: Any) -> str:
        result = await self.render("system_prompt", variables)
        return result.rendered

    async def render_planner_prompt(self, **variables: Any) -> str:
        result = await self.render("planner_prompt", variables)
        return result.rendered

    async def render_executor_prompt(self, **variables: Any) -> str:
        result = await self.render("executor_prompt", variables)
        return result.rendered

    async def render_memory_prompt(self, **variables: Any) -> str:
        result = await self.render("memory_prompt", variables)
        return result.rendered

    async def render_rag_prompt(self, **variables: Any) -> str:
        result = await self.render("rag_prompt", variables)
        return result.rendered

    async def render_tool_prompt(self, **variables: Any) -> str:
        result = await self.render("tool_prompt", variables)
        return result.rendered

    async def render_web_prompt(self, **variables: Any) -> str:
        result = await self.render("web_prompt", variables)
        return result.rendered

    async def render_computer_prompt(self, **variables: Any) -> str:
        result = await self.render("computer_prompt", variables)
        return result.rendered

    async def render_reviewer_prompt(self, **variables: Any) -> str:
        result = await self.render("reviewer_prompt", variables)
        return result.rendered

    async def render_manager_prompt(self, **variables: Any) -> str:
        result = await self.render("manager_prompt", variables)
        return result.rendered

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    async def reload(self, name: Optional[str] = None) -> Dict[str, bool]:
        await self._ensure_initialized()
        results: Dict[str, bool] = {}
        names = [name] if name else PROMPT_NAMES
        for n in names:
            try:
                pv = await self._load_prompt(n)
                await self._cache.invalidate(n)
                results[n] = True
                logger.info("Reloaded prompt '%s' v%d", n, pv.version)
            except Exception as exc:
                logger.error("Failed to reload prompt '%s': %s", n, exc)
                results[n] = False
        return results

    async def _auto_reload_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.reload_interval)
            try:
                await self.reload()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Auto-reload failed: %s", exc)

    # ------------------------------------------------------------------
    # Custom prompt registration
    # ------------------------------------------------------------------

    async def register_prompt(self, name: str, content: str) -> PromptVersion:
        await self._ensure_initialized()
        checksum = hashlib.sha256(content.encode()).hexdigest()
        version = self._version_counter.get(name, 0) + 1
        self._version_counter[name] = version
        pv = PromptVersion(
            name=name,
            content=content,
            version=version,
            checksum=checksum,
            loaded_at=time.time(),
            source="custom",
        )
        self._prompts[name] = pv
        await self._cache.invalidate(name)
        return pv

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_version(self, name: str) -> Optional[PromptVersion]:
        return self._prompts.get(name)

    def list_prompts(self) -> List[str]:
        return list(self._prompts.keys())

    def get_all_versions(self) -> Dict[str, int]:
        return {n: pv.version for n, pv in self._prompts.items()}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        missing = [n for n in PROMPT_NAMES if n not in self._prompts]
        cache_size = await self._cache.size()
        return {
            "healthy": len(missing) == 0,
            "loaded_prompts": len(self._prompts),
            "missing_prompts": missing,
            "cache_size": cache_size,
            "jinja2_available": _JINJA2_AVAILABLE,
            "prompts_dir": self._config.prompts_dir,
            "auto_reload": self._config.auto_reload,
            "versions": self.get_all_versions(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.load()

    async def __aenter__(self) -> "PromptConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    async def shutdown(self) -> None:
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        self._initialized = False

    def __repr__(self) -> str:
        return f"PromptConnector(loaded={len(self._prompts)}, initialized={self._initialized})"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_prompt_connector() -> PromptConnector:
    return PromptConnector.get_instance()


async def render_prompt(name: str, **variables: Any) -> str:
    connector = get_prompt_connector()
    result = await connector.render(name, variables)
    return result.rendered
