"""
Prompt Registry
Thread-safe, async-capable Jinja2 prompt registry with versioning,
render caching, hot-reload, and health checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    select_autoescape,
)

logger = logging.getLogger("prompt_registry")


class PromptRegistryError(Exception):
    """Base exception for prompt registry errors."""


class PromptNotFoundError(PromptRegistryError):
    """Raised when a requested prompt or version cannot be located."""


class PromptRenderError(PromptRegistryError):
    """Raised when a prompt template fails to compile or render."""


class PromptValidationError(PromptRegistryError):
    """Raised when a prompt fails validation on registration."""


@dataclass(frozen=True)
class PromptVersion:
    """Immutable record of a single prompt version."""

    version: str
    content: str
    checksum: str
    registered_at: float
    source_path: Optional[str] = None


@dataclass
class PromptEntry:
    """A named prompt with its version history and compiled template cache."""

    name: str
    versions: Dict[str, PromptVersion] = field(default_factory=dict)
    latest_version: Optional[str] = None
    compiled: Dict[str, Template] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class HealthStatus:
    """Result of a registry-wide health check."""

    healthy: bool
    total_prompts: int
    total_versions: int
    failed_prompts: List[str]
    checked_at: float
    details: Dict[str, Any] = field(default_factory=dict)


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class PromptRegistry:
    """
    Thread-safe registry for loading, versioning, caching, and rendering
    Jinja2 prompt templates used across an agent runtime.
    """

    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        autoescape: bool = False,
        strict_undefined: bool = True,
        cache_size: int = 256,
    ) -> None:
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._entries: Dict[str, PromptEntry] = {}
        self._failed: Dict[str, str] = {}
        self._template_dir = Path(template_dir) if template_dir else None

        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)) if self._template_dir else None,
            autoescape=select_autoescape(disabled_extensions=("jinja2",)) if autoescape else False,
            undefined=StrictUndefined if strict_undefined else None,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._max_render_cache = cache_size
        self._render_cache: Dict[str, str] = {}
        self._render_cache_order: List[str] = []

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        content: Optional[str] = None,
        *,
        path: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        set_latest: bool = True,
    ) -> PromptVersion:
        """
        Register a new prompt or a new version of an existing prompt.

        Exactly one of `content` or `path` must be provided. If `version`
        is omitted, the content checksum is used as the version id.
        """
        if content is None and path is None:
            raise PromptValidationError(
                f"register('{name}'): either content or path must be provided"
            )
        if content is not None and path is not None:
            raise PromptValidationError(
                f"register('{name}'): provide only one of content or path"
            )

        source_path: Optional[str] = None
        if path is not None:
            file_path = Path(path)
            if not file_path.is_file():
                raise PromptValidationError(f"register('{name}'): path not found: {file_path}")
            content = file_path.read_text(encoding="utf-8")
            source_path = str(file_path)

        if not content or not content.strip():
            raise PromptValidationError(f"register('{name}'): content is empty")

        checksum = _checksum(content)
        resolved_version = version or checksum

        prompt_version = PromptVersion(
            version=resolved_version,
            content=content,
            checksum=checksum,
            registered_at=time.time(),
            source_path=source_path,
        )

        with self._lock:
            entry = self._entries.setdefault(name, PromptEntry(name=name))
            entry.versions[resolved_version] = prompt_version
            entry.compiled.pop(resolved_version, None)
            if set_latest or entry.latest_version is None:
                entry.latest_version = resolved_version
            self._failed.pop(name, None)

        logger.info("Registered prompt '%s' version '%s'", name, resolved_version)
        return prompt_version

    def register_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*.jinja2",
    ) -> List[str]:
        """Bulk-register every template file in a directory matching `pattern`."""
        target_dir = Path(directory) if directory else self._template_dir
        if target_dir is None:
            raise PromptValidationError("register_directory: no directory configured")

        registered: List[str] = []
        for file_path in sorted(target_dir.glob(pattern)):
            name = file_path.stem
            try:
                self.register(name, path=file_path)
                registered.append(name)
            except PromptRegistryError as exc:
                with self._lock:
                    self._failed[name] = str(exc)
                logger.error("Failed to register '%s': %s", name, exc)
        return registered

    # ------------------------------------------------------------------ #
    # Loading / lookup
    # ------------------------------------------------------------------ #

    def load(self, name: str, version: Optional[str] = None) -> PromptVersion:
        """Retrieve a registered prompt version (latest by default)."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise PromptNotFoundError(f"Prompt '{name}' is not registered")
            target_version = version or entry.latest_version
            if target_version is None or target_version not in entry.versions:
                raise PromptNotFoundError(f"Prompt '{name}' has no version '{target_version}'")
            return entry.versions[target_version]

    def list_prompts(self) -> List[str]:
        """List all registered prompt names."""
        with self._lock:
            return sorted(self._entries.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all version ids registered for a given prompt."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise PromptNotFoundError(f"Prompt '{name}' is not registered")
            return sorted(entry.versions.keys())

    # ------------------------------------------------------------------ #
    # Compilation / rendering
    # ------------------------------------------------------------------ #

    def _compile(self, name: str, prompt_version: PromptVersion) -> Template:
        with self._lock:
            entry = self._entries[name]
            compiled = entry.compiled.get(prompt_version.version)
            if compiled is not None:
                return compiled
            try:
                compiled = self._env.from_string(prompt_version.content)
            except TemplateError as exc:
                raise PromptRenderError(
                    f"Failed to compile prompt '{name}@{prompt_version.version}': {exc}"
                ) from exc
            entry.compiled[prompt_version.version] = compiled
            return compiled

    def render(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        version: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Render a registered prompt template synchronously with the given context."""
        context = context or {}
        prompt_version = self.load(name, version=version)
        cache_key = self._cache_key(name, prompt_version.version, context)

        if use_cache:
            with self._lock:
                cached = self._render_cache.get(cache_key)
                if cached is not None:
                    return cached

        template = self._compile(name, prompt_version)
        try:
            rendered = template.render(**context)
        except TemplateError as exc:
            raise PromptRenderError(
                f"Failed to render prompt '{name}@{prompt_version.version}': {exc}"
            ) from exc

        if use_cache:
            self._store_cache(cache_key, rendered)
        return rendered

    async def render_async(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        version: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Async-safe wrapper around render(), suitable for async runtimes."""
        async with self._async_lock:
            return await asyncio.to_thread(
                self.render, name, context, version=version, use_cache=use_cache
            )

    def _cache_key(self, name: str, version: str, context: Dict[str, Any]) -> str:
        try:
            stable = repr(sorted(context.items(), key=lambda kv: str(kv[0])))
        except Exception:
            stable = repr(context)
        return f"{name}:{version}:{_checksum(stable)}"

    def _store_cache(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._render_cache:
                self._render_cache_order.remove(key)
            self._render_cache[key] = value
            self._render_cache_order.append(key)
            while len(self._render_cache_order) > self._max_render_cache:
                oldest = self._render_cache_order.pop(0)
                self._render_cache.pop(oldest, None)

    # ------------------------------------------------------------------ #
    # Reload / invalidation
    # ------------------------------------------------------------------ #

    def reload(self, name: Optional[str] = None) -> List[str]:
        """
        Reload prompt(s) from their original source file on disk.
        If `name` is omitted, reloads every prompt that has a known source path.
        """
        reloaded: List[str] = []
        with self._lock:
            targets = [name] if name else list(self._entries.keys())
            for target in targets:
                entry = self._entries.get(target)
                if entry is None or entry.latest_version is None:
                    continue
                current = entry.versions[entry.latest_version]
                if not current.source_path:
                    continue
                try:
                    self.register(target, path=current.source_path)
                    self.clear_cache(target)
                    reloaded.append(target)
                except PromptRegistryError as exc:
                    self._failed[target] = str(exc)
                    logger.error("Failed to reload '%s': %s", target, exc)
        return reloaded

    async def reload_async(self, name: Optional[str] = None) -> List[str]:
        """Async-safe wrapper around reload()."""
        async with self._async_lock:
            return await asyncio.to_thread(self.reload, name)

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear the rendered-output cache for one prompt, or for all prompts."""
        with self._lock:
            if name is None:
                self._render_cache.clear()
                self._render_cache_order.clear()
                return
            stale = [key for key in self._render_cache if key.startswith(f"{name}:")]
            for key in stale:
                self._render_cache.pop(key, None)
                if key in self._render_cache_order:
                    self._render_cache_order.remove(key)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health_check(self) -> HealthStatus:
        """Validate that every registered prompt version still compiles."""
        failed: List[str] = []
        total_versions = 0

        with self._lock:
            names = list(self._entries.keys())
            for name in names:
                entry = self._entries[name]
                total_versions += len(entry.versions)
                for version_key, prompt_version in entry.versions.items():
                    try:
                        self._compile(name, prompt_version)
                    except PromptRenderError:
                        failed.append(f"{name}@{version_key}")
            for name, reason in self._failed.items():
                failed.append(f"{name}: {reason}")

            status = HealthStatus(
                healthy=len(failed) == 0,
                total_prompts=len(names),
                total_versions=total_versions,
                failed_prompts=failed,
                checked_at=time.time(),
                details={"render_cache_size": len(self._render_cache)},
            )

        logger.info(
            "Health check: healthy=%s prompts=%d versions=%d failed=%d",
            status.healthy,
            status.total_prompts,
            status.total_versions,
            len(failed),
        )
        return status

    async def health_check_async(self) -> HealthStatus:
        """Async-safe wrapper around health_check()."""
        return await asyncio.to_thread(self.health_check)


_default_registry: Optional[PromptRegistry] = None
_default_registry_lock = threading.Lock()


def get_default_registry(template_dir: Optional[Union[str, Path]] = None) -> PromptRegistry:
    """Return a process-wide singleton PromptRegistry instance."""
    global _default_registry
    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = PromptRegistry(template_dir=template_dir)
        return _default_registry
