"""
registry.py
Test registration, discovery, and health utilities for the AI Operating
System test suite.  Integrates with pytest collection via a custom marker
so that registered tests can be filtered from the CLI.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import pytest

T = TypeVar("T")

logger = logging.getLogger("test_registry")

_THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------- #


@contextmanager
def _on_sys_path(directory: Path) -> Iterator[None]:
    """Temporarily add *directory* to ``sys.path`` if not already present."""
    resolved = directory.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
        try:
            yield
        finally:
            sys.path.remove(str(resolved))
    else:
        yield


# ---------------------------------------------------------------------- #
# Exceptions
# ---------------------------------------------------------------------- #


class TestRegistryError(Exception):
    """Base exception for test registry errors."""


class TestSuiteNotFoundError(TestRegistryError):
    """Raised when a requested test suite is not registered."""


# ---------------------------------------------------------------------- #
# Per-function registration (decorator-based)
# ---------------------------------------------------------------------- #

_registry: Dict[str, "TestEntry"] = {}
"""Global in-process registry of all explicitly registered test functions."""


@dataclass
class TestEntry:
    """Metadata for a single registered test case."""

    test_id: str
    description: str
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    timeout_seconds: Optional[float] = None
    module: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.test_id)


def register(
    description: str = "",
    tags: Optional[Sequence[str]] = None,
    owner: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a test function in the global registry.

    Usage::

        @registry.register(description="Validates JWT lifecycle", tags=["security", "auth"])
        def test_jwt_flow(mock_security):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        test_id = f"{func.__module__}.{func.__qualname__}"
        _registry[test_id] = TestEntry(
            test_id=test_id,
            description=description or (func.__doc__ or "").strip(),
            tags=list(tags) if tags else [],
            owner=owner,
            timeout_seconds=timeout_seconds,
            module=func.__module__,
        )
        return func
    return decorator


def unregister(test_id: str) -> bool:
    """Remove a test from the global registry.  Returns ``True`` if present."""
    return _registry.pop(test_id, None) is not None


def clear() -> None:
    """Remove all registered entries."""
    _registry.clear()


def registered_tests() -> List[TestEntry]:
    """Return a snapshot of all currently registered test entries."""
    return sorted(_registry.values(), key=lambda e: e.test_id)


def registered_test_ids() -> List[str]:
    """Return just the IDs of registered tests."""
    return sorted(_registry.keys())


# ---------------------------------------------------------------------- #
# Function-level discovery
# ---------------------------------------------------------------------- #


@dataclass
class DiscoveredTest:
    """A test discovered via module scanning (not necessarily registered)."""

    module: str
    name: str
    func: Callable[..., Any]
    is_async: bool
    markers: List[str] = field(default_factory=list)


def discover_in_module(module_name: str, search_path: Optional[Path] = None) -> List[DiscoveredTest]:
    """Return all callables prefixed with ``test_`` in *module_name*.

    If *search_path* is provided, it is temporarily added to ``sys.path``
    so that the module can be imported even if it lives outside the normal
    import path.
    """
    if search_path is not None:
        with _on_sys_path(search_path):
            module = importlib.import_module(module_name)
    else:
        module = importlib.import_module(module_name)
    tests: List[DiscoveredTest] = []
    for name, obj in inspect.getmembers(module, inspect.iscoroutinefunction):
        if name.startswith("test_"):
            markers = _extract_markers(obj)
            tests.append(DiscoveredTest(module=module_name, name=name, func=obj, is_async=True, markers=markers))
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_") and not inspect.iscoroutinefunction(obj):
            markers = _extract_markers(obj)
            tests.append(DiscoveredTest(module=module_name, name=name, func=obj, is_async=False, markers=markers))
    return tests


def discover_package(package_name: str) -> Dict[str, List[DiscoveredTest]]:
    """Recursively discover tests in *package_name* and its subpackages.

    Returns a mapping of module name to its discovered tests.  The
    package's parent directory is temporarily added to ``sys.path`` to
    ensure child modules can be imported.
    """
    package = importlib.import_module(package_name)
    pkg_dir = Path(package.__file__).resolve().parent if package.__file__ else Path.cwd()
    parent_dir = pkg_dir.parent
    result: Dict[str, List[DiscoveredTest]] = {}

    def _walk(pkg_root: Path, prefix: str) -> None:
        for mod_info in pkgutil.walk_packages(path=[str(pkg_root)], prefix=prefix, onerror=lambda _: None):
            if mod_info.ispkg:
                continue
            if mod_info.name.startswith("test_") or "_test" in mod_info.name:
                try:
                    tests = discover_in_module(mod_info.name, search_path=parent_dir)
                    if tests:
                        result[mod_info.name] = tests
                except Exception:
                    continue

    with _on_sys_path(parent_dir):
        _walk(pkg_dir, prefix=package_name + ".")
    return result


def discover_in_directory(directory: str, pattern: str = "test_*.py") -> Dict[str, List[DiscoveredTest]]:
    """Scan *directory* for test files matching *pattern* and discover tests.

    *directory* is temporarily added to ``sys.path`` so that each matching
    file can be imported as a top-level module.
    """
    result: Dict[str, List[DiscoveredTest]] = {}
    root = Path(directory).resolve()
    for py_file in sorted(root.rglob(pattern)):
        module_name = str(py_file.relative_to(root).with_suffix("")).replace("\\", ".").replace("/", ".")
        try:
            tests = discover_in_module(module_name, search_path=root)
            if tests:
                result[module_name] = tests
        except Exception:
            continue
    return result


def _extract_markers(func: Callable[..., Any]) -> List[str]:
    """Return marker names attached to *func* via ``pytest.mark``."""
    marks: List[str] = []
    if hasattr(func, "pytestmark"):
        if isinstance(func.pytestmark, list):
            marks = [m.name for m in func.pytestmark if hasattr(m, "name")]
    return marks


# ---------------------------------------------------------------------- #
# Pytest integration helpers
# ---------------------------------------------------------------------- #


def pytest_register_marker() -> pytest.MarkDecorator:
    """Return a pytest marker that marks a test as **registered**.

    Use from a conftest.py::

        register_marker = registry.pytest_register_marker()

        @register_marker
        def test_something():
            ...
    """
    return pytest.mark.registered()


def pytest_collect_tests() -> List[TestEntry]:
    """Return registered tests for custom pytest collection plugins.

    This is intended to be called from a ``pytest_collect_file`` or
    ``pytest_pycollect_makeitem`` hook implementation.
    """
    return registered_tests()


# ---------------------------------------------------------------------- #
# In-process runner (for decorated tests)
# ---------------------------------------------------------------------- #


@dataclass
class TestSuiteResult:
    """Result of running a set of tests through the in-process registry runner."""

    total: int
    passed: int
    failed: int
    duration_seconds: float
    failures: List[Tuple[str, str]] = field(default_factory=list)


async def run_registered(
    fixtures: Optional[Dict[str, Any]] = None,
    filter_tags: Optional[Sequence[str]] = None,
) -> TestSuiteResult:
    """Execute all (or filtered) registered tests in-process.

    This is **not** a replacement for ``pytest`` itself; it is useful for
    health checks, pre-commit gates, or embedding in other tooling where
    a full pytest invocation is undesirable.

    .. caution::
        Tests are called as plain coroutines / functions.  Pytest fixtures
        are **not** resolved.  Provide pre-built objects via *fixtures* or
        rely on tests that do not require fixtures.
    """
    entries = registered_tests()
    if filter_tags:
        tag_set = set(filter_tags)
        entries = [e for e in entries if tag_set & set(e.tags)]

    start = time.monotonic()
    passed = 0
    failed = 0
    failures: List[Tuple[str, str]] = []
    _fixtures = fixtures or {}

    for entry in entries:
        *mod_parts, func_name = entry.test_id.rsplit(".", 1)
        mod_name = ".".join(mod_parts) if mod_parts else None
        if mod_name is None:
            failed += 1
            failures.append((entry.test_id, "cannot determine module from test_id"))
            continue

        try:
            with _on_sys_path(_THIS_DIR):
                module = importlib.import_module(mod_name)
            func = getattr(module, func_name, None)
            if func is None:
                failed += 1
                failures.append((entry.test_id, f"function {func_name} not found in {mod_name}"))
                continue

            if inspect.iscoroutinefunction(func):
                await func(**_fixtures)
            else:
                func(**_fixtures)
            passed += 1
        except SystemExit:
            failed += 1
            failures.append((entry.test_id, "raised SystemExit"))
        except Exception as exc:
            failed += 1
            failures.append((entry.test_id, str(exc)))

    duration = time.monotonic() - start
    return TestSuiteResult(
        total=len(entries),
        passed=passed,
        failed=failed,
        duration_seconds=duration,
        failures=failures,
    )


# ---------------------------------------------------------------------- #
# Health (function-level registry)
# ---------------------------------------------------------------------- #


@dataclass
class RegistryHealth:
    """Health snapshot of the function-level test registry."""

    healthy: bool
    registered_count: int
    discovered_count: Optional[int] = None
    modules_scanned: Optional[List[str]] = None
    issues: List[str] = field(default_factory=list)


def check_registered_health(
    package: Optional[str] = None,
    directory: Optional[str] = None,
) -> RegistryHealth:
    """Perform a health check on the function-level registry and optionally discover tests.

    Reports:
    - Number of registered tests.
    - Number of discovered tests (if *package* or *directory* is provided).
    - Any module import errors or naming issues.
    """
    issues: List[str] = []
    reg_count = len(_registry)

    discovered: Dict[str, List[DiscoveredTest]] = {}
    if package:
        try:
            discovered = discover_package(package)
        except Exception as exc:
            issues.append(f"failed to discover package '{package}': {exc}")
    if directory:
        try:
            discovered = discover_in_directory(directory)
        except Exception as exc:
            issues.append(f"failed to discover directory '{directory}': {exc}")

    discovered_count = sum(len(tests) for tests in discovered.values()) if discovered else None
    modules_scanned = list(discovered.keys()) if discovered else None

    for mod, tests in discovered.items():
        for test in tests:
            test_id = f"{mod}.{test.name}"
            if test_id in _registry:
                stored = _registry[test_id]
                if stored.description != (test.func.__doc__ or "").strip():
                    issues.append(f"description mismatch for {test_id}")

    healthy = len(issues) == 0
    return RegistryHealth(
        healthy=healthy,
        registered_count=reg_count,
        discovered_count=discovered_count,
        modules_scanned=modules_scanned,
        issues=issues,
    )


def list_registered_summary() -> str:
    """Return a human-readable summary of all registered tests."""
    lines: List[str] = [f"Registered tests: {len(_registry)}"]
    for entry in registered_tests():
        tags = f" [{', '.join(entry.tags)}]" if entry.tags else ""
        owner = f" @{entry.owner}" if entry.owner else ""
        lines.append(f"  {entry.test_id}{tags}{owner}")
        if entry.description:
            lines.append(f"    {entry.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Suite-level class (thread-safe, file-based, subprocess execution)
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class TestSuite:
    """A registered pytest test suite (typically a single test file)."""

    name: str
    path: str
    markers: Sequence[str] = field(default_factory=tuple)
    registered_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TestRunResult:
    """Outcome of running a single test suite via pytest subprocess."""

    suite: str
    exit_code: int
    passed: bool
    duration_seconds: float
    stdout: str
    stderr: str
    ran_at: float


@dataclass(frozen=True)
class HealthStatus:
    """Result of a registry-wide health check on suite-level registry."""

    healthy: bool
    total_suites: int
    discoverable_suites: int
    missing_suites: List[str]
    checked_at: float


class TestRegistry:
    """
    Thread-safe registry for discovering, registering, and executing
    pytest test suites (files), with cached run results and health reporting.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        self._lock = threading.RLock()
        self._suites: Dict[str, TestSuite] = {}
        self._last_results: Dict[str, TestRunResult] = {}
        self._root_dir = Path(root_dir) if root_dir else Path.cwd()

    # ------------------------------------------------------------------ #
    # Registration / discovery
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        path: Union[str, Path],
        markers: Optional[Sequence[str]] = None,
    ) -> TestSuite:
        """Register a single test suite by name and file path."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = (self._root_dir / file_path).resolve()

        suite = TestSuite(name=name, path=str(file_path), markers=tuple(markers or ()))
        with self._lock:
            self._suites[name] = suite
        logger.info("Registered test suite '%s' at '%s'", name, file_path)
        return suite

    def discover(self, pattern: str = "test_*.py") -> List[str]:
        """Discover and register every test file under root_dir matching `pattern`.

        The paths yielded by ``glob`` already include *root_dir*, so they
        are stored directly without joining again to avoid duplication.
        """
        discovered: List[str] = []
        with self._lock:
            for file_path in sorted(self._root_dir.glob(pattern)):
                name = file_path.stem
                resolved = file_path.resolve()
                if name not in self._suites:
                    self._suites[name] = TestSuite(name=name, path=str(resolved), markers=())
                discovered.append(name)
        logger.info("Discovered %d test suite(s) matching '%s'", len(discovered), pattern)
        return discovered

    def unregister(self, name: str) -> bool:
        """Remove a suite from the registry."""
        with self._lock:
            existed = self._suites.pop(name, None) is not None
            self._last_results.pop(name, None)
            return existed

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> TestSuite:
        with self._lock:
            suite = self._suites.get(name)
            if suite is None:
                raise TestSuiteNotFoundError(f"Test suite '{name}' is not registered")
            return suite

    def list_suites(self) -> List[str]:
        with self._lock:
            return sorted(self._suites.keys())

    def last_result(self, name: str) -> Optional[TestRunResult]:
        with self._lock:
            return self._last_results.get(name)

    # ------------------------------------------------------------------ #
    # Execution (pytest subprocess)
    # ------------------------------------------------------------------ #

    def run(self, name: str, extra_args: Optional[Sequence[str]] = None) -> TestRunResult:
        """Run a single registered suite in a pytest subprocess."""
        suite = self.get(name)
        args = [sys.executable, "-m", "pytest", suite.path, "-q"]
        for marker in suite.markers:
            args += ["-m", marker]
        if extra_args:
            args += list(extra_args)

        started = time.time()
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(self._root_dir),
        )
        duration = time.time() - started

        result = TestRunResult(
            suite=name,
            exit_code=completed.returncode,
            passed=completed.returncode == 0,
            duration_seconds=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
            ran_at=time.time(),
        )
        with self._lock:
            self._last_results[name] = result
        logger.info(
            "Ran test suite '%s': passed=%s duration=%.2fs exit_code=%d",
            name, result.passed, duration, result.exit_code,
        )
        return result

    def run_all(self, extra_args: Optional[Sequence[str]] = None) -> List[TestRunResult]:
        """Run every registered suite sequentially and return all results."""
        with self._lock:
            names = list(self._suites.keys())
        return [self.run(name, extra_args=extra_args) for name in names]

    def run_matching(self, pattern: str, extra_args: Optional[Sequence[str]] = None) -> List[TestRunResult]:
        """Run only suites whose name matches a simple glob-style pattern."""
        import fnmatch

        with self._lock:
            names = [name for name in self._suites if fnmatch.fnmatch(name, pattern)]
        return [self.run(name, extra_args=extra_args) for name in names]

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health_check(self) -> HealthStatus:
        """Verify every registered suite's source file still exists on disk."""
        missing: List[str] = []
        with self._lock:
            suites = list(self._suites.values())
            for suite in suites:
                if not Path(suite.path).is_file():
                    missing.append(suite.name)

            status = HealthStatus(
                healthy=len(missing) == 0,
                total_suites=len(suites),
                discoverable_suites=len(suites) - len(missing),
                missing_suites=missing,
                checked_at=time.time(),
            )
        logger.info(
            "Test registry health check: healthy=%s total=%d missing=%d",
            status.healthy, status.total_suites, len(missing),
        )
        return status

    def summary(self) -> Dict[str, int]:
        """Aggregate pass/fail counts across the most recent run of each suite."""
        with self._lock:
            results = list(self._last_results.values())
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        return {"total": len(results), "passed": passed, "failed": failed}


# ---------------------------------------------------------------------- #
# Singleton
# ---------------------------------------------------------------------- #

_default_test_registry: Optional[TestRegistry] = None
_default_test_registry_lock = threading.Lock()


def get_default_test_registry(root_dir: Optional[Union[str, Path]] = None) -> TestRegistry:
    """Return a process-wide singleton TestRegistry instance."""
    global _default_test_registry  # noqa: PLW0603
    with _default_test_registry_lock:
        if _default_test_registry is None:
            _default_test_registry = TestRegistry(root_dir=root_dir)
        return _default_test_registry
