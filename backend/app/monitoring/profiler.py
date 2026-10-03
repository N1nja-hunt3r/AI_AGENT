from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import resource
import time
import tracemalloc
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional, TypeVar


class ProfileKind(str, Enum):
    EXECUTION = "execution"
    MEMORY = "memory"
    LATENCY = "latency"
    TOKEN = "token"


class ProfilerError(Exception):
    pass


@dataclass
class ProfileResult:
    profile_id: str
    kind: ProfileKind
    name: str
    started_at: float
    ended_at: float
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "kind": self.kind.value,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 6),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


_T = TypeVar("_T")


class Profiler:
    """Execution, memory, latency, and token usage profiling with cProfile and tracemalloc."""

    def __init__(self, *, max_results: int = 5000) -> None:
        self.max_results = max_results
        self._results: list[ProfileResult] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._profile_count = 0
        self._measure_count = 0

    def _store(self, result: ProfileResult) -> ProfileResult:
        self._results.append(result)
        if len(self._results) > self.max_results:
            self._results = self._results[-self.max_results :]
        return result

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        kind: ProfileKind = ProfileKind.EXECUTION,
        enable_cprofile: bool = False,
        enable_memory_trace: bool = False,
    ) -> Iterator[dict[str, Any]]:
        self._profile_count += 1
        profile_id = str(uuid.uuid4())
        started_at = time.time()
        cprofiler: Optional[cProfile.Profile] = None
        mem_was_tracing = tracemalloc.is_tracing()

        if enable_cprofile:
            cprofiler = cProfile.Profile()
            cprofiler.enable()

        if enable_memory_trace and not mem_was_tracing:
            tracemalloc.start()

        cpu_start = self._cpu_time()
        context: dict[str, Any] = {}

        try:
            yield context
        finally:
            ended_at = time.time()
            cpu_end = self._cpu_time()
            metadata: dict[str, Any] = dict(context)
            metadata["cpu_seconds"] = round(cpu_end - cpu_start, 6)

            if cprofiler is not None:
                cprofiler.disable()
                stream = io.StringIO()
                stats = pstats.Stats(cprofiler, stream=stream).sort_stats("cumulative")
                stats.print_stats(15)
                metadata["cprofile_top15"] = stream.getvalue()

            if enable_memory_trace:
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    metadata["memory_current_bytes"] = current
                    metadata["memory_peak_bytes"] = peak
                    if not mem_was_tracing:
                        tracemalloc.stop()

            result = ProfileResult(
                profile_id=profile_id, kind=kind, name=name,
                started_at=started_at, ended_at=ended_at,
                duration_seconds=ended_at - started_at, metadata=metadata,
            )
            self._store(result)

    @staticmethod
    def _cpu_time() -> float:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_utime + usage.ru_stime
        except (AttributeError, OSError):
            return 0.0

    def measure(
        self,
        func: Callable[..., _T],
        *args: Any,
        name: Optional[str] = None,
        kind: ProfileKind = ProfileKind.LATENCY,
        **kwargs: Any,
    ) -> tuple[_T, ProfileResult]:
        self._measure_count += 1
        label: str = name if name is not None else getattr(func, "__name__", "callable")
        profile_id = str(uuid.uuid4())
        started_at = time.time()
        cpu_start = self._cpu_time()

        try:
            tracemalloc_was_active = tracemalloc.is_tracing()
            if not tracemalloc_was_active:
                tracemalloc.start()
            result_value = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            if not tracemalloc_was_active:
                tracemalloc.stop()
        except Exception:
            ended_at = time.time()
            cpu_end = self._cpu_time()
            error_result = ProfileResult(
                profile_id=profile_id, kind=kind, name=label,
                started_at=started_at, ended_at=ended_at,
                duration_seconds=ended_at - started_at,
                metadata={"cpu_seconds": round(cpu_end - cpu_start, 6), "error": True},
            )
            self._store(error_result)
            raise

        ended_at = time.time()
        cpu_end = self._cpu_time()
        profile_result = ProfileResult(
            profile_id=profile_id, kind=kind, name=label,
            started_at=started_at, ended_at=ended_at,
            duration_seconds=ended_at - started_at,
            metadata={
                "cpu_seconds": round(cpu_end - cpu_start, 6),
                "memory_current_bytes": current,
                "memory_peak_bytes": peak,
            },
        )
        self._store(profile_result)
        return result_value, profile_result

    def record_token_profile(
        self,
        name: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        model: Optional[str] = None,
    ) -> ProfileResult:
        total = prompt_tokens + completion_tokens
        tokens_per_second = total / duration_seconds if duration_seconds > 0 else 0.0
        now = time.time()
        result = ProfileResult(
            profile_id=str(uuid.uuid4()), kind=ProfileKind.TOKEN, name=name,
            started_at=now - duration_seconds, ended_at=now, duration_seconds=duration_seconds,
            metadata={
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": total, "tokens_per_second": round(tokens_per_second, 4),
                "model": model,
            },
        )
        return self._store(result)

    def get_results(
        self,
        *,
        kind: Optional[ProfileKind] = None,
        name: Optional[str] = None,
        limit: int = 200,
    ) -> list[ProfileResult]:
        items = self._results
        if kind is not None:
            items = [r for r in items if r.kind == kind]
        if name is not None:
            items = [r for r in items if r.name == name]
        return items[-limit:]

    def summary(self, *, kind: Optional[ProfileKind] = None, name: Optional[str] = None) -> dict[str, Any]:
        results = self.get_results(kind=kind, name=name, limit=self.max_results)
        if not results:
            return {"count": 0}
        durations = [r.duration_seconds for r in results]
        return {
            "count": len(results),
            "total_duration_seconds": round(sum(durations), 6),
            "mean_duration_seconds": round(sum(durations) / len(durations), 6),
            "min_duration_seconds": round(min(durations), 6),
            "max_duration_seconds": round(max(durations), 6),
        }

    async def measure_async(
        self,
        func: Callable[..., _T],
        *args: Any,
        name: Optional[str] = None,
        kind: ProfileKind = ProfileKind.LATENCY,
        **kwargs: Any,
    ) -> tuple[_T, ProfileResult]:
        async with self._lock:
            return await asyncio.to_thread(
                lambda: self.measure(func, *args, name=name, kind=kind, **kwargs)
            )

    def health_check(self) -> HealthStatus:
        try:
            with self.profile("__health_check__", kind=ProfileKind.EXECUTION) as ctx:
                ctx["probe"] = True
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "stored_result_count": len(self._results),
                "profile_count": self._profile_count,
                "measure_count": self._measure_count,
                "tracemalloc_active": tracemalloc.is_tracing(),
            }
            return HealthStatus(healthy=True, component="profiler", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="profiler", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Profiler",
    "ProfileResult",
    "ProfileKind",
    "ProfilerError",
    "HealthStatus",
]
