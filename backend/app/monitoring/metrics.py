from __future__ import annotations

import asyncio
import json
import math
import os
import resource
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class MetricsError(Exception):
    pass


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _label_key(labels: Optional[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


@dataclass
class HistogramData:
    count: int = 0
    sum: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    values: deque[float] = field(default_factory=lambda: deque(maxlen=2000))

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.values.append(value)

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": round(self.sum, 6),
            "mean": round(self.mean(), 6),
            "min": round(self.minimum, 6) if self.count else 0.0,
            "max": round(self.maximum, 6) if self.count else 0.0,
            "p50": round(self.percentile(50), 6),
            "p95": round(self.percentile(95), 6),
            "p99": round(self.percentile(99), 6),
        }


class Metrics:
    """Production metrics collector: counters, gauges, histograms, resource usage tracking."""

    def __init__(self, *, namespace: str = "app", max_histogram_samples: int = 2000) -> None:
        self.namespace = namespace
        self.max_histogram_samples = max_histogram_samples
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], HistogramData]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._op_count = 0

    def increment(self, name: str, value: float = 1.0, *, labels: Optional[dict[str, str]] = None) -> None:
        self._op_count += 1
        key = _label_key(labels)
        self._counters[name][key] += value

    def decrement(self, name: str, value: float = 1.0, *, labels: Optional[dict[str, str]] = None) -> None:
        self.increment(name, -value, labels=labels)

    def gauge_set(self, name: str, value: float, *, labels: Optional[dict[str, str]] = None) -> None:
        self._op_count += 1
        key = _label_key(labels)
        self._gauges[name][key] = value

    def observe(self, name: str, value: float, *, labels: Optional[dict[str, str]] = None) -> None:
        self._op_count += 1
        key = _label_key(labels)
        bucket = self._histograms[name]
        if key not in bucket:
            bucket[key] = HistogramData(values=deque(maxlen=self.max_histogram_samples))
        bucket[key].add(value)

    def record(
        self,
        name: str,
        value: float,
        *,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        if metric_type == MetricType.COUNTER:
            self.increment(name, value, labels=labels)
        elif metric_type == MetricType.GAUGE:
            self.gauge_set(name, value, labels=labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.observe(name, value, labels=labels)
        else:
            raise MetricsError(f"unknown metric type: {metric_type}")

    @contextmanager
    def time_execution(self, name: str, *, labels: Optional[dict[str, str]] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe(f"{name}.duration_seconds", elapsed, labels=labels)

    def record_execution_time(self, name: str, duration_seconds: float, *, labels: Optional[dict[str, str]] = None) -> None:
        self.observe(f"{name}.duration_seconds", duration_seconds, labels=labels)

    def record_token_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: Optional[str] = None,
    ) -> None:
        labels = {"model": model} if model else None
        self.increment("tokens.prompt", prompt_tokens, labels=labels)
        self.increment("tokens.completion", completion_tokens, labels=labels)
        self.increment("tokens.total", prompt_tokens + completion_tokens, labels=labels)

    def record_memory_usage(self) -> float:
        try:
            usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            usage_mb = usage_kb / 1024.0 if os.uname().sysname != "Darwin" else usage_kb / (1024.0 * 1024.0)
        except (AttributeError, OSError):
            usage_mb = 0.0
        self.gauge_set("process.memory_mb", usage_mb)
        return usage_mb

    def record_cpu_usage(self) -> float:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_seconds = usage.ru_utime + usage.ru_stime
        except (AttributeError, OSError):
            cpu_seconds = 0.0
        self.gauge_set("process.cpu_seconds", cpu_seconds)
        return cpu_seconds

    def get_counter(self, name: str, *, labels: Optional[dict[str, str]] = None) -> float:
        return self._counters.get(name, {}).get(_label_key(labels), 0.0)

    def get_gauge(self, name: str, *, labels: Optional[dict[str, str]] = None) -> Optional[float]:
        return self._gauges.get(name, {}).get(_label_key(labels))

    def get_histogram(self, name: str, *, labels: Optional[dict[str, str]] = None) -> Optional[dict[str, Any]]:
        data = self._histograms.get(name, {}).get(_label_key(labels))
        return data.to_dict() if data else None

    def reset(self, name: Optional[str] = None) -> None:
        if name is None:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            return
        self._counters.pop(name, None)
        self._gauges.pop(name, None)
        self._histograms.pop(name, None)

    def export(self, *, fmt: str = "json") -> str:
        snapshot = self._snapshot()
        if fmt == "json":
            return json.dumps(snapshot, default=str, sort_keys=True)
        if fmt == "prometheus":
            return self._to_prometheus(snapshot)
        raise MetricsError(f"unsupported export format: {fmt}")

    def _snapshot(self) -> dict[str, Any]:
        counters = {
            name: [{"labels": dict(k), "value": v} for k, v in series.items()]
            for name, series in self._counters.items()
        }
        gauges = {
            name: [{"labels": dict(k), "value": v} for k, v in series.items()]
            for name, series in self._gauges.items()
        }
        histograms = {
            name: [{"labels": dict(k), **data.to_dict()} for k, data in series.items()]
            for name, series in self._histograms.items()
        }
        return {
            "namespace": self.namespace,
            "timestamp": time.time(),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def _to_prometheus(self, snapshot: dict[str, Any]) -> str:
        lines: list[str] = []

        def fmt_labels(labels: dict[str, str]) -> str:
            if not labels:
                return ""
            inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return "{" + inner + "}"

        for name, series in snapshot["counters"].items():
            metric_name = f"{self.namespace}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} counter")
            for entry in series:
                lines.append(f"{metric_name}{fmt_labels(entry['labels'])} {entry['value']}")

        for name, series in snapshot["gauges"].items():
            metric_name = f"{self.namespace}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            for entry in series:
                lines.append(f"{metric_name}{fmt_labels(entry['labels'])} {entry['value']}")

        for name, series in snapshot["histograms"].items():
            metric_name = f"{self.namespace}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} summary")
            for entry in series:
                base_labels = entry["labels"]
                for q in ("p50", "p95", "p99"):
                    q_labels = {**base_labels, "quantile": q}
                    lines.append(f"{metric_name}{fmt_labels(q_labels)} {entry[q]}")
                lines.append(f"{metric_name}_sum{fmt_labels(base_labels)} {entry['sum']}")
                lines.append(f"{metric_name}_count{fmt_labels(base_labels)} {entry['count']}")

        return "\n".join(lines) + "\n"

    async def increment_async(self, name: str, value: float = 1.0, *, labels: Optional[dict[str, str]] = None) -> None:
        async with self._lock:
            self.increment(name, value, labels=labels)

    async def observe_async(self, name: str, value: float, *, labels: Optional[dict[str, str]] = None) -> None:
        async with self._lock:
            self.observe(name, value, labels=labels)

    async def record_async(self, name: str, value: float, **kwargs: Any) -> None:
        async with self._lock:
            self.record(name, value, **kwargs)

    async def export_async(self, *, fmt: str = "json") -> str:
        async with self._lock:
            return self.export(fmt=fmt)

    def health_check(self) -> HealthStatus:
        try:
            self.gauge_set("__health_check__", 1.0)
            value = self.get_gauge("__health_check__")
            self._gauges.pop("__health_check__", None)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "namespace": self.namespace,
                "counter_metric_count": len(self._counters),
                "gauge_metric_count": len(self._gauges),
                "histogram_metric_count": len(self._histograms),
                "operation_count": self._op_count,
            }
            return HealthStatus(healthy=value == 1.0, component="metrics", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="metrics", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "Metrics",
    "MetricType",
    "MetricsError",
    "HistogramData",
    "HealthStatus",
]
