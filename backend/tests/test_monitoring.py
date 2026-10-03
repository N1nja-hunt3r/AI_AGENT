"""
test_monitoring.py
Tests for the monitoring subsystem: structured logging, alert thresholds,
dashboard aggregation, and metrics collection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest


@dataclass
class LogEntry:
    level: str
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


class MockLogger:
    def __init__(self) -> None:
        self.entries: List[LogEntry] = []

    def log(self, level: str, message: str, **context: Any) -> LogEntry:
        entry = LogEntry(level=level, message=message, context=context)
        self.entries.append(entry)
        return entry

    def info(self, message: str, **context: Any) -> LogEntry:
        return self.log("info", message, **context)

    def error(self, message: str, **context: Any) -> LogEntry:
        return self.log("error", message, **context)

    def filter(self, level: str) -> List[LogEntry]:
        return [entry for entry in self.entries if entry.level == level]


class MockAlertManager:
    def __init__(self) -> None:
        self.thresholds: Dict[str, float] = {}
        self.fired: List[Dict[str, Any]] = []

    def set_threshold(self, metric: str, max_value: float) -> None:
        self.thresholds[metric] = max_value

    def evaluate(self, metric: str, value: float) -> bool:
        threshold = self.thresholds.get(metric)
        if threshold is not None and value > threshold:
            self.fired.append({"metric": metric, "value": value, "threshold": threshold})
            return True
        return False


class MockMetrics:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timings: Dict[str, List[float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def record_timing(self, name: str, seconds: float) -> None:
        self.timings.setdefault(name, []).append(seconds)

    def average_timing(self, name: str) -> float:
        values = self.timings.get(name, [])
        return sum(values) / len(values) if values else 0.0


class MockDashboard:
    def __init__(self, metrics: MockMetrics, alerts: MockAlertManager) -> None:
        self.metrics = metrics
        self.alerts = alerts

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.metrics.counters),
            "gauges": dict(self.metrics.gauges),
            "active_alerts": list(self.alerts.fired),
        }


@pytest.fixture
def logger() -> MockLogger:
    return MockLogger()


@pytest.fixture
def alerts() -> MockAlertManager:
    return MockAlertManager()


@pytest.fixture
def metrics() -> MockMetrics:
    return MockMetrics()


@pytest.fixture
def dashboard(metrics: MockMetrics, alerts: MockAlertManager) -> MockDashboard:
    return MockDashboard(metrics, alerts)


def test_logger_records_info_entry(logger: MockLogger) -> None:
    entry = logger.info("task completed", task_id="T1")
    assert entry in logger.entries
    assert entry.level == "info"
    assert entry.context["task_id"] == "T1"


def test_logger_filters_by_level(logger: MockLogger) -> None:
    logger.info("started")
    logger.error("failed to connect")
    errors = logger.filter("error")
    assert len(errors) == 1
    assert errors[0].message == "failed to connect"


def test_alert_fires_when_threshold_exceeded(alerts: MockAlertManager) -> None:
    alerts.set_threshold("latency_ms", 500)
    fired = alerts.evaluate("latency_ms", 800)
    assert fired is True
    assert alerts.fired[0]["metric"] == "latency_ms"


def test_alert_does_not_fire_within_threshold(alerts: MockAlertManager) -> None:
    alerts.set_threshold("latency_ms", 500)
    assert alerts.evaluate("latency_ms", 200) is False
    assert alerts.fired == []


def test_alert_with_no_threshold_never_fires(alerts: MockAlertManager) -> None:
    assert alerts.evaluate("unconfigured_metric", 10_000) is False


def test_metrics_increment_counter(metrics: MockMetrics) -> None:
    metrics.increment("requests_total")
    metrics.increment("requests_total", amount=4)
    assert metrics.counters["requests_total"] == 5


def test_metrics_set_gauge(metrics: MockMetrics) -> None:
    metrics.set_gauge("queue_depth", 12.0)
    assert metrics.gauges["queue_depth"] == 12.0


def test_metrics_average_timing(metrics: MockMetrics) -> None:
    metrics.record_timing("execute_duration", 1.0)
    metrics.record_timing("execute_duration", 3.0)
    assert metrics.average_timing("execute_duration") == 2.0


def test_metrics_average_timing_with_no_data_is_zero(metrics: MockMetrics) -> None:
    assert metrics.average_timing("never_recorded") == 0.0


def test_dashboard_snapshot_includes_counters_gauges_and_alerts(
    dashboard: MockDashboard, metrics: MockMetrics, alerts: MockAlertManager
) -> None:
    metrics.increment("tasks_completed")
    metrics.set_gauge("active_agents", 3.0)
    alerts.set_threshold("error_rate", 0.1)
    alerts.evaluate("error_rate", 0.5)

    snapshot = dashboard.snapshot()
    assert snapshot["counters"]["tasks_completed"] == 1
    assert snapshot["gauges"]["active_agents"] == 3.0
    assert snapshot["active_alerts"]


@pytest.mark.asyncio
async def test_monitoring_components_compose_in_async_workflow(
    logger: MockLogger, metrics: MockMetrics, alerts: MockAlertManager
) -> None:
    async def run_task() -> None:
        logger.info("task started")
        metrics.increment("tasks_started")
        alerts.set_threshold("tasks_started", 0)
        alerts.evaluate("tasks_started", metrics.counters["tasks_started"])

    await run_task()
    assert metrics.counters["tasks_started"] == 1
    assert alerts.fired