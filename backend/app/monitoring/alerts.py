from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class AlertCategory(str, Enum):
    THRESHOLD = "threshold"
    ERROR = "error"
    SECURITY = "security"
    RESOURCE = "resource"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class AlertsError(Exception):
    pass


class AlertNotFoundError(AlertsError):
    pass


NotifyFn = Callable[["Alert"], Union[None, Awaitable[None]]]


@dataclass
class Alert:
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    source: str
    state: AlertState = AlertState.ACTIVE
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    resolved_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "state": self.state.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_by": self.resolved_by,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThresholdRule:
    name: str
    metric_name: str
    threshold: float
    comparison: str = "gt"
    severity: AlertSeverity = AlertSeverity.WARNING

    def evaluate(self, value: float) -> bool:
        ops = {
            "gt": lambda a, b: a > b,
            "gte": lambda a, b: a >= b,
            "lt": lambda a, b: a < b,
            "lte": lambda a, b: a <= b,
            "eq": lambda a, b: a == b,
        }
        if self.comparison not in ops:
            raise AlertsError(f"unsupported comparison operator '{self.comparison}'")
        return ops[self.comparison](value, self.threshold)


class AlertManager:
    """Threshold/error/security/resource alerting with notification dispatch and resolution tracking."""

    def __init__(self, *, max_history: int = 10_000) -> None:
        self.max_history = max_history
        self._alerts: dict[str, Alert] = {}
        self._history: list[Alert] = []
        self._threshold_rules: dict[str, ThresholdRule] = {}
        self._notifiers: list[NotifyFn] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._trigger_count = 0
        self._resolve_count = 0
        self._notify_count = 0
        self._notify_failures = 0

    def add_notifier(self, notifier: NotifyFn) -> None:
        self._notifiers.append(notifier)

    def remove_notifier(self, notifier: NotifyFn) -> bool:
        if notifier in self._notifiers:
            self._notifiers.remove(notifier)
            return True
        return False

    def register_threshold_rule(self, rule: ThresholdRule) -> None:
        self._threshold_rules[rule.name] = rule

    def evaluate_threshold(self, rule_name: str, value: float, *, source: str = "metrics") -> Optional[Alert]:
        rule = self._threshold_rules.get(rule_name)
        if rule is None:
            raise AlertsError(f"no threshold rule registered under name '{rule_name}'")
        if rule.evaluate(value):
            return self.trigger(
                category=AlertCategory.THRESHOLD,
                severity=rule.severity,
                title=f"Threshold breached: {rule.name}",
                message=f"metric '{rule.metric_name}' value {value} {rule.comparison} threshold {rule.threshold}",
                source=source,
                metadata={"metric_name": rule.metric_name, "value": value, "threshold": rule.threshold},
            )
        return None

    def trigger(
        self,
        *,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Alert:
        self._trigger_count += 1
        alert = Alert(
            alert_id=str(uuid.uuid4()), category=category, severity=severity,
            title=title, message=message, source=source, metadata=metadata or {},
        )
        self._alerts[alert.alert_id] = alert
        self._archive(alert)
        return alert

    def trigger_error_alert(self, message: str, *, source: str, severity: AlertSeverity = AlertSeverity.ERROR, metadata: Optional[dict[str, Any]] = None) -> Alert:
        return self.trigger(category=AlertCategory.ERROR, severity=severity, title="Error detected", message=message, source=source, metadata=metadata)

    def trigger_security_alert(self, message: str, *, source: str, severity: AlertSeverity = AlertSeverity.CRITICAL, metadata: Optional[dict[str, Any]] = None) -> Alert:
        return self.trigger(category=AlertCategory.SECURITY, severity=severity, title="Security event", message=message, source=source, metadata=metadata)

    def trigger_resource_alert(self, message: str, *, source: str, severity: AlertSeverity = AlertSeverity.WARNING, metadata: Optional[dict[str, Any]] = None) -> Alert:
        return self.trigger(category=AlertCategory.RESOURCE, severity=severity, title="Resource alert", message=message, source=source, metadata=metadata)

    def _archive(self, alert: Alert) -> None:
        self._history.append(alert)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def notify(self, alert_id: str) -> int:
        alert = self._alerts.get(alert_id) or next((a for a in self._history if a.alert_id == alert_id), None)
        if alert is None:
            raise AlertNotFoundError(f"no alert with id {alert_id}")
        delivered = 0
        for notifier in self._notifiers:
            self._notify_count += 1
            try:
                result = notifier(alert)
                if asyncio.iscoroutine(result):
                    result.close()
                delivered += 1
            except Exception:
                self._notify_failures += 1
        return delivered

    async def notify_async(self, alert_id: str) -> int:
        alert = self._alerts.get(alert_id) or next((a for a in self._history if a.alert_id == alert_id), None)
        if alert is None:
            raise AlertNotFoundError(f"no alert with id {alert_id}")
        delivered = 0
        for notifier in self._notifiers:
            self._notify_count += 1
            try:
                result = notifier(alert)
                if asyncio.iscoroutine(result):
                    await result
                delivered += 1
            except Exception:
                self._notify_failures += 1
        return delivered

    def acknowledge(self, alert_id: str, *, acknowledged_by: str) -> Alert:
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise AlertNotFoundError(f"no active alert with id {alert_id}")
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.metadata["acknowledged_by"] = acknowledged_by
        return alert

    def resolve(self, alert_id: str, *, resolved_by: str, resolution_note: Optional[str] = None) -> Alert:
        self._resolve_count += 1
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise AlertNotFoundError(f"no active alert with id {alert_id}")
        alert.state = AlertState.RESOLVED
        alert.resolved_at = time.time()
        alert.resolved_by = resolved_by
        if resolution_note:
            alert.metadata["resolution_note"] = resolution_note
        self._alerts.pop(alert_id, None)
        return alert

    def list_active(
        self,
        *,
        category: Optional[AlertCategory] = None,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
    ) -> list[Alert]:
        items = list(self._alerts.values())
        if category is not None:
            items = [a for a in items if a.category == category]
        if severity is not None:
            items = [a for a in items if a.severity == severity]
        if source is not None:
            items = [a for a in items if a.source == source]
        return sorted(items, key=lambda a: a.created_at)

    def get(self, alert_id: str) -> Optional[Alert]:
        if alert_id in self._alerts:
            return self._alerts[alert_id]
        return next((a for a in self._history if a.alert_id == alert_id), None)

    def history(self, limit: int = 200, *, state: Optional[AlertState] = None) -> list[Alert]:
        items = self._history if state is None else [a for a in self._history if a.state == state]
        return items[-limit:]

    async def trigger_async(self, **kwargs: Any) -> Alert:
        async with self._lock:
            return self.trigger(**kwargs)

    async def resolve_async(self, alert_id: str, **kwargs: Any) -> Alert:
        async with self._lock:
            return self.resolve(alert_id, **kwargs)

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "active_alert_count": len(self._alerts),
                "history_count": len(self._history),
                "threshold_rule_count": len(self._threshold_rules),
                "notifier_count": len(self._notifiers),
                "trigger_count": self._trigger_count,
                "resolve_count": self._resolve_count,
                "notify_count": self._notify_count,
                "notify_failures": self._notify_failures,
            }
            return HealthStatus(healthy=True, component="alerts", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="alerts", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "AlertManager",
    "Alert",
    "AlertCategory",
    "AlertSeverity",
    "AlertState",
    "ThresholdRule",
    "AlertsError",
    "AlertNotFoundError",
    "HealthStatus",
]
