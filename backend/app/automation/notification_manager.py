from __future__ import annotations

import asyncio
import json
import smtplib
import time
import uuid
from dataclasses import dataclass, field
from email.message import EmailMessage
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


class NotificationChannel(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    PUSH = "push"


class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SCHEDULED = "scheduled"


class NotificationError(Exception):
    pass


class ChannelNotConfiguredError(NotificationError):
    pass


PushSenderFn = Callable[[str, str, dict[str, Any]], Union[bool, Awaitable[bool]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = True
    from_address: str = "noreply@example.com"


@dataclass
class NotificationRecord:
    notification_id: str
    channel: NotificationChannel
    target: str
    subject: Optional[str]
    message: str
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    error: Optional[str] = None
    scheduled_for: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "notification_id": self.notification_id, "channel": self.channel.value, "target": self.target,
            "subject": self.subject, "status": self.status.value, "created_at": self.created_at,
            "sent_at": self.sent_at, "error": self.error, "scheduled_for": self.scheduled_for,
        }


class NotificationManager:
    """Sends notifications via email, webhook, Slack, Discord, and push channels."""

    def __init__(
        self,
        *,
        email_config: Optional[EmailConfig] = None,
        push_sender: Optional[PushSenderFn] = None,
        http_timeout_seconds: float = 10.0,
        max_history: int = 5000,
    ) -> None:
        self.email_config = email_config
        self.push_sender = push_sender
        self.http_timeout_seconds = http_timeout_seconds
        self.max_history = max_history
        self._history: list[NotificationRecord] = []
        self._scheduled: dict[str, NotificationRecord] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._send_count = 0
        self._failure_count = 0

    def _archive(self, record: NotificationRecord) -> None:
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    async def _send_email(self, target: str, subject: Optional[str], message: str) -> None:
        if self.email_config is None:
            raise ChannelNotConfiguredError("email channel is not configured")

        def _send_sync() -> None:
            cfg = self.email_config
            assert cfg is not None
            msg = EmailMessage()
            msg["From"] = cfg.from_address
            msg["To"] = target
            msg["Subject"] = subject or "Notification"
            msg.set_content(message)

            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=self.http_timeout_seconds) as server:
                if cfg.use_tls:
                    server.starttls()
                if cfg.username and cfg.password:
                    server.login(cfg.username, cfg.password)
                server.send_message(msg)

        await asyncio.to_thread(_send_sync)

    def _http_post_json(self, url: str, payload: dict[str, Any]) -> None:
        if not _HAS_URLLIB:
            raise NotificationError("urllib is unavailable in this environment")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.http_timeout_seconds) as resp:
            if resp.status >= 400:
                raise NotificationError(f"webhook returned HTTP {resp.status}")

    async def _send_webhook(self, target: str, message: str, *, payload_override: Optional[dict[str, Any]] = None) -> None:
        payload = payload_override if payload_override is not None else {"text": message}
        await asyncio.to_thread(self._http_post_json, target, payload)

    async def _send_slack(self, target: str, message: str) -> None:
        await asyncio.to_thread(self._http_post_json, target, {"text": message})

    async def _send_discord(self, target: str, message: str) -> None:
        await asyncio.to_thread(self._http_post_json, target, {"content": message})

    async def _send_push(self, target: str, subject: Optional[str], message: str) -> None:
        if self.push_sender is None:
            raise ChannelNotConfiguredError("push channel is not configured")
        result = self.push_sender(target, subject or "", {"body": message})
        if asyncio.iscoroutine(result):
            ok = await result
        else:
            ok = result
        if not ok:
            raise NotificationError("push sender reported failure")

    async def send(
        self,
        channel: NotificationChannel,
        target: str,
        message: str,
        *,
        subject: Optional[str] = None,
        payload_override: Optional[dict[str, Any]] = None,
    ) -> NotificationRecord:
        self._send_count += 1
        record = NotificationRecord(
            notification_id=str(uuid.uuid4()), channel=channel, target=target, subject=subject, message=message,
        )

        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email(target, subject, message)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(target, message, payload_override=payload_override)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack(target, message)
            elif channel == NotificationChannel.DISCORD:
                await self._send_discord(target, message)
            elif channel == NotificationChannel.PUSH:
                await self._send_push(target, subject, message)
            else:
                raise NotificationError(f"unsupported channel: {channel}")

            record.status = NotificationStatus.SENT
            record.sent_at = time.time()
        except Exception as exc:
            self._failure_count += 1
            record.status = NotificationStatus.FAILED
            record.error = str(exc)

        self._archive(record)
        return record

    async def broadcast(
        self,
        channel: NotificationChannel,
        targets: list[str],
        message: str,
        *,
        subject: Optional[str] = None,
        payload_override: Optional[dict[str, Any]] = None,
    ) -> list[NotificationRecord]:
        results = await asyncio.gather(
            *(self.send(channel, target, message, subject=subject, payload_override=payload_override) for target in targets)
        )
        return list(results)

    async def broadcast_multi_channel(
        self,
        channel_targets: dict[NotificationChannel, list[str]],
        message: str,
        *,
        subject: Optional[str] = None,
    ) -> list[NotificationRecord]:
        tasks = []
        for channel, targets in channel_targets.items():
            for target in targets:
                tasks.append(self.send(channel, target, message, subject=subject))
        return list(await asyncio.gather(*tasks))

    def schedule(
        self,
        channel: NotificationChannel,
        target: str,
        message: str,
        *,
        send_at: float,
        subject: Optional[str] = None,
    ) -> NotificationRecord:
        record = NotificationRecord(
            notification_id=str(uuid.uuid4()), channel=channel, target=target, subject=subject,
            message=message, status=NotificationStatus.SCHEDULED, scheduled_for=send_at,
        )
        self._scheduled[record.notification_id] = record
        return record

    async def dispatch_due_scheduled(self) -> list[NotificationRecord]:
        now = time.time()
        due_ids = [nid for nid, rec in self._scheduled.items() if rec.scheduled_for is not None and rec.scheduled_for <= now]
        dispatched: list[NotificationRecord] = []
        for nid in due_ids:
            rec = self._scheduled.pop(nid)
            result = await self.send(rec.channel, rec.target, rec.message, subject=rec.subject)
            dispatched.append(result)
        return dispatched

    def cancel_scheduled(self, notification_id: str) -> bool:
        return self._scheduled.pop(notification_id, None) is not None

    def list_scheduled(self) -> list[NotificationRecord]:
        return list(self._scheduled.values())

    def history(self, limit: int = 200, *, channel: Optional[NotificationChannel] = None, status: Optional[NotificationStatus] = None) -> list[NotificationRecord]:
        items = self._history
        if channel is not None:
            items = [r for r in items if r.channel == channel]
        if status is not None:
            items = [r for r in items if r.status == status]
        return items[-limit:]

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "send_count": self._send_count,
                "failure_count": self._failure_count,
                "history_count": len(self._history),
                "scheduled_count": len(self._scheduled),
                "email_configured": self.email_config is not None,
                "push_configured": self.push_sender is not None,
            }
            return HealthStatus(healthy=True, component="notification_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="notification_manager", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "NotificationManager",
    "NotificationRecord",
    "NotificationChannel",
    "NotificationStatus",
    "EmailConfig",
    "NotificationError",
    "ChannelNotConfiguredError",
    "HealthStatus",
]
