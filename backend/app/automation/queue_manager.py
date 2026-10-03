from __future__ import annotations

import asyncio
import heapq
import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QueueMode(str, Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    DELAYED = "delayed"


class MessageState(str, Enum):
    QUEUED = "queued"
    DELAYED = "delayed"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"


class QueueManagerError(Exception):
    pass


class MessageNotFoundError(QueueManagerError):
    pass


class QueueEmptyError(QueueManagerError):
    pass


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Message:
    message_id: str
    payload: Any
    priority: int = 100
    state: MessageState = MessageState.QUEUED
    enqueued_at: float = field(default_factory=time.time)
    available_at: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id, "priority": self.priority, "state": self.state.value,
            "enqueued_at": self.enqueued_at, "available_at": self.available_at,
            "retry_count": self.retry_count, "max_retries": self.max_retries, "last_error": self.last_error,
        }


@dataclass(order=True)
class _HeapEntry:
    available_at: float
    priority: int
    seq: int
    message_id: str = field(compare=False)


class QueueManager:
    """Priority/FIFO/delayed message queue with retries, dead-letter queue, and metrics."""

    def __init__(self, *, default_max_retries: int = 3, default_retry_delay_seconds: float = 5.0) -> None:
        self.default_max_retries = default_max_retries
        self.default_retry_delay_seconds = default_retry_delay_seconds
        self._messages: dict[str, Message] = {}
        self._heap: list[_HeapEntry] = []
        self._seq_counter = itertools.count()
        self._dead_letter: list[Message] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._push_count = 0
        self._pop_count = 0
        self._retry_count = 0
        self._dlq_count = 0
        self._purge_count = 0

    def push(
        self,
        payload: Any,
        *,
        priority: int = 100,
        delay_seconds: float = 0.0,
        max_retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
    ) -> Message:
        self._push_count += 1
        now = time.time()
        message = Message(
            message_id=str(uuid.uuid4()), payload=payload, priority=priority,
            state=MessageState.DELAYED if delay_seconds > 0 else MessageState.QUEUED,
            available_at=now + delay_seconds,
            max_retries=max_retries if max_retries is not None else self.default_max_retries,
            retry_delay_seconds=retry_delay_seconds if retry_delay_seconds is not None else self.default_retry_delay_seconds,
        )
        self._messages[message.message_id] = message
        heapq.heappush(self._heap, _HeapEntry(message.available_at, message.priority, next(self._seq_counter), message.message_id))
        return message

    def pop(self) -> Optional[Message]:
        self._pop_count += 1
        now = time.time()
        skipped: list[_HeapEntry] = []

        result: Optional[Message] = None
        while self._heap:
            entry = heapq.heappop(self._heap)
            message = self._messages.get(entry.message_id)
            if message is None or message.state in (MessageState.COMPLETED, MessageState.DEAD_LETTERED, MessageState.IN_FLIGHT):
                continue
            if message.available_at > now:
                skipped.append(entry)
                continue
            message.state = MessageState.IN_FLIGHT
            result = message
            break

        for entry in skipped:
            heapq.heappush(self._heap, entry)

        return result

    def peek(self) -> Optional[Message]:
        now = time.time()
        for entry in sorted(self._heap):
            message = self._messages.get(entry.message_id)
            if message is not None and message.state in (MessageState.QUEUED, MessageState.DELAYED) and message.available_at <= now:
                return message
        return None

    def ack(self, message_id: str) -> bool:
        message = self._messages.get(message_id)
        if message is None:
            raise MessageNotFoundError(f"no message with id {message_id}")
        message.state = MessageState.COMPLETED
        return True

    def retry(self, message_id: str, *, error: Optional[str] = None) -> Message:
        self._retry_count += 1
        message = self._messages.get(message_id)
        if message is None:
            raise MessageNotFoundError(f"no message with id {message_id}")

        message.retry_count += 1
        message.last_error = error

        if message.retry_count > message.max_retries:
            message.state = MessageState.DEAD_LETTERED
            self._dead_letter.append(message)
            self._dlq_count += 1
            return message

        message.state = MessageState.RETRYING
        message.available_at = time.time() + message.retry_delay_seconds
        heapq.heappush(self._heap, _HeapEntry(message.available_at, message.priority, next(self._seq_counter), message.message_id))
        message.state = MessageState.DELAYED
        return message

    def nack(self, message_id: str, *, requeue: bool = True, error: Optional[str] = None) -> Message:
        if requeue:
            return self.retry(message_id, error=error)
        message = self._messages.get(message_id)
        if message is None:
            raise MessageNotFoundError(f"no message with id {message_id}")
        message.state = MessageState.DEAD_LETTERED
        message.last_error = error
        self._dead_letter.append(message)
        self._dlq_count += 1
        return message

    def purge(self, *, state: Optional[MessageState] = None) -> int:
        self._purge_count += 1
        if state is None:
            count = len(self._messages)
            self._messages.clear()
            self._heap.clear()
            self._dead_letter.clear()
            return count

        to_remove = [mid for mid, m in self._messages.items() if m.state == state]
        for mid in to_remove:
            self._messages.pop(mid, None)
        if state == MessageState.DEAD_LETTERED:
            self._dead_letter = [m for m in self._dead_letter if m.message_id not in to_remove]
        self._heap = [e for e in self._heap if e.message_id not in to_remove]
        heapq.heapify(self._heap)
        return len(to_remove)

    def get(self, message_id: str) -> Optional[Message]:
        return self._messages.get(message_id)

    def list_dead_letter(self, *, limit: int = 200) -> list[Message]:
        return self._dead_letter[-limit:]

    def requeue_dead_letter(self, message_id: str) -> Message:
        message = next((m for m in self._dead_letter if m.message_id == message_id), None)
        if message is None:
            raise MessageNotFoundError(f"no dead-lettered message with id {message_id}")
        message.state = MessageState.QUEUED
        message.retry_count = 0
        message.available_at = time.time()
        self._dead_letter.remove(message)
        heapq.heappush(self._heap, _HeapEntry(message.available_at, message.priority, next(self._seq_counter), message.message_id))
        return message

    def size(self, *, state: Optional[MessageState] = None) -> int:
        if state is None:
            return sum(1 for m in self._messages.values() if m.state not in (MessageState.COMPLETED,))
        return sum(1 for m in self._messages.values() if m.state == state)

    def metrics(self) -> dict[str, Any]:
        return {
            "total_messages": len(self._messages),
            "queued": self.size(state=MessageState.QUEUED),
            "delayed": self.size(state=MessageState.DELAYED),
            "in_flight": self.size(state=MessageState.IN_FLIGHT),
            "completed": self.size(state=MessageState.COMPLETED),
            "dead_lettered": len(self._dead_letter),
            "push_count": self._push_count,
            "pop_count": self._pop_count,
            "retry_count": self._retry_count,
            "dlq_count": self._dlq_count,
            "purge_count": self._purge_count,
        }

    async def push_async(self, payload: Any, **kwargs: Any) -> Message:
        async with self._lock:
            return self.push(payload, **kwargs)

    async def pop_async(self) -> Optional[Message]:
        async with self._lock:
            return self.pop()

    async def retry_async(self, message_id: str, **kwargs: Any) -> Message:
        async with self._lock:
            return self.retry(message_id, **kwargs)

    async def purge_async(self, **kwargs: Any) -> int:
        async with self._lock:
            return self.purge(**kwargs)

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                **self.metrics(),
            }
            healthy = self.size(state=MessageState.DEAD_LETTERED) < max(100, len(self._messages))
            return HealthStatus(healthy=healthy, component="queue_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="queue_manager", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "QueueManager",
    "Message",
    "MessageState",
    "QueueMode",
    "QueueManagerError",
    "MessageNotFoundError",
    "QueueEmptyError",
    "HealthStatus",
]
