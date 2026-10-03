from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


class BusError(Exception):
    pass


class SubscriberNotFoundError(BusError):
    pass


class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Message:
    message_id: str
    sender: str
    topic: str
    payload: Any
    recipient: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    sent_at_epoch: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    source: str
    data: Any
    occurred_at_epoch: float = field(default_factory=time.time)


MessageHandler = Callable[[Message], Awaitable[None]]
EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class _Subscription:
    subscriber_id: str
    topic: str
    handler: MessageHandler


@dataclass
class _EventSubscription:
    subscriber_id: str
    event_type: str
    handler: EventHandler


class AgentQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)

    async def put(self, message: Message) -> None:
        await self._queue.put(message)

    async def get(self, timeout_seconds: Optional[float] = None) -> Message:
        if timeout_seconds is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


class CommunicationBus:
    def __init__(self) -> None:
        self._queues: dict[str, AgentQueue] = {}
        self._topic_subscriptions: list[_Subscription] = []
        self._event_subscriptions: list[_EventSubscription] = []
        self._message_log: list[Message] = []
        self._event_log: list[Event] = []
        self._lock = asyncio.Lock()
        self._sequence = 0

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}-{int(time.time() * 1000)}"

    async def register_agent(self, agent_id: str, queue_maxsize: int = 0) -> AgentQueue:
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = AgentQueue(maxsize=queue_maxsize)
            return self._queues[agent_id]

    async def unregister_agent(self, agent_id: str) -> None:
        async with self._lock:
            self._queues.pop(agent_id, None)
            self._topic_subscriptions = [
                sub for sub in self._topic_subscriptions if sub.subscriber_id != agent_id
            ]
            self._event_subscriptions = [
                sub for sub in self._event_subscriptions if sub.subscriber_id != agent_id
            ]

    async def send(
        self,
        sender: str,
        recipient: str,
        topic: str,
        payload: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        message = Message(
            message_id=self._next_id("msg"),
            sender=sender,
            topic=topic,
            payload=payload,
            recipient=recipient,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        async with self._lock:
            queue = self._queues.get(recipient)
            self._message_log.append(message)
        if queue is None:
            raise SubscriberNotFoundError(f"agent '{recipient}' is not registered")
        await queue.put(message)
        await self._dispatch_topic_handlers(message)
        return message

    async def broadcast(
        self,
        sender: str,
        topic: str,
        payload: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude: Optional[set[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Message]:
        exclusions = exclude or set()
        async with self._lock:
            recipients = [r for r in self._queues.keys() if r != sender and r not in exclusions]
            queues = {r: self._queues[r] for r in recipients}

        sent_messages: list[Message] = []
        for recipient in recipients:
            message = Message(
                message_id=self._next_id("msg"),
                sender=sender,
                topic=topic,
                payload=payload,
                recipient=recipient,
                priority=priority,
                metadata=metadata or {},
            )
            async with self._lock:
                self._message_log.append(message)
            await queues[recipient].put(message)
            sent_messages.append(message)

        broadcast_message = Message(
            message_id=self._next_id("msg"),
            sender=sender,
            topic=topic,
            payload=payload,
            recipient=None,
            priority=priority,
            metadata=metadata or {},
        )
        await self._dispatch_topic_handlers(broadcast_message)
        return sent_messages

    async def receive(
        self, agent_id: str, timeout_seconds: Optional[float] = None
    ) -> Message:
        async with self._lock:
            queue = self._queues.get(agent_id)
        if queue is None:
            raise SubscriberNotFoundError(f"agent '{agent_id}' is not registered")
        return await queue.get(timeout_seconds)

    async def subscribe_topic(
        self, subscriber_id: str, topic: str, handler: MessageHandler
    ) -> None:
        async with self._lock:
            self._topic_subscriptions.append(
                _Subscription(subscriber_id=subscriber_id, topic=topic, handler=handler)
            )

    async def unsubscribe_topic(self, subscriber_id: str, topic: str) -> None:
        async with self._lock:
            self._topic_subscriptions = [
                sub
                for sub in self._topic_subscriptions
                if not (sub.subscriber_id == subscriber_id and sub.topic == topic)
            ]

    async def _dispatch_topic_handlers(self, message: Message) -> None:
        async with self._lock:
            matching = [
                sub.handler
                for sub in self._topic_subscriptions
                if sub.topic == message.topic
            ]
        await asyncio.gather(
            *(self._safe_invoke_message_handler(handler, message) for handler in matching),
            return_exceptions=True,
        )

    @staticmethod
    async def _safe_invoke_message_handler(handler: MessageHandler, message: Message) -> None:
        try:
            await handler(message)
        except Exception:
            pass

    async def publish_event(
        self, source: str, event_type: str, data: Any
    ) -> Event:
        event = Event(
            event_id=self._next_id("evt"),
            event_type=event_type,
            source=source,
            data=data,
        )
        async with self._lock:
            self._event_log.append(event)
            matching = [
                sub.handler
                for sub in self._event_subscriptions
                if sub.event_type == event_type
            ]
        await asyncio.gather(
            *(self._safe_invoke_event_handler(handler, event) for handler in matching),
            return_exceptions=True,
        )
        return event

    @staticmethod
    async def _safe_invoke_event_handler(handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:
            pass

    async def subscribe_event(
        self, subscriber_id: str, event_type: str, handler: EventHandler
    ) -> None:
        async with self._lock:
            self._event_subscriptions.append(
                _EventSubscription(
                    subscriber_id=subscriber_id, event_type=event_type, handler=handler
                )
            )

    async def unsubscribe_event(self, subscriber_id: str, event_type: str) -> None:
        async with self._lock:
            self._event_subscriptions = [
                sub
                for sub in self._event_subscriptions
                if not (sub.subscriber_id == subscriber_id and sub.event_type == event_type)
            ]

    async def message_history(
        self,
        topic: Optional[str] = None,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Message]:
        async with self._lock:
            results = self._message_log
            if topic is not None:
                results = [m for m in results if m.topic == topic]
            if sender is not None:
                results = [m for m in results if m.sender == sender]
            if recipient is not None:
                results = [m for m in results if m.recipient == recipient]
            if limit is not None:
                results = results[-limit:]
            return list(results)

    async def event_history(
        self, event_type: Optional[str] = None, limit: Optional[int] = None
    ) -> list[Event]:
        async with self._lock:
            results = self._event_log
            if event_type is not None:
                results = [e for e in results if e.event_type == event_type]
            if limit is not None:
                results = results[-limit:]
            return list(results)

    def queue_depth(self, agent_id: str) -> int:
        queue = self._queues.get(agent_id)
        if queue is None:
            raise SubscriberNotFoundError(f"agent '{agent_id}' is not registered")
        return queue.qsize()

    def registered_agents(self) -> list[str]:
        return list(self._queues.keys())
