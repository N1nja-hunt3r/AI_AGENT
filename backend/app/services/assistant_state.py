"""
assistant_state.py

Unified assistant state machine.

States:
  OFF -> Starting -> Initializing -> Ready -> Listening -> Thinking
  -> Researching -> Coding -> Speaking -> Idle -> Sleep

Events drive transitions between states. An idle timeout triggers
automatic sleep after configurable inactivity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    OFF = "off"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    THINKING = "thinking"
    RESEARCHING = "researching"
    CODING = "coding"
    SPEAKING = "speaking"
    IDLE = "idle"
    SLEEP = "sleep"
    ERROR = "error"


VALID_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.OFF: {AssistantState.STARTING},
    AssistantState.STARTING: {AssistantState.INITIALIZING, AssistantState.ERROR},
    AssistantState.INITIALIZING: {AssistantState.READY, AssistantState.ERROR},
    AssistantState.READY: {AssistantState.LISTENING, AssistantState.SLEEP, AssistantState.ERROR},
    AssistantState.LISTENING: {AssistantState.THINKING, AssistantState.READY, AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.THINKING: {AssistantState.RESEARCHING, AssistantState.CODING, AssistantState.SPEAKING, AssistantState.READY, AssistantState.ERROR},
    AssistantState.RESEARCHING: {AssistantState.THINKING, AssistantState.SPEAKING, AssistantState.READY, AssistantState.ERROR},
    AssistantState.CODING: {AssistantState.THINKING, AssistantState.SPEAKING, AssistantState.READY, AssistantState.ERROR},
    AssistantState.SPEAKING: {AssistantState.LISTENING, AssistantState.IDLE, AssistantState.READY, AssistantState.ERROR},
    AssistantState.IDLE: {AssistantState.LISTENING, AssistantState.SLEEP, AssistantState.READY, AssistantState.ERROR},
    AssistantState.SLEEP: {AssistantState.LISTENING, AssistantState.READY, AssistantState.OFF, AssistantState.ERROR},
    AssistantState.ERROR: {AssistantState.READY, AssistantState.OFF, AssistantState.SLEEP},
}


StateChangeCallback = Callable[[AssistantState, AssistantState, Optional[str]], None]


@dataclass
class StateConfig:
    idle_timeout_seconds: int = 10
    sleep_timeout_seconds: int = 300
    enable_sleep: bool = True
    auto_listen_after_speak: bool = True


@dataclass
class StateEvent:
    from_state: AssistantState
    to_state: AssistantState
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class AssistantStateMachine:
    def __init__(
        self,
        config: Optional[StateConfig] = None,
        on_state_change: Optional[StateChangeCallback] = None,
    ) -> None:
        self._config = config or StateConfig()
        self._state = AssistantState.OFF
        self._on_state_change = on_state_change
        self._event_history: list[StateEvent] = []
        self._idle_timer: Optional[asyncio.Task[None]] = None
        self._sleep_timer: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._listeners: dict[str, list[Callable[[StateEvent], None]]] = {}

    @property
    def current_state(self) -> AssistantState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.value

    def is_available(self) -> bool:
        return self._state in {
            AssistantState.READY,
            AssistantState.LISTENING,
            AssistantState.IDLE,
        }

    def is_active(self) -> bool:
        return self._state not in {
            AssistantState.OFF,
            AssistantState.SLEEP,
            AssistantState.ERROR,
        }

    async def transition(
        self,
        target: AssistantState,
        reason: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        async with self._lock:
            current = self._state

            if target == current:
                return True

            if not force and target not in VALID_TRANSITIONS.get(current, set()):
                logger.warning(
                    "Invalid state transition: %s -> %s",
                    current.value, target.value,
                )
                return False

            self._state = target
            event = StateEvent(
                from_state=current,
                to_state=target,
                reason=reason,
            )
            self._event_history.append(event)

            logger.info(
                "State change: %s -> %s%s",
                current.value, target.value,
                f" ({reason})" if reason else "",
            )

        self._notify_listeners(event)
        if self._on_state_change:
            self._on_state_change(current, target, reason)

        self._handle_auto_transitions(target)
        return True

    def _handle_auto_transitions(self, state: AssistantState) -> None:
        if state in {AssistantState.READY, AssistantState.IDLE}:
            self._start_idle_timer()
        else:
            self._cancel_idle_timer()

        if state == AssistantState.SLEEP:
            self._cancel_idle_timer()
            self._cancel_sleep_timer()
        elif state == AssistantState.OFF:
            self._cancel_idle_timer()
            self._cancel_sleep_timer()

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if self._config.idle_timeout_seconds <= 0:
            return
        async def _timer() -> None:
            await asyncio.sleep(self._config.idle_timeout_seconds)
            if self._state == AssistantState.IDLE:
                await self.transition(
                    AssistantState.SLEEP,
                    reason=f"No activity for {self._config.idle_timeout_seconds}s",
                )
            elif self._state == AssistantState.READY:
                await self.transition(
                    AssistantState.IDLE,
                    reason="No activity detected",
                )
        self._idle_timer = asyncio.create_task(_timer())

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer and not self._idle_timer.done():
            self._idle_timer.cancel()
        self._idle_timer = None

    def _cancel_sleep_timer(self) -> None:
        if self._sleep_timer and not self._sleep_timer.done():
            self._sleep_timer.cancel()
        self._sleep_timer = None

    def _notify_listeners(self, event: StateEvent) -> None:
        for key in ("*", event.to_state.value):
            listeners = self._listeners.get(key, [])
            for listener in listeners:
                try:
                    listener(event)
                except Exception:
                    logger.exception("State listener error for %s", key)

    def on_state(self, state: str, callback: Callable[[StateEvent], None]) -> None:
        self._listeners.setdefault(state, []).append(callback)

    def get_history(self, limit: int = 10) -> list[StateEvent]:
        return self._event_history[-limit:]

    def get_config(self) -> StateConfig:
        return self._config

    def set_idle_timeout(self, seconds: int) -> None:
        self._config.idle_timeout_seconds = seconds

    def ping(self) -> None:
        if self._state in {AssistantState.READY, AssistantState.IDLE}:
            self._start_idle_timer()


state_machine = AssistantStateMachine()
