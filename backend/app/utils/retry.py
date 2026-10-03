"""
retry.py

Retry decorators supporting exponential backoff with jitter, for both
synchronous and asynchronous callables, plus a configurable timeout
decorator.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExceptionTypes = Union[Type[BaseException], Tuple[Type[BaseException], ...]]


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Failed after {attempts} attempts: {last_exception}")


class RetryTimeoutError(Exception):
    """Raised when a function call exceeds its allowed execution time."""


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry/backoff behavior."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    exceptions: ExceptionTypes = (Exception,)

    def compute_delay(self, attempt: int) -> float:
        """Compute the delay before the given attempt (1-indexed), with jitter."""
        raw_delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        capped_delay = min(raw_delay, self.max_delay)
        jitter = capped_delay * self.jitter_factor
        return max(0.0, capped_delay + random.uniform(-jitter, jitter))


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter_factor: float = 0.1,
    exceptions: ExceptionTypes = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    reraise_as: Optional[Type[BaseException]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a synchronous function with exponential backoff."""
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        jitter_factor=jitter_factor,
        exceptions=exceptions,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except policy.exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= policy.max_attempts:
                        break
                    if on_retry:
                        on_retry(attempt, exc)
                    delay = policy.compute_delay(attempt)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                        func.__name__,
                        attempt,
                        policy.max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None
            if reraise_as is not None:
                raise reraise_as(str(last_exc)) from last_exc
            raise RetryError(policy.max_attempts, last_exc)

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter_factor: float = 0.1,
    exceptions: ExceptionTypes = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    reraise_as: Optional[Type[BaseException]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: retry an async function with exponential backoff."""
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        jitter_factor=jitter_factor,
        exceptions=exceptions,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except policy.exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= policy.max_attempts:
                        break
                    if on_retry:
                        on_retry(attempt, exc)
                    delay = policy.compute_delay(attempt)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                        func.__name__,
                        attempt,
                        policy.max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            assert last_exc is not None
            if reraise_as is not None:
                raise reraise_as(str(last_exc)) from last_exc
            raise RetryError(policy.max_attempts, last_exc)

        return wrapper

    return decorator


def timeout(seconds: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: enforce a timeout on a synchronous or async function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError as exc:
                    raise RetryTimeoutError(
                        f"{func.__name__} timed out after {seconds}s"
                    ) from exc

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = func(*args, **kwargs)
            elapsed = time.monotonic() - start
            if elapsed > seconds:
                logger.warning(
                    "%s exceeded timeout budget: %.2fs > %.2fs",
                    func.__name__,
                    elapsed,
                    seconds,
                )
            return result

        return sync_wrapper

    return decorator


def retry_call(
    func: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> T:
    """Imperatively retry a callable using a RetryPolicy instance."""
    active_policy = policy or RetryPolicy()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, active_policy.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except active_policy.exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt >= active_policy.max_attempts:
                break
            time.sleep(active_policy.compute_delay(attempt))
    assert last_exc is not None
    raise RetryError(active_policy.max_attempts, last_exc)
