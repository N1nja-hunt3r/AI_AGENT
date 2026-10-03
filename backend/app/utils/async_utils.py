"""
async_utils.py

Asynchronous utility helpers: gather-with-concurrency, timeouts,
cancellation-safe wrappers, and async retry execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Coroutine, Iterable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskTimeoutError(Exception):
    """Raised when an awaited coroutine exceeds its allotted timeout."""


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    *,
    timeout_message: Optional[str] = None,
) -> T:
    """Run a coroutine with a timeout, raising TaskTimeoutError on expiry."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        message = timeout_message or f"Operation timed out after {timeout}s"
        raise TaskTimeoutError(message) from exc


async def gather_with_concurrency(
    coros: Iterable[Awaitable[T]],
    limit: int,
    *,
    return_exceptions: bool = False,
) -> List[Any]:
    """Run coroutines concurrently, bounded by a semaphore limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    semaphore = asyncio.Semaphore(limit)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(_bounded(c) for c in coros), return_exceptions=return_exceptions
    )


async def gather_with_timeout(
    coros: Iterable[Awaitable[T]],
    timeout: float,
    *,
    return_exceptions: bool = True,
) -> List[Any]:
    """Gather coroutines, enforcing an overall timeout for the whole batch."""
    return await asyncio.wait_for(
        asyncio.gather(*coros, return_exceptions=return_exceptions),
        timeout=timeout,
    )


async def safe_cancel(task: "asyncio.Task[Any]", *, grace_period: float = 1.0) -> None:
    """Cancel a task and await its completion, suppressing CancelledError."""
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=grace_period)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Error while cancelling task %r", task)


async def cancel_all(tasks: Iterable["asyncio.Task[Any]"], *, grace_period: float = 1.0) -> None:
    """Cancel a collection of tasks concurrently."""
    await asyncio.gather(
        *(safe_cancel(t, grace_period=grace_period) for t in tasks),
        return_exceptions=True,
    )


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    **kwargs: Any,
) -> T:
    """Retry an async callable with exponential backoff."""
    attempt = 0
    current_delay = delay
    while True:
        try:
            return await func(*args, **kwargs)
        except exceptions as exc:  # type: ignore[misc]
            attempt += 1
            if attempt > retries:
                logger.error("retry_async exhausted after %d attempts: %s", attempt, exc)
                raise
            if on_retry:
                on_retry(attempt, exc)
            logger.warning(
                "retry_async attempt %d/%d failed: %s. Retrying in %.2fs",
                attempt,
                retries,
                exc,
                current_delay,
            )
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * backoff, max_delay)


async def run_in_background(
    coro: Coroutine[Any, Any, T],
    *,
    name: Optional[str] = None,
    error_handler: Optional[Callable[[BaseException], None]] = None,
) -> "asyncio.Task[T]":
    """Schedule a coroutine as a background task with error logging."""

    async def _wrapped() -> T:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            if error_handler:
                error_handler(exc)
            else:
                logger.exception("Background task %r failed", name)
            raise

    return asyncio.create_task(_wrapped(), name=name)


async def wait_for_condition(
    predicate: Callable[[], bool],
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll a synchronous predicate until True or timeout elapses."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(poll_interval)
    return predicate()


class AsyncRateLimiter:
    """Simple token-bucket style async rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        if max_calls <= 0 or period_seconds <= 0:
            raise ValueError("max_calls and period_seconds must be positive")
        self._max_calls = max_calls
        self._period = period_seconds
        self._semaphore = asyncio.Semaphore(max_calls)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        asyncio.get_event_loop().call_later(self._period, self._semaphore.release)

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None
