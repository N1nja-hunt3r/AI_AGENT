"""
middleware_loader.py

Defines and registers core HTTP middleware: security headers, request
logging, monitoring instrumentation and rate limiting.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("middleware_loader")

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code and latency for every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attaches standard security headers to every response."""

    _HEADERS: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for key, value in self._HEADERS.items():
            response.headers.setdefault(key, value)
        return response


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Tracks per-route request counts and cumulative latency."""

    def __init__(self, app: FastAPI, metrics_sink: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        super().__init__(app)
        self.metrics_sink: Dict[str, Dict[str, float]] = (
            metrics_sink if metrics_sink is not None else defaultdict(lambda: {"count": 0.0, "total_ms": 0.0})
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        key = f"{request.method} {request.url.path}"
        bucket = self.metrics_sink[key]
        bucket["count"] += 1
        bucket["total_ms"] += duration_ms
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """A simple in-memory fixed-window rate limiter, keyed by client IP."""

    def __init__(self, app: FastAPI, max_requests: int = 120, window_seconds: int = 60) -> None:
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_id = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self._window_seconds
        hits = [t for t in self._hits[client_id] if t > window_start]
        hits.append(now)
        self._hits[client_id] = hits

        if len(hits) > self._max_requests:
            return Response(
                content='{"detail": "Rate limit exceeded."}',
                status_code=429,
                media_type="application/json",
            )
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """Extracts JWT from Authorization header, verifies via auth_service, sets request.state.user."""

    EXEMPT_PATHS: List[str] = [
        "/", "/health", "/health/live", "/health/ready", "/metrics",
        "/api/v1/health",         "/api/v1/chat",
        "/api/v1/voice",
        "/api/v1/auth/signup", "/api/v1/auth/login",
        "/api/v1/auth/refresh", "/api/v1/auth/oauth",
        "/docs", "/redoc", "/openapi.json",
    ]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        for exempt in self.EXEMPT_PATHS:
            if request.url.path.startswith(exempt):
                return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Not authenticated"})

        token = auth_header[len("Bearer "):]
        auth_service = getattr(request.app.state, "auth_service", None)
        if auth_service is None:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": "Auth service unavailable"})

        try:
            payload = await auth_service.verify_access_token(token)
            payload["id"] = payload.get("sub", "")
            payload["name"] = payload.get("username", "")
            request.state.user = payload
        except Exception:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid or expired token"})

        return await call_next(request)


class MiddlewareLoader:
    """Registers the standard middleware stack onto a FastAPI application."""

    def __init__(
        self,
        enable_logging: bool = True,
        enable_security_headers: bool = True,
        enable_monitoring: bool = True,
        enable_rate_limit: bool = True,
        rate_limit_max_requests: int = 120,
        rate_limit_window_seconds: int = 60,
    ) -> None:
        self._enable_logging = enable_logging
        self._enable_security_headers = enable_security_headers
        self._enable_monitoring = enable_monitoring
        self._enable_rate_limit = enable_rate_limit
        self._rate_limit_max_requests = rate_limit_max_requests
        self._rate_limit_window_seconds = rate_limit_window_seconds
        self.monitoring_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "total_ms": 0.0}
        )
        self._registered: List[str] = []

    def register_middleware(self, app: FastAPI) -> List[str]:
        # Starlette applies middleware in reverse order of addition, so the
        # last middleware added runs first (outermost).
        app.add_middleware(AuthMiddleware)
        self._registered.append("auth")

        if self._enable_rate_limit:
            app.add_middleware(
                RateLimitMiddleware,
                max_requests=self._rate_limit_max_requests,
                window_seconds=self._rate_limit_window_seconds,
            )
            self._registered.append("rate_limit")

        if self._enable_monitoring:
            app.add_middleware(MonitoringMiddleware, metrics_sink=self.monitoring_metrics)
            self._registered.append("monitoring")

        if self._enable_security_headers:
            app.add_middleware(SecurityHeadersMiddleware)
            self._registered.append("security_headers")

        if self._enable_logging:
            app.add_middleware(RequestLoggingMiddleware)
            self._registered.append("request_logging")

        logger.info("Registered middleware: %s", self._registered)
        return list(self._registered)

    def list_middleware(self) -> List[str]:
        return list(self._registered)


_middleware_loader: Optional[MiddlewareLoader] = None


def get_middleware_loader() -> MiddlewareLoader:
    global _middleware_loader
    if _middleware_loader is None:
        _middleware_loader = MiddlewareLoader()
    return _middleware_loader
