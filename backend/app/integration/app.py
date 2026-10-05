"""
app.py

Production FastAPI application entrypoint: lifespan-managed system
bootstrap, router and middleware registration, dependency injection,
CORS, WebSockets, health/metrics endpoints and global exception
handling.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.integration.bootstrap import BootstrapReport, SystemBootstrap, get_bootstrap
from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.middleware_loader import MiddlewareLoader, get_middleware_loader
from app.integration.router_loader import RouterLoader, get_router_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("app")

APP_START_TIME = time.time()


def _load_allowed_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    if "*" in origins:
        logger.warning("CORS_ORIGINS contains wildcard '*'; falling back to explicit local origins")
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: DependencyContainer = get_container()
    bootstrap: SystemBootstrap = await get_bootstrap()

    app.state.container = container
    app.state.bootstrap = bootstrap
    app.state.websocket_clients = set()

    logger.info("Application lifespan startup beginning.")
    report: BootstrapReport = await bootstrap.initialize()
    app.state.bootstrap_report = report

    # Expose kernel components on app.state for health router
    if bootstrap.controller:
        app.state.controller = bootstrap.controller
        app.state.planner = getattr(bootstrap.controller, "planner", None)
        app.state.executor = getattr(bootstrap.controller, "executor", None)

    # Map container keys to app.state attr names for the health router
    _state_attrs = {
        "capability:tool_capability": "tool_registry",
        "capability:rag_capability": "rag_service",
        "capability:memory_capability": "memory_capability",
        "capability:computer_capability": "computer_service",
        "security:auth_provider": "auth_service",
        "security:validator": "security_service",
        "monitoring:health_monitor": "monitoring_service",
        "database:connection": "database",
        "vector_db:connection": "vectordb",
    }
    for key, attr in _state_attrs.items():
        val = container.try_resolve(key)
        if val is not None:
            setattr(app.state, attr, val)

    # Wire agent_loader as agent_manager (has list_agents + health_check)
    if bootstrap.agent_loader is not None:
        app.state.agent_manager = bootstrap.agent_loader

    # Wire settings_service from config manager
    from app.utils.config import ConfigManager
    app.state.settings_service = ConfigManager()

    # Wire artifact_service with in-memory store
    from app.integration.artifact_store import InMemoryArtifactStore
    app.state.artifact_service = InMemoryArtifactStore()

    # Wire in-memory memory capability for chat router
    class _InMemoryMemory:
        def __init__(self) -> None:
            self._store: Dict[str, List[Dict[str, Any]]] = {}
        async def recall(self, session_id: str, user_id: str, query: str) -> List[Dict[str, Any]]:
            return self._store.get(session_id, [])
        async def store(self, session_id: str, user_id: str, role: str, content: str) -> None:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append({"role": role, "content": content, "user_id": user_id})
        async def health_check(self) -> bool:
            return True
    app.state.memory_capability = _InMemoryMemory()

    # Wire redis to in-memory cache (no real Redis required)
    from app.utils.cache_utils import LRUTTLCache
    app.state.redis = LRUTTLCache()

    # Wire automation engine from bootstrap if available
    if hasattr(bootstrap, "automation") and bootstrap.automation is not None:
        app.state.automation_service = bootstrap.automation

    # Wire TTS and STT services (voice router) with provider registry
    from app.services.tts import TextToSpeech
    from app.services.speech_to_text import SpeechToText

    provider_registry = container.try_resolve("provider:registry")
    if provider_registry is not None:
        app.state.tts_service = TextToSpeech.from_registry(provider_registry)
        app.state.stt_service = SpeechToText.from_registry(provider_registry)
        logger.info("TTS/STT services wired with provider registry")
    else:
        app.state.tts_service = TextToSpeech()
        app.state.stt_service = SpeechToText()
        logger.warning("TTS/STT services created without provider registry (mock mode)")

    # Wire greeting, state machine, and wake word services
    from app.services.greeting import GreetingService
    from app.services.assistant_state import state_machine
    from app.services.wake_word import wake_word_service
    app.state.greeting_service = GreetingService()
    app.state.state_machine = state_machine
    app.state.wake_word_service = wake_word_service

    if not report.ok:
        logger.warning("Bootstrap completed with errors: %s", report.errors)
    else:
        logger.info("Bootstrap completed successfully in %.2fs.", report.duration_seconds)

    try:
        yield
    finally:
        logger.info("Application lifespan shutdown beginning.")
        shutdown_report = await bootstrap.shutdown()
        logger.info(
            "Shutdown completed in %.2fs (ok=%s).",
            shutdown_report.duration_seconds,
            shutdown_report.ok,
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI System API",
        version="1.0.0",
        description="Production API surface for the AI system runtime.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_load_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    middleware_loader: MiddlewareLoader = get_middleware_loader()
    middleware_loader.register_middleware(app)

    router_loader: RouterLoader = get_router_loader()
    router_loader.register_routers(app)

    register_exception_handlers(app)
    register_core_routes(app)
    register_legacy_events(app)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.warning("HTTPException on %s %s: %s", request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred.",
                "path": request.url.path,
            },
        )


def register_core_routes(app: FastAPI) -> None:
    @app.get("/", tags=["system"])
    async def root() -> Dict[str, Any]:
        return {
            "app": "AI Agent Backend",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    @app.get("/health", tags=["system"])
    async def health(request: Request) -> Dict[str, Any]:
        bootstrap: SystemBootstrap = request.app.state.bootstrap
        component_health = await bootstrap.health_check()
        overall_ok = bootstrap.is_ready
        return {
            "status": "ok" if overall_ok else "degraded",
            "uptime_seconds": round(time.time() - APP_START_TIME, 2),
            "components": component_health,
        }

    @app.get("/health/live", tags=["system"])
    async def liveness() -> Dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["system"])
    async def readiness(request: Request) -> JSONResponse:
        bootstrap: SystemBootstrap = request.app.state.bootstrap
        if bootstrap.is_ready:
            return JSONResponse(status_code=200, content={"status": "ready"})
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.get("/metrics", tags=["system"])
    async def metrics(request: Request) -> Dict[str, Any]:
        middleware_loader: MiddlewareLoader = get_middleware_loader()
        bootstrap: SystemBootstrap = request.app.state.bootstrap
        monitoring_metrics: Dict[str, Dict[str, float]] = dict(middleware_loader.monitoring_metrics)

        route_metrics = {
            route: {
                "count": int(values["count"]),
                "avg_latency_ms": round(values["total_ms"] / values["count"], 3) if values["count"] else 0.0,
            }
            for route, values in monitoring_metrics.items()
        }

        return {
            "uptime_seconds": round(time.time() - APP_START_TIME, 2),
            "ready": bootstrap.is_ready,
            "routes": route_metrics,
            "websocket_connections": len(getattr(request.app.state, "websocket_clients", set())),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        websocket.app.state.websocket_clients.add(websocket)
        logger.info(
            "WebSocket client connected (total=%d).", len(websocket.app.state.websocket_clients)
        )
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_json({"type": "echo", "data": data, "ts": time.time()})
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        finally:
            websocket.app.state.websocket_clients.discard(websocket)


def register_legacy_events(app: FastAPI) -> None:
    """
    Registers classic startup/shutdown event hooks alongside the lifespan
    context manager, for lightweight signals (e.g. external integrations)
    that specifically depend on the event-based API.
    """

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Startup event fired; lifespan-managed bootstrap is authoritative.")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Shutdown event fired; lifespan-managed teardown is authoritative.")


app = create_app()
