"""
controller_connector.py - Production-grade controller connector with DI, middleware, routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RequestStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ControllerMode(str, Enum):
    SINGLE = "single"
    MULTI_AGENT = "multi_agent"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    mode: ControllerMode = ControllerMode.SINGLE
    max_iterations: int = 10
    budget_usd: Optional[float] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    request_id: str
    status: RequestStatus
    result: Any = None
    plan: Optional[List[Dict[str, Any]]] = None
    steps_executed: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ControllerConnectorConfig:
    mode: ControllerMode = ControllerMode.SINGLE
    max_iterations: int = 10
    enable_planning: bool = True
    enable_review: bool = True
    enable_replanning: bool = True
    max_replan_attempts: int = 3
    execution_timeout: float = 300.0
    enable_middleware: bool = True
    enable_budget: bool = True
    default_budget_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ControllerConnectorError(Exception):
    pass

class RequestTimeoutError(ControllerConnectorError):
    pass

class BudgetExceededError(ControllerConnectorError):
    pass

class PlanningError(ControllerConnectorError):
    pass


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class _Middleware:
    async def before_request(self, request: AgentRequest) -> AgentRequest:
        return request

    async def after_response(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        return response

    async def on_error(self, request: AgentRequest, error: Exception) -> None:
        pass


class LoggingMiddleware(_Middleware):
    async def before_request(self, request: AgentRequest) -> AgentRequest:
        logger.info("Request [%s] content='%.80s'", request.request_id, request.content)
        return request

    async def after_response(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        logger.info(
            "Response [%s] status=%s latency=%.1fms cost=$%.4f",
            request.request_id, response.status.value, response.latency_ms, response.cost_usd,
        )
        return response

    async def on_error(self, request: AgentRequest, error: Exception) -> None:
        logger.error("Request [%s] error: %s", request.request_id, error)


class BudgetMiddleware(_Middleware):
    def __init__(self, budget_manager: Any = None) -> None:
        self._bm = budget_manager

    async def before_request(self, request: AgentRequest) -> AgentRequest:
        if self._bm and request.budget_usd:
            try:
                await self._bm.set_budget(request.request_id, request.budget_usd)
            except Exception:
                pass
        return request

    async def after_response(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        if self._bm:
            try:
                await self._bm.record_spend(request.request_id, response.cost_usd)
            except Exception:
                pass
        return response


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class _Router:
    def __init__(self) -> None:
        self._routes: Dict[str, Callable[..., Any]] = {}

    def register(self, pattern: str, handler: Callable[..., Any]) -> None:
        self._routes[pattern] = handler

    async def route(self, request: AgentRequest) -> Optional[Callable[..., Any]]:
        content_lower = request.content.lower()
        for pattern, handler in self._routes.items():
            if pattern.lower() in content_lower:
                return handler
        return None


# ---------------------------------------------------------------------------
# Planner adapter
# ---------------------------------------------------------------------------

class _PlannerAdapter:
    def __init__(self, planner: Any, prompt_connector: Any, llm_connector: Any) -> None:
        self._planner = planner
        self._prompt = prompt_connector
        self._llm = llm_connector

    async def plan(self, request: AgentRequest) -> List[Dict[str, Any]]:
        if self._planner and hasattr(self._planner, "plan"):
            try:
                return await self._planner.plan(request.content, context=request.context)
            except Exception as exc:
                logger.warning("Planner failed, using LLM fallback: %s", exc)

        if self._llm and self._prompt:
            try:
                prompt_result = await self._prompt.render(
                    "planner_prompt",
                    {"goal": request.content, "tools": [t.get("function", {}).get("name", "") for t in request.tools]},
                )
                messages = [{"role": "user", "content": prompt_result.rendered}]
                response = await self._llm.complete(messages=messages)
                return [{"step": 1, "description": response.text, "status": "pending"}]
            except Exception as exc:
                logger.warning("LLM planning failed: %s", exc)

        return [{"step": 1, "description": request.content, "status": "pending"}]


# ---------------------------------------------------------------------------
# Executor adapter
# ---------------------------------------------------------------------------

class _ExecutorAdapter:
    def __init__(self, executor: Any, llm_connector: Any, capability_connector: Any) -> None:
        self._executor = executor
        self._llm = llm_connector
        self._caps = capability_connector

    async def execute_step(
        self,
        step: Dict[str, Any],
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._executor and hasattr(self._executor, "execute_step"):
            try:
                return await self._executor.execute_step(step, context=context)
            except Exception as exc:
                logger.warning("Executor step failed, using LLM fallback: %s", exc)

        if self._llm:
            try:
                messages = [
                    {"role": "system", "content": "Execute the following step and return the result."},
                    {"role": "user", "content": step.get("description", str(step))},
                ]
                if request.tools:
                    response = await self._llm.complete(messages=messages, tools=request.tools)
                else:
                    response = await self._llm.complete(messages=messages)
                return {
                    "step": step.get("step", 0),
                    "result": response.text,
                    "tokens": response.total_tokens,
                    "cost": response.cost_usd,
                    "tool_calls": [tc.__dict__ for tc in response.tool_calls],
                }
            except Exception as exc:
                return {"step": step.get("step", 0), "result": None, "error": str(exc)}

        return {"step": step.get("step", 0), "result": "No executor available"}


# ---------------------------------------------------------------------------
# Reviewer adapter
# ---------------------------------------------------------------------------

class _ReviewerAdapter:
    def __init__(self, reviewer: Any, llm_connector: Any, prompt_connector: Any) -> None:
        self._reviewer = reviewer
        self._llm = llm_connector
        self._prompt = prompt_connector

    async def review(
        self,
        task: str,
        output: str,
        plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._reviewer and hasattr(self._reviewer, "review"):
            try:
                return await self._reviewer.review(task=task, output=output)
            except Exception as exc:
                logger.warning("Reviewer failed: %s", exc)

        if self._llm and self._prompt:
            try:
                prompt_result = await self._prompt.render(
                    "reviewer_prompt", {"task": task, "output": output}
                )
                messages = [{"role": "user", "content": prompt_result.rendered}]
                response = await self._llm.complete(messages=messages)
                return {
                    "approved": True,
                    "feedback": response.text,
                    "score": 7,
                    "tokens": response.total_tokens,
                }
            except Exception as exc:
                logger.warning("LLM review failed: %s", exc)

        return {"approved": True, "feedback": "No reviewer configured", "score": 5}


# ---------------------------------------------------------------------------
# ControllerConnector – Singleton
# ---------------------------------------------------------------------------

class ControllerConnector:
    _instance: Optional["ControllerConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config = ControllerConnectorConfig()
        self._planner_adapter: Optional[_PlannerAdapter] = None
        self._executor_adapter: Optional[_ExecutorAdapter] = None
        self._reviewer_adapter: Optional[_ReviewerAdapter] = None
        self._router = _Router()
        self._middleware: List[_Middleware] = []
        self._initialized = False
        self._request_history: Dict[str, AgentResponse] = {}
        self._active_requests: Dict[str, AgentRequest] = {}
        self._lock_requests = asyncio.Lock()

        # Injected dependencies
        self._llm: Any = None
        self._planner: Any = None
        self._executor: Any = None
        self._reviewer: Any = None
        self._prompt_connector: Any = None
        self._budget_manager: Any = None
        self._context_engine: Any = None
        self._capability_connector: Any = None
        self._replanner: Any = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ControllerConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "ControllerConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: Optional[ControllerConnectorConfig] = None,
        llm_connector: Any = None,
        planner: Any = None,
        executor: Any = None,
        reviewer: Any = None,
        prompt_connector: Any = None,
        budget_manager: Any = None,
        context_engine: Any = None,
        capability_connector: Any = None,
        replanner: Any = None,
    ) -> "ControllerConnector":
        if self._initialized:
            return self
        if config:
            self._config = config

        self._llm = llm_connector
        self._planner = planner
        self._executor = executor
        self._reviewer = reviewer
        self._prompt_connector = prompt_connector
        self._budget_manager = budget_manager
        self._context_engine = context_engine
        self._capability_connector = capability_connector
        self._replanner = replanner

        # Build adapters
        self._planner_adapter = _PlannerAdapter(planner, prompt_connector, llm_connector)
        self._executor_adapter = _ExecutorAdapter(executor, llm_connector, capability_connector)
        self._reviewer_adapter = _ReviewerAdapter(reviewer, llm_connector, prompt_connector)

        # Default middleware
        self._middleware.append(LoggingMiddleware())
        if budget_manager:
            self._middleware.append(BudgetMiddleware(budget_manager))

        self._initialized = True
        logger.info(
            "ControllerConnector initialized: mode=%s planning=%s review=%s",
            self._config.mode.value,
            self._config.enable_planning,
            self._config.enable_review,
        )
        return self

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def add_middleware(self, mw: _Middleware) -> None:
        self._middleware.append(mw)

    async def _run_before(self, request: AgentRequest) -> AgentRequest:
        for mw in self._middleware:
            try:
                request = await mw.before_request(request)
            except Exception as exc:
                logger.warning("Middleware before_request error: %s", exc)
        return request

    async def _run_after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        for mw in reversed(self._middleware):
            try:
                response = await mw.after_response(request, response)
            except Exception as exc:
                logger.warning("Middleware after_response error: %s", exc)
        return response

    async def _run_on_error(self, request: AgentRequest, error: Exception) -> None:
        for mw in self._middleware:
            try:
                await mw.on_error(request, error)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Handle request
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        content: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        budget_usd: Optional[float] = None,
        max_iterations: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResponse:
        await self._ensure_initialized()

        request = AgentRequest(
            request_id=request_id or str(uuid.uuid4()),
            content=content,
            context=context or {},
            session_id=session_id,
            agent_id=agent_id,
            tools=tools or [],
            budget_usd=budget_usd or self._config.default_budget_usd,
            max_iterations=max_iterations or self._config.max_iterations,
            mode=self._config.mode,
            metadata=kwargs,
        )

        async with self._lock_requests:
            self._active_requests[request.request_id] = request

        t0 = time.perf_counter()
        response = AgentResponse(request_id=request.request_id, status=RequestStatus.PENDING)

        try:
            request = await self._run_before(request)
            response = await asyncio.wait_for(
                self._process_request(request),
                timeout=self._config.execution_timeout,
            )
        except asyncio.TimeoutError:
            response.status = RequestStatus.FAILED
            response.error = f"Request timed out after {self._config.execution_timeout}s"
            await self._run_on_error(request, asyncio.TimeoutError(response.error))
        except BudgetExceededError as exc:
            response.status = RequestStatus.FAILED
            response.error = str(exc)
            await self._run_on_error(request, exc)
        except Exception as exc:
            response.status = RequestStatus.FAILED
            response.error = f"{type(exc).__name__}: {exc}"
            logger.error("Request [%s] unhandled error:\n%s", request.request_id, traceback.format_exc())
            await self._run_on_error(request, exc)
        finally:
            response.latency_ms = (time.perf_counter() - t0) * 1000
            async with self._lock_requests:
                self._active_requests.pop(request.request_id, None)
            self._request_history[request.request_id] = response
            response = await self._run_after(request, response)

        return response

    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        response = AgentResponse(request_id=request.request_id, status=RequestStatus.PLANNING)
        total_tokens = 0
        total_cost = 0.0

        # Optional: route to custom handler
        handler = await self._router.route(request)
        if handler:
            result = await handler(request)
            response.status = RequestStatus.COMPLETED
            response.result = result
            return response

        # Optional: enrich context
        messages: List[Dict[str, Any]] = []
        if self._context_engine and hasattr(self._context_engine, "build_context"):
            try:
                messages = await self._context_engine.build_context(
                    query=request.content,
                    session_id=request.session_id,
                    agent_id=request.agent_id,
                )
            except Exception as exc:
                logger.warning("Context engine failed: %s", exc)

        # Planning
        plan: List[Dict[str, Any]] = []
        if self._config.enable_planning and self._planner_adapter:
            response.status = RequestStatus.PLANNING
            plan = await self._planner_adapter.plan(request)
            response.plan = plan
        else:
            plan = [{"step": 1, "description": request.content, "status": "pending"}]
            response.plan = plan

        # Execution loop
        response.status = RequestStatus.EXECUTING
        results: List[Any] = []
        replan_count = 0
        exec_context = {**request.context, "messages": messages}

        for iteration in range(request.max_iterations):
            pending_steps = [s for s in plan if s.get("status") == "pending"]
            if not pending_steps:
                break

            step = pending_steps[0]
            exec_result = await self._executor_adapter.execute_step(step, request, exec_context)
            step["status"] = "done"
            step["result"] = exec_result.get("result")
            results.append(exec_result)
            total_tokens += exec_result.get("tokens", 0)
            total_cost += exec_result.get("cost", 0.0)
            response.steps_executed += 1
            exec_context["last_result"] = exec_result.get("result")

            # Budget check
            if request.budget_usd and total_cost >= request.budget_usd:
                raise BudgetExceededError(
                    f"Budget ${request.budget_usd:.4f} exceeded at ${total_cost:.4f}"
                )

        # Review
        combined_result = "\n".join(str(r.get("result", "")) for r in results)
        if self._config.enable_review and self._reviewer_adapter:
            response.status = RequestStatus.REVIEWING
            review = await self._reviewer_adapter.review(
                task=request.content,
                output=combined_result,
                plan=plan,
            )
            total_tokens += review.get("tokens", 0)

            # Replan if review not approved
            if not review.get("approved", True) and self._config.enable_replanning:
                if replan_count < self._config.max_replan_attempts:
                    replan_count += 1
                    response.status = RequestStatus.REPLANNING
                    if self._replanner and hasattr(self._replanner, "replan"):
                        try:
                            plan = await self._replanner.replan(
                                original_plan=plan,
                                feedback=review.get("feedback", ""),
                                request=request,
                            )
                            response.plan = plan
                        except Exception as exc:
                            logger.warning("Replanner failed: %s", exc)

        response.result = combined_result
        response.total_tokens = total_tokens
        response.cost_usd = total_cost
        response.status = RequestStatus.COMPLETED
        return response

    # ------------------------------------------------------------------
    # Router registration
    # ------------------------------------------------------------------

    def register_route(self, pattern: str, handler: Callable[..., Any]) -> None:
        self._router.register(pattern, handler)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        components: Dict[str, bool] = {
            "llm_connector": self._llm is not None,
            "planner": self._planner is not None or self._planner_adapter is not None,
            "executor": self._executor is not None or self._executor_adapter is not None,
            "reviewer": self._reviewer is not None,
            "prompt_connector": self._prompt_connector is not None,
            "budget_manager": self._budget_manager is not None,
            "context_engine": self._context_engine is not None,
            "capability_connector": self._capability_connector is not None,
        }

        if self._llm and hasattr(self._llm, "health_check"):
            try:
                llm_health = await self._llm.health_check()
                components["llm_healthy"] = isinstance(llm_health, dict) and any(llm_health.values())
            except Exception:
                components["llm_healthy"] = False

        return {
            "healthy": True,
            "initialized": self._initialized,
            "mode": self._config.mode.value,
            "active_requests": len(self._active_requests),
            "total_handled": len(self._request_history),
            "components": components,
            "middleware_count": len(self._middleware),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        async with self._lock_requests:
            cancelled = list(self._active_requests.keys())
        for rid in cancelled:
            logger.warning("Cancelling active request %s on shutdown", rid)
        self._initialized = False
        logger.info("ControllerConnector shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def get_request_history(self, request_id: str) -> Optional[AgentResponse]:
        return self._request_history.get(request_id)

    def get_active_requests(self) -> List[str]:
        return list(self._active_requests.keys())

    async def cancel_request(self, request_id: str) -> bool:
        async with self._lock_requests:
            if request_id in self._active_requests:
                self._active_requests.pop(request_id)
                self._request_history[request_id] = AgentResponse(
                    request_id=request_id,
                    status=RequestStatus.CANCELLED,
                    error="Cancelled by user",
                )
                return True
        return False

    async def __aenter__(self) -> "ControllerConnector":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return (
            f"ControllerConnector(mode={self._config.mode.value}, "
            f"active={len(self._active_requests)}, "
            f"history={len(self._request_history)})"
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def get_controller_connector() -> ControllerConnector:
    return ControllerConnector.get_instance()


async def handle_request(content: str, **kwargs: Any) -> AgentResponse:
    return await get_controller_connector().handle_request(content, **kwargs)
