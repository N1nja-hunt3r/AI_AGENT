"""
kernel_initializer.py

Initializes the core reasoning kernel: Controller, Planner, Executor,
Router, PromptManager, ContextEngine, kernel-level Middleware chain and
Replanner, wiring them together and exposing aggregate health checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus, Registry, get_registry
from app.wiring.provider_initializer import create_llm_client

logger = logging.getLogger("kernel_initializer")

KernelMiddlewareFn = Callable[
    [Dict[str, Any], Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
    Awaitable[Dict[str, Any]],
]


class PromptManager:
    """Manages prompt templates used throughout the kernel."""

    def __init__(self) -> None:
        self._templates: Dict[str, str] = {}

    def register(self, name: str, template: str) -> None:
        self._templates[name] = template

    def render(self, name: str, **kwargs: Any) -> str:
        template = self._templates.get(name, "")
        return template.format(**kwargs)

    async def health_check(self) -> bool:
        return True


class ContextEngine:
    """Maintains conversational and task context across kernel cycles."""

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._context)

    async def health_check(self) -> bool:
        return True


class Planner:
    """Produces an execution plan for a given goal/context."""

    def __init__(self) -> None:
        self._plans: Dict[str, Dict[str, Any]] = {}

    async def plan(self, goal: str, context: ContextEngine) -> List[Dict[str, Any]]:
        return [{"step": 1, "action": "execute", "goal": goal}]

    async def create_plan(
        self,
        session_id: str,
        user_id: str,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
    ) -> Dict[str, Any]:
        plan_id = f"plan_{uuid4().hex[:12]}"
        steps = [{"step_id": f"step_{i}", "description": f"Step {i} of {max_steps}", "tool": None, "depends_on": []} for i in range(1, max_steps + 1)]
        plan = {"plan_id": plan_id, "session_id": session_id, "steps": steps, "goal": goal}
        self._plans[plan_id] = plan
        return plan

    async def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        return self._plans.get(plan_id)

    async def health_check(self) -> bool:
        return True


class Replanner:
    """Revises an existing plan when execution deviates or fails."""

    async def replan(
        self,
        original_plan: List[Dict[str, Any]],
        failure_reason: str,
        context: ContextEngine,
    ) -> List[Dict[str, Any]]:
        return original_plan

    async def health_check(self) -> bool:
        return True


class Executor:
    """Executes individual plan steps."""

    async def execute(self, step: Dict[str, Any], context: ContextEngine) -> Dict[str, Any]:
        return {"step": step, "status": "completed"}

    async def execute_steps(
        self,
        session_id: str,
        user_id: str,
        steps: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for step in steps:
            if dry_run:
                results.append({"step_id": step.get("step_id", ""), "status": "dry_run", "output": None, "error": None})
            else:
                results.append({"step_id": step.get("step_id", ""), "status": "completed", "output": {"result": "success"}, "error": None})
        return results

    async def health_check(self) -> bool:
        return True


class KernelRouter:
    """Routes a request to the appropriate agent/capability inside the kernel."""

    def __init__(self) -> None:
        self._routes: Dict[str, str] = {}

    def add_route(self, intent: str, target: str) -> None:
        self._routes[intent] = target

    def resolve(self, intent: str) -> Optional[str]:
        return self._routes.get(intent)

    async def health_check(self) -> bool:
        return True


class KernelMiddlewareChain:
    """A chain of kernel-level middleware applied around request handling."""

    def __init__(self) -> None:
        self._chain: List[KernelMiddlewareFn] = []

    def use(self, middleware: KernelMiddlewareFn) -> None:
        self._chain.append(middleware)

    async def run(
        self,
        payload: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        async def build(index: int) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
            if index >= len(self._chain):
                return handler

            next_fn = await build(index + 1)

            async def current(data: Dict[str, Any]) -> Dict[str, Any]:
                return await self._chain[index](data, next_fn)

            return current

        pipeline = await build(0)
        return await pipeline(payload)

    async def health_check(self) -> bool:
        return True


class Controller:
    """Top-level kernel controller orchestrating plan -> execute -> replan."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        replanner: Replanner,
        router: KernelRouter,
        context_engine: ContextEngine,
        prompt_manager: PromptManager,
        middleware: KernelMiddlewareChain,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.replanner = replanner
        self.router = router
        self.context_engine = context_engine
        self.prompt_manager = prompt_manager
        self.middleware = middleware
        self.last_usage: Dict[str, int] = {}

    async def run(self, goal: str) -> Dict[str, Any]:
        async def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            plan = await self.planner.plan(payload["goal"], self.context_engine)
            results: List[Dict[str, Any]] = []
            for step in plan:
                try:
                    result = await self.executor.execute(step, self.context_engine)
                except Exception as exc:  # pragma: no cover - defensive
                    plan = await self.replanner.replan(plan, str(exc), self.context_engine)
                    continue
                results.append(result)
            return {"goal": payload["goal"], "results": results}

        return await self.middleware.run({"goal": goal}, handler)

    async def _get_llm_client(self) -> Any:
        client = getattr(self, "_llm_client", None)
        if client is None:
            from app.integration.dependency_container import get_container
            container = get_container()
            existing = container.try_resolve("provider:registry")
            client = create_llm_client(registry=existing)
            self._llm_client = client
        return client

    async def handle_chat(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            prompt, system_prompt = self._build_chat_prompt(messages, memory_context)
            llm_client = await self._get_llm_client()
            content = await llm_client.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens or 512,
            )
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return {"content": content, "usage": self.last_usage}
        except Exception as exc:
            logger.exception("handle_chat failed")
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return {
                "content": f"Error: {exc}",
                "usage": self.last_usage,
            }

    async def stream_chat(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        chunk_id = str(uuid4())
        t0 = time.monotonic()
        token_count = 0
        first_token_yielded = False

        try:
            t1 = time.monotonic()
            prompt, system_prompt = self._build_chat_prompt(messages, memory_context)
            t2 = time.monotonic()
            logger.info(
                "stream_chat session=%s stage=build_prompt duration_ms=%.1f msg_count=%d",
                session_id, (t2 - t1) * 1000, len(messages),
            )

            llm_client = await self._get_llm_client()
            t3 = time.monotonic()
            logger.info(
                "stream_chat session=%s stage=get_llm_client duration_ms=%.1f",
                session_id, (t3 - t2) * 1000,
            )

            logger.info(
                "stream_chat session=%s stage=before_llm_call setup_ms=%.1f",
                session_id, (t3 - t0) * 1000,
            )

            async for token in llm_client.stream(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens or 16384,
            ):
                if not first_token_yielded:
                    first_token_yielded = True
                    ttft = time.monotonic() - t3
                    logger.info(
                        "stream_chat session=%s stage=first_token ttft_s=%.2f",
                        session_id, ttft,
                    )
                yield {"delta": token, "id": chunk_id, "done": False}
                token_count += 1

            t4 = time.monotonic()
            total = t4 - t0
            logger.info(
                "stream_chat session=%s stage=complete token_count=%d total_s=%.2f tokens_per_sec=%.1f",
                session_id, token_count, total, token_count / total if total > 0 else 0,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception(
                "stream_chat session=%s stage=error after_s=%.2f token_count=%d first_token_yielded=%s",
                session_id, elapsed, token_count, first_token_yielded,
            )
            fallback = f"Error: {exc}"
            for char in fallback:
                yield {"delta": char, "id": chunk_id, "done": False}
                token_count += 1

        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        }
        yield {"delta": "", "id": chunk_id, "done": True}

        if first_token_yielded:
            logger.info(
                "stream_chat session=%s stage=done ttft_s=%.2f total_s=%.2f",
                session_id, time.monotonic() - t3, time.monotonic() - t0,
            )

    def _build_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_history: int = 20,
        max_prompt_chars: int = 100_000,
    ) -> Tuple[str, Optional[str]]:
        system_prompt = None
        parts: List[str] = []

        # Include memory context as system-level context
        if memory_context:
            memory_lines: List[str] = []
            for mem in memory_context[-10:]:  # only last 10 memory items
                role = mem.get("role", "user")
                content = mem.get("content", "")
                memory_lines.append(f"[Previous {role}]: {content}")
            if memory_lines:
                parts.append("## Conversation History\n" + "\n".join(memory_lines))

        # Limit to last N messages
        for msg in messages[-max_history:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        prompt = "\n".join(parts)

        # Safeguard: truncate prompt to max_prompt_chars
        if len(prompt) > max_prompt_chars:
            prompt = prompt[-max_prompt_chars:]
            prompt = "## Truncated (showing recent context)\n" + prompt

        return prompt, system_prompt
        if not prompt.endswith(":"):
            prompt += "\nAssistant:"
        return prompt, system_prompt

    async def health_check(self) -> bool:
        checks = await asyncio.gather(
            self.planner.health_check(),
            self.executor.health_check(),
            self.replanner.health_check(),
            self.router.health_check(),
            self.context_engine.health_check(),
            self.prompt_manager.health_check(),
            self.middleware.health_check(),
        )
        return all(checks)


class KernelInitializer:
    """Builds and wires the full kernel component graph."""

    def __init__(self, container: Optional[DependencyContainer] = None) -> None:
        self._container = container or get_container()
        self._registry: Registry = get_registry("kernel")
        self._controller: Optional[Controller] = None
        self._initialized = False

    async def initialize(self) -> Controller:
        if self._initialized and self._controller is not None:
            return self._controller

        prompt_manager = PromptManager()
        context_engine = ContextEngine()
        planner = Planner()
        replanner = Replanner()
        executor = Executor()
        router = KernelRouter()
        middleware = KernelMiddlewareChain()

        controller = Controller(
            planner=planner,
            executor=executor,
            replanner=replanner,
            router=router,
            context_engine=context_engine,
            prompt_manager=prompt_manager,
            middleware=middleware,
        )

        for name, component in (
            ("prompt_manager", prompt_manager),
            ("context_engine", context_engine),
            ("planner", planner),
            ("replanner", replanner),
            ("executor", executor),
            ("router", router),
            ("middleware", middleware),
            ("controller", controller),
        ):
            self._registry.register(
                name,
                component,
                metadata={"kind": "kernel"},
                health_check=getattr(component, "health_check", None),
                overwrite=True,
            )
            self._container.register_singleton(f"kernel:{name}", component)

        self._controller = controller
        self._initialized = True
        logger.info("Kernel initialized successfully.")
        return controller

    @property
    def controller(self) -> Optional[Controller]:
        return self._controller

    async def health_check(self) -> Dict[str, HealthStatus]:
        return await self._registry.health_check_all()

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_kernel_initializer: Optional[KernelInitializer] = None


def get_kernel_initializer() -> KernelInitializer:
    global _kernel_initializer
    if _kernel_initializer is None:
        _kernel_initializer = KernelInitializer()
    return _kernel_initializer
