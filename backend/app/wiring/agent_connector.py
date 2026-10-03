"""
agent_connector.py - Production-grade agent connector with multi-agent orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    MANAGER = "manager"
    RESEARCH = "research"
    CODER = "coder"
    REVIEWER = "reviewer"
    MEMORY = "memory"
    PLANNER = "planner"
    CUSTOM = "custom"

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    AGENT = "agent"

class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    sender_agent: Optional[str] = None
    receiver_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    thread_id: Optional[str] = None

    def to_llm_message(self) -> Dict[str, Any]:
        role = self.role.value
        if role == "agent":
            role = "assistant"
        return {"role": role, "content": self.content}


@dataclass
class AgentTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    agent_type: AgentType = AgentType.MANAGER
    assigned_to: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    task_id: str
    agent_id: str
    agent_type: AgentType
    success: bool
    output: Any
    error: Optional[str] = None
    messages: List[AgentMessage] = field(default_factory=list)
    tool_calls_made: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentConfig:
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.CUSTOM
    name: str = ""
    system_prompt: str = ""
    model: Optional[str] = None
    provider_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_iterations: int = 10
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    rag_enabled: bool = False
    timeout: float = 300.0
    retry_count: int = 2
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStats:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgentConnectorError(Exception):
    pass

class AgentNotFoundError(AgentConnectorError):
    pass

class AgentExecutionError(AgentConnectorError):
    pass

class AgentTimeoutError(AgentConnectorError):
    pass

class AgentDisabledError(AgentConnectorError):
    pass

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: Dict[AgentType, str] = {
    AgentType.MANAGER: """You are a Manager Agent responsible for orchestrating tasks among specialized agents.
Your role:
- Understand the user's request and decompose it into subtasks.
- Delegate subtasks to appropriate specialized agents.
- Synthesize results from agents into a coherent final response.
- Maintain high-level oversight and quality control.
Always be clear, decisive, and efficient.""",

    AgentType.RESEARCH: """You are a Research Agent specialized in information gathering and analysis.
Your role:
- Search for and retrieve relevant information.
- Analyze data from multiple sources.
- Provide accurate, well-cited summaries.
- Identify key facts, trends, and insights.
Always be thorough, accurate, and cite your sources.""",

    AgentType.CODER: """You are a Coder Agent specialized in software development.
Your role:
- Write clean, efficient, well-documented code.
- Debug and fix code issues.
- Implement algorithms and data structures.
- Follow best practices and design patterns.
- Review and improve existing code.
Always write production-quality code with proper error handling.""",

    AgentType.REVIEWER: """You are a Reviewer Agent specialized in quality assurance and critique.
Your role:
- Review code, documents, and outputs for quality and correctness.
- Identify bugs, logical errors, and improvements.
- Provide constructive, detailed feedback.
- Verify outputs against requirements.
- Ensure standards and best practices are followed.
Always be thorough, fair, and constructive.""",

    AgentType.MEMORY: """You are a Memory Agent specialized in information retention and retrieval.
Your role:
- Store important information for future reference.
- Retrieve relevant past context when needed.
- Summarize and consolidate information.
- Manage memory hierarchies (short-term, long-term).
- Connect related pieces of information.
Always be precise about what is stored and retrieved.""",

    AgentType.PLANNER: """You are a Planner Agent specialized in strategic planning and task decomposition.
Your role:
- Create detailed, actionable plans.
- Break complex goals into step-by-step tasks.
- Identify dependencies and critical paths.
- Estimate effort and resources needed.
- Adapt plans based on new information.
Always create clear, feasible, and well-structured plans.""",
}

# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._stats = AgentStats()
        self._message_history: List[AgentMessage] = []
        self._status = AgentStatus.IDLE
        self._llm_connector: Any = None
        self._tool_connector: Any = None
        self._memory_connector: Any = None

    async def _get_llm(self) -> Any:
        if self._llm_connector is None:
            try:
                from llm_connector import get_connector
                self._llm_connector = get_connector()
            except ImportError:
                logger.warning("llm_connector not available")
        return self._llm_connector

    async def _get_tools(self) -> Any:
        if self._tool_connector is None:
            try:
                from tool_connector import get_tool_connector
                self._tool_connector = get_tool_connector()
            except ImportError:
                logger.warning("tool_connector not available")
        return self._tool_connector

    async def _get_memory(self) -> Any:
        if self._memory_connector is None:
            try:
                from memory_connector import get_memory_connector
                self._memory_connector = get_memory_connector()
            except ImportError:
                logger.warning("memory_connector not available")
        return self._memory_connector

    async def _build_messages(
        self, task: AgentTask, additional_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        system = self.config.system_prompt or _SYSTEM_PROMPTS.get(self.config.agent_type, "")
        if additional_context:
            system += f"\n\nContext:\n{additional_context}"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in self._message_history[-20:]:
            messages.append(msg.to_llm_message())
        messages.append({"role": "user", "content": task.description})
        return messages

    async def _fetch_memory_context(self, task: AgentTask) -> str:
        if not self.config.memory_enabled:
            return ""
        try:
            memory = await self._get_memory()
            if memory:
                await memory._ensure_initialized()
                results = await memory.retrieve(
                    query=task.description, top_k=5,
                    agent_id=self.config.agent_id,
                )
                if results:
                    return "\n".join(f"- {r.entry.content}" for r in results)
        except Exception as exc:
            logger.warning("Memory fetch failed for agent %s: %s", self.config.name, exc)
        return ""

    async def _store_to_memory(self, task: AgentTask, output: str) -> None:
        if not self.config.memory_enabled:
            return
        try:
            memory = await self._get_memory()
            if memory:
                await memory._ensure_initialized()
                await memory.store(
                    content=f"Task: {task.description[:200]}\nResult: {output[:500]}",
                    agent_id=self.config.agent_id,
                    importance=0.6,
                )
        except Exception as exc:
            logger.warning("Memory store failed for agent %s: %s", self.config.name, exc)

    async def run(self, task: AgentTask) -> AgentResult:
        self._status = AgentStatus.RUNNING
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._execute(task),
                timeout=self.config.timeout,
            )
            self._status = AgentStatus.COMPLETED
            self._stats.total_tasks += 1
            self._stats.successful_tasks += 1
            return result
        except asyncio.TimeoutError:
            self._status = AgentStatus.FAILED
            self._stats.total_tasks += 1
            self._stats.failed_tasks += 1
            return AgentResult(
                task_id=task.id,
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=False,
                output=None,
                error=f"Agent timeout after {self.config.timeout}s",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            self._status = AgentStatus.FAILED
            self._stats.total_tasks += 1
            self._stats.failed_tasks += 1
            return AgentResult(
                task_id=task.id,
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=False,
                output=None,
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    async def _execute(self, task: AgentTask) -> AgentResult:
        t0 = time.perf_counter()
        llm = await self._get_llm()
        if not llm:
            return AgentResult(
                task_id=task.id,
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=False,
                output=None,
                error="LLM connector unavailable",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        memory_ctx = await self._fetch_memory_context(task)
        messages = await self._build_messages(task, additional_context=memory_ctx)

        tool_schemas = []
        if self.config.tools:
            tools_connector = await self._get_tools()
            if tools_connector:
                tool_schemas = tools_connector.get_openai_schemas(names=self.config.tools)

        kwargs: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.provider_key:
            kwargs["provider_key"] = self.config.provider_key

        total_tokens = 0
        total_cost = 0.0
        tool_calls_made = 0

        try:
            if tool_schemas:
                tools_connector = await self._get_tools()

                async def _tool_executor(tc: Any) -> Any:
                    nonlocal tool_calls_made
                    tool_calls_made += 1
                    if tools_connector:
                        from tool_connector import ToolCall as TC
                        call = TC(id=tc.id, name=tc.name, arguments=tc.arguments)
                        result = await tools_connector.execute_tool_call(call)
                        return result.output if result.success else f"Error: {result.error}"
                    return "Tool executor unavailable"

                response = await llm.complete_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_executor=_tool_executor,
                    max_rounds=self.config.max_iterations,
                    **kwargs,
                )
            else:
                response = await llm.complete(messages=messages, **kwargs)

            output = response.text
            total_tokens = response.total_tokens
            total_cost = response.cost_usd

            agent_msg = AgentMessage(
                role=MessageRole.AGENT,
                content=output,
                sender_agent=self.config.agent_id,
            )
            self._message_history.append(agent_msg)

            await self._store_to_memory(task, output)

            self._stats.total_tokens += total_tokens
            self._stats.total_cost_usd += total_cost
            latency = (time.perf_counter() - t0) * 1000
            self._stats.total_latency_ms += latency

            return AgentResult(
                task_id=task.id,
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=True,
                output=output,
                messages=[agent_msg],
                tool_calls_made=tool_calls_made,
                tokens_used=total_tokens,
                cost_usd=total_cost,
                latency_ms=latency,
            )
        except Exception as exc:
            raise AgentExecutionError(f"Agent {self.config.name} execution failed: {exc}") from exc

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def stats(self) -> AgentStats:
        return self._stats


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------

class ManagerAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.MANAGER,
            name="ManagerAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.MANAGER],
            temperature=0.5,
        )
        super().__init__(cfg)

    async def delegate(
        self,
        subtask: str,
        agent_type: AgentType,
        connector: "AgentConnector",
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        task = AgentTask(
            description=subtask,
            agent_type=agent_type,
            context=context or {},
        )
        return await connector.run_agent(agent_type=agent_type, task=task)


class ResearchAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.RESEARCH,
            name="ResearchAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.RESEARCH],
            tools=["web_search"],
            temperature=0.3,
        )
        super().__init__(cfg)


class CoderAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.CODER,
            name="CoderAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.CODER],
            tools=["calculator", "terminal", "file_reader"],
            temperature=0.2,
            max_tokens=8192,
        )
        super().__init__(cfg)


class ReviewerAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.REVIEWER,
            name="ReviewerAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.REVIEWER],
            temperature=0.3,
        )
        super().__init__(cfg)


class MemoryAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.MEMORY,
            name="MemoryAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.MEMORY],
            memory_enabled=True,
            temperature=0.2,
        )
        super().__init__(cfg)

    async def store(self, content: str, importance: float = 0.7, **kwargs: Any) -> bool:
        try:
            memory = await self._get_memory()
            if memory:
                await memory._ensure_initialized()
                await memory.store(content=content, importance=importance, **kwargs)
                return True
        except Exception as exc:
            logger.warning("MemoryAgent store failed: %s", exc)
        return False

    async def recall(self, query: str, top_k: int = 5) -> List[str]:
        try:
            memory = await self._get_memory()
            if memory:
                await memory._ensure_initialized()
                results = await memory.retrieve(query=query, top_k=top_k)
                return [r.entry.content for r in results]
        except Exception as exc:
            logger.warning("MemoryAgent recall failed: %s", exc)
        return []


class PlannerAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        cfg = config or AgentConfig(
            agent_type=AgentType.PLANNER,
            name="PlannerAgent",
            system_prompt=_SYSTEM_PROMPTS[AgentType.PLANNER],
            temperature=0.4,
        )
        super().__init__(cfg)

    async def create_plan(self, goal: str) -> List[Dict[str, Any]]:
        task = AgentTask(
            description=f"Create a detailed step-by-step plan for: {goal}\n"
                        f"Return a JSON array of steps with 'step', 'description', 'agent_type' keys.",
            agent_type=AgentType.PLANNER,
        )
        result = await self.run(task)
        if result.success and result.output:
            try:
                import re
                json_match = re.search(r'\[.*\]', result.output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass
        return [{"step": 1, "description": goal, "agent_type": "manager"}]


# ---------------------------------------------------------------------------
# AgentConnector – Singleton
# ---------------------------------------------------------------------------

class AgentConnector:
    _instance: Optional["AgentConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._type_to_agent: Dict[AgentType, str] = {}
        self._initialized = False
        self._message_bus: asyncio.Queue = asyncio.Queue()
        self._stats: Dict[str, AgentStats] = {}
        self._event_hooks: List[Callable] = []

    @classmethod
    def get_instance(cls) -> "AgentConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "AgentConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def initialize(self, register_defaults: bool = True) -> "AgentConnector":
        if self._initialized:
            return self
        if register_defaults:
            self._register_defaults()
        self._initialized = True
        logger.info("AgentConnector initialized with %d agents", len(self._agents))
        return self

    def _register_defaults(self) -> None:
        self.register_agent(ManagerAgent())
        self.register_agent(ResearchAgent())
        self.register_agent(CoderAgent())
        self.register_agent(ReviewerAgent())
        self.register_agent(MemoryAgent())
        self.register_agent(PlannerAgent())

    def _ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent: BaseAgent,
        agent_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        aid = agent_id or agent.config.agent_id
        if aid in self._agents and not overwrite:
            logger.warning("Agent already registered: %s", aid)
            return aid
        self._agents[aid] = agent
        self._type_to_agent[agent.config.agent_type] = aid
        self._stats[aid] = agent.stats
        logger.debug("Registered agent: %s type=%s", aid, agent.config.agent_type.value)
        return aid

    def register_custom_agent(
        self,
        config: AgentConfig,
        agent_class: type = BaseAgent,
        overwrite: bool = False,
    ) -> str:
        agent = agent_class(config)
        return self.register_agent(agent, overwrite=overwrite)

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            atype = agent.config.agent_type
            if self._type_to_agent.get(atype) == agent_id:
                del self._type_to_agent[atype]
            self._stats.pop(agent_id, None)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def get_agent_by_type(self, agent_type: AgentType) -> Optional[BaseAgent]:
        aid = self._type_to_agent.get(agent_type)
        return self._agents.get(aid) if aid else None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        task: AgentTask,
        agent_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
    ) -> AgentResult:
        self._ensure()
        agent: Optional[BaseAgent] = None

        if agent_id:
            agent = self._agents.get(agent_id)
        elif agent_type:
            agent = self.get_agent_by_type(agent_type)

        if not agent:
            return AgentResult(
                task_id=task.id,
                agent_id=agent_id or "",
                agent_type=agent_type or AgentType.CUSTOM,
                success=False,
                output=None,
                error=f"Agent not found: {agent_id or agent_type}",
            )

        if not agent.config.enabled:
            return AgentResult(
                task_id=task.id,
                agent_id=agent.config.agent_id,
                agent_type=agent.config.agent_type,
                success=False,
                output=None,
                error=f"Agent disabled: {agent.config.name}",
            )

        await self._fire_event("task_start", task, agent)
        result = await agent.run(task)
        await self._fire_event("task_end", task, agent, result)
        return result

    async def run_task(
        self,
        description: str,
        agent_type: AgentType = AgentType.MANAGER,
        context: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        task = AgentTask(
            description=description,
            agent_type=agent_type,
            context=context or {},
            priority=priority,
            metadata={"session_id": session_id} if session_id else {},
        )
        return await self.run_agent(task=task, agent_type=agent_type)

    async def run_parallel(
        self,
        tasks: List[Tuple[AgentTask, AgentType]],
    ) -> List[AgentResult]:
        coros = [self.run_agent(task=t, agent_type=at) for t, at in tasks]
        return await asyncio.gather(*coros, return_exceptions=False)

    async def run_pipeline(
        self,
        description: str,
        pipeline: List[AgentType],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AgentResult]:
        results: List[AgentResult] = []
        current_context = dict(context or {})
        current_input = description

        for agent_type in pipeline:
            task = AgentTask(
                description=current_input,
                agent_type=agent_type,
                context=current_context,
            )
            result = await self.run_agent(task=task, agent_type=agent_type)
            results.append(result)
            if result.success and result.output:
                current_input = result.output
                current_context["previous_output"] = result.output

        return results

    async def manager_run(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Run through manager agent which orchestrates other agents."""
        manager = self.get_agent_by_type(AgentType.MANAGER)
        if not manager or not isinstance(manager, ManagerAgent):
            return await self.run_task(description, agent_type=AgentType.MANAGER, context=context)

        planner = self.get_agent_by_type(AgentType.PLANNER)
        plan: List[Dict[str, Any]] = []
        if planner and isinstance(planner, PlannerAgent):
            try:
                plan = await planner.create_plan(description)
            except Exception as exc:
                logger.warning("Planning failed: %s", exc)

        if not plan:
            return await self.run_task(description, agent_type=AgentType.MANAGER, context=context)

        results: List[AgentResult] = []
        for step in plan[:5]:
            step_desc = step.get("description", description)
            step_type_str = step.get("agent_type", "manager")
            try:
                step_type = AgentType(step_type_str)
            except ValueError:
                step_type = AgentType.MANAGER
            result = await self.run_task(step_desc, agent_type=step_type, context=context)
            results.append(result)
            if not result.success:
                break

        combined_output = "\n\n".join(
            f"Step {i+1}: {r.output}" for i, r in enumerate(results) if r.success and r.output
        )
        final_task = AgentTask(
            description=f"Synthesize these results for: {description}\n\nResults:\n{combined_output}",
            agent_type=AgentType.MANAGER,
            context=context or {},
        )
        return await self.run_agent(task=final_task, agent_type=AgentType.MANAGER)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event_hook(self, hook: Callable) -> None:
        self._event_hooks.append(hook)

    async def _fire_event(self, event: str, *args: Any) -> None:
        for hook in self._event_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(event, *args)
                else:
                    hook(event, *args)
            except Exception as exc:
                logger.warning("Event hook error: %s", exc)

    # ------------------------------------------------------------------
    # Info & health
    # ------------------------------------------------------------------

    def list_agents(self) -> List[Dict[str, Any]]:
        self._ensure()
        return [
            {
                "id": aid,
                "name": agent.config.name,
                "type": agent.config.agent_type.value,
                "status": agent.status.value,
                "enabled": agent.config.enabled,
                "tools": agent.config.tools,
                "total_tasks": agent.stats.total_tasks,
                "success_rate": (
                    agent.stats.successful_tasks / agent.stats.total_tasks
                    if agent.stats.total_tasks else 0.0
                ),
            }
            for aid, agent in self._agents.items()
        ]

    async def health_check(self) -> Dict[str, Any]:
        self._ensure()
        agent_health = {}
        for aid, agent in self._agents.items():
            agent_health[aid] = {
                "name": agent.config.name,
                "type": agent.config.agent_type.value,
                "status": agent.status.value,
                "enabled": agent.config.enabled,
                "total_tasks": agent.stats.total_tasks,
                "success_rate": (
                    agent.stats.successful_tasks / agent.stats.total_tasks
                    if agent.stats.total_tasks else 0.0
                ),
                "total_cost_usd": agent.stats.total_cost_usd,
            }
        total_tasks = sum(a.stats.total_tasks for a in self._agents.values())
        total_success = sum(a.stats.successful_tasks for a in self._agents.values())
        return {
            "healthy": True,
            "total_agents": len(self._agents),
            "enabled_agents": sum(1 for a in self._agents.values() if a.config.enabled),
            "total_tasks": total_tasks,
            "overall_success_rate": total_success / total_tasks if total_tasks else 0.0,
            "agents": agent_health,
        }

    def __repr__(self) -> str:
        return f"AgentConnector(agents={list(self._agents.keys())})"


def get_agent_connector() -> AgentConnector:
    connector = AgentConnector.get_instance()
    if not connector._initialized:
        connector.initialize()
    return connector


async def run_task(description: str, agent_type: AgentType = AgentType.MANAGER, **kwargs: Any) -> AgentResult:
    return await get_agent_connector().run_task(description=description, agent_type=agent_type, **kwargs)


async def manager_run(description: str, **kwargs: Any) -> AgentResult:
    return await get_agent_connector().manager_run(description=description, **kwargs)
