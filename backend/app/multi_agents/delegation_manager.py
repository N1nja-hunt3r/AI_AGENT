from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.multi_agents.base_agent import (
    Agent,
    AgentResult,
    AgentTask,
)


class DelegationError(Exception):
    pass


class NoEligibleAgentError(DelegationError):
    pass


class AssignmentNotFoundError(DelegationError):
    pass


class AssignmentState(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REASSIGNED = "reassigned"
    ABANDONED = "abandoned"


class SelectionStrategy(Enum):
    LEAST_LOADED = "least_loaded"
    HIGHEST_PRIORITY = "highest_priority"
    ROUND_ROBIN = "round_robin"


@dataclass
class AgentMetrics:
    agent_name: str
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_seconds: float = 0.0
    last_assigned_epoch: Optional[float] = None
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def average_duration_seconds(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_duration_seconds / self.tasks_completed


@dataclass
class Assignment:
    assignment_id: str
    task: AgentTask
    agent_name: str
    state: AssignmentState
    assigned_at_epoch: float
    attempt: int = 1
    result: Optional[AgentResult] = None
    error: Optional[str] = None
    completed_at_epoch: Optional[float] = None


class DelegationManager:
    def __init__(
        self,
        max_retries: int = 2,
        max_consecutive_failures: int = 3,
        selection_strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED,
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self._metrics: dict[str, AgentMetrics] = {}
        self._assignments: dict[str, Assignment] = {}
        self._max_retries = max_retries
        self._max_consecutive_failures = max_consecutive_failures
        self._selection_strategy = selection_strategy
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        self._sequence = 0

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}-{int(time.time() * 1000)}"

    async def register_agent(self, agent: Agent) -> None:
        async with self._lock:
            self._agents[agent.name] = agent
            if agent.name not in self._metrics:
                self._metrics[agent.name] = AgentMetrics(agent_name=agent.name)

    async def unregister_agent(self, agent_name: str) -> None:
        async with self._lock:
            self._agents.pop(agent_name, None)

    async def _eligible_agents(self, required_capability: Optional[str]) -> list[Agent]:
        async with self._lock:
            candidates = list(self._agents.values())

        eligible = []
        for agent in candidates:
            if not agent.is_available:
                continue
            if required_capability is not None and not agent.supports(required_capability):
                continue
            metrics = self._metrics.get(agent.name)
            if metrics and metrics.consecutive_failures >= self._max_consecutive_failures:
                continue
            eligible.append(agent)
        return eligible

    async def select_agent(
        self, required_capability: Optional[str] = None
    ) -> Agent:
        eligible = await self._eligible_agents(required_capability)
        if not eligible:
            raise NoEligibleAgentError(
                f"no eligible agent found for capability '{required_capability}'"
            )

        if self._selection_strategy == SelectionStrategy.HIGHEST_PRIORITY:
            return max(eligible, key=lambda a: int(a.priority))

        if self._selection_strategy == SelectionStrategy.ROUND_ROBIN:
            async with self._lock:
                self._round_robin_index = (self._round_robin_index + 1) % len(eligible)
                return eligible[self._round_robin_index]

        def load_key(agent: Agent) -> tuple[int, float]:
            metrics = self._metrics.get(agent.name) or AgentMetrics(agent_name=agent.name)
            return (metrics.tasks_assigned - metrics.tasks_completed - metrics.tasks_failed,
                    -metrics.success_rate)

        return min(eligible, key=load_key)

    async def assign_task(
        self, task: AgentTask, required_capability: Optional[str] = None
    ) -> Assignment:
        agent = await self.select_agent(required_capability)
        assignment_id = self._next_id("assign")
        assignment = Assignment(
            assignment_id=assignment_id,
            task=task,
            agent_name=agent.name,
            state=AssignmentState.ASSIGNED,
            assigned_at_epoch=time.time(),
        )
        async with self._lock:
            self._assignments[assignment_id] = assignment
            metrics = self._metrics.setdefault(agent.name, AgentMetrics(agent_name=agent.name))
            metrics.tasks_assigned += 1
            metrics.last_assigned_epoch = time.time()
        return assignment

    async def execute_assignment(self, assignment_id: str) -> AgentResult:
        async with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise AssignmentNotFoundError(f"assignment '{assignment_id}' not found")
            agent = self._agents.get(assignment.agent_name)

        if agent is None:
            return await self._handle_failure(
                assignment_id, "assigned agent is no longer registered"
            )

        assignment.state = AssignmentState.RUNNING
        start = time.monotonic()
        try:
            result = await agent.execute(assignment.task)
            duration = time.monotonic() - start

            async with self._lock:
                metrics = self._metrics.setdefault(
                    agent.name, AgentMetrics(agent_name=agent.name)
                )
                if result.success:
                    metrics.tasks_completed += 1
                    metrics.consecutive_failures = 0
                    metrics.total_duration_seconds += duration
                    assignment.state = AssignmentState.COMPLETED
                else:
                    metrics.tasks_failed += 1
                    metrics.consecutive_failures += 1
                    assignment.state = AssignmentState.FAILED
                    assignment.error = result.error
                assignment.result = result
                assignment.completed_at_epoch = time.time()

            if not result.success and assignment.attempt <= self._max_retries:
                return await self._retry_assignment(assignment)

            return result
        except Exception as exc:
            return await self._handle_failure(assignment_id, str(exc))

    async def _handle_failure(self, assignment_id: str, error: str) -> AgentResult:
        async with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise AssignmentNotFoundError(f"assignment '{assignment_id}' not found")
            assignment.state = AssignmentState.FAILED
            assignment.error = error
            assignment.completed_at_epoch = time.time()
            metrics = self._metrics.setdefault(
                assignment.agent_name, AgentMetrics(agent_name=assignment.agent_name)
            )
            metrics.tasks_failed += 1
            metrics.consecutive_failures += 1

        if assignment.attempt <= self._max_retries:
            return await self._retry_assignment(assignment)

        return AgentResult(
            task_id=assignment.task.task_id,
            agent_name=assignment.agent_name,
            action=assignment.task.action,
            success=False,
            error=error,
            duration_seconds=0.0,
        )

    async def _retry_assignment(self, failed_assignment: Assignment) -> AgentResult:
        try:
            new_agent = await self.select_agent()
        except NoEligibleAgentError:
            return AgentResult(
                task_id=failed_assignment.task.task_id,
                agent_name=failed_assignment.agent_name,
                action=failed_assignment.task.action,
                success=False,
                error="no eligible agent available for retry",
                duration_seconds=0.0,
            )

        new_assignment_id = self._next_id("assign")
        new_assignment = Assignment(
            assignment_id=new_assignment_id,
            task=failed_assignment.task,
            agent_name=new_agent.name,
            state=AssignmentState.ASSIGNED,
            assigned_at_epoch=time.time(),
            attempt=failed_assignment.attempt + 1,
        )
        async with self._lock:
            self._assignments[new_assignment_id] = new_assignment
            failed_assignment.state = AssignmentState.REASSIGNED
            metrics = self._metrics.setdefault(
                new_agent.name, AgentMetrics(agent_name=new_agent.name)
            )
            metrics.tasks_assigned += 1

        return await self.execute_assignment(new_assignment_id)

    async def get_assignment(self, assignment_id: str) -> Assignment:
        async with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise AssignmentNotFoundError(f"assignment '{assignment_id}' not found")
            return assignment

    async def get_metrics(self, agent_name: str) -> AgentMetrics:
        async with self._lock:
            return self._metrics.get(agent_name) or AgentMetrics(agent_name=agent_name)

    async def get_all_metrics(self) -> dict[str, AgentMetrics]:
        async with self._lock:
            return dict(self._metrics)

    async def abandon_assignment(self, assignment_id: str) -> None:
        async with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise AssignmentNotFoundError(f"assignment '{assignment_id}' not found")
            assignment.state = AssignmentState.ABANDONED
            assignment.completed_at_epoch = time.time()
