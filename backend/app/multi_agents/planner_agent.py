from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

from app.multi_agents.base_agent import (
    Agent,
    AgentMetadata,
    AgentPriority,
    AgentTask,
    HealthCheckResult,
    HealthStatus,
)


class PlannerError(Exception):
    pass


class CircularDependencyError(PlannerError):
    pass


class PlanNotFoundError(PlannerError):
    pass


class StepState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepPriority(IntEnum):
    LOW = 0
    NORMAL = 50
    HIGH = 100


@dataclass
class PlanStep:
    step_id: str
    description: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    priority: StepPriority = StepPriority.NORMAL
    state: StepState = StepState.PENDING
    fallback_step_ids: tuple[str, ...] = field(default_factory=tuple)
    is_fallback: bool = False


@dataclass
class ExecutionPlan:
    plan_id: str
    goal: str
    steps: dict[str, PlanStep]
    created_at_epoch: float = field(default_factory=time.time)
    version: int = 1


GoalDecomposer = Any


class PlannerAgent(Agent):
    def __init__(
        self,
        decomposer: Optional[GoalDecomposer] = None,
        metadata: Optional[AgentMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or AgentMetadata(
                name="planner_agent",
                version="1.0.0",
                description="Decomposes goals into ordered, dependency-aware execution plans",
                priority=AgentPriority.HIGH,
                capabilities=("planning", "replanning"),
                timeout_seconds=120.0,
                approval_required=False,
            )
        )
        self._decomposer = decomposer
        self._plans: dict[str, ExecutionPlan] = {}

    async def _on_initialize(self) -> None:
        self._plans = {}

    async def _on_shutdown(self) -> None:
        self._plans.clear()

    async def _on_health_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            agent_name=self.name,
            status=HealthStatus.HEALTHY,
            latency_seconds=None,
            checked_at_epoch=time.time(),
            detail=f"active_plans={len(self._plans)}",
        )

    async def _on_execute(self, task: AgentTask) -> Any:
        action = task.action
        params = task.parameters

        if action == "create_plan":
            return self.create_plan(str(params["goal"]), params.get("context", {}))
        if action == "next_steps":
            return self.next_steps(str(params["plan_id"]))
        if action == "mark_step":
            return self.mark_step(
                str(params["plan_id"]),
                str(params["step_id"]),
                StepState(params["state"]),
            )
        if action == "replan":
            return self.replan(str(params["plan_id"]), params.get("reason", ""))
        if action == "get_fallback":
            return self.get_fallback_steps(str(params["plan_id"]), str(params["step_id"]))

        raise ValueError(f"unknown planner action: {action}")

    def create_plan(self, goal: str, context: dict[str, Any]) -> ExecutionPlan:
        plan_id = f"plan-{uuid.uuid4().hex[:12]}"
        raw_steps = (
            self._decomposer(goal, context)
            if self._decomposer is not None
            else self._default_decompose(goal, context)
        )

        steps: dict[str, PlanStep] = {}
        for raw in raw_steps:
            step_id = raw.get("step_id") or f"step-{uuid.uuid4().hex[:8]}"
            steps[step_id] = PlanStep(
                step_id=step_id,
                description=raw.get("description", ""),
                action=raw["action"],
                parameters=raw.get("parameters", {}),
                depends_on=tuple(raw.get("depends_on", ())),
                priority=StepPriority(raw.get("priority", StepPriority.NORMAL)),
                fallback_step_ids=tuple(raw.get("fallback_step_ids", ())),
                is_fallback=bool(raw.get("is_fallback", False)),
            )

        if not steps:
            raise PlannerError(f"goal '{goal}' produced no steps")

        self._validate_dependencies(steps)
        plan = ExecutionPlan(plan_id=plan_id, goal=goal, steps=steps)
        self._plans[plan_id] = plan
        self._refresh_ready_states(plan)
        return plan

    @staticmethod
    def _default_decompose(goal: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "step_id": f"step-{uuid.uuid4().hex[:8]}",
                "description": goal,
                "action": "execute_goal",
                "parameters": {"goal": goal, **context},
                "depends_on": [],
                "priority": StepPriority.NORMAL,
            }
        ]

    def _validate_dependencies(self, steps: dict[str, PlanStep]) -> None:
        visited: set[str] = set()
        in_progress: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in in_progress:
                raise CircularDependencyError(
                    f"circular dependency detected involving step '{step_id}'"
                )
            in_progress.add(step_id)
            step = steps.get(step_id)
            if step is None:
                raise PlannerError(f"step '{step_id}' referenced but not defined")
            for dep in step.depends_on:
                if dep not in steps:
                    raise PlannerError(
                        f"step '{step_id}' depends on undefined step '{dep}'"
                    )
                visit(dep)
            in_progress.discard(step_id)
            visited.add(step_id)

        for step_id in steps:
            visit(step_id)

    def _refresh_ready_states(self, plan: ExecutionPlan) -> None:
        for step in plan.steps.values():
            if step.state != StepState.PENDING:
                continue
            dependencies_met = all(
                plan.steps[dep].state == StepState.DONE for dep in step.depends_on
            )
            if dependencies_met:
                step.state = StepState.READY

    def _get_plan(self, plan_id: str) -> ExecutionPlan:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise PlanNotFoundError(f"plan '{plan_id}' not found")
        return plan

    def next_steps(self, plan_id: str) -> list[PlanStep]:
        plan = self._get_plan(plan_id)
        self._refresh_ready_states(plan)
        ready = [s for s in plan.steps.values() if s.state == StepState.READY]
        return sorted(ready, key=lambda s: int(s.priority), reverse=True)

    def mark_step(self, plan_id: str, step_id: str, state: StepState) -> PlanStep:
        plan = self._get_plan(plan_id)
        step = plan.steps.get(step_id)
        if step is None:
            raise PlannerError(f"step '{step_id}' not found in plan '{plan_id}'")
        step.state = state
        self._refresh_ready_states(plan)
        return step

    def get_fallback_steps(self, plan_id: str, step_id: str) -> list[PlanStep]:
        plan = self._get_plan(plan_id)
        step = plan.steps.get(step_id)
        if step is None:
            raise PlannerError(f"step '{step_id}' not found in plan '{plan_id}'")
        return [plan.steps[fid] for fid in step.fallback_step_ids if fid in plan.steps]

    def replan(self, plan_id: str, reason: str = "") -> ExecutionPlan:
        plan = self._get_plan(plan_id)
        failed_steps = [s for s in plan.steps.values() if s.state == StepState.FAILED]

        if not failed_steps:
            plan.version += 1
            return plan

        for step in failed_steps:
            fallback_steps = self.get_fallback_steps(plan_id, step.step_id)
            if fallback_steps:
                for fallback in fallback_steps:
                    fallback.state = StepState.PENDING
                step.state = StepState.SKIPPED
            else:
                step.state = StepState.PENDING

        plan.version += 1
        self._refresh_ready_states(plan)
        return plan

    def prioritize(self, plan_id: str, step_id: str, priority: StepPriority) -> PlanStep:
        plan = self._get_plan(plan_id)
        step = plan.steps.get(step_id)
        if step is None:
            raise PlannerError(f"step '{step_id}' not found in plan '{plan_id}'")
        step.priority = priority
        return step

    def get_plan(self, plan_id: str) -> ExecutionPlan:
        return self._get_plan(plan_id)

    def is_complete(self, plan_id: str) -> bool:
        plan = self._get_plan(plan_id)
        return all(
            s.state in (StepState.DONE, StepState.SKIPPED) for s in plan.steps.values()
        )
