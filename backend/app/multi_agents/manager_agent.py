from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.multi_agents.base_agent import (
    Agent,
    AgentMetadata,
    AgentPriority,
    AgentResult,
    AgentTask,
    HealthCheckResult,
    HealthStatus,
)
from app.multi_agents.delegation_manager import (
    DelegationManager,
    NoEligibleAgentError,
)
from app.multi_agents.shared_blackboard import SharedBlackboard, TaskState


class ManagerError(Exception):
    pass


class DecompositionError(ManagerError):
    pass


class SubtaskState(Enum):
    PENDING = "pending"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Subtask:
    subtask_id: str
    description: str
    action: str
    parameters: dict[str, Any]
    required_capability: Optional[str]
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    state: SubtaskState = SubtaskState.PENDING
    assignment_id: Optional[str] = None
    result: Optional[AgentResult] = None


@dataclass
class Plan:
    plan_id: str
    goal: str
    subtasks: dict[str, Subtask]
    created_at_epoch: float = field(default_factory=time.time)
    replan_count: int = 0


TaskDecomposer = Any


class ManagerAgent(Agent):
    def __init__(
        self,
        delegation_manager: DelegationManager,
        blackboard: SharedBlackboard,
        decomposer: Optional[TaskDecomposer] = None,
        max_replans: int = 2,
        metadata: Optional[AgentMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or AgentMetadata(
                name="manager_agent",
                version="1.0.0",
                description="Decomposes goals into subtasks and delegates to specialized agents",
                priority=AgentPriority.HIGH,
                capabilities=("planning", "delegation", "monitoring"),
                timeout_seconds=600.0,
                approval_required=False,
                max_concurrent_tasks=5,
            )
        )
        self._delegation_manager = delegation_manager
        self._blackboard = blackboard
        self._decomposer = decomposer
        self._plans: dict[str, Plan] = {}

    async def _on_initialize(self) -> None:
        self._plans = {}

    async def _on_shutdown(self) -> None:
        self._plans.clear()

    async def _on_health_check(self) -> HealthCheckResult:
        stuck_plans = [
            p for p in self._plans.values()
            if any(s.state == SubtaskState.PENDING for s in p.subtasks.values())
            and time.time() - p.created_at_epoch > 3600
        ]
        if stuck_plans:
            return HealthCheckResult(
                agent_name=self.name,
                status=HealthStatus.DEGRADED,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=f"{len(stuck_plans)} plans appear stalled",
            )
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

        if action == "run_goal":
            return await self.run_goal(str(params["goal"]), params.get("context", {}))
        if action == "get_plan_status":
            return await self.get_plan_status(str(params["plan_id"]))
        if action == "replan":
            return await self.replan(str(params["plan_id"]))

        raise ValueError(f"unknown manager action: {action}")

    def decompose_goal(self, goal: str, context: dict[str, Any]) -> Plan:
        plan_id = f"plan-{uuid.uuid4().hex[:12]}"

        if self._decomposer is not None:
            raw_subtasks = self._decomposer(goal, context)
        else:
            raw_subtasks = self._default_decompose(goal, context)

        subtasks: dict[str, Subtask] = {}
        for raw in raw_subtasks:
            subtask_id = raw.get("subtask_id") or f"sub-{uuid.uuid4().hex[:8]}"
            subtasks[subtask_id] = Subtask(
                subtask_id=subtask_id,
                description=raw.get("description", ""),
                action=raw["action"],
                parameters=raw.get("parameters", {}),
                required_capability=raw.get("required_capability"),
                depends_on=tuple(raw.get("depends_on", ())),
            )

        if not subtasks:
            raise DecompositionError(f"goal '{goal}' produced no subtasks")

        return Plan(plan_id=plan_id, goal=goal, subtasks=subtasks)

    @staticmethod
    def _default_decompose(goal: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "subtask_id": f"sub-{uuid.uuid4().hex[:8]}",
                "description": goal,
                "action": "execute_goal",
                "parameters": {"goal": goal, **context},
                "required_capability": None,
                "depends_on": [],
            }
        ]

    def _ready_subtasks(self, plan: Plan) -> list[Subtask]:
        ready = []
        for subtask in plan.subtasks.values():
            if subtask.state != SubtaskState.PENDING:
                continue
            dependencies_met = all(
                plan.subtasks[dep_id].state == SubtaskState.COMPLETED
                for dep_id in subtask.depends_on
                if dep_id in plan.subtasks
            )
            if dependencies_met:
                ready.append(subtask)
        return ready

    async def run_goal(self, goal: str, context: dict[str, Any]) -> dict[str, Any]:
        plan = self.decompose_goal(goal, context)
        self._plans[plan.plan_id] = plan

        await self._blackboard.set_state(
            f"plan:{plan.plan_id}", {"goal": goal, "status": "running"}, updated_by=self.name
        )

        max_iterations = len(plan.subtasks) * 3 + 1
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            ready = self._ready_subtasks(plan)
            if not ready:
                break

            for subtask in ready:
                await self._delegate_subtask(plan, subtask)

            if all(
                s.state in (SubtaskState.COMPLETED, SubtaskState.FAILED)
                for s in plan.subtasks.values()
            ):
                break

        failed = [s for s in plan.subtasks.values() if s.state == SubtaskState.FAILED]
        if failed and plan.replan_count < 2:
            replanned = await self.replan(plan.plan_id)
            if replanned:
                return await self._summarize_plan(plan)

        return await self._summarize_plan(plan)

    async def _delegate_subtask(self, plan: Plan, subtask: Subtask) -> None:
        await self._blackboard.set_task_status(
            subtask.subtask_id, TaskState.ASSIGNED, detail=subtask.description
        )

        agent_task = AgentTask(
            task_id=subtask.subtask_id,
            action=subtask.action,
            parameters=subtask.parameters,
            context={"plan_id": plan.plan_id, "goal": plan.goal},
        )

        try:
            assignment = await self._delegation_manager.assign_task(
                agent_task, required_capability=subtask.required_capability
            )
            subtask.state = SubtaskState.DELEGATED
            subtask.assignment_id = assignment.assignment_id

            await self._blackboard.set_task_status(
                subtask.subtask_id,
                TaskState.IN_PROGRESS,
                assigned_to=assignment.agent_name,
            )

            result = await self._delegation_manager.execute_assignment(assignment.assignment_id)
            subtask.result = result

            if result.success:
                subtask.state = SubtaskState.COMPLETED
                await self._blackboard.set_task_status(
                    subtask.subtask_id, TaskState.COMPLETED, progress=1.0
                )
                await self._blackboard.put_artifact(
                    name=f"result:{subtask.subtask_id}",
                    content=result.result,
                    content_type="agent_result",
                    created_by=result.agent_name,
                )
            else:
                subtask.state = SubtaskState.FAILED
                await self._blackboard.set_task_status(
                    subtask.subtask_id, TaskState.FAILED, detail=result.error or ""
                )
        except NoEligibleAgentError as exc:
            subtask.state = SubtaskState.FAILED
            await self._blackboard.set_task_status(
                subtask.subtask_id, TaskState.FAILED, detail=str(exc)
            )

    async def replan(self, plan_id: str) -> bool:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ManagerError(f"plan '{plan_id}' not found")

        failed_subtasks = [s for s in plan.subtasks.values() if s.state == SubtaskState.FAILED]
        if not failed_subtasks:
            return False

        plan.replan_count += 1
        for subtask in failed_subtasks:
            subtask.state = SubtaskState.PENDING
            subtask.assignment_id = None
            subtask.result = None

        ready = self._ready_subtasks(plan)
        for subtask in ready:
            await self._delegate_subtask(plan, subtask)

        return True

    async def get_plan_status(self, plan_id: str) -> dict[str, Any]:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ManagerError(f"plan '{plan_id}' not found")
        return await self._summarize_plan(plan)

    async def _summarize_plan(self, plan: Plan) -> dict[str, Any]:
        completed = sum(1 for s in plan.subtasks.values() if s.state == SubtaskState.COMPLETED)
        failed = sum(1 for s in plan.subtasks.values() if s.state == SubtaskState.FAILED)
        total = len(plan.subtasks)
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "total_subtasks": total,
            "completed": completed,
            "failed": failed,
            "replan_count": plan.replan_count,
            "subtasks": {
                sid: {
                    "description": s.description,
                    "state": s.state.value,
                    "error": s.result.error if s.result else None,
                }
                for sid, s in plan.subtasks.items()
            },
        }
