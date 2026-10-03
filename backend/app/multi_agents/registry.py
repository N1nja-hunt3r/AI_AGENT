"""Agent Registry: registration, discovery, health, messaging, and task assignment hub."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    runtime_checkable,
)

logger = logging.getLogger("registry")


class AgentStatus(str, Enum):
    REGISTERING = "registering"
    INITIALIZING = "initializing"
    IDLE = "idle"
    AVAILABLE = "available"
    BUSY = "busy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    OFFLINE = "offline"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@runtime_checkable
class AgentLike(Protocol):
    async def health_check(self) -> Dict[str, Any]: ...
    async def handle_task(self, task: Any) -> Any: ...
    async def receive_message(self, message: Any, sender: str) -> Any: ...


@runtime_checkable
class CommunicationBusLike(Protocol):
    async def send(self, recipient: str, message: Any, sender: str) -> Any: ...
    async def broadcast(self, message: Any, sender: str, exclude: Optional[Sequence[str]] = None) -> Dict[str, Any]: ...


@runtime_checkable
class SharedBlackboardLike(Protocol):
    async def set(self, key: str, value: Any) -> None: ...
    async def get(self, key: str) -> Any: ...


@runtime_checkable
class DelegationManagerLike(Protocol):
    async def assign_task(self, agent_id: str, task: Any) -> Any: ...


@dataclass
class AgentMetadata:
    agent_id: str
    name: str
    version: str = "1.0.0"
    capabilities: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.REGISTERING
    instance: Optional[Any] = None
    max_concurrent_tasks: int = 1
    current_task_count: int = 0
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        return (
            self.status in (AgentStatus.IDLE, AgentStatus.AVAILABLE)
            and self.current_task_count < self.max_concurrent_tasks
        )

    def load_factor(self) -> float:
        if self.max_concurrent_tasks <= 0:
            return 1.0
        return self.current_task_count / self.max_concurrent_tasks


@dataclass
class HealthCheckResult:
    agent_id: str
    healthy: bool
    status: AgentStatus
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _version_tuple(version: str) -> tuple:
    parts = []
    for p in version.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _version_satisfies(version: str, min_version: Optional[str]) -> bool:
    if not min_version:
        return True
    return _version_tuple(version) >= _version_tuple(min_version)


class AgentRegistry:
    """Central registry for agent registration, discovery, messaging, and task routing."""

    def __init__(
        self,
        communication_bus: Optional[CommunicationBusLike] = None,
        shared_blackboard: Optional[SharedBlackboardLike] = None,
        delegation_manager: Optional[DelegationManagerLike] = None,
        heartbeat_timeout_seconds: float = 60.0,
    ) -> None:
        self.registry_id: str = f"registry-{uuid.uuid4().hex[:8]}"
        self.communication_bus = communication_bus
        self.shared_blackboard = shared_blackboard
        self.delegation_manager = delegation_manager
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

        self._agents: Dict[str, AgentMetadata] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._rr_counters: Dict[str, int] = {}
        self._initialized = False
        self._metrics: Dict[str, int] = {
            "registrations": 0,
            "unregistrations": 0,
            "messages_sent": 0,
            "broadcasts": 0,
            "tasks_assigned": 0,
            "health_checks": 0,
        }

    # ---------- lifecycle ----------

    async def initialize(self) -> None:
        async with self._async_lock:
            if self._initialized:
                return
            self._initialized = True
            logger.info("Registry %s initialized.", self.registry_id)

    async def shutdown(self) -> None:
        async with self._async_lock:
            with self._lock:
                agent_ids = list(self._agents.keys())
                for agent_id in agent_ids:
                    self._agents[agent_id].status = AgentStatus.SHUTDOWN
            for agent_id in agent_ids:
                instance = self._agents[agent_id].instance
                shutdown_fn = getattr(instance, "shutdown", None)
                if callable(shutdown_fn):
                    try:
                        result = shutdown_fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Error shutting down agent %s: %s", agent_id, exc)
            self._initialized = False
            logger.info("Registry %s shut down.", self.registry_id)

    # ---------- registration ----------

    def register(
        self,
        agent_id: str,
        name: Optional[str] = None,
        instance: Optional[Any] = None,
        capabilities: Optional[Sequence[str]] = None,
        version: str = "1.0.0",
        priority: int = 0,
        dependencies: Optional[Sequence[str]] = None,
        max_concurrent_tasks: int = 1,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMetadata:
        with self._lock:
            meta = AgentMetadata(
                agent_id=agent_id,
                name=name or agent_id,
                version=version,
                capabilities=set(capabilities or []),
                tags=set(tags or []),
                priority=priority,
                dependencies=list(dependencies or []),
                status=AgentStatus.IDLE,
                instance=instance,
                max_concurrent_tasks=max_concurrent_tasks,
                metadata=metadata or {},
            )
            self._agents[agent_id] = meta
            self._metrics["registrations"] += 1

            missing = [d for d in meta.dependencies if d not in self._agents]
            if missing:
                logger.warning(
                    "Agent %s registered with unresolved dependencies: %s", agent_id, missing
                )

            logger.info("Registered agent '%s' (capabilities=%s)", agent_id, meta.capabilities)
            return meta

    async def register_async(
        self,
        agent_id: str,
        name: Optional[str] = None,
        instance: Optional[Any] = None,
        capabilities: Optional[Sequence[str]] = None,
        version: str = "1.0.0",
        priority: int = 0,
        dependencies: Optional[Sequence[str]] = None,
        max_concurrent_tasks: int = 1,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMetadata:
        async with self._async_lock:
            return self.register(
                agent_id, name, instance, capabilities, version, priority,
                dependencies, max_concurrent_tasks, tags, metadata,
            )

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].status = AgentStatus.OFFLINE
                del self._agents[agent_id]
                self._metrics["unregistrations"] += 1
                logger.info("Unregistered agent '%s'", agent_id)
                return True
            return False

    async def unregister_async(self, agent_id: str) -> bool:
        async with self._async_lock:
            return self.unregister(agent_id)

    def get_unresolved_dependencies(self, agent_id: str) -> List[str]:
        with self._lock:
            meta = self._agents.get(agent_id)
            if not meta:
                return []
            return [d for d in meta.dependencies if d not in self._agents]

    # ---------- discovery / lookup ----------

    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        with self._lock:
            return self._agents.get(agent_id)

    def discover(
        self,
        capability: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        min_version: Optional[str] = None,
    ) -> List[AgentMetadata]:
        with self._lock:
            results = list(self._agents.values())

        if capability is not None:
            results = [a for a in results if capability in a.capabilities]
        if tag is not None:
            results = [a for a in results if tag in a.tags]
        if status is not None:
            results = [a for a in results if a.status == status]
        if min_version is not None:
            results = [a for a in results if _version_satisfies(a.version, min_version)]

        return sorted(results, key=lambda a: a.priority, reverse=True)

    def find_by_capability(
        self, capability: str, min_version: Optional[str] = None
    ) -> List[AgentMetadata]:
        return self.discover(capability=capability, min_version=min_version)

    def find_available_agents(self, capability: Optional[str] = None) -> List[AgentMetadata]:
        with self._lock:
            candidates = list(self._agents.values())
        if capability is not None:
            candidates = [a for a in candidates if capability in a.capabilities]
        available = [a for a in candidates if a.is_available()]
        return sorted(available, key=lambda a: (a.load_factor(), -a.priority))

    def find_best_agent(
        self,
        capability: Optional[str] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> Optional[AgentMetadata]:
        exclude_set = set(exclude or [])
        candidates = self.find_available_agents(capability)
        candidates = [a for a in candidates if a.agent_id not in exclude_set]
        if not candidates:
            return None

        max_priority = max(a.priority for a in candidates)
        top_priority = [a for a in candidates if a.priority == max_priority]
        top_priority.sort(key=lambda a: a.load_factor())

        min_load = top_priority[0].load_factor()
        least_loaded = [a for a in top_priority if a.load_factor() == min_load]

        if len(least_loaded) == 1:
            return least_loaded[0]

        rr_key = capability or "__any__"
        with self._lock:
            idx = self._rr_counters.get(rr_key, 0)
            chosen = least_loaded[idx % len(least_loaded)]
            self._rr_counters[rr_key] = idx + 1
        return chosen

    # ---------- status / metadata ----------

    def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        with self._lock:
            meta = self._agents.get(agent_id)
            if not meta:
                return False
            meta.status = status
            meta.last_heartbeat = time.time()
            return True

    async def update_status_async(self, agent_id: str, status: AgentStatus) -> bool:
        async with self._async_lock:
            return self.update_status(agent_id, status)

    def update_metadata(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        with self._lock:
            meta = self._agents.get(agent_id)
            if not meta:
                return False
            meta.metadata.update(metadata)
            return True

    def heartbeat(self, agent_id: str) -> bool:
        with self._lock:
            meta = self._agents.get(agent_id)
            if not meta:
                return False
            meta.last_heartbeat = time.time()
            return True

    # ---------- health checks ----------

    async def health_check(self, agent_id: str) -> HealthCheckResult:
        meta = self.get_agent(agent_id)
        if not meta:
            return HealthCheckResult(
                agent_id=agent_id, healthy=False, status=AgentStatus.OFFLINE,
                details={"error": "agent not registered"},
            )

        self._metrics["health_checks"] += 1
        instance = meta.instance
        health_fn = getattr(instance, "health_check", None) if instance else None

        if callable(health_fn):
            try:
                result = health_fn()
                if asyncio.iscoroutine(result):
                    result = await result
                details = result if isinstance(result, dict) else {"raw": result}
                healthy = details.get("status", "healthy") in ("healthy", True)
                if healthy:
                    self.heartbeat(agent_id)
                return HealthCheckResult(
                    agent_id=agent_id, healthy=healthy, status=meta.status, details=details
                )
            except Exception as exc:  # noqa: BLE001
                self.update_status(agent_id, AgentStatus.ERROR)
                return HealthCheckResult(
                    agent_id=agent_id, healthy=False, status=AgentStatus.ERROR,
                    details={"error": str(exc)},
                )

        age = time.time() - meta.last_heartbeat
        healthy = age <= self.heartbeat_timeout_seconds and meta.status not in (
            AgentStatus.OFFLINE, AgentStatus.ERROR, AgentStatus.SHUTDOWN,
        )
        return HealthCheckResult(
            agent_id=agent_id, healthy=healthy, status=meta.status,
            details={"heartbeat_age_seconds": age},
        )

    async def health_check_all(self) -> Dict[str, HealthCheckResult]:
        with self._lock:
            agent_ids = list(self._agents.keys())
        results = await asyncio.gather(*(self.health_check(aid) for aid in agent_ids))
        return dict(zip(agent_ids, results))

    def health_status(self, agent_id: str) -> Optional[AgentStatus]:
        meta = self.get_agent(agent_id)
        return meta.status if meta else None

    # ---------- messaging ----------

    async def send_message(self, agent_id: str, message: Any, sender: str = "registry") -> Any:
        meta = self.get_agent(agent_id)
        if not meta:
            raise KeyError(f"Agent '{agent_id}' is not registered.")

        self._metrics["messages_sent"] += 1

        if self.communication_bus is not None:
            return await self.communication_bus.send(agent_id, message, sender)

        receive_fn = getattr(meta.instance, "receive_message", None) if meta.instance else None
        if callable(receive_fn):
            result = receive_fn(message, sender)
            if asyncio.iscoroutine(result):
                return await result
            return result

        logger.warning("No communication channel available for agent '%s'.", agent_id)
        return None

    async def broadcast(self, message: Any, sender: str = "registry", exclude: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        exclude_set = set(exclude or [])
        self._metrics["broadcasts"] += 1

        if self.communication_bus is not None:
            return await self.communication_bus.broadcast(message, sender, exclude=list(exclude_set))

        with self._lock:
            targets = [aid for aid in self._agents if aid not in exclude_set]

        results: Dict[str, Any] = {}
        for agent_id in targets:
            try:
                results[agent_id] = await self.send_message(agent_id, message, sender)
            except Exception as exc:  # noqa: BLE001
                results[agent_id] = {"error": str(exc)}
        return results

    # ---------- task assignment ----------

    async def assign_task(
        self,
        task: Any,
        capability: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta: Optional[AgentMetadata]
        if agent_id is not None:
            meta = self.get_agent(agent_id)
        else:
            meta = self.find_best_agent(capability)

        if meta is None:
            return {"status": "no_agent_available", "task": task}

        with self._lock:
            meta.current_task_count += 1
            meta.status = AgentStatus.BUSY if meta.current_task_count >= meta.max_concurrent_tasks else AgentStatus.AVAILABLE

        self._metrics["tasks_assigned"] += 1

        try:
            if self.delegation_manager is not None:
                result = await self.delegation_manager.assign_task(meta.agent_id, task)
            else:
                handle_fn = getattr(meta.instance, "handle_task", None) if meta.instance else None
                if callable(handle_fn):
                    result = handle_fn(task)
                    if asyncio.iscoroutine(result):
                        result = await result
                else:
                    result = await self.send_message(meta.agent_id, task, sender="registry")

            return {"status": "assigned", "agent_id": meta.agent_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            self.update_status(meta.agent_id, AgentStatus.ERROR)
            return {"status": "error", "agent_id": meta.agent_id, "error": str(exc)}
        finally:
            with self._lock:
                meta.current_task_count = max(0, meta.current_task_count - 1)
                if meta.status != AgentStatus.ERROR:
                    meta.status = AgentStatus.IDLE if meta.current_task_count == 0 else AgentStatus.AVAILABLE

    # ---------- blackboard integration ----------

    async def publish_to_blackboard(self, key: str, value: Any) -> bool:
        if self.shared_blackboard is None:
            return False
        await self.shared_blackboard.set(key, value)
        return True

    async def read_from_blackboard(self, key: str) -> Any:
        if self.shared_blackboard is None:
            return None
        return await self.shared_blackboard.get(key)

    # ---------- metrics / introspection ----------

    def get_metrics(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._metrics)

    def list_agents(self) -> List[AgentMetadata]:
        with self._lock:
            return list(self._agents.values())

    def agent_count(self) -> int:
        with self._lock:
            return len(self._agents)
