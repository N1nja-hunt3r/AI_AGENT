"""
conftest.py
Shared pytest fixtures for the AI Operating System test suite: mock LLM,
vector DB, database, capabilities, agents, memory, tools, automation,
security, and a FastAPI app/TestClient wired to those mocks.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ---------------------------------------------------------------------- #
# Mock LLM
# ---------------------------------------------------------------------- #

class MockLLM:
    """In-memory stand-in for an LLM client supporting generate() and stream()."""

    def __init__(self, default_response: str = "mock response") -> None:
        self.default_response = default_response
        self.calls: List[Dict[str, Any]] = []
        self.fail_next: bool = False

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("mock llm failure")
        return str(kwargs.get("response_override", self.default_response))

    async def stream(self, prompt: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "stream": True})
        for token in self.default_response.split():
            await asyncio.sleep(0)
            yield token + " "


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


# ---------------------------------------------------------------------- #
# Mock Vector DB
# ---------------------------------------------------------------------- #

class MockVectorDB:
    """In-memory stand-in for a vector database backend."""

    def __init__(self, backend: str = "memory") -> None:
        self.backend = backend
        self.store: Dict[str, Dict[str, Any]] = {}

    async def upsert(self, doc_id: str, vector: Sequence[float], metadata: Dict[str, Any]) -> None:
        self.store[doc_id] = {"vector": list(vector), "metadata": metadata}

    async def query(self, vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return [
            {"id": doc_id, "score": 1.0, "metadata": entry["metadata"]}
            for doc_id, entry in list(self.store.items())[:top_k]
        ]

    async def delete(self, doc_id: str) -> bool:
        return self.store.pop(doc_id, None) is not None


@pytest.fixture
def mock_vector_db() -> MockVectorDB:
    return MockVectorDB()


@pytest.fixture
def vector_db_factory() -> Callable[[str], MockVectorDB]:
    def _factory(backend: str = "memory") -> MockVectorDB:
        return MockVectorDB(backend=backend)
    return _factory


class MockRetriever:
    """Embeds a query and retrieves matching vectors from a vector DB."""

    def __init__(self, vector_db: MockVectorDB, embed_fn: Callable[[str], List[float]]) -> None:
        self.vector_db = vector_db
        self.embed_fn = embed_fn

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = self.embed_fn(query)
        return await self.vector_db.query(vector, top_k=top_k)


def mock_embed(text: str) -> List[float]:
    """Deterministic pseudo-embedding suitable only for tests."""
    return [float(len(text) % 7), float(sum(map(ord, text)) % 13)]


@pytest.fixture
def mock_embed_fn() -> Callable[[str], List[float]]:
    return mock_embed


@pytest.fixture
def mock_retriever(mock_vector_db: MockVectorDB, mock_embed_fn: Callable[[str], List[float]]) -> MockRetriever:
    return MockRetriever(mock_vector_db, mock_embed_fn)


def chunk_text(text: str, chunk_size: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] or [""]


@pytest.fixture
def chunker() -> Callable[[str, int], List[str]]:
    return chunk_text


def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(candidate: Dict[str, Any]) -> int:
        text = candidate.get("metadata", {}).get("text", "")
        return sum(1 for word in query.lower().split() if word in text.lower())
    return sorted(candidates, key=score, reverse=True)


@pytest.fixture
def reranker() -> Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    return rerank


# ---------------------------------------------------------------------- #
# Mock Database
# ---------------------------------------------------------------------- #

class MockDatabase:
    """In-memory stand-in for a relational/document database."""

    def __init__(self) -> None:
        self.tables: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.backups: List[Dict[str, Any]] = []
        self.cache: Dict[str, Dict[str, Any]] = {}

    async def create(self, table: str, record: Dict[str, Any]) -> str:
        record_id = str(record.get("id") or uuid.uuid4())
        self.tables.setdefault(table, {})[record_id] = {**record, "id": record_id}
        return record_id

    async def read(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        return self.tables.get(table, {}).get(record_id)

    async def update(self, table: str, record_id: str, fields: Dict[str, Any]) -> bool:
        record = self.tables.get(table, {}).get(record_id)
        if record is None:
            return False
        record.update(fields)
        return True

    async def delete(self, table: str, record_id: str) -> bool:
        return self.tables.get(table, {}).pop(record_id, None) is not None

    async def open_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"user_id": user_id, "created_at": time.time(), "active": True}
        return session_id

    async def close_session(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        session["active"] = False
        return True

    async def backup(self) -> str:
        backup_id = str(uuid.uuid4())
        snapshot = {table: dict(records) for table, records in self.tables.items()}
        self.backups.append({"id": backup_id, "snapshot": snapshot})
        return backup_id

    async def restore(self, backup_id: str) -> bool:
        for backup in self.backups:
            if backup["id"] == backup_id:
                self.tables = {table: dict(records) for table, records in backup["snapshot"].items()}
                return True
        return False

    async def cache_set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self.cache[key] = {"value": value, "expires_at": (time.time() + ttl) if ttl is not None else None}

    async def cache_get(self, key: str) -> Optional[Any]:
        entry = self.cache.get(key)
        if entry is None:
            return None
        if entry["expires_at"] is not None and entry["expires_at"] < time.time():
            self.cache.pop(key, None)
            return None
        return entry["value"]


@pytest.fixture
def mock_database() -> MockDatabase:
    return MockDatabase()


# ---------------------------------------------------------------------- #
# Mock Capabilities
# ---------------------------------------------------------------------- #

class MockCapability:
    def __init__(self, name: str, handler: Callable[..., Any]) -> None:
        self.name = name
        self.handler = handler
        self.invocations: int = 0

    async def invoke(self, **kwargs: Any) -> Any:
        self.invocations += 1
        result = self.handler(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


class MockCapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: Dict[str, MockCapability] = {}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        self._capabilities[name] = MockCapability(name, handler)

    def get(self, name: str) -> MockCapability:
        if name not in self._capabilities:
            raise KeyError(f"capability '{name}' not registered")
        return self._capabilities[name]

    def list(self) -> List[str]:
        return sorted(self._capabilities.keys())


@pytest.fixture
def mock_capabilities() -> MockCapabilityRegistry:
    registry = MockCapabilityRegistry()
    registry.register("memory", lambda **kw: {"status": "ok", "source": "memory"})
    registry.register("rag", lambda **kw: {"status": "ok", "source": "rag"})
    registry.register("tool", lambda **kw: {"status": "ok", "source": "tool"})
    registry.register("web", lambda **kw: {"status": "ok", "source": "web"})
    registry.register("computer", lambda **kw: {"status": "ok", "source": "computer"})
    return registry


# ---------------------------------------------------------------------- #
# Mock Agents
# ---------------------------------------------------------------------- #

class MockAgent:
    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role
        self.status = "idle"
        self.history: List[Dict[str, Any]] = []

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = "in_progress"
        await asyncio.sleep(0)
        result = {
            "agent": self.name,
            "task_id": task.get("id"),
            "output": f"{self.name} completed {task.get('description', '')}".strip(),
            "status": "completed",
        }
        self.history.append(result)
        self.status = "idle"
        return result


class MockAgentRegistry:
    def __init__(self) -> None:
        self.agents: Dict[str, MockAgent] = {
            "manager": MockAgent("manager", "delegation"),
            "coder": MockAgent("coder", "code_generation"),
            "research": MockAgent("research", "research"),
            "memory": MockAgent("memory", "memory_management"),
            "planner": MockAgent("planner", "planning"),
            "reviewer": MockAgent("reviewer", "review"),
        }

    def get(self, name: str) -> MockAgent:
        if name not in self.agents:
            raise KeyError(f"agent '{name}' not found")
        return self.agents[name]

    def list(self) -> List[str]:
        return sorted(self.agents.keys())


@pytest.fixture
def mock_agents() -> MockAgentRegistry:
    return MockAgentRegistry()


# ---------------------------------------------------------------------- #
# Mock Memory
# ---------------------------------------------------------------------- #

@dataclass
class MemoryRecord:
    id: str
    content: str
    timestamp: float
    tier: str  # "short" | "long"


class MockMemoryStore:
    def __init__(self) -> None:
        self.short_term: List[MemoryRecord] = []
        self.long_term: List[MemoryRecord] = []

    def add_short(self, content: str) -> MemoryRecord:
        record = MemoryRecord(id=str(uuid.uuid4()), content=content, timestamp=time.time(), tier="short")
        self.short_term.append(record)
        return record

    def add_long(self, content: str) -> MemoryRecord:
        record = MemoryRecord(id=str(uuid.uuid4()), content=content, timestamp=time.time(), tier="long")
        self.long_term.append(record)
        return record

    def promote(self, record_id: str) -> bool:
        for index, record in enumerate(self.short_term):
            if record.id == record_id:
                promoted = self.short_term.pop(index)
                self.long_term.append(MemoryRecord(promoted.id, promoted.content, promoted.timestamp, "long"))
                return True
        return False

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        pool = self.short_term + self.long_term
        matches = [record for record in pool if query.lower() in record.content.lower()]
        return (matches or pool)[:limit]

    def compress(self, tier: str = "long") -> str:
        records = self.long_term if tier == "long" else self.short_term
        return " | ".join(record.content for record in records)


@pytest.fixture
def mock_memory() -> MockMemoryStore:
    return MockMemoryStore()


# ---------------------------------------------------------------------- #
# Mock Tools
# ---------------------------------------------------------------------- #

async def _calculator(expression: str) -> float:
    allowed_chars = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed_chars:
        raise ValueError("invalid expression")
    return eval(expression)  # noqa: S307 - safe: input restricted to numeric/operator characters above


async def _weather(location: str) -> Dict[str, Any]:
    return {"location": location, "temperature_c": 21.0, "condition": "clear"}


async def _terminal(command: str) -> Dict[str, Any]:
    if command.strip().lower().startswith(("rm ", "rm -")):
        raise PermissionError("destructive command blocked")
    return {"command": command, "stdout": f"executed: {command}", "exit_code": 0}


async def _search(query: str) -> List[Dict[str, Any]]:
    return [{"title": f"Result for {query}", "url": "https://example.com", "snippet": "mock result"}]


class MockToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {
            "calculator": _calculator,
            "weather": _weather,
            "terminal": _terminal,
            "search": _search,
        }
        self.call_log: List[Dict[str, Any]] = []

    async def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self.tools:
            raise KeyError(f"tool '{name}' not registered")
        self.call_log.append({"tool": name, "kwargs": kwargs})
        return await self.tools[name](**kwargs)


@pytest.fixture
def mock_tools() -> MockToolRegistry:
    return MockToolRegistry()


# ---------------------------------------------------------------------- #
# Mock Automation
# ---------------------------------------------------------------------- #

@dataclass
class ScheduledJob:
    id: str
    name: str
    run_at: float
    triggered: bool = False


class MockAutomationEngine:
    def __init__(self) -> None:
        self.scheduler: Dict[str, ScheduledJob] = {}
        self.queue: List[Dict[str, Any]] = []
        self.workflows: Dict[str, List[str]] = {}
        self.trigger_log: List[str] = []

    def schedule(self, name: str, run_at: float) -> ScheduledJob:
        job = ScheduledJob(id=str(uuid.uuid4()), name=name, run_at=run_at)
        self.scheduler[job.id] = job
        return job

    def enqueue(self, task: Dict[str, Any]) -> None:
        self.queue.append(task)

    def dequeue(self) -> Optional[Dict[str, Any]]:
        return self.queue.pop(0) if self.queue else None

    def define_workflow(self, name: str, steps: List[str]) -> None:
        self.workflows[name] = steps

    async def run_workflow(self, name: str) -> List[str]:
        executed: List[str] = []
        for step in self.workflows.get(name, []):
            await asyncio.sleep(0)
            executed.append(step)
        return executed

    def fire_trigger(self, job_id: str) -> bool:
        job = self.scheduler.get(job_id)
        if job is None:
            return False
        job.triggered = True
        self.trigger_log.append(job_id)
        return True


@pytest.fixture
def mock_automation() -> MockAutomationEngine:
    return MockAutomationEngine()


# ---------------------------------------------------------------------- #
# Mock Security
# ---------------------------------------------------------------------- #

class MockSecurity:
    def __init__(self) -> None:
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.sandbox_log: List[str] = []
        self._secret = "test-secret-key"

    def request_approval(self, action: str) -> str:
        approval_id = str(uuid.uuid4())
        self.pending_approvals[approval_id] = {"action": action, "approved": False}
        return approval_id

    def approve(self, approval_id: str) -> bool:
        record = self.pending_approvals.get(approval_id)
        if record is None:
            return False
        record["approved"] = True
        return True

    def is_approved(self, approval_id: str) -> bool:
        record = self.pending_approvals.get(approval_id)
        return bool(record and record["approved"])

    def check_rate_limit(self, key: str, max_calls: int, window_seconds: float) -> bool:
        now = time.time()
        calls = [t for t in self.rate_limits.get(key, []) if now - t < window_seconds]
        calls.append(now)
        self.rate_limits[key] = calls
        return len(calls) <= max_calls

    def run_in_sandbox(self, command: str) -> Dict[str, Any]:
        self.sandbox_log.append(command)
        return {"command": command, "isolated": True, "exit_code": 0}

    def issue_jwt(self, subject: str, ttl_seconds: float = 3600) -> str:
        payload = f"{subject}:{time.time() + ttl_seconds}:{self._secret}"
        return "tok_" + str(abs(hash(payload)))

    def validate_jwt(self, token: str) -> bool:
        return isinstance(token, str) and token.startswith("tok_")

    def validate_input(self, value: str, max_length: int = 10_000) -> bool:
        if not isinstance(value, str) or len(value) > max_length:
            return False
        banned = ("<script", "drop table", "../")
        return not any(token in value.lower() for token in banned)


@pytest.fixture
def mock_security() -> MockSecurity:
    return MockSecurity()


# ---------------------------------------------------------------------- #
# Planner / Executor / Router / Controller
# ---------------------------------------------------------------------- #

@dataclass
class PlanNode:
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    owner: str = "direct_reasoning"
    fallback: Optional[str] = None


@dataclass
class Plan:
    objective: str
    nodes: List[PlanNode] = field(default_factory=list)

    def node(self, node_id: str) -> PlanNode:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(node_id)

    def topological_order(self) -> List[str]:
        order: List[str] = []
        visited: set = set()
        visiting: set = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise ValueError(f"circular dependency detected at '{node_id}'")
            visiting.add(node_id)
            for dependency in self.node(node_id).depends_on:
                visit(dependency)
            visiting.discard(node_id)
            visited.add(node_id)
            order.append(node_id)

        for candidate in self.nodes:
            visit(candidate.id)
        return order


class Planner:
    def __init__(self, llm: MockLLM) -> None:
        self.llm = llm

    async def decompose(self, objective: str, max_subtasks: int = 5) -> Plan:
        await self.llm.generate(f"decompose: {objective}")
        steps = [step.strip() for step in objective.split(" and ") if step.strip()] or [objective]
        steps = steps[:max_subtasks]
        nodes = [
            PlanNode(id=f"T{i + 1}", description=step, depends_on=[f"T{i}"] if i > 0 else [])
            for i, step in enumerate(steps)
        ]
        return Plan(objective=objective, nodes=nodes)

    def add_fallback(self, plan: Plan, node_id: str, fallback: str) -> None:
        plan.node(node_id).fallback = fallback

    async def replan(self, plan: Plan, failed_node_id: str, reason: str) -> Plan:
        failed = plan.node(failed_node_id)
        remaining = [n for n in plan.nodes if n.id != failed_node_id]
        replacement = PlanNode(
            id=f"{failed.id}-r",
            description=f"retry: {failed.description} (after: {reason})",
            depends_on=list(failed.depends_on),
            owner=failed.owner,
            fallback=failed.fallback,
        )
        new_nodes = remaining + [replacement]
        for n in new_nodes:
            n.depends_on = [dep if dep != failed_node_id else replacement.id for dep in n.depends_on]
        return Plan(objective=plan.objective, nodes=new_nodes)


@pytest.fixture
def planner(mock_llm: MockLLM) -> Planner:
    return Planner(mock_llm)


@dataclass
class ExecutionResult:
    node_id: str
    status: str  # "success" | "failed"
    output: Any = None
    error: Optional[str] = None


class Executor:
    def __init__(self, tools: MockToolRegistry, security: MockSecurity, max_retries: int = 2) -> None:
        self.tools = tools
        self.security = security
        self.max_retries = max_retries
        self.artifacts: Dict[str, Any] = {}
        self.rolled_back: List[str] = []

    async def execute(
        self,
        plan: Plan,
        handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> List[ExecutionResult]:
        handlers = handlers or {}
        results: List[ExecutionResult] = []
        for node_id in plan.topological_order():
            node = plan.node(node_id)
            handler = handlers.get(node_id)
            try:
                output = await self.retry(handler) if handler else node.description
                results.append(ExecutionResult(node_id=node_id, status="success", output=output))
                self.artifacts[node_id] = output
            except Exception as exc:  # noqa: BLE001 - intentionally broad for execution isolation
                results.append(ExecutionResult(node_id=node_id, status="failed", error=str(exc)))
                if node.fallback:
                    self.artifacts[node_id] = node.fallback
        return results

    async def retry(self, handler: Callable[..., Any]) -> Any:
        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        assert last_error is not None
        raise last_error

    async def request_approval(self, action: str) -> str:
        return self.security.request_approval(action)

    async def rollback(self, node_id: str) -> bool:
        if node_id in self.artifacts:
            del self.artifacts[node_id]
            self.rolled_back.append(node_id)
            return True
        return False


@pytest.fixture
def executor(mock_tools: MockToolRegistry, mock_security: MockSecurity) -> Executor:
    return Executor(mock_tools, mock_security)


class Router:
    KEYWORDS: Dict[str, tuple] = {
        "memory": ("remember", "previously", "preference", "earlier"),
        "rag": ("document", "according to the file", "in the report"),
        "tool": ("calculate", "compute", "run command"),
        "web": ("latest", "current", "today", "news"),
        "computer": ("click", "screenshot", "open the browser", "type into"),
    }

    def route(self, query: str) -> str:
        lowered = query.lower()
        for path, keywords in self.KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return path
        return "direct_reasoning"

    async def route_async(self, query: str) -> str:
        await asyncio.sleep(0)
        return self.route(query)


@pytest.fixture
def router() -> Router:
    return Router()


class Controller:
    def __init__(self, planner: Planner, executor: Executor, router: Router) -> None:
        self.planner = planner
        self.executor = executor
        self.router = router
        self.context: Dict[str, Any] = {}

    async def handle(
        self,
        objective: str,
        handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> Dict[str, Any]:
        path = await self.router.route_async(objective)
        plan = await self.planner.decompose(objective)
        results = await self.executor.execute(plan, handlers=handlers)

        self.context["last_objective"] = objective
        self.context["last_path"] = path

        failed = [r for r in results if r.status == "failed"]
        if failed:
            replanned = await self.planner.replan(plan, failed[0].node_id, failed[0].error or "unknown error")
            retry_results = await self.executor.execute(replanned)
            results = results + retry_results

        return {"objective": objective, "path": path, "results": results}


@pytest.fixture
def controller(planner: Planner, executor: Executor, router: Router) -> Controller:
    return Controller(planner, executor, router)


# ---------------------------------------------------------------------- #
# FastAPI app / TestClient
# ---------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def build_app(llm: MockLLM, database: MockDatabase, security: MockSecurity) -> FastAPI:
    sessions: Dict[str, List[str]] = {}
    startup_events: List[str] = []
    shutdown_events: List[str] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup_events.append("startup")
        yield
        shutdown_events.append("shutdown")

    app = FastAPI(lifespan=lifespan)
    app.state.container = {"llm": llm, "database": database, "security": security}
    app.state.startup_events = startup_events
    app.state.shutdown_events = shutdown_events

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/route")
    async def route_query(query: str) -> Dict[str, str]:
        return {"path": Router().route(query)}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        if not security.validate_input(payload.message):
            raise HTTPException(status_code=400, detail="invalid input")
        session_id = payload.session_id or str(uuid.uuid4())
        reply = await llm.generate(payload.message)
        sessions.setdefault(session_id, []).append(payload.message)
        return ChatResponse(session_id=session_id, reply=reply)

    @app.post("/stream")
    async def stream(payload: ChatRequest) -> StreamingResponse:
        if not security.validate_input(payload.message):
            raise HTTPException(status_code=400, detail="invalid input")

        async def token_generator():
            async for token in llm.stream(payload.message):
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                if not security.validate_input(data):
                    await websocket.send_json({"event": "error", "detail": "invalid input"})
                    continue
                reply = await llm.generate(data)
                await websocket.send_json({"event": "message", "reply": reply})
        except WebSocketDisconnect:
            return

    return app


@pytest.fixture
def app(mock_llm: MockLLM, mock_database: MockDatabase, mock_security: MockSecurity) -> FastAPI:
    return build_app(mock_llm, mock_database, mock_security)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app: FastAPI):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac