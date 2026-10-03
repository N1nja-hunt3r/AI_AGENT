from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Sequence

from app.computer_use.base_computer import (
    CapabilityMetadata,
    CapabilityPriority,
    ComputerCapability,
    ExecutionRequest,
    ExecutionResult,
    HealthCheckResult,
    HealthStatus,
)


# ---------------------------------------------------------------------------
# Terminal execution domain
# ---------------------------------------------------------------------------

class TerminalExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DENIED = "denied"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass(frozen=True)
class CommandResult:
    command: str
    args: tuple[str, ...]
    status: TerminalExecutionStatus
    returncode: Optional[int]
    stdout: str
    stderr: str
    duration_seconds: float
    workdir: str
    request_id: str
    rolled_back: bool = False
    rollback_error: Optional[str] = None
    executed_at_epoch: float = field(default_factory=time.time)


@dataclass
class AuditEvent:
    timestamp: float
    request_id: str
    event_type: str
    command: str
    args: tuple[str, ...]
    detail: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "request_id": self.request_id,
                "event_type": self.event_type,
                "command": self.command,
                "args": list(self.args),
                "detail": self.detail,
            },
            sort_keys=True,
        )


class AuditLogger:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self._logger = logging.getLogger("terminal.audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler: logging.Handler = (
                logging.FileHandler(log_path) if log_path else logging.StreamHandler()
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> None:
        with self._lock:
            self._logger.info(event.to_json())


class CommandWhitelist:
    def __init__(self, allowed_commands: Sequence[str]) -> None:
        self._allowed = frozenset(allowed_commands)

    def is_allowed(self, command: str) -> bool:
        return command in self._allowed

    def allowed_commands(self) -> frozenset[str]:
        return self._allowed


class ApprovalProvider:
    async def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> bool:
        raise NotImplementedError


class AutoApprovalProvider(ApprovalProvider):
    async def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> bool:
        return True


class CallbackApprovalProvider(ApprovalProvider):
    def __init__(
        self,
        callback: Callable[[str, tuple[str, ...], str], bool],
        timeout_seconds: float = 30.0,
    ) -> None:
        self._callback = callback
        self._timeout_seconds = timeout_seconds

    async def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> bool:
        import asyncio

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._callback, command, args, request_id),
                timeout=self._timeout_seconds,
            )
        except (TimeoutError, Exception):
            return False


class RollbackHandler:
    def snapshot(self, workdir: str) -> str:
        raise NotImplementedError

    def restore(self, snapshot_id: str, workdir: str) -> None:
        raise NotImplementedError


class FilesystemCopyRollbackHandler(RollbackHandler):
    def __init__(self, snapshot_root: Optional[str] = None) -> None:
        self._snapshot_root = snapshot_root or tempfile.mkdtemp(prefix="terminal_snapshots_")

    def snapshot(self, workdir: str) -> str:
        snapshot_id = uuid.uuid4().hex
        dest = Path(self._snapshot_root) / snapshot_id
        src = Path(workdir)
        if src.exists():
            self._copy_tree(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return snapshot_id

    def restore(self, snapshot_id: str, workdir: str) -> None:
        src = Path(self._snapshot_root) / snapshot_id
        dest = Path(workdir)
        if not src.exists():
            raise FileNotFoundError(f"snapshot {snapshot_id} not found")
        if dest.exists():
            for child in dest.iterdir():
                self._remove(child)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            target = dest / child.name
            if child.is_dir():
                self._copy_tree(child, target)
            else:
                target.write_bytes(child.read_bytes())

    @staticmethod
    def _copy_tree(src: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                FilesystemCopyRollbackHandler._copy_tree(item, target)
            else:
                target.write_bytes(item.read_bytes())

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir():
            for child in path.iterdir():
                FilesystemCopyRollbackHandler._remove(child)
            path.rmdir()
        else:
            path.unlink()


@dataclass
class SandboxConfig:
    workdir: str
    env: dict[str, str] = field(default_factory=dict)
    use_chroot: bool = False
    chroot_path: Optional[str] = None
    drop_privileges_uid: Optional[int] = None
    drop_privileges_gid: Optional[int] = None
    resource_cpu_seconds: Optional[int] = None
    resource_memory_bytes: Optional[int] = None
    network_disabled: bool = True


class SandboxExecutor:
    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    def _preexec_fn(self) -> Callable[[], None]:
        config = self._config

        def setup() -> None:
            try:
                os.setsid()
            except OSError:
                pass

            if config.resource_cpu_seconds is not None:
                import resource

                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (config.resource_cpu_seconds, config.resource_cpu_seconds),
                )
            if config.resource_memory_bytes is not None:
                import resource

                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (config.resource_memory_bytes, config.resource_memory_bytes),
                )

            if config.use_chroot and config.chroot_path:
                os.chroot(config.chroot_path)
                os.chdir("/")

            if config.drop_privileges_gid is not None:
                os.setgid(config.drop_privileges_gid)
            if config.drop_privileges_uid is not None:
                os.setuid(config.drop_privileges_uid)

        return setup

    def build_env(self) -> dict[str, str]:
        base_env = {"PATH": "/usr/bin:/bin", "HOME": self._config.workdir, "LANG": "C.UTF-8"}
        if self._config.network_disabled:
            base_env["NO_PROXY"] = "*"
        base_env.update(self._config.env)
        return base_env

    def run(
        self, full_command: list[str], timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            full_command,
            cwd=self._config.workdir,
            env=self.build_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=self._preexec_fn() if os.name == "posix" else None,
            start_new_session=(os.name != "posix"),
        )


class TerminalCapability(ComputerCapability):
    def __init__(
        self,
        sandbox_config: SandboxConfig,
        whitelist: CommandWhitelist,
        audit_logger: Optional[AuditLogger] = None,
        approval_provider: Optional[ApprovalProvider] = None,
        rollback_handler: Optional[RollbackHandler] = None,
        history_max_entries: int = 500,
        metadata: Optional[CapabilityMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="terminal",
                version="1.0.0",
                description="Executes whitelisted shell commands in a sandbox",
                priority=CapabilityPriority.NORMAL,
                timeout_seconds=30.0,
                approval_required=True,
            )
        )
        self._sandbox_config = sandbox_config
        self._sandbox = SandboxExecutor(sandbox_config)
        self._whitelist = whitelist
        self._audit = audit_logger or AuditLogger()
        self._approval = approval_provider or AutoApprovalProvider()
        self._rollback = rollback_handler
        self._history: Deque[CommandResult] = deque(maxlen=history_max_entries)
        self._history_lock = threading.Lock()
        self._exec_lock = threading.Lock()

    async def _on_initialize(self) -> None:
        Path(self._sandbox_config.workdir).mkdir(parents=True, exist_ok=True)

    async def _on_shutdown(self) -> None:
        pass

    async def _on_health_check(self) -> HealthCheckResult:
        workdir = Path(self._sandbox_config.workdir)
        if not workdir.exists() or not workdir.is_dir():
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=f"workdir '{workdir}' is missing",
            )
        if not os.access(workdir, os.W_OK):
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.DEGRADED,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail="workdir is not writable",
            )
        return HealthCheckResult(
            capability_name=self.name,
            status=HealthStatus.HEALTHY,
            latency_seconds=None,
            checked_at_epoch=time.time(),
        )

    async def _on_execute(self, request: ExecutionRequest) -> Any:
        action = request.action
        params = request.parameters

        if action == "run":
            return await self.run(
                str(params["command_line"]),
                timeout_seconds=request.timeout_seconds,
                enable_rollback=bool(params.get("enable_rollback", False)),
            )
        if action == "history":
            return self.history(int(params.get("limit", 50)))

        raise ValueError(f"unknown terminal action: {action}")

    def history(self, limit: int = 50) -> list[CommandResult]:
        with self._history_lock:
            items = list(self._history)
        return items[-limit:]

    async def run(
        self,
        command_line: str,
        timeout_seconds: Optional[float] = None,
        enable_rollback: bool = False,
    ) -> CommandResult:
        import asyncio

        request_id = uuid.uuid4().hex
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds

        try:
            tokens = shlex.split(command_line)
        except ValueError as exc:
            result = CommandResult(
                command=command_line,
                args=(),
                status=TerminalExecutionStatus.ERROR,
                returncode=None,
                stdout="",
                stderr=f"failed to parse command: {exc}",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )
            self._record(result)
            return result

        if not tokens:
            result = CommandResult(
                command=command_line,
                args=(),
                status=TerminalExecutionStatus.ERROR,
                returncode=None,
                stdout="",
                stderr="empty command",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )
            self._record(result)
            return result

        command, args = tokens[0], tuple(tokens[1:])

        self._audit.log(AuditEvent(time.time(), request_id, "requested", command, args))

        if not self._whitelist.is_allowed(command):
            self._audit.log(
                AuditEvent(time.time(), request_id, "blocked", command, args, "not whitelisted")
            )
            result = CommandResult(
                command=command,
                args=args,
                status=TerminalExecutionStatus.BLOCKED,
                returncode=None,
                stdout="",
                stderr=f"command '{command}' is not whitelisted",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )
            self._record(result)
            return result

        approved = await self._approval.request_approval(command, args, request_id)
        self._audit.log(
            AuditEvent(
                time.time(),
                request_id,
                "approval_decision",
                command,
                args,
                "approved" if approved else "denied",
            )
        )
        if not approved:
            result = CommandResult(
                command=command,
                args=args,
                status=TerminalExecutionStatus.DENIED,
                returncode=None,
                stdout="",
                stderr="execution denied",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )
            self._record(result)
            return result

        snapshot_id: Optional[str] = None
        if enable_rollback and self._rollback is not None:
            try:
                snapshot_id = await asyncio.to_thread(
                    self._rollback.snapshot, self._sandbox_config.workdir
                )
                self._audit.log(
                    AuditEvent(time.time(), request_id, "snapshot_created", command, args, snapshot_id)
                )
            except Exception as exc:
                self._audit.log(
                    AuditEvent(time.time(), request_id, "snapshot_failed", command, args, str(exc))
                )

        result = await asyncio.to_thread(
            self._execute_sync, command, args, timeout, request_id, snapshot_id
        )
        self._record(result)
        return result

    def _execute_sync(
        self,
        command: str,
        args: tuple[str, ...],
        timeout: float,
        request_id: str,
        snapshot_id: Optional[str],
    ) -> CommandResult:
        with self._exec_lock:
            start = time.monotonic()
            try:
                completed = self._sandbox.run([command, *args], timeout_seconds=timeout)
                duration = time.monotonic() - start
                status = (
                    TerminalExecutionStatus.SUCCESS
                    if completed.returncode == 0
                    else TerminalExecutionStatus.FAILED
                )
                self._audit.log(
                    AuditEvent(
                        time.time(),
                        request_id,
                        "executed",
                        command,
                        args,
                        f"returncode={completed.returncode} duration={duration:.4f}",
                    )
                )
                rolled_back = False
                rollback_error: Optional[str] = None
                if status == TerminalExecutionStatus.FAILED and snapshot_id and self._rollback:
                    rolled_back, rollback_error = self._attempt_rollback(
                        snapshot_id, request_id, command, args
                    )
                return CommandResult(
                    command=command,
                    args=args,
                    status=status,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    duration_seconds=duration,
                    workdir=self._sandbox_config.workdir,
                    request_id=request_id,
                    rolled_back=rolled_back,
                    rollback_error=rollback_error,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start
                self._audit.log(
                    AuditEvent(time.time(), request_id, "timeout", command, args, f"timeout={timeout}")
                )
                rolled_back = False
                rollback_error = None
                if snapshot_id and self._rollback:
                    rolled_back, rollback_error = self._attempt_rollback(
                        snapshot_id, request_id, command, args
                    )
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                return CommandResult(
                    command=command,
                    args=args,
                    status=TerminalExecutionStatus.TIMEOUT,
                    returncode=None,
                    stdout=stdout,
                    stderr=stderr or f"command timed out after {timeout}s",
                    duration_seconds=duration,
                    workdir=self._sandbox_config.workdir,
                    request_id=request_id,
                    rolled_back=rolled_back,
                    rollback_error=rollback_error,
                )
            except Exception as exc:
                duration = time.monotonic() - start
                self._audit.log(AuditEvent(time.time(), request_id, "error", command, args, str(exc)))
                rolled_back = False
                rollback_error = None
                if snapshot_id and self._rollback:
                    rolled_back, rollback_error = self._attempt_rollback(
                        snapshot_id, request_id, command, args
                    )
                return CommandResult(
                    command=command,
                    args=args,
                    status=TerminalExecutionStatus.ERROR,
                    returncode=None,
                    stdout="",
                    stderr=str(exc),
                    duration_seconds=duration,
                    workdir=self._sandbox_config.workdir,
                    request_id=request_id,
                    rolled_back=rolled_back,
                    rollback_error=rollback_error,
                )

    def _attempt_rollback(
        self, snapshot_id: str, request_id: str, command: str, args: tuple[str, ...]
    ) -> tuple[bool, Optional[str]]:
        assert self._rollback is not None
        try:
            self._rollback.restore(snapshot_id, self._sandbox_config.workdir)
            self._audit.log(
                AuditEvent(time.time(), request_id, "rollback_success", command, args, snapshot_id)
            )
            return True, None
        except Exception as exc:
            self._audit.log(
                AuditEvent(time.time(), request_id, "rollback_failed", command, args, str(exc))
            )
            return False, str(exc)

    def _record(self, result: CommandResult) -> None:
        with self._history_lock:
            self._history.append(result)


# ---------------------------------------------------------------------------
# Registry domain
# ---------------------------------------------------------------------------

class RegistryError(Exception):
    pass


class CapabilityAlreadyRegisteredError(RegistryError):
    pass


class CapabilityNotRegisteredError(RegistryError):
    pass


class CircularDependencyError(RegistryError):
    pass


class MissingDependencyError(RegistryError):
    pass


@dataclass(frozen=True)
class RegistryEntry:
    capability: ComputerCapability
    registered_at_epoch: float


@dataclass(frozen=True)
class DiscoveryFilter:
    tag: Optional[str] = None
    name_prefix: Optional[str] = None
    min_priority: Optional[int] = None
    ready_only: bool = False


class CapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._lock = threading.Lock()

    async def register(
        self, capability: ComputerCapability, replace: bool = False
    ) -> None:
        with self._lock:
            name = capability.name
            if name in self._entries and not replace:
                raise CapabilityAlreadyRegisteredError(
                    f"capability '{name}' is already registered"
                )
            self._validate_dependencies_present(capability)
            self._entries[name] = RegistryEntry(
                capability=capability, registered_at_epoch=time.time()
            )

    async def unregister(self, name: str, shutdown: bool = True) -> None:
        with self._lock:
            entry = self._entries.pop(name, None)
            if entry is None:
                raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")
            dependents = self._find_dependents_locked(name)
            if dependents:
                self._entries[name] = entry
                raise RegistryError(
                    f"cannot unregister '{name}': required by {sorted(dependents)}"
                )
            if shutdown:
                await entry.capability.shutdown()

    def _validate_dependencies_present(self, capability: ComputerCapability) -> None:
        for dependency in capability.metadata.dependencies:
            if dependency not in self._entries and dependency != capability.name:
                pass

    def _find_dependents_locked(self, name: str) -> set[str]:
        dependents: set[str] = set()
        for other_name, entry in self._entries.items():
            if name in entry.capability.metadata.dependencies:
                dependents.add(other_name)
        return dependents

    def get(self, name: str) -> ComputerCapability:
        entry = self._entries.get(name)
        if entry is None:
            raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")
        return entry.capability

    def discover(self, filter_: Optional[DiscoveryFilter] = None) -> list[ComputerCapability]:
        capabilities = [entry.capability for entry in self._entries.values()]

        if filter_ is None:
            return self._sort_by_priority(capabilities)

        filtered = capabilities
        if filter_.tag is not None:
            filtered = [c for c in filtered if filter_.tag in c.metadata.tags]
        if filter_.name_prefix is not None:
            filtered = [c for c in filtered if c.name.startswith(filter_.name_prefix)]
        if filter_.min_priority is not None:
            filtered = [c for c in filtered if int(c.priority) >= filter_.min_priority]
        if filter_.ready_only:
            filtered = [c for c in filtered if c.is_ready]

        return self._sort_by_priority(filtered)

    @staticmethod
    def _sort_by_priority(capabilities: list[ComputerCapability]) -> list[ComputerCapability]:
        return sorted(capabilities, key=lambda c: int(c.priority), reverse=True)

    def resolve_dependency_order(self, names: Optional[list[str]] = None) -> list[str]:
        target_names = names if names is not None else list(self._entries.keys())
        for name in target_names:
            if name not in self._entries:
                raise CapabilityNotRegisteredError(f"capability '{name}' is not registered")

        visited: set[str] = set()
        in_progress: set[str] = set()
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in in_progress:
                raise CircularDependencyError(
                    f"circular dependency detected involving '{name}'"
                )
            in_progress.add(name)
            entry = self._entries.get(name)
            if entry is None:
                raise MissingDependencyError(f"dependency '{name}' is not registered")
            for dependency in entry.capability.metadata.dependencies:
                if dependency not in self._entries:
                    raise MissingDependencyError(
                        f"'{name}' depends on unregistered capability '{dependency}'"
                    )
                visit(dependency)
            in_progress.discard(name)
            visited.add(name)
            ordered.append(name)

        for name in target_names:
            visit(name)

        return ordered

    async def initialize_all(self, names: Optional[list[str]] = None) -> dict[str, Optional[str]]:
        order = self.resolve_dependency_order(names)
        results: dict[str, Optional[str]] = {}
        for name in order:
            capability = self.get(name)
            try:
                await capability.initialize()
                results[name] = None
            except Exception as exc:
                results[name] = str(exc)
        return results

    async def shutdown_all(self, names: Optional[list[str]] = None) -> dict[str, Optional[str]]:
        order = self.resolve_dependency_order(names)
        results: dict[str, Optional[str]] = {}
        for name in reversed(order):
            capability = self.get(name)
            try:
                await capability.shutdown()
                results[name] = None
            except Exception as exc:
                results[name] = str(exc)
        return results

    async def execute(self, name: str, request: ExecutionRequest) -> ExecutionResult:
        capability = self.get(name)
        return await capability.execute(request)

    async def health_check(self, name: str) -> HealthCheckResult:
        capability = self.get(name)
        return await capability.health_check()

    async def health_check_all(
        self, names: Optional[list[str]] = None
    ) -> dict[str, HealthCheckResult]:
        import asyncio

        target_names = names if names is not None else list(self._entries.keys())
        capabilities = [(name, self.get(name)) for name in target_names]

        results = await asyncio.gather(
            *(capability.health_check() for _, capability in capabilities),
            return_exceptions=True,
        )

        health_map: dict[str, HealthCheckResult] = {}
        for (name, _), result in zip(capabilities, results):
            if isinstance(result, HealthCheckResult):
                health_map[name] = result
            else:
                health_map[name] = HealthCheckResult(
                    capability_name=name,
                    status=HealthStatus.UNKNOWN,
                    latency_seconds=None,
                    checked_at_epoch=time.time(),
                    detail=str(result),
                )
        return health_map

    def list_names(self) -> list[str]:
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
