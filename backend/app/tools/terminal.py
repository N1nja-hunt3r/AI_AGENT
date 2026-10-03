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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence


class ApprovalDecision(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


class ExecutionStatus(Enum):
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
    status: ExecutionStatus
    returncode: Optional[int]
    stdout: str
    stderr: str
    duration_seconds: float
    workdir: str
    request_id: str
    rolled_back: bool = False
    rollback_error: Optional[str] = None


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
            if log_path:
                handler: logging.Handler = logging.FileHandler(log_path)
            else:
                handler = logging.StreamHandler()
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
    def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> ApprovalDecision:
        raise NotImplementedError


class AutoApprovalProvider(ApprovalProvider):
    def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> ApprovalDecision:
        return ApprovalDecision.APPROVED


class CallbackApprovalProvider(ApprovalProvider):
    def __init__(
        self,
        callback: Callable[[str, tuple[str, ...], str], bool],
        timeout_seconds: float = 30.0,
    ) -> None:
        self._callback = callback
        self._timeout_seconds = timeout_seconds

    def request_approval(
        self, command: str, args: tuple[str, ...], request_id: str
    ) -> ApprovalDecision:
        result: dict[str, Optional[bool]] = {"value": None}

        def run() -> None:
            try:
                result["value"] = bool(self._callback(command, args, request_id))
            except Exception:
                result["value"] = False

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(self._timeout_seconds)
        if thread.is_alive():
            return ApprovalDecision.TIMEOUT
        if result["value"]:
            return ApprovalDecision.APPROVED
        return ApprovalDecision.DENIED


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
        base_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": self._config.workdir,
            "LANG": "C.UTF-8",
        }
        if self._config.network_disabled:
            base_env["NO_PROXY"] = "*"
        base_env.update(self._config.env)
        return base_env

    def run(
        self,
        full_command: list[str],
        timeout_seconds: float,
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


class SafeTerminal:
    def __init__(
        self,
        whitelist: CommandWhitelist,
        sandbox_config: SandboxConfig,
        audit_logger: Optional[AuditLogger] = None,
        approval_provider: Optional[ApprovalProvider] = None,
        rollback_handler: Optional[RollbackHandler] = None,
        default_timeout_seconds: float = 30.0,
        require_approval: bool = True,
    ) -> None:
        self._whitelist = whitelist
        self._sandbox = SandboxExecutor(sandbox_config)
        self._sandbox_config = sandbox_config
        self._audit = audit_logger or AuditLogger()
        self._approval = approval_provider or AutoApprovalProvider()
        self._rollback = rollback_handler
        self._default_timeout_seconds = default_timeout_seconds
        self._require_approval = require_approval
        self._lock = threading.Lock()

    def execute(
        self,
        command_line: str,
        timeout_seconds: Optional[float] = None,
        enable_rollback: bool = False,
    ) -> CommandResult:
        request_id = uuid.uuid4().hex
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds

        try:
            tokens = shlex.split(command_line)
        except ValueError as exc:
            self._audit.log(
                AuditEvent(
                    timestamp=time.time(),
                    request_id=request_id,
                    event_type="parse_error",
                    command=command_line,
                    args=(),
                    detail=str(exc),
                )
            )
            return CommandResult(
                command=command_line,
                args=(),
                status=ExecutionStatus.ERROR,
                returncode=None,
                stdout="",
                stderr=f"failed to parse command: {exc}",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )

        if not tokens:
            return CommandResult(
                command=command_line,
                args=(),
                status=ExecutionStatus.ERROR,
                returncode=None,
                stdout="",
                stderr="empty command",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )

        command, args = tokens[0], tuple(tokens[1:])

        self._audit.log(
            AuditEvent(
                timestamp=time.time(),
                request_id=request_id,
                event_type="requested",
                command=command,
                args=args,
            )
        )

        if not self._whitelist.is_allowed(command):
            self._audit.log(
                AuditEvent(
                    timestamp=time.time(),
                    request_id=request_id,
                    event_type="blocked",
                    command=command,
                    args=args,
                    detail="command not in whitelist",
                )
            )
            return CommandResult(
                command=command,
                args=args,
                status=ExecutionStatus.BLOCKED,
                returncode=None,
                stdout="",
                stderr=f"command '{command}' is not whitelisted",
                duration_seconds=0.0,
                workdir=self._sandbox_config.workdir,
                request_id=request_id,
            )

        if self._require_approval:
            decision = self._approval.request_approval(command, args, request_id)
            self._audit.log(
                AuditEvent(
                    timestamp=time.time(),
                    request_id=request_id,
                    event_type="approval_decision",
                    command=command,
                    args=args,
                    detail=decision.value,
                )
            )
            if decision != ApprovalDecision.APPROVED:
                status = (
                    ExecutionStatus.TIMEOUT
                    if decision == ApprovalDecision.TIMEOUT
                    else ExecutionStatus.DENIED
                )
                return CommandResult(
                    command=command,
                    args=args,
                    status=status,
                    returncode=None,
                    stdout="",
                    stderr=f"execution {decision.value}",
                    duration_seconds=0.0,
                    workdir=self._sandbox_config.workdir,
                    request_id=request_id,
                )

        snapshot_id: Optional[str] = None
        if enable_rollback and self._rollback is not None:
            try:
                snapshot_id = self._rollback.snapshot(self._sandbox_config.workdir)
                self._audit.log(
                    AuditEvent(
                        timestamp=time.time(),
                        request_id=request_id,
                        event_type="snapshot_created",
                        command=command,
                        args=args,
                        detail=snapshot_id,
                    )
                )
            except Exception as exc:
                self._audit.log(
                    AuditEvent(
                        timestamp=time.time(),
                        request_id=request_id,
                        event_type="snapshot_failed",
                        command=command,
                        args=args,
                        detail=str(exc),
                    )
                )

        with self._lock:
            start = time.monotonic()
            try:
                completed = self._sandbox.run([command, *args], timeout_seconds=timeout)
                duration = time.monotonic() - start
                status = (
                    ExecutionStatus.SUCCESS if completed.returncode == 0 else ExecutionStatus.FAILED
                )
                self._audit.log(
                    AuditEvent(
                        timestamp=time.time(),
                        request_id=request_id,
                        event_type="executed",
                        command=command,
                        args=args,
                        detail=f"returncode={completed.returncode} duration={duration:.4f}",
                    )
                )
                rolled_back = False
                rollback_error: Optional[str] = None
                if status == ExecutionStatus.FAILED and snapshot_id is not None and self._rollback:
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
                    AuditEvent(
                        timestamp=time.time(),
                        request_id=request_id,
                        event_type="timeout",
                        command=command,
                        args=args,
                        detail=f"timeout_seconds={timeout}",
                    )
                )
                rolled_back = False
                rollback_error = None
                if snapshot_id is not None and self._rollback:
                    rolled_back, rollback_error = self._attempt_rollback(
                        snapshot_id, request_id, command, args
                    )
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                return CommandResult(
                    command=command,
                    args=args,
                    status=ExecutionStatus.TIMEOUT,
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
                self._audit.log(
                    AuditEvent(
                        timestamp=time.time(),
                        request_id=request_id,
                        event_type="error",
                        command=command,
                        args=args,
                        detail=str(exc),
                    )
                )
                rolled_back = False
                rollback_error = None
                if snapshot_id is not None and self._rollback:
                    rolled_back, rollback_error = self._attempt_rollback(
                        snapshot_id, request_id, command, args
                    )
                return CommandResult(
                    command=command,
                    args=args,
                    status=ExecutionStatus.ERROR,
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
        self,
        snapshot_id: str,
        request_id: str,
        command: str,
        args: tuple[str, ...],
    ) -> tuple[bool, Optional[str]]:
        assert self._rollback is not None
        try:
            self._rollback.restore(snapshot_id, self._sandbox_config.workdir)
            self._audit.log(
                AuditEvent(
                    timestamp=time.time(),
                    request_id=request_id,
                    event_type="rollback_success",
                    command=command,
                    args=args,
                    detail=snapshot_id,
                )
            )
            return True, None
        except Exception as exc:
            self._audit.log(
                AuditEvent(
                    timestamp=time.time(),
                    request_id=request_id,
                    event_type="rollback_failed",
                    command=command,
                    args=args,
                    detail=str(exc),
                )
            )
            return False, str(exc)


def build_default_terminal(
    workdir: str,
    allowed_commands: Sequence[str],
    audit_log_path: Optional[str] = None,
    approval_callback: Optional[Callable[[str, tuple[str, ...], str], bool]] = None,
    default_timeout_seconds: float = 30.0,
    enable_rollback_handler: bool = True,
) -> SafeTerminal:
    Path(workdir).mkdir(parents=True, exist_ok=True)

    whitelist = CommandWhitelist(allowed_commands)
    sandbox_config = SandboxConfig(workdir=workdir)
    audit_logger = AuditLogger(audit_log_path)
    approval_provider: ApprovalProvider = (
        CallbackApprovalProvider(approval_callback) if approval_callback else AutoApprovalProvider()
    )
    rollback_handler = FilesystemCopyRollbackHandler() if enable_rollback_handler else None

    return SafeTerminal(
        whitelist=whitelist,
        sandbox_config=sandbox_config,
        audit_logger=audit_logger,
        approval_provider=approval_provider,
        rollback_handler=rollback_handler,
        default_timeout_seconds=default_timeout_seconds,
        require_approval=approval_callback is not None,
    )
