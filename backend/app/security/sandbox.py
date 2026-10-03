from __future__ import annotations

import asyncio
import os
import platform
import resource
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"
    DENIED = "denied"


class SandboxError(Exception):
    pass


class CommandNotWhitelistedError(SandboxError):
    pass


class ResourceLimitError(SandboxError):
    pass


@dataclass
class ResourceLimits:
    cpu_seconds: int = 5
    memory_bytes: int = 256 * 1024 * 1024
    max_processes: int = 16
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_open_files: int = 64
    wall_timeout_seconds: float = 10.0


@dataclass
class ExecutionResult:
    execution_id: str
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    duration_seconds: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    work_dir: Optional[str] = None


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


DEFAULT_COMMAND_WHITELIST: frozenset[str] = frozenset({
    "python3", "python", "node", "ls", "cat", "echo", "pwd",
    "grep", "head", "tail", "wc", "sort", "uniq", "find",
})

_DANGEROUS_TOKENS: frozenset[str] = frozenset({
    ";", "&&", "||", "|", "`", "$(", ">", ">>", "<", "&",
})


def _set_resource_limits(limits: ResourceLimits) -> None:
    """Run inside child process (preexec_fn) to apply OS-level limits. POSIX only.

    Each limit is applied independently and failures are swallowed individually so
    that one unsupported/denied limit (common in containers, e.g. RLIMIT_NPROC or
    RLIMIT_AS under cgroup-constrained or already-near-limit environments) does not
    prevent the rest of the limits -- or the exec itself -- from being applied.

    Session/process-group isolation is handled by passing start_new_session=True to
    Popen rather than calling os.setsid() here, since Popen already performs that
    setsid() internally before invoking preexec_fn in some platforms/timings, and a
    redundant call would raise.
    """
    def _try_set(res_name: int, value: int) -> None:
        try:
            resource.setrlimit(res_name, (value, value))
        except (ValueError, OSError):
            pass

    _try_set(resource.RLIMIT_CPU, limits.cpu_seconds)
    _try_set(resource.RLIMIT_AS, limits.memory_bytes)
    _try_set(resource.RLIMIT_NPROC, limits.max_processes)
    _try_set(resource.RLIMIT_FSIZE, limits.max_file_size_bytes)
    _try_set(resource.RLIMIT_NOFILE, limits.max_open_files)


class Sandbox:
    """Restricted command execution with filesystem/process isolation and resource limits."""

    def __init__(
        self,
        *,
        command_whitelist: Optional[frozenset[str]] = None,
        default_limits: Optional[ResourceLimits] = None,
        isolated_root: Optional[str | Path] = None,
        env_allowlist: Optional[frozenset[str]] = None,
    ) -> None:
        self.command_whitelist = command_whitelist if command_whitelist is not None else DEFAULT_COMMAND_WHITELIST
        self.default_limits = default_limits or ResourceLimits()
        self.isolated_root = Path(isolated_root) if isolated_root else Path(tempfile.gettempdir()) / "sandbox_root"
        self.isolated_root.mkdir(parents=True, exist_ok=True)
        self.env_allowlist = env_allowlist if env_allowlist is not None else frozenset({"PATH", "LANG", "LC_ALL"})
        self._active: dict[str, subprocess.Popen[bytes]] = {}
        self._history: list[ExecutionResult] = []
        self._created_at = time.time()
        self._execution_count = 0
        self._is_posix = platform.system() != "Windows"
        self._lock = asyncio.Lock()

    def _validate_command(self, command: list[str]) -> None:
        if not command:
            raise CommandNotWhitelistedError("empty command")
        binary = Path(command[0]).name
        if binary not in self.command_whitelist:
            raise CommandNotWhitelistedError(f"command '{binary}' is not whitelisted")
        joined = " ".join(command)
        for token in _DANGEROUS_TOKENS:
            if token in joined:
                raise CommandNotWhitelistedError(f"dangerous shell metacharacter detected: '{token}'")

    def _build_env(self, extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k in self.env_allowlist}
        if extra_env:
            env.update({k: v for k, v in extra_env.items() if k in self.env_allowlist})
        return env

    def isolate(self, execution_id: Optional[str] = None) -> Path:
        """Create and return an isolated filesystem working directory for one execution."""
        eid = execution_id or str(uuid.uuid4())
        work_dir = self.isolated_root / eid
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(work_dir, 0o700)
        except OSError:
            pass
        return work_dir

    def execute(
        self,
        command: list[str] | str,
        *,
        limits: Optional[ResourceLimits] = None,
        work_dir: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        self._execution_count += 1
        execution_id = str(uuid.uuid4())
        cmd_list = shlex.split(command) if isinstance(command, str) else list(command)
        active_limits = limits or self.default_limits

        try:
            self._validate_command(cmd_list)
        except CommandNotWhitelistedError as exc:
            result = ExecutionResult(execution_id=execution_id, status=ExecutionStatus.DENIED, error=str(exc))
            self._history.append(result)
            return result

        isolated_dir = Path(work_dir) if work_dir else self.isolate(execution_id)
        started_at = time.time()
        preexec = (lambda: _set_resource_limits(active_limits)) if self._is_posix else None

        try:
            proc = subprocess.Popen(
                cmd_list,
                cwd=str(isolated_dir),
                env=self._build_env(env),
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec,
                start_new_session=self._is_posix,
            )
        except (OSError, ValueError) as exc:
            result = ExecutionResult(
                execution_id=execution_id, status=ExecutionStatus.FAILED,
                error=str(exc), started_at=started_at, finished_at=time.time(),
                work_dir=str(isolated_dir),
            )
            self._history.append(result)
            return result

        self._active[execution_id] = proc

        try:
            stdout_b, stderr_b = proc.communicate(
                input=stdin_data.encode() if stdin_data else None,
                timeout=active_limits.wall_timeout_seconds,
            )
            status = ExecutionStatus.COMPLETED if proc.returncode == 0 else ExecutionStatus.FAILED
            result = ExecutionResult(
                execution_id=execution_id,
                status=status,
                stdout=stdout_b.decode(errors="replace"),
                stderr=stderr_b.decode(errors="replace"),
                return_code=proc.returncode,
                started_at=started_at,
                finished_at=time.time(),
                work_dir=str(isolated_dir),
            )
        except subprocess.TimeoutExpired:
            self.terminate(execution_id)
            result = ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.TIMEOUT,
                error=f"execution exceeded wall timeout of {active_limits.wall_timeout_seconds}s",
                started_at=started_at,
                finished_at=time.time(),
                work_dir=str(isolated_dir),
            )
        finally:
            result.duration_seconds = round((result.finished_at or time.time()) - started_at, 4)
            self._active.pop(execution_id, None)
            self._history.append(result)

        return result

    def terminate(self, execution_id: str) -> bool:
        proc = self._active.get(execution_id)
        if proc is None:
            return False
        try:
            if self._is_posix:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if self._is_posix:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:
                    proc.kill()
                proc.wait(timeout=3)
            return True
        except Exception:
            return False
        finally:
            self._active.pop(execution_id, None)

    def terminate_all(self) -> int:
        count = 0
        for eid in list(self._active.keys()):
            if self.terminate(eid):
                count += 1
        return count

    async def execute_async(
        self,
        command: list[str] | str,
        *,
        limits: Optional[ResourceLimits] = None,
        work_dir: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        async with self._lock:
            pass
        return await asyncio.to_thread(
            self.execute, command, limits=limits, work_dir=work_dir, env=env, stdin_data=stdin_data
        )

    async def terminate_async(self, execution_id: str) -> bool:
        return await asyncio.to_thread(self.terminate, execution_id)

    def history(self, limit: int = 100) -> list[ExecutionResult]:
        return self._history[-limit:]

    def health_check(self) -> HealthStatus:
        try:
            probe = self.execute(["echo", "ok"], limits=ResourceLimits(wall_timeout_seconds=3.0))
            healthy = probe.status == ExecutionStatus.COMPLETED and "ok" in probe.stdout
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "execution_count": self._execution_count,
                "active_executions": len(self._active),
                "history_size": len(self._history),
                "whitelist_size": len(self.command_whitelist),
                "isolated_root": str(self.isolated_root),
                "posix": self._is_posix,
                "probe_status": probe.status.value,
            }
            return HealthStatus(healthy=healthy, component="sandbox", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="sandbox", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Sandbox",
    "ExecutionResult",
    "ExecutionStatus",
    "ResourceLimits",
    "SandboxError",
    "CommandNotWhitelistedError",
    "ResourceLimitError",
    "HealthStatus",
]
