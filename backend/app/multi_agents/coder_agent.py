from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from app.multi_agents.base_agent import (
    Agent,
    AgentMetadata,
    AgentPriority,
    AgentTask,
    HealthCheckResult,
    HealthStatus,
)


class CoderError(Exception):
    pass


class PatchApplyError(CoderError):
    pass


class TestExecutionError(CoderError):
    pass


class ChangeType(Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    REFACTOR = "refactor"


@dataclass(frozen=True)
class GeneratedFile:
    path: str
    content: str
    change_type: ChangeType
    language: str = ""


@dataclass(frozen=True)
class CodePatch:
    patch_id: str
    file_path: str
    diff_text: str
    original_content: str
    new_content: str
    created_at_epoch: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TestResult:
    test_name: str
    passed: bool
    output: str
    duration_seconds: float
    error: Optional[str] = None


@dataclass(frozen=True)
class TestRunSummary:
    total: int
    passed: int
    failed: int
    results: tuple[TestResult, ...]
    duration_seconds: float


class ReviewSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class CodeReviewFinding:
    severity: ReviewSeverity
    message: str
    file_path: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class CodeReviewResult:
    findings: tuple[CodeReviewFinding, ...]
    approved: bool
    quality_score: float


@dataclass(frozen=True)
class GitStatus:
    branch: str
    is_dirty: bool
    staged_files: tuple[str, ...]
    unstaged_files: tuple[str, ...]
    untracked_files: tuple[str, ...]


CodeGeneratorFn = Callable[[str, dict[str, Any]], Awaitable[list[GeneratedFile]]]
TestRunnerFn = Callable[[list[str]], Awaitable[TestRunSummary]]
GitStatusFn = Callable[[], Awaitable[GitStatus]]
GitCommitFn = Callable[[str, list[str]], Awaitable[str]]


class CoderAgent(Agent):
    def __init__(
        self,
        code_generator_fn: Optional[CodeGeneratorFn] = None,
        test_runner_fn: Optional[TestRunnerFn] = None,
        git_status_fn: Optional[GitStatusFn] = None,
        git_commit_fn: Optional[GitCommitFn] = None,
        metadata: Optional[AgentMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or AgentMetadata(
                name="coder_agent",
                version="1.0.0",
                description="Generates, refactors, tests, reviews, and patches code with git awareness",
                priority=AgentPriority.NORMAL,
                capabilities=("code_generation", "refactoring", "testing", "code_review"),
                timeout_seconds=180.0,
                approval_required=True,
            )
        )
        self._code_generator_fn = code_generator_fn
        self._test_runner_fn = test_runner_fn
        self._git_status_fn = git_status_fn
        self._git_commit_fn = git_commit_fn
        self._generated_files_cache: dict[str, str] = {}
        self._patch_sequence = 0

    async def _on_initialize(self) -> None:
        self._generated_files_cache = {}
        self._patch_sequence = 0

    async def _on_shutdown(self) -> None:
        self._generated_files_cache.clear()

    async def _on_health_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            agent_name=self.name,
            status=HealthStatus.HEALTHY,
            latency_seconds=None,
            checked_at_epoch=time.time(),
            detail=f"cached_files={len(self._generated_files_cache)}",
        )

    async def _on_execute(self, task: AgentTask) -> Any:
        action = task.action
        params = task.parameters

        if action == "generate_code":
            return await self.generate_code(str(params["spec"]), params.get("context", {}))
        if action == "refactor":
            return await self.refactor(str(params["file_path"]), str(params["original_content"]), str(params["instructions"]))
        if action == "run_tests":
            return await self.run_tests(list(params.get("test_paths", [])))
        if action == "review_code":
            return self.review_code(str(params["file_path"]), str(params["content"]))
        if action == "create_patch":
            return self.create_patch(
                str(params["file_path"]),
                str(params["original_content"]),
                str(params["new_content"]),
            )
        if action == "git_status":
            return await self.git_status()
        if action == "git_commit":
            return await self.git_commit(str(params["message"]), list(params.get("files", [])))

        raise ValueError(f"unknown coder action: {action}")

    async def generate_code(
        self, spec: str, context: dict[str, Any]
    ) -> list[GeneratedFile]:
        if self._code_generator_fn is None:
            raise CoderError("no code generator function configured")
        files = await self._code_generator_fn(spec, context)
        for generated_file in files:
            self._generated_files_cache[generated_file.path] = generated_file.content
        return files

    async def refactor(
        self, file_path: str, original_content: str, instructions: str
    ) -> CodePatch:
        if self._code_generator_fn is None:
            raise CoderError("no code generator function configured for refactoring")

        files = await self._code_generator_fn(
            instructions, {"file_path": file_path, "original_content": original_content}
        )
        if not files:
            raise CoderError("refactor produced no output")

        new_content = files[0].content
        return self.create_patch(file_path, original_content, new_content)

    def create_patch(
        self, file_path: str, original_content: str, new_content: str
    ) -> CodePatch:
        self._patch_sequence += 1
        diff_lines = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        diff_text = "".join(diff_lines)
        return CodePatch(
            patch_id=f"patch-{self._patch_sequence}-{int(time.time() * 1000)}",
            file_path=file_path,
            diff_text=diff_text,
            original_content=original_content,
            new_content=new_content,
        )

    @staticmethod
    def apply_patch(patch: CodePatch, current_content: str) -> str:
        if current_content != patch.original_content:
            raise PatchApplyError(
                f"current content of '{patch.file_path}' does not match patch base; cannot apply cleanly"
            )
        return patch.new_content

    async def run_tests(self, test_paths: list[str]) -> TestRunSummary:
        if self._test_runner_fn is None:
            raise TestExecutionError("no test runner function configured")
        start = time.monotonic()
        try:
            summary = await self._test_runner_fn(test_paths)
            return summary
        except Exception as exc:
            duration = time.monotonic() - start
            return TestRunSummary(
                total=0,
                passed=0,
                failed=0,
                results=(
                    TestResult(
                        test_name="test_execution",
                        passed=False,
                        output="",
                        duration_seconds=duration,
                        error=str(exc),
                    ),
                ),
                duration_seconds=duration,
            )

    def review_code(self, file_path: str, content: str) -> CodeReviewResult:
        findings: list[CodeReviewFinding] = []
        lines = content.splitlines()

        for index, line in enumerate(lines, start=1):
            stripped = line.rstrip()
            if len(line) > 120:
                findings.append(
                    CodeReviewFinding(
                        severity=ReviewSeverity.WARNING,
                        message="line exceeds 120 characters",
                        file_path=file_path,
                        line_number=index,
                        suggestion="break this line into multiple lines",
                    )
                )
            if line != stripped and line.strip() != "":
                findings.append(
                    CodeReviewFinding(
                        severity=ReviewSeverity.INFO,
                        message="trailing whitespace detected",
                        file_path=file_path,
                        line_number=index,
                    )
                )
            if "TODO" in line or "FIXME" in line:
                findings.append(
                    CodeReviewFinding(
                        severity=ReviewSeverity.INFO,
                        message="unresolved TODO/FIXME marker",
                        file_path=file_path,
                        line_number=index,
                    )
                )
            if "except:" in line or "except Exception:" in line:
                findings.append(
                    CodeReviewFinding(
                        severity=ReviewSeverity.WARNING,
                        message="overly broad exception handling",
                        file_path=file_path,
                        line_number=index,
                        suggestion="catch specific exception types",
                    )
                )

        error_count = sum(1 for f in findings if f.severity == ReviewSeverity.ERROR)
        warning_count = sum(1 for f in findings if f.severity == ReviewSeverity.WARNING)
        quality_score = max(0.0, 1.0 - (error_count * 0.2 + warning_count * 0.05))
        approved = error_count == 0 and quality_score >= 0.7

        return CodeReviewResult(
            findings=tuple(findings),
            approved=approved,
            quality_score=quality_score,
        )

    async def git_status(self) -> GitStatus:
        if self._git_status_fn is None:
            raise CoderError("no git status function configured")
        return await self._git_status_fn()

    async def git_commit(self, message: str, files: list[str]) -> str:
        if self._git_commit_fn is None:
            raise CoderError("no git commit function configured")
        return await self._git_commit_fn(message, files)
