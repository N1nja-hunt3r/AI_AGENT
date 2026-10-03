"""Reviewer Agent: validates, scores, and approves outputs produced by other agents."""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger("reviewer_agent")


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    PENDING = "pending"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Severity = Severity.MEDIUM
    field: Optional[str] = None


@dataclass
class Improvement:
    suggestion: str
    rationale: str = ""
    priority: Severity = Severity.MEDIUM


@dataclass
class QualityScore:
    overall: float
    correctness: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    consistency: float = 0.0
    safety: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    review_id: str
    status: ReviewStatus
    quality: QualityScore
    issues: List[ValidationIssue] = field(default_factory=list)
    improvements: List[Improvement] = field(default_factory=list)
    consistent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ApprovalDecision:
    approval_id: str
    status: ApprovalStatus
    review: ReviewResult
    reasons: List[str] = field(default_factory=list)
    approver: str = "reviewer_agent"
    created_at: float = field(default_factory=time.time)


ValidatorFn = Callable[[Any], List[ValidationIssue]]


class ReviewerAgent:
    """Agent responsible for reviewing, validating, scoring, and approving outputs."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        quality_threshold: float = 0.7,
        approval_threshold: float = 0.8,
        max_history: int = 1000,
    ) -> None:
        self.agent_id: str = agent_id or f"reviewer-{uuid.uuid4().hex[:8]}"
        self.quality_threshold = quality_threshold
        self.approval_threshold = approval_threshold
        self.max_history = max_history
        self._lock = asyncio.Lock()
        self._history: List[ReviewResult] = []
        self._custom_validators: List[ValidatorFn] = []
        self._metrics: Dict[str, int] = {
            "reviews_total": 0,
            "passed": 0,
            "failed": 0,
            "needs_revision": 0,
            "approved": 0,
            "rejected": 0,
        }

    # ---------- validator management ----------

    def register_validator(self, validator: ValidatorFn) -> None:
        self._custom_validators.append(validator)

    # ---------- core validation ----------

    def validate_result(
        self,
        output: Any,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        criteria = criteria or {}

        if output is None:
            issues.append(
                ValidationIssue(
                    code="EMPTY_OUTPUT",
                    message="Output is None.",
                    severity=Severity.CRITICAL,
                )
            )
            return issues

        if isinstance(output, str) and not output.strip():
            issues.append(
                ValidationIssue(
                    code="EMPTY_STRING",
                    message="Output string is empty or whitespace only.",
                    severity=Severity.HIGH,
                )
            )

        required_fields = criteria.get("required_fields")
        if required_fields and isinstance(output, dict):
            for f_name in required_fields:
                if f_name not in output or output[f_name] in (None, "", [], {}):
                    issues.append(
                        ValidationIssue(
                            code="MISSING_FIELD",
                            message=f"Required field '{f_name}' is missing or empty.",
                            severity=Severity.HIGH,
                            field=f_name,
                        )
                    )

        min_length = criteria.get("min_length")
        if min_length is not None and isinstance(output, (str, list, dict)):
            length = len(output)
            if length < min_length:
                issues.append(
                    ValidationIssue(
                        code="TOO_SHORT",
                        message=f"Output length {length} is below minimum {min_length}.",
                        severity=Severity.MEDIUM,
                    )
                )

        max_length = criteria.get("max_length")
        if max_length is not None and isinstance(output, (str, list, dict)):
            length = len(output)
            if length > max_length:
                issues.append(
                    ValidationIssue(
                        code="TOO_LONG",
                        message=f"Output length {length} exceeds maximum {max_length}.",
                        severity=Severity.LOW,
                    )
                )

        pattern = criteria.get("pattern")
        if pattern and isinstance(output, str):
            if not re.search(pattern, output):
                issues.append(
                    ValidationIssue(
                        code="PATTERN_MISMATCH",
                        message=f"Output does not match required pattern '{pattern}'.",
                        severity=Severity.MEDIUM,
                    )
                )

        forbidden_terms = criteria.get("forbidden_terms")
        if forbidden_terms and isinstance(output, str):
            lowered = output.lower()
            for term in forbidden_terms:
                if term.lower() in lowered:
                    issues.append(
                        ValidationIssue(
                            code="FORBIDDEN_TERM",
                            message=f"Output contains forbidden term '{term}'.",
                            severity=Severity.HIGH,
                        )
                    )

        for validator in self._custom_validators:
            try:
                issues.extend(validator(output))
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    ValidationIssue(
                        code="VALIDATOR_ERROR",
                        message=f"Custom validator raised an exception: {exc}",
                        severity=Severity.MEDIUM,
                    )
                )

        return issues

    async def validate_result_async(
        self, output: Any, criteria: Optional[Dict[str, Any]] = None
    ) -> List[ValidationIssue]:
        return await asyncio.to_thread(self.validate_result, output, criteria)

    # ---------- consistency checks ----------

    def check_consistency(
        self,
        outputs: Sequence[Any],
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[bool, List[ValidationIssue]]:
        issues: List[ValidationIssue] = []
        if not outputs or len(outputs) < 2:
            return True, issues

        if key_fn is None:
            def key_fn(x: Any) -> Any:
                return x

        keyed = [key_fn(o) for o in outputs]
        first = keyed[0]
        consistent = True
        for idx, value in enumerate(keyed[1:], start=1):
            if value != first:
                consistent = False
                issues.append(
                    ValidationIssue(
                        code="INCONSISTENT_OUTPUT",
                        message=f"Output at index {idx} differs from baseline output.",
                        severity=Severity.MEDIUM,
                    )
                )

        if all(isinstance(o, (int, float)) for o in outputs):
            values = [float(o) for o in outputs]
            if len(values) > 1 and statistics.pstdev(values) > (statistics.mean(values) * 0.5 + 1e-9):
                issues.append(
                    ValidationIssue(
                        code="HIGH_VARIANCE",
                        message="Numeric outputs show high variance across samples.",
                        severity=Severity.LOW,
                    )
                )

        return consistent, issues

    async def check_consistency_async(
        self,
        outputs: Sequence[Any],
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[bool, List[ValidationIssue]]:
        return await asyncio.to_thread(self.check_consistency, outputs, key_fn)

    # ---------- quality scoring ----------

    def score_quality(
        self,
        output: Any,
        issues: Optional[List[ValidationIssue]] = None,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> QualityScore:
        issues = issues if issues is not None else self.validate_result(output, criteria)

        severity_penalty = {
            Severity.LOW: 0.03,
            Severity.MEDIUM: 0.08,
            Severity.HIGH: 0.2,
            Severity.CRITICAL: 0.5,
        }
        penalty = sum(severity_penalty[i.severity] for i in issues)
        correctness = max(0.0, 1.0 - penalty)

        completeness = 1.0
        if isinstance(output, (str, list, dict)):
            length = len(output)
            completeness = min(1.0, length / 50.0) if length else 0.0

        clarity = 1.0
        if isinstance(output, str):
            words = output.split()
            avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
            clarity = 1.0 if 2.5 <= avg_word_len <= 9 else 0.7

        consistency = 1.0 if not any(i.code == "INCONSISTENT_OUTPUT" for i in issues) else 0.5

        safety = 1.0 if not any(i.code == "FORBIDDEN_TERM" for i in issues) else 0.2

        overall = round(
            0.35 * correctness
            + 0.2 * completeness
            + 0.15 * clarity
            + 0.15 * consistency
            + 0.15 * safety,
            4,
        )

        return QualityScore(
            overall=overall,
            correctness=round(correctness, 4),
            completeness=round(completeness, 4),
            clarity=round(clarity, 4),
            consistency=round(consistency, 4),
            safety=round(safety, 4),
            details={"issue_count": len(issues)},
        )

    async def score_quality_async(
        self,
        output: Any,
        issues: Optional[List[ValidationIssue]] = None,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> QualityScore:
        return await asyncio.to_thread(self.score_quality, output, issues, criteria)

    # ---------- improvement suggestions ----------

    def suggest_improvements(
        self, output: Any, issues: Optional[List[ValidationIssue]] = None
    ) -> List[Improvement]:
        issues = issues if issues is not None else self.validate_result(output)
        improvements: List[Improvement] = []

        for issue in issues:
            if issue.code in ("EMPTY_OUTPUT", "EMPTY_STRING"):
                improvements.append(
                    Improvement(
                        suggestion="Provide a non-empty, substantive output.",
                        rationale="Output was empty or missing content.",
                        priority=Severity.CRITICAL,
                    )
                )
            elif issue.code == "MISSING_FIELD":
                improvements.append(
                    Improvement(
                        suggestion=f"Populate the missing field '{issue.field}'.",
                        rationale=issue.message,
                        priority=Severity.HIGH,
                    )
                )
            elif issue.code == "TOO_SHORT":
                improvements.append(
                    Improvement(
                        suggestion="Expand the output with more detail or supporting content.",
                        rationale=issue.message,
                        priority=Severity.MEDIUM,
                    )
                )
            elif issue.code == "TOO_LONG":
                improvements.append(
                    Improvement(
                        suggestion="Condense the output to remove redundancy.",
                        rationale=issue.message,
                        priority=Severity.LOW,
                    )
                )
            elif issue.code == "PATTERN_MISMATCH":
                improvements.append(
                    Improvement(
                        suggestion="Reformat the output to match the required pattern.",
                        rationale=issue.message,
                        priority=Severity.MEDIUM,
                    )
                )
            elif issue.code == "FORBIDDEN_TERM":
                improvements.append(
                    Improvement(
                        suggestion="Remove or replace disallowed terminology.",
                        rationale=issue.message,
                        priority=Severity.HIGH,
                    )
                )
            elif issue.code == "INCONSISTENT_OUTPUT":
                improvements.append(
                    Improvement(
                        suggestion="Reconcile differing outputs to ensure a single consistent result.",
                        rationale=issue.message,
                        priority=Severity.MEDIUM,
                    )
                )
            else:
                improvements.append(
                    Improvement(
                        suggestion=f"Address issue: {issue.message}",
                        rationale=issue.code,
                        priority=issue.severity,
                    )
                )

        return improvements

    async def suggest_improvements_async(
        self, output: Any, issues: Optional[List[ValidationIssue]] = None
    ) -> List[Improvement]:
        return await asyncio.to_thread(self.suggest_improvements, output, issues)

    # ---------- full review ----------

    def review_output(
        self,
        output: Any,
        criteria: Optional[Dict[str, Any]] = None,
        related_outputs: Optional[Sequence[Any]] = None,
    ) -> ReviewResult:
        review_id = f"rev-{uuid.uuid4().hex[:10]}"
        issues = self.validate_result(output, criteria)

        consistent = True
        if related_outputs:
            consistent, consistency_issues = self.check_consistency([output, *related_outputs])
            issues.extend(consistency_issues)

        quality = self.score_quality(output, issues, criteria)
        improvements = self.suggest_improvements(output, issues)

        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        has_high = any(i.severity == Severity.HIGH for i in issues)

        if has_critical:
            status = ReviewStatus.FAILED
        elif quality.overall < self.quality_threshold or has_high:
            status = ReviewStatus.NEEDS_REVISION
        else:
            status = ReviewStatus.PASSED

        result = ReviewResult(
            review_id=review_id,
            status=status,
            quality=quality,
            issues=issues,
            improvements=improvements,
            consistent=consistent,
            metadata={"criteria": criteria or {}},
        )

        self._record(result)
        return result

    async def review_output_async(
        self,
        output: Any,
        criteria: Optional[Dict[str, Any]] = None,
        related_outputs: Optional[Sequence[Any]] = None,
    ) -> ReviewResult:
        return await asyncio.to_thread(self.review_output, output, criteria, related_outputs)

    def _record(self, result: ReviewResult) -> None:
        self._history.append(result)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        self._metrics["reviews_total"] += 1
        self._metrics[result.status.value] = self._metrics.get(result.status.value, 0) + 1

    # ---------- approval ----------

    def approve(
        self,
        output: Any,
        criteria: Optional[Dict[str, Any]] = None,
        related_outputs: Optional[Sequence[Any]] = None,
        require_review: Optional[ReviewResult] = None,
    ) -> ApprovalDecision:
        review = require_review or self.review_output(output, criteria, related_outputs)
        approval_id = f"appr-{uuid.uuid4().hex[:10]}"
        reasons: List[str] = []

        if review.status == ReviewStatus.FAILED:
            status = ApprovalStatus.REJECTED
            reasons.append("Critical validation issues present.")
        elif review.quality.overall >= self.approval_threshold and not any(
            i.severity in (Severity.HIGH, Severity.CRITICAL) for i in review.issues
        ):
            status = ApprovalStatus.APPROVED
            reasons.append("Quality score meets approval threshold.")
        elif review.quality.overall >= self.quality_threshold:
            status = ApprovalStatus.CONDITIONAL
            reasons.append("Quality acceptable but below approval threshold; revisions recommended.")
        else:
            status = ApprovalStatus.REJECTED
            reasons.append("Quality score below minimum threshold.")

        decision = ApprovalDecision(
            approval_id=approval_id,
            status=status,
            review=review,
            reasons=reasons,
        )

        if status == ApprovalStatus.APPROVED:
            self._metrics["approved"] += 1
        elif status == ApprovalStatus.REJECTED:
            self._metrics["rejected"] += 1

        return decision

    async def approve_async(
        self,
        output: Any,
        criteria: Optional[Dict[str, Any]] = None,
        related_outputs: Optional[Sequence[Any]] = None,
        require_review: Optional[ReviewResult] = None,
    ) -> ApprovalDecision:
        async with self._lock:
            return await asyncio.to_thread(
                self.approve, output, criteria, related_outputs, require_review
            )

    # ---------- introspection ----------

    def get_history(self, limit: int = 50) -> List[ReviewResult]:
        return self._history[-limit:]

    def get_metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    async def health_check(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "reviews_total": self._metrics["reviews_total"],
            "history_size": len(self._history),
            "timestamp": time.time(),
        }
