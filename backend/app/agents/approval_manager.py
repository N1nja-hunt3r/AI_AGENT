"""
Approval Manager - Human approval workflow for the AI Agent Platform.

This module provides comprehensive approval management including:
- Human-in-the-loop approval requests
- Dangerous action detection and gating
- Computer/terminal action approval
- File deletion safeguards
- Configurable timeouts and escalation
- Approval status tracking and audit logging
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# ENUMS
# =============================================================================


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()
    ESCALATED = auto()
    AUTO_APPROVED = auto()
    AUTO_REJECTED = auto()


class ActionCategory(Enum):
    """Categories of actions requiring approval."""

    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    DESTRUCTIVE = "destructive"


class ActionType(Enum):
    """Types of actions that may require approval."""

    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_MOVE = "file_move"
    FILE_PERMISSION = "file_permission"
    DIRECTORY_DELETE = "directory_delete"

    # Terminal/shell operations
    TERMINAL_COMMAND = "terminal_command"
    SHELL_SCRIPT = "shell_script"
    PROCESS_KILL = "process_kill"
    SERVICE_CONTROL = "service_control"

    # System operations
    SYSTEM_CONFIG = "system_config"
    NETWORK_CONFIG = "network_config"
    USER_MANAGEMENT = "user_management"
    PACKAGE_INSTALL = "package_install"
    PACKAGE_REMOVE = "package_remove"

    # Code operations
    CODE_EXECUTION = "code_execution"
    CODE_DEPLOY = "code_deploy"
    DATABASE_WRITE = "database_write"
    DATABASE_DELETE = "database_delete"
    DATABASE_SCHEMA = "database_schema"

    # External operations
    API_CALL = "api_call"
    EMAIL_SEND = "email_send"
    PAYMENT = "payment"
    DATA_EXPORT = "data_export"

    # Agent operations
    AGENT_SPAWN = "agent_spawn"
    TOOL_INSTALL = "tool_install"
    PERMISSION_GRANT = "permission_grant"

    # Generic
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class EscalationLevel(Enum):
    """Escalation levels for approval requests."""

    NONE = auto()
    NOTIFY = auto()
    REQUIRE_SENIOR = auto()
    REQUIRE_ADMIN = auto()
    BLOCK = auto()


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ApprovalError(Exception):
    """Base exception for approval errors."""

    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class ApprovalTimeoutError(ApprovalError):
    """Approval request timed out."""

    pass


class ApprovalRejectedError(ApprovalError):
    """Approval was rejected."""

    def __init__(
        self,
        message: str,
        request_id: str | None = None,
        reason: str = "",
    ) -> None:
        super().__init__(message, request_id)
        self.reason = reason


class ApprovalRequiredError(ApprovalError):
    """Action requires approval but none was requested."""

    pass


# =============================================================================
# ACTION DETECTION
# =============================================================================


@dataclass(frozen=True)
class DangerousPattern:
    """
    Pattern for detecting dangerous actions.

    Attributes:
        pattern: Regex pattern to match
        action_type: Type of action
        category: Risk category
        description: Human-readable description
        requires_approval: Whether approval is required
    """

    pattern: str
    action_type: ActionType
    category: ActionCategory
    description: str
    requires_approval: bool = True

    def matches(self, content: str) -> bool:
        """Check if content matches pattern."""
        return bool(re.search(self.pattern, content, re.IGNORECASE | re.MULTILINE))


# Default dangerous patterns for terminal commands
DANGEROUS_TERMINAL_PATTERNS: tuple[DangerousPattern, ...] = (
    # Destructive file operations
    DangerousPattern(
        pattern=r"\brm\s+(-[rf]+\s+)*(/|~|\$HOME|\*)",
        action_type=ActionType.FILE_DELETE,
        category=ActionCategory.DESTRUCTIVE,
        description="Recursive or root file deletion",
    ),
    DangerousPattern(
        pattern=r"\brm\s+-[rf]*\s+",
        action_type=ActionType.FILE_DELETE,
        category=ActionCategory.HIGH_RISK,
        description="File deletion with force/recursive flags",
    ),
    DangerousPattern(
        pattern=r"\brmdir\b",
        action_type=ActionType.DIRECTORY_DELETE,
        category=ActionCategory.MEDIUM_RISK,
        description="Directory deletion",
    ),
    # Disk operations
    DangerousPattern(
        pattern=r"\b(mkfs|fdisk|parted|dd\s+if=)\b",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.DESTRUCTIVE,
        description="Disk formatting or low-level operations",
    ),
    DangerousPattern(
        pattern=r"\bdd\b.*\bof=/dev/",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.DESTRUCTIVE,
        description="Direct disk write",
    ),
    # Permission changes
    DangerousPattern(
        pattern=r"\bchmod\s+(-R\s+)?(777|666|a\+[rwx])",
        action_type=ActionType.FILE_PERMISSION,
        category=ActionCategory.HIGH_RISK,
        description="Overly permissive file permissions",
    ),
    DangerousPattern(
        pattern=r"\bchown\s+-R\s+",
        action_type=ActionType.FILE_PERMISSION,
        category=ActionCategory.MEDIUM_RISK,
        description="Recursive ownership change",
    ),
    # System control
    DangerousPattern(
        pattern=r"\b(shutdown|reboot|init\s+[0-6]|poweroff)\b",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.CRITICAL,
        description="System shutdown or reboot",
    ),
    DangerousPattern(
        pattern=r"\bsystemctl\s+(stop|disable|mask)\b",
        action_type=ActionType.SERVICE_CONTROL,
        category=ActionCategory.HIGH_RISK,
        description="Service stop or disable",
    ),
    # Process control
    DangerousPattern(
        pattern=r"\bkill\s+(-9\s+)?(-1|1\b)",
        action_type=ActionType.PROCESS_KILL,
        category=ActionCategory.CRITICAL,
        description="Kill init or all processes",
    ),
    DangerousPattern(
        pattern=r"\bkillall\b",
        action_type=ActionType.PROCESS_KILL,
        category=ActionCategory.HIGH_RISK,
        description="Kill processes by name",
    ),
    # Network operations
    DangerousPattern(
        pattern=r"\b(iptables|ufw|firewall-cmd)\s+.*(DROP|REJECT|delete)",
        action_type=ActionType.NETWORK_CONFIG,
        category=ActionCategory.HIGH_RISK,
        description="Firewall rule modification",
    ),
    DangerousPattern(
        pattern=r"\bifconfig\s+\w+\s+down\b",
        action_type=ActionType.NETWORK_CONFIG,
        category=ActionCategory.HIGH_RISK,
        description="Network interface down",
    ),
    # User management
    DangerousPattern(
        pattern=r"\b(userdel|groupdel)\b",
        action_type=ActionType.USER_MANAGEMENT,
        category=ActionCategory.HIGH_RISK,
        description="User or group deletion",
    ),
    DangerousPattern(
        pattern=r"\bpasswd\s+",
        action_type=ActionType.USER_MANAGEMENT,
        category=ActionCategory.HIGH_RISK,
        description="Password change",
    ),
    # Package management
    DangerousPattern(
        pattern=r"\b(apt|yum|dnf|pacman)\s+(remove|purge|autoremove)\b",
        action_type=ActionType.PACKAGE_REMOVE,
        category=ActionCategory.MEDIUM_RISK,
        description="Package removal",
    ),
    # Dangerous commands
    DangerousPattern(
        pattern=r"\bcurl\b.*\|\s*(bash|sh|zsh)\b",
        action_type=ActionType.SHELL_SCRIPT,
        category=ActionCategory.CRITICAL,
        description="Piping remote content to shell",
    ),
    DangerousPattern(
        pattern=r"\bwget\b.*-O\s*-\s*\|\s*(bash|sh)\b",
        action_type=ActionType.SHELL_SCRIPT,
        category=ActionCategory.CRITICAL,
        description="Piping downloaded content to shell",
    ),
    DangerousPattern(
        pattern=r"\beval\b",
        action_type=ActionType.CODE_EXECUTION,
        category=ActionCategory.HIGH_RISK,
        description="Dynamic code evaluation",
    ),
    # Sudo operations
    DangerousPattern(
        pattern=r"\bsudo\s+",
        action_type=ActionType.TERMINAL_COMMAND,
        category=ActionCategory.MEDIUM_RISK,
        description="Elevated privilege execution",
    ),
    # Environment manipulation
    DangerousPattern(
        pattern=r"\bexport\s+(PATH|LD_LIBRARY_PATH|LD_PRELOAD)=",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.MEDIUM_RISK,
        description="Critical environment variable modification",
    ),
)

# Dangerous patterns for file paths
DANGEROUS_PATH_PATTERNS: tuple[DangerousPattern, ...] = (
    DangerousPattern(
        pattern=r"^/$",
        action_type=ActionType.FILE_DELETE,
        category=ActionCategory.DESTRUCTIVE,
        description="Root directory operation",
    ),
    DangerousPattern(
        pattern=r"^/etc/",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.HIGH_RISK,
        description="System configuration directory",
    ),
    DangerousPattern(
        pattern=r"^/boot/",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.CRITICAL,
        description="Boot directory",
    ),
    DangerousPattern(
        pattern=r"^/usr/(bin|sbin|lib)/",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.HIGH_RISK,
        description="System binary directory",
    ),
    DangerousPattern(
        pattern=r"^~?/\.(ssh|gnupg|config)/",
        action_type=ActionType.SYSTEM_CONFIG,
        category=ActionCategory.HIGH_RISK,
        description="Sensitive user configuration",
    ),
    DangerousPattern(
        pattern=r"\.(pem|key|crt|p12|pfx)$",
        action_type=ActionType.FILE_DELETE,
        category=ActionCategory.CRITICAL,
        description="Cryptographic key file",
    ),
    DangerousPattern(
        pattern=r"(password|secret|credential|token)",
        action_type=ActionType.FILE_DELETE,
        category=ActionCategory.HIGH_RISK,
        description="Potentially sensitive file",
    ),
)


class ActionDetector:
    """
    Detects dangerous actions and classifies risk.

    Analyzes commands, file paths, and operations to determine
    risk level and approval requirements.
    """

    def __init__(
        self,
        terminal_patterns: Sequence[DangerousPattern] | None = None,
        path_patterns: Sequence[DangerousPattern] | None = None,
        custom_patterns: Sequence[DangerousPattern] | None = None,
    ) -> None:
        """
        Initialize action detector.

        Args:
            terminal_patterns: Patterns for terminal commands
            path_patterns: Patterns for file paths
            custom_patterns: Additional custom patterns
        """
        self._terminal_patterns = list(terminal_patterns or DANGEROUS_TERMINAL_PATTERNS)
        self._path_patterns = list(path_patterns or DANGEROUS_PATH_PATTERNS)
        self._custom_patterns = list(custom_patterns or [])

    def analyze_command(self, command: str) -> list[DangerousPattern]:
        """
        Analyze a terminal command for dangerous patterns.

        Args:
            command: Command string to analyze

        Returns:
            List of matched dangerous patterns
        """
        matches: list[DangerousPattern] = []

        for pattern in self._terminal_patterns + self._custom_patterns:
            if pattern.matches(command):
                matches.append(pattern)

        return matches

    def analyze_path(self, path: str, operation: ActionType) -> list[DangerousPattern]:
        """
        Analyze a file path for dangerous patterns.

        Args:
            path: File path to analyze
            operation: Type of operation on the path

        Returns:
            List of matched dangerous patterns
        """
        matches: list[DangerousPattern] = []

        for pattern in self._path_patterns:
            if pattern.matches(path):
                # Adjust category based on operation
                if operation in {ActionType.FILE_DELETE, ActionType.DIRECTORY_DELETE}:
                    matches.append(pattern)
                elif operation == ActionType.FILE_WRITE and pattern.category.value >= ActionCategory.HIGH_RISK.value:
                    matches.append(pattern)

        return matches

    def get_risk_category(
        self,
        action_type: ActionType,
        command: str | None = None,
        path: str | None = None,
    ) -> ActionCategory:
        """
        Determine overall risk category for an action.

        Args:
            action_type: Type of action
            command: Optional command string
            path: Optional file path

        Returns:
            Highest risk category found
        """
        categories: list[ActionCategory] = [ActionCategory.SAFE]

        # Check command patterns
        if command:
            for pattern in self.analyze_command(command):
                categories.append(pattern.category)

        # Check path patterns
        if path:
            for pattern in self.analyze_path(path, action_type):
                categories.append(pattern.category)

        # Default categories by action type
        default_categories = {
            ActionType.FILE_DELETE: ActionCategory.MEDIUM_RISK,
            ActionType.DIRECTORY_DELETE: ActionCategory.HIGH_RISK,
            ActionType.TERMINAL_COMMAND: ActionCategory.LOW_RISK,
            ActionType.SHELL_SCRIPT: ActionCategory.MEDIUM_RISK,
            ActionType.CODE_EXECUTION: ActionCategory.MEDIUM_RISK,
            ActionType.DATABASE_DELETE: ActionCategory.HIGH_RISK,
            ActionType.DATABASE_SCHEMA: ActionCategory.HIGH_RISK,
            ActionType.SYSTEM_CONFIG: ActionCategory.HIGH_RISK,
            ActionType.PAYMENT: ActionCategory.CRITICAL,
        }

        if action_type in default_categories:
            categories.append(default_categories[action_type])

        # Return highest risk
        category_order = [
            ActionCategory.SAFE,
            ActionCategory.LOW_RISK,
            ActionCategory.MEDIUM_RISK,
            ActionCategory.HIGH_RISK,
            ActionCategory.CRITICAL,
            ActionCategory.DESTRUCTIVE,
        ]

        return max(categories, key=lambda c: category_order.index(c))

    def requires_approval(
        self,
        action_type: ActionType,
        command: str | None = None,
        path: str | None = None,
        threshold: ActionCategory = ActionCategory.MEDIUM_RISK,
    ) -> bool:
        """
        Check if action requires approval.

        Args:
            action_type: Type of action
            command: Optional command string
            path: Optional file path
            threshold: Minimum category requiring approval

        Returns:
            True if approval is required
        """
        category = self.get_risk_category(action_type, command, path)

        category_order = [
            ActionCategory.SAFE,
            ActionCategory.LOW_RISK,
            ActionCategory.MEDIUM_RISK,
            ActionCategory.HIGH_RISK,
            ActionCategory.CRITICAL,
            ActionCategory.DESTRUCTIVE,
        ]

        return category_order.index(category) >= category_order.index(threshold)

    def add_pattern(self, pattern: DangerousPattern) -> None:
        """Add a custom pattern."""
        self._custom_patterns.append(pattern)


# =============================================================================
# APPROVAL REQUEST
# =============================================================================


@dataclass
class ApprovalContext:
    """
    Context information for an approval request.

    Attributes:
        session_id: Associated session
        request_id: Associated request
        agent_id: Agent requesting approval
        user_id: User who initiated the action
        environment: Execution environment
        previous_actions: Recent actions for context
        metadata: Additional context data
    """

    session_id: str | None = None
    request_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    environment: str = "unknown"
    previous_actions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """
    Request for human approval.

    Attributes:
        approval_id: Unique request identifier
        action_type: Type of action
        category: Risk category
        title: Short title
        description: Detailed description
        action_details: Specific action details
        context: Approval context
        status: Current status
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        timeout_seconds: Timeout duration
        requires_reason: Whether rejection requires reason
        auto_approve_after: Auto-approve after N seconds (None = never)
        escalation_level: Current escalation level
        approver_id: ID of approver (if approved/rejected)
        approved_at: Approval timestamp
        rejection_reason: Reason for rejection
        audit_log: Audit trail
    """

    approval_id: str
    action_type: ActionType
    category: ActionCategory
    title: str
    description: str
    action_details: dict[str, Any]
    context: ApprovalContext
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    timeout_seconds: int = 300
    requires_reason: bool = False
    auto_approve_after: int | None = None
    escalation_level: EscalationLevel = EscalationLevel.NONE
    approver_id: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str = ""
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set expiration time if not set."""
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(seconds=self.timeout_seconds)

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == ApprovalStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining before expiration."""
        if self.expires_at is None:
            return timedelta(seconds=self.timeout_seconds)
        remaining = self.expires_at - datetime.now(timezone.utc)
        return max(remaining, timedelta(0))

    @property
    def is_approved(self) -> bool:
        """Check if request was approved."""
        return self.status in {ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED}

    @property
    def is_rejected(self) -> bool:
        """Check if request was rejected."""
        return self.status in {
            ApprovalStatus.REJECTED,
            ApprovalStatus.AUTO_REJECTED,
            ApprovalStatus.TIMEOUT,
        }

    def add_audit_entry(self, action: str, details: dict[str, Any] | None = None) -> None:
        """Add entry to audit log."""
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details or {},
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approval_id": self.approval_id,
            "action_type": self.action_type.value,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "action_details": self.action_details,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "timeout_seconds": self.timeout_seconds,
            "time_remaining_seconds": self.time_remaining.total_seconds(),
            "escalation_level": self.escalation_level.name,
            "approver_id": self.approver_id,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "context": {
                "session_id": self.context.session_id,
                "agent_id": self.context.agent_id,
                "user_id": self.context.user_id,
                "environment": self.context.environment,
            },
        }


# =============================================================================
# APPROVAL HANDLERS
# =============================================================================


class ApprovalHandler(Protocol):
    """Protocol for approval request handlers."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval from handler."""
        ...

    async def cancel_request(self, approval_id: str) -> bool:
        """Cancel a pending request."""
        ...


class ConsoleApprovalHandler:
    """
    Console-based approval handler for CLI environments.

    Prompts user via console for approval decisions.
    """

    def __init__(
        self,
        auto_reject_on_timeout: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._auto_reject = auto_reject_on_timeout
        self._logger = logger or logging.getLogger(__name__)
        self._pending: dict[str, asyncio.Event] = {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval via console."""
        self._pending[request.approval_id] = asyncio.Event()

        # Display request
        print("\n" + "=" * 60)
        print(f"🔐 APPROVAL REQUIRED: {request.title}")
        print("=" * 60)
        print(f"Category: {request.category.value.upper()}")
        print(f"Action: {request.action_type.value}")
        print(f"Description: {request.description}")
        print("\nDetails:")
        for key, value in request.action_details.items():
            print(f"  {key}: {value}")
        print(f"\nTimeout: {request.timeout_seconds} seconds")
        print("=" * 60)

        try:
            # Wait for input with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input("Approve? [y/N]: ").strip().lower(),
                ),
                timeout=request.timeout_seconds,
            )

            if response in {"y", "yes", "approve"}:
                return ApprovalStatus.APPROVED
            else:
                return ApprovalStatus.REJECTED

        except asyncio.TimeoutError:
            print("\n⏰ Approval request timed out")
            return ApprovalStatus.TIMEOUT if self._auto_reject else ApprovalStatus.PENDING

        finally:
            self._pending.pop(request.approval_id, None)

    async def cancel_request(self, approval_id: str) -> bool:
        """Cancel a pending request."""
        if approval_id in self._pending:
            self._pending[approval_id].set()
            return True
        return False


class CallbackApprovalHandler:
    """
    Callback-based approval handler.

    Invokes callbacks for approval requests, useful for
    integrating with external systems (Slack, email, etc.).
    """

    def __init__(
        self,
        on_request: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
        on_decision: Callable[[ApprovalRequest, ApprovalStatus], Awaitable[None]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._on_request = on_request
        self._on_decision = on_decision
        self._logger = logger or logging.getLogger(__name__)
        self._pending: dict[str, asyncio.Future[ApprovalStatus]] = {}
        self._lock = asyncio.Lock()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval via callback."""
        async with self._lock:
            future: asyncio.Future[ApprovalStatus] = asyncio.Future()
            self._pending[request.approval_id] = future

        # Notify via callback
        if self._on_request:
            try:
                await self._on_request(request)
            except Exception as e:
                self._logger.error(f"Approval request callback error: {e}")

        # Wait for decision
        try:
            status = await asyncio.wait_for(
                future,
                timeout=request.timeout_seconds,
            )
        except asyncio.TimeoutError:
            status = ApprovalStatus.TIMEOUT
        finally:
            async with self._lock:
                self._pending.pop(request.approval_id, None)

        # Notify decision
        if self._on_decision:
            try:
                await self._on_decision(request, status)
            except Exception as e:
                self._logger.error(f"Approval decision callback error: {e}")

        return status

    async def submit_decision(
        self,
        approval_id: str,
        approved: bool,
        approver_id: str | None = None,
        reason: str = "",
    ) -> bool:
        """
        Submit a decision for a pending request.

        Args:
            approval_id: Request identifier
            approved: Whether to approve
            approver_id: ID of approver
            reason: Reason for decision

        Returns:
            True if decision was submitted
        """
        async with self._lock:
            future = self._pending.get(approval_id)
            if future and not future.done():
                status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
                future.set_result(status)
                return True
            return False

    async def cancel_request(self, approval_id: str) -> bool:
        """Cancel a pending request."""
        async with self._lock:
            future = self._pending.get(approval_id)
            if future and not future.done():
                future.set_result(ApprovalStatus.CANCELLED)
                return True
            return False


class AutoApprovalHandler:
    """
    Automatic approval handler based on rules.

    Useful for testing or low-risk environments.
    """

    def __init__(
        self,
        auto_approve_categories: set[ActionCategory] | None = None,
        auto_reject_categories: set[ActionCategory] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._auto_approve = auto_approve_categories or {
            ActionCategory.SAFE,
            ActionCategory.LOW_RISK,
        }
        self._auto_reject = auto_reject_categories or {
            ActionCategory.DESTRUCTIVE,
        }
        self._delay = delay_seconds

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Auto-approve or reject based on category."""
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if request.category in self._auto_approve:
            return ApprovalStatus.AUTO_APPROVED
        elif request.category in self._auto_reject:
            return ApprovalStatus.AUTO_REJECTED
        else:
            # Default to pending (will timeout)
            return ApprovalStatus.PENDING

    async def cancel_request(self, approval_id: str) -> bool:
        """Cancel not supported for auto handler."""
        return False


# =============================================================================
# APPROVAL STORE
# =============================================================================


class ApprovalStore(Protocol):
    """Protocol for approval request storage."""

    async def save(self, request: ApprovalRequest) -> None:
        """Save approval request."""
        ...

    async def load(self, approval_id: str) -> ApprovalRequest | None:
        """Load approval request."""
        ...

    async def update(self, request: ApprovalRequest) -> None:
        """Update approval request."""
        ...

    async def list_pending(self) -> list[ApprovalRequest]:
        """List pending requests."""
        ...

    async def list_by_session(self, session_id: str) -> list[ApprovalRequest]:
        """List requests by session."""
        ...


class InMemoryApprovalStore:
    """In-memory approval store."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()

    async def save(self, request: ApprovalRequest) -> None:
        async with self._lock:
            self._requests[request.approval_id] = request

    async def load(self, approval_id: str) -> ApprovalRequest | None:
        return self._requests.get(approval_id)

    async def update(self, request: ApprovalRequest) -> None:
        async with self._lock:
            self._requests[request.approval_id] = request

    async def list_pending(self) -> list[ApprovalRequest]:
        return [r for r in self._requests.values() if r.is_pending]

    async def list_by_session(self, session_id: str) -> list[ApprovalRequest]:
        return [
            r for r in self._requests.values()
            if r.context.session_id == session_id
        ]

    async def list_all(self) -> list[ApprovalRequest]:
        return list(self._requests.values())

    async def delete(self, approval_id: str) -> bool:
        async with self._lock:
            if approval_id in self._requests:
                del self._requests[approval_id]
                return True
            return False


# =============================================================================
# APPROVAL POLICY
# =============================================================================


@dataclass
class ApprovalPolicy:
    """
    Policy configuration for approval requirements.

    Attributes:
        name: Policy name
        enabled: Whether policy is active
        approval_threshold: Minimum category requiring approval
        timeout_seconds: Default timeout
        require_reason_on_reject: Require reason for rejection
        auto_approve_safe: Auto-approve safe actions
        escalation_rules: Rules for escalation
        allowed_approvers: List of allowed approver IDs
        blocked_actions: Actions that are always blocked
    """

    name: str = "default"
    enabled: bool = True
    approval_threshold: ActionCategory = ActionCategory.MEDIUM_RISK
    timeout_seconds: int = 300
    require_reason_on_reject: bool = False
    auto_approve_safe: bool = True
    escalation_rules: dict[ActionCategory, EscalationLevel] = field(
        default_factory=lambda: {
            ActionCategory.CRITICAL: EscalationLevel.REQUIRE_SENIOR,
            ActionCategory.DESTRUCTIVE: EscalationLevel.REQUIRE_ADMIN,
        }
    )
    allowed_approvers: set[str] = field(default_factory=set)
    blocked_actions: set[ActionType] = field(default_factory=set)

    def requires_approval(self, category: ActionCategory) -> bool:
        """Check if category requires approval."""
        if not self.enabled:
            return False

        if self.auto_approve_safe and category == ActionCategory.SAFE:
            return False

        category_order = [
            ActionCategory.SAFE,
            ActionCategory.LOW_RISK,
            ActionCategory.MEDIUM_RISK,
            ActionCategory.HIGH_RISK,
            ActionCategory.CRITICAL,
            ActionCategory.DESTRUCTIVE,
        ]

        return category_order.index(category) >= category_order.index(self.approval_threshold)

    def get_escalation_level(self, category: ActionCategory) -> EscalationLevel:
        """Get escalation level for category."""
        return self.escalation_rules.get(category, EscalationLevel.NONE)

    def is_action_blocked(self, action_type: ActionType) -> bool:
        """Check if action type is blocked."""
        return action_type in self.blocked_actions

    def can_approve(self, approver_id: str) -> bool:
        """Check if approver is allowed."""
        if not self.allowed_approvers:
            return True
        return approver_id in self.allowed_approvers


# =============================================================================
# APPROVAL MANAGER
# =============================================================================


@dataclass
class ApprovalManagerConfig:
    """
    Configuration for ApprovalManager.

    Attributes:
        default_timeout_seconds: Default approval timeout
        max_pending_requests: Maximum pending requests
        enable_audit_log: Enable audit logging
        cleanup_expired_interval: Interval for cleanup
    """

    default_timeout_seconds: int = 300
    max_pending_requests: int = 100
    enable_audit_log: bool = True
    cleanup_expired_interval: int = 60


class ApprovalManager:
    """
    Central approval management system.

    Manages approval requests, policies, and handlers
    for human-in-the-loop workflows.
    """

    def __init__(
        self,
        config: ApprovalManagerConfig | None = None,
        handler: ApprovalHandler | None = None,
        store: ApprovalStore | None = None,
        policy: ApprovalPolicy | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize approval manager.

        Args:
            config: Manager configuration
            handler: Approval handler
            store: Approval store
            policy: Approval policy
            logger: Optional logger
        """
        self._config = config or ApprovalManagerConfig()
        self._handler = handler or CallbackApprovalHandler()
        self._store = store or InMemoryApprovalStore()
        self._policy = policy or ApprovalPolicy()
        self._logger = logger or logging.getLogger(__name__)

        # Components
        self._detector = ActionDetector()

        # State
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    # -------------------------------------------------------------------------
    # Request Creation
    # -------------------------------------------------------------------------

    async def request_approval(
        self,
        action_type: ActionType,
        title: str,
        description: str,
        action_details: dict[str, Any] | None = None,
        context: ApprovalContext | None = None,
        timeout_seconds: int | None = None,
        command: str | None = None,
        path: str | None = None,
    ) -> ApprovalRequest:
        """
        Create and submit an approval request.

        Args:
            action_type: Type of action
            title: Short title
            description: Detailed description
            action_details: Specific action details
            context: Approval context
            timeout_seconds: Custom timeout
            command: Command string for analysis
            path: File path for analysis

        Returns:
            ApprovalRequest with final status

        Raises:
            ApprovalRequiredError: If action is blocked
            ApprovalTimeoutError: If request times out
            ApprovalRejectedError: If request is rejected
        """
        # Check if action is blocked
        if self._policy.is_action_blocked(action_type):
            raise ApprovalRequiredError(
                f"Action type {action_type.value} is blocked by policy"
            )

        # Determine risk category
        category = self._detector.get_risk_category(action_type, command, path)

        # Check if approval is required
        if not self._policy.requires_approval(category):
            # Auto-approve
            request = ApprovalRequest(
                approval_id=str(uuid4()),
                action_type=action_type,
                category=category,
                title=title,
                description=description,
                action_details=action_details or {},
                context=context or ApprovalContext(),
                status=ApprovalStatus.AUTO_APPROVED,
                timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
            )
            request.add_audit_entry("auto_approved", {"reason": "below_threshold"})
            await self._store.save(request)
            return request

        # Create request
        request = ApprovalRequest(
            approval_id=str(uuid4()),
            action_type=action_type,
            category=category,
            title=title,
            description=description,
            action_details=action_details or {},
            context=context or ApprovalContext(),
            timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
            requires_reason=self._policy.require_reason_on_reject,
            escalation_level=self._policy.get_escalation_level(category),
        )

        # Add command/path to details
        if command:
            request.action_details["command"] = command
        if path:
            request.action_details["path"] = path

        request.add_audit_entry("created", {
            "category": category.value,
            "escalation": request.escalation_level.name,
        })

        # Save request
        await self._store.save(request)

        self._logger.info(
            f"Approval requested: {request.approval_id} - {title} ({category.value})"
        )

        # Submit to handler
        try:
            status = await self._handler.request_approval(request)
            request.status = status
            request.approved_at = datetime.now(timezone.utc) if request.is_approved else None

            request.add_audit_entry(
                "decision",
                {"status": status.name},
            )

        except Exception as e:
            self._logger.error(f"Approval handler error: {e}")
            request.status = ApprovalStatus.TIMEOUT
            request.add_audit_entry("error", {"message": str(e)})

        # Update stored request
        await self._store.update(request)

        # Handle result
        if request.status == ApprovalStatus.TIMEOUT:
            raise ApprovalTimeoutError(
                f"Approval request timed out: {title}",
                request_id=request.approval_id,
            )

        if request.is_rejected:
            raise ApprovalRejectedError(
                f"Approval rejected: {title}",
                request_id=request.approval_id,
                reason=request.rejection_reason,
            )

        return request

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def request_terminal_approval(
        self,
        command: str,
        context: ApprovalContext | None = None,
        timeout_seconds: int | None = None,
    ) -> ApprovalRequest:
        """
        Request approval for terminal command execution.

        Args:
            command: Command to execute
            context: Approval context
            timeout_seconds: Custom timeout

        Returns:
            ApprovalRequest with final status
        """
        # Analyze command
        patterns = self._detector.analyze_command(command)
        descriptions = [p.description for p in patterns]

        return await self.request_approval(
            action_type=ActionType.TERMINAL_COMMAND,
            title=f"Execute: {command[:50]}{'...' if len(command) > 50 else ''}",
            description=f"Terminal command execution. Detected patterns: {', '.join(descriptions) or 'none'}",
            action_details={
                "command": command,
                "detected_patterns": [p.pattern for p in patterns],
            },
            context=context,
            timeout_seconds=timeout_seconds,
            command=command,
        )

    async def request_file_delete_approval(
        self,
        path: str,
        recursive: bool = False,
        context: ApprovalContext | None = None,
        timeout_seconds: int | None = None,
    ) -> ApprovalRequest:
        """
        Request approval for file deletion.

        Args:
            path: File or directory path
            recursive: Whether deletion is recursive
            context: Approval context
            timeout_seconds: Custom timeout

        Returns:
            ApprovalRequest with final status
        """
        action_type = ActionType.DIRECTORY_DELETE if recursive else ActionType.FILE_DELETE

        return await self.request_approval(
            action_type=action_type,
            title=f"Delete: {path}",
            description=f"{'Recursive ' if recursive else ''}file deletion",
            action_details={
                "path": path,
                "recursive": recursive,
            },
            context=context,
            timeout_seconds=timeout_seconds,
            path=path,
        )

    async def request_code_execution_approval(
        self,
        code: str,
        language: str = "python",
        context: ApprovalContext | None = None,
        timeout_seconds: int | None = None,
    ) -> ApprovalRequest:
        """
        Request approval for code execution.

        Args:
            code: Code to execute
            language: Programming language
            context: Approval context
            timeout_seconds: Custom timeout

        Returns:
            ApprovalRequest with final status
        """
        # Truncate code for display
        code_preview = code[:200] + "..." if len(code) > 200 else code

        return await self.request_approval(
            action_type=ActionType.CODE_EXECUTION,
            title=f"Execute {language} code",
            description=f"Code execution in {language}",
            action_details={
                "language": language,
                "code_preview": code_preview,
                "code_length": len(code),
                "code_hash": hashlib.sha256(code.encode()).hexdigest()[:12],
            },
            context=context,
            timeout_seconds=timeout_seconds,
        )

    async def request_api_call_approval(
        self,
        url: str,
        method: str = "GET",
        context: ApprovalContext | None = None,
        timeout_seconds: int | None = None,
    ) -> ApprovalRequest:
        """
        Request approval for external API call.

        Args:
            url: API URL
            method: HTTP method
            context: Approval context
            timeout_seconds: Custom timeout

        Returns:
            ApprovalRequest with final status
        """
        return await self.request_approval(
            action_type=ActionType.API_CALL,
            title=f"{method} {url[:50]}",
            description=f"External API call: {method} request",
            action_details={
                "url": url,
                "method": method,
            },
            context=context,
            timeout_seconds=timeout_seconds,
        )

    # -------------------------------------------------------------------------
    # Check Methods (Non-blocking)
    # -------------------------------------------------------------------------

    def check_requires_approval(
        self,
        action_type: ActionType,
        command: str | None = None,
        path: str | None = None,
    ) -> tuple[bool, ActionCategory]:
        """
        Check if action requires approval without requesting.

        Args:
            action_type: Type of action
            command: Optional command string
            path: Optional file path

        Returns:
            Tuple of (requires_approval, category)
        """
        category = self._detector.get_risk_category(action_type, command, path)
        requires = self._policy.requires_approval(category)
        return requires, category

    def analyze_command(self, command: str) -> dict[str, Any]:
        """
        Analyze a command for dangerous patterns.

        Args:
            command: Command to analyze

        Returns:
            Analysis results
        """
        patterns = self._detector.analyze_command(command)
        category = self._detector.get_risk_category(
            ActionType.TERMINAL_COMMAND,
            command=command,
        )

        return {
            "command": command,
            "category": category.value,
            "requires_approval": self._policy.requires_approval(category),
            "patterns_matched": [
                {
                    "pattern": p.pattern,
                    "description": p.description,
                    "category": p.category.value,
                }
                for p in patterns
            ],
        }

    # -------------------------------------------------------------------------
    # Decision Submission
    # -------------------------------------------------------------------------

    async def approve(
        self,
        approval_id: str,
        approver_id: str,
        comment: str = "",
    ) -> bool:
        """
        Approve a pending request.

        Args:
            approval_id: Request identifier
            approver_id: ID of approver
            comment: Optional comment

        Returns:
            True if approved successfully
        """
        request = await self._store.load(approval_id)
        if request is None or not request.is_pending:
            return False

        # Check approver permission
        if not self._policy.can_approve(approver_id):
            self._logger.warning(f"Unauthorized approver: {approver_id}")
            return False

        # Submit decision to handler
        if isinstance(self._handler, CallbackApprovalHandler):
            return await self._handler.submit_decision(
                approval_id,
                approved=True,
                approver_id=approver_id,
            )

        # Direct update
        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approved_at = datetime.now(timezone.utc)
        request.add_audit_entry("approved", {
            "approver_id": approver_id,
            "comment": comment,
        })

        await self._store.update(request)
        return True

    async def reject(
        self,
        approval_id: str,
        approver_id: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a pending request.

        Args:
            approval_id: Request identifier
            approver_id: ID of rejector
            reason: Reason for rejection

        Returns:
            True if rejected successfully
        """
        request = await self._store.load(approval_id)
        if request is None or not request.is_pending:
            return False

        # Check if reason is required
        if request.requires_reason and not reason:
            self._logger.warning("Rejection reason required but not provided")
            return False

        # Submit decision to handler
        if isinstance(self._handler, CallbackApprovalHandler):
            return await self._handler.submit_decision(
                approval_id,
                approved=False,
                approver_id=approver_id,
                reason=reason,
            )

        # Direct update
        request.status = ApprovalStatus.REJECTED
        request.approver_id = approver_id
        request.rejection_reason = reason
        request.add_audit_entry("rejected", {
            "approver_id": approver_id,
            "reason": reason,
        })

        await self._store.update(request)
        return True

    async def cancel(self, approval_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            approval_id: Request identifier

        Returns:
            True if cancelled successfully
        """
        request = await self._store.load(approval_id)
        if request is None or not request.is_pending:
            return False

        # Cancel via handler
        await self._handler.cancel_request(approval_id)

        request.status = ApprovalStatus.CANCELLED
        request.add_audit_entry("cancelled")
        await self._store.update(request)

        return True

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def get_request(self, approval_id: str) -> ApprovalRequest | None:
        """Get approval request by ID."""
        return await self._store.load(approval_id)

    async def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending requests."""
        return await self._store.list_pending()

    async def get_session_requests(self, session_id: str) -> list[ApprovalRequest]:
        """Get requests for a session."""
        return await self._store.list_by_session(session_id)

    async def get_status(self, approval_id: str) -> ApprovalStatus | None:
        """Get status of a request."""
        request = await self._store.load(approval_id)
        return request.status if request else None

    # -------------------------------------------------------------------------
    # Policy Management
    # -------------------------------------------------------------------------

    def set_policy(self, policy: ApprovalPolicy) -> None:
        """Set approval policy."""
        self._policy = policy

    def get_policy(self) -> ApprovalPolicy:
        """Get current policy."""
        return self._policy

    def add_blocked_action(self, action_type: ActionType) -> None:
        """Add action type to blocked list."""
        self._policy.blocked_actions.add(action_type)

    def remove_blocked_action(self, action_type: ActionType) -> None:
        """Remove action type from blocked list."""
        self._policy.blocked_actions.discard(action_type)

    def add_dangerous_pattern(self, pattern: DangerousPattern) -> None:
        """Add custom dangerous pattern."""
        self._detector.add_pattern(pattern)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """
        Clean up expired requests.

        Returns:
            Number of requests cleaned up
        """
        if isinstance(self._store, InMemoryApprovalStore):
            requests = await self._store.list_all()
            cleaned = 0

            for request in requests:
                if request.is_expired and request.is_pending:
                    request.status = ApprovalStatus.TIMEOUT
                    request.add_audit_entry("expired")
                    await self._store.update(request)
                    cleaned += 1

            return cleaned

        return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get approval statistics."""
        if isinstance(self._store, InMemoryApprovalStore):
            requests = await self._store.list_all()

            by_status: dict[str, int] = {}
            by_category: dict[str, int] = {}
            by_action: dict[str, int] = {}

            for request in requests:
                status_name = request.status.name
                by_status[status_name] = by_status.get(status_name, 0) + 1

                category_name = request.category.value
                by_category[category_name] = by_category.get(category_name, 0) + 1

                action_name = request.action_type.value
                by_action[action_name] = by_action.get(action_name, 0) + 1

            pending = await self._store.list_pending()

            return {
                "total_requests": len(requests),
                "pending_requests": len(pending),
                "by_status": by_status,
                "by_category": by_category,
                "by_action_type": by_action,
                "policy_enabled": self._policy.enabled,
                "approval_threshold": self._policy.approval_threshold.value,
            }

        return {}


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_approval_manager(
    handler_type: str = "callback",
    approval_threshold: ActionCategory = ActionCategory.MEDIUM_RISK,
    timeout_seconds: int = 300,
    auto_approve_safe: bool = True,
    blocked_actions: set[ActionType] | None = None,
    logger: logging.Logger | None = None,
) -> ApprovalManager:
    """
    Factory function to create configured ApprovalManager.

    Args:
        handler_type: Type of handler ("callback", "console", "auto")
        approval_threshold: Minimum category requiring approval
        timeout_seconds: Default timeout
        auto_approve_safe: Auto-approve safe actions
        blocked_actions: Actions to block
        logger: Optional logger

    Returns:
        Configured ApprovalManager
    """
    # Create handler
    handler: ApprovalHandler
    if handler_type == "console":
        handler = ConsoleApprovalHandler()
    elif handler_type == "auto":
        handler = AutoApprovalHandler()
    else:
        handler = CallbackApprovalHandler()

    # Create policy
    policy = ApprovalPolicy(
        approval_threshold=approval_threshold,
        timeout_seconds=timeout_seconds,
        auto_approve_safe=auto_approve_safe,
        blocked_actions=blocked_actions or set(),
    )

    # Create config
    config = ApprovalManagerConfig(
        default_timeout_seconds=timeout_seconds,
    )

    return ApprovalManager(
        config=config,
        handler=handler,
        policy=policy,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ApprovalStatus",
    "ActionCategory",
    "ActionType",
    "EscalationLevel",
    # Exceptions
    "ApprovalError",
    "ApprovalTimeoutError",
    "ApprovalRejectedError",
    "ApprovalRequiredError",
    # Detection
    "DangerousPattern",
    "ActionDetector",
    "DANGEROUS_TERMINAL_PATTERNS",
    "DANGEROUS_PATH_PATTERNS",
    # Request
    "ApprovalContext",
    "ApprovalRequest",
    # Handlers
    "ApprovalHandler",
    "ConsoleApprovalHandler",
    "CallbackApprovalHandler",
    "AutoApprovalHandler",
    # Store
    "ApprovalStore",
    "InMemoryApprovalStore",
    # Policy
    "ApprovalPolicy",
    # Manager
    "ApprovalManagerConfig",
    "ApprovalManager",
    # Factory
    "create_approval_manager",
]
