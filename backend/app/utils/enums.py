"""
enums.py

Typed enumerations shared across the application: task/agent lifecycle
states, capability types, memory types, user roles, and security levels.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RETRYING = "retrying"


@unique
class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_HUMAN = "waiting_for_human"
    ERROR = "error"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


@unique
class CapabilityType(str, Enum):
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    PLANNING = "planning"
    MEMORY_RECALL = "memory_recall"
    WEB_SEARCH = "web_search"
    FILE_IO = "file_io"
    REASONING = "reasoning"


@unique
class ExecutionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"
    ABORTED = "aborted"


@unique
class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    SCRATCHPAD = "scratchpad"


@unique
class UserRole(str, Enum):
    ADMIN = "admin"
    OWNER = "owner"
    MEMBER = "member"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    GUEST = "guest"
    SERVICE_ACCOUNT = "service_account"


@unique
class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"
