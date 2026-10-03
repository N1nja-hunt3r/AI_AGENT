"""
exceptions.py

Typed exception hierarchy for the application. All custom exceptions
derive from BaseAppException to allow uniform handling and structured
error metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Root exception for all application-specific errors."""

    default_message: str = "An application error occurred."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message: str = message or self.default_message
        self.code: str = code or self.__class__.__name__
        self.details: Dict[str, Any] = details or {}
        self.cause: Optional[BaseException] = cause
        self.timestamp: datetime = datetime.now(timezone.utc)
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class AgentException(BaseAppException):
    """Raised for agent lifecycle, planning, or reasoning failures."""

    default_message = "Agent execution error."


class AgentInitializationError(AgentException):
    default_message = "Agent failed to initialize."


class AgentTimeoutError(AgentException):
    default_message = "Agent execution timed out."


class MemoryException(BaseAppException):
    """Raised for memory storage, retrieval, or consistency failures."""

    default_message = "Memory subsystem error."


class MemoryNotFoundError(MemoryException):
    default_message = "Requested memory entry was not found."


class MemoryCapacityExceededError(MemoryException):
    default_message = "Memory capacity exceeded."


class ToolException(BaseAppException):
    """Raised for tool invocation or tool-result parsing failures."""

    default_message = "Tool execution error."


class ToolNotFoundError(ToolException):
    default_message = "Requested tool was not found."


class ToolExecutionTimeoutError(ToolException):
    default_message = "Tool execution timed out."


class SecurityException(BaseAppException):
    """Raised for authentication, authorization, or policy violations."""

    default_message = "Security violation."


class AuthenticationError(SecurityException):
    default_message = "Authentication failed."


class AuthorizationError(SecurityException):
    default_message = "Authorization denied."


class PolicyViolationError(SecurityException):
    default_message = "Action violates a configured security policy."


class AutomationException(BaseAppException):
    """Raised for workflow/automation orchestration failures."""

    default_message = "Automation workflow error."


class WorkflowValidationError(AutomationException):
    default_message = "Workflow definition is invalid."


class WorkflowExecutionError(AutomationException):
    default_message = "Workflow execution failed."


class DatabaseException(BaseAppException):
    """Raised for relational/database connectivity or query errors."""

    default_message = "Database error."


class DatabaseConnectionError(DatabaseException):
    default_message = "Failed to connect to the database."


class DatabaseIntegrityError(DatabaseException):
    default_message = "Database integrity constraint violated."


class VectorDBException(BaseAppException):
    """Raised for vector database connectivity, indexing, or query errors."""

    default_message = "Vector database error."


class VectorDimensionMismatchError(VectorDBException):
    default_message = "Vector embedding dimension mismatch."


class VectorIndexNotFoundError(VectorDBException):
    default_message = "Requested vector index was not found."


class ConfigurationException(BaseAppException):
    """Raised for invalid or missing application configuration."""

    default_message = "Configuration error."


class ValidationException(BaseAppException):
    """Raised for general input/schema validation failures."""

    default_message = "Validation error."


class RateLimitExceededError(BaseAppException):
    """Raised when a request exceeds configured rate limits."""

    default_message = "Rate limit exceeded."
