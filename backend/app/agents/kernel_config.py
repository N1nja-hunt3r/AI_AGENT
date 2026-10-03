"""
Kernel Config - Centralized configuration for the AI Agent Platform.

This module provides comprehensive configuration management including:
- Execution settings and timeouts
- Model configurations and routing
- Budget limits and thresholds
- Feature flags and toggles
- Environment variable integration
- Configuration validation and defaults
- Hot-reload support for dynamic updates
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# ENUMS
# =============================================================================


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(Enum):
    """LLM providers."""

    NVIDIA = "nvidia"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class StorageBackend(Enum):
    """Storage backend types."""

    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    REDIS = "redis"
    S3 = "s3"
    DATABASE = "database"


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================


@dataclass
class TimeoutConfig:
    """
    Timeout configuration for various operations.

    All values in seconds unless otherwise specified.
    """

    # Request timeouts
    request_timeout: float = 30.0
    llm_timeout: float = 120.0
    tool_timeout: float = 60.0
    api_timeout: float = 30.0

    # Execution timeouts
    step_timeout: float = 300.0
    plan_timeout: float = 1800.0
    session_timeout: float = 3600.0

    # Connection timeouts
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0

    # Approval timeouts
    approval_timeout: float = 300.0
    escalation_timeout: float = 600.0

    # Retry settings
    retry_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_backoff_factor: float = 2.0

    def get_timedelta(self, name: str) -> timedelta:
        """Get timeout as timedelta."""
        value = getattr(self, name, 30.0)
        return timedelta(seconds=value)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "request_timeout": self.request_timeout,
            "llm_timeout": self.llm_timeout,
            "tool_timeout": self.tool_timeout,
            "api_timeout": self.api_timeout,
            "step_timeout": self.step_timeout,
            "plan_timeout": self.plan_timeout,
            "session_timeout": self.session_timeout,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
            "approval_timeout": self.approval_timeout,
            "escalation_timeout": self.escalation_timeout,
            "retry_delay": self.retry_delay,
            "retry_max_delay": self.retry_max_delay,
            "retry_backoff_factor": self.retry_backoff_factor,
        }


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig:
    """
    Configuration for a single model.

    Attributes:
        model_id: Model identifier
        provider: Model provider
        api_key_env: Environment variable for API key
        endpoint: Custom endpoint URL
        max_tokens: Maximum tokens per request
        temperature: Default temperature
        top_p: Default top_p
        context_window: Context window size
        cost_per_1k_input: Cost per 1000 input tokens
        cost_per_1k_output: Cost per 1000 output tokens
        rate_limit_rpm: Requests per minute limit
        rate_limit_tpm: Tokens per minute limit
        supports_streaming: Whether model supports streaming
        supports_functions: Whether model supports function calling
        supports_vision: Whether model supports vision
        metadata: Additional model metadata
    """

    model_id: str
    provider: ModelProvider = ModelProvider.OPENAI
    api_key_env: str = "OPENAI_API_KEY"
    endpoint: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    context_window: int = 8192
    cost_per_1k_input: Decimal = Decimal("0.001")
    cost_per_1k_output: Decimal = Decimal("0.002")
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 90000
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        """Get API key from environment."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "api_key_env": self.api_key_env,
            "endpoint": self.endpoint,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "cost_per_1k_input": str(self.cost_per_1k_input),
            "cost_per_1k_output": str(self.cost_per_1k_output),
            "rate_limit_rpm": self.rate_limit_rpm,
            "rate_limit_tpm": self.rate_limit_tpm,
            "supports_streaming": self.supports_streaming,
            "supports_functions": self.supports_functions,
            "supports_vision": self.supports_vision,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            provider=ModelProvider(data.get("provider", "openai")),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            endpoint=data.get("endpoint"),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            context_window=data.get("context_window", 8192),
            cost_per_1k_input=Decimal(data.get("cost_per_1k_input", "0.001")),
            cost_per_1k_output=Decimal(data.get("cost_per_1k_output", "0.002")),
            rate_limit_rpm=data.get("rate_limit_rpm", 60),
            rate_limit_tpm=data.get("rate_limit_tpm", 90000),
            supports_streaming=data.get("supports_streaming", True),
            supports_functions=data.get("supports_functions", True),
            supports_vision=data.get("supports_vision", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModelsConfig:
    """
    Configuration for all models.

    Attributes:
        default_model: Default model ID
        fallback_model: Fallback model ID
        models: Dictionary of model configurations
        routing_strategy: Model routing strategy
    """

    default_model: str = "deepseek-ai/deepseek-v4-pro"
    fallback_model: str = "qwen/qwen3.5-122b-a10b"
    models: dict[str, ModelConfig] = field(default_factory=dict)
    routing_strategy: str = "cost_optimized"

    def __post_init__(self) -> None:
        """Initialize default models if empty."""
        if not self.models:
            self.models = self._default_models()

    def _default_models(self) -> dict[str, ModelConfig]:
        """Get default model configurations (NVIDIA NIM primary)."""
        return {
            "deepseek-ai/deepseek-v4-pro": ModelConfig(
                model_id="deepseek-ai/deepseek-v4-pro",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_DEEPSEEK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=16384,
                context_window=131072,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=False,
            ),
            "deepseek-ai/deepseek-v4-flash": ModelConfig(
                model_id="deepseek-ai/deepseek-v4-flash",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_FAST_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=16384,
                context_window=131072,
                supports_streaming=True,
                supports_functions=True,
            ),
            "meta/llama-3.3-70b-instruct": ModelConfig(
                model_id="meta/llama-3.3-70b-instruct",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_LLAMA_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=4096,
                context_window=128000,
                supports_streaming=True,
                supports_functions=True,
            ),
            "qwen/qwen3.5-397b-a17b": ModelConfig(
                model_id="qwen/qwen3.5-397b-a17b",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_QWEN_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=16384,
                context_window=131072,
                supports_streaming=True,
                supports_functions=True,
            ),
            "meta/llama-3.2-90b-vision-instruct": ModelConfig(
                model_id="meta/llama-3.2-90b-vision-instruct",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_VISION_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=512,
                context_window=128000,
                supports_streaming=False,
                supports_functions=False,
                supports_vision=True,
            ),
            "nvidia/nv-embed-v1": ModelConfig(
                model_id="nvidia/nv-embed-v1",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_EMBED_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=0,
                context_window=2048,
                supports_streaming=False,
                supports_functions=False,
            ),
            "openai/whisper-large-v3": ModelConfig(
                model_id="openai/whisper-large-v3",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_STT_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=0,
                context_window=0,
                supports_streaming=False,
                supports_functions=False,
            ),
            "magpie-tts-multilingual": ModelConfig(
                model_id="magpie-tts-multilingual",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_TTS_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=0,
                context_window=0,
                supports_streaming=False,
                supports_functions=False,
            ),
            # ── Fallback models ──────────────────────────────────────────────
            "qwen/qwen3.5-122b-a10b": ModelConfig(
                model_id="qwen/qwen3.5-122b-a10b",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_DEEPSEEK_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=16384,
                context_window=131072,
                supports_streaming=True,
                supports_functions=True,
            ),
            "meta/llama-3.1-70b-instruct": ModelConfig(
                model_id="meta/llama-3.1-70b-instruct",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_LLAMA_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=4096,
                context_window=128000,
                supports_streaming=True,
                supports_functions=True,
            ),
            "qwen/qwen3-next-80b-a3b-instruct": ModelConfig(
                model_id="qwen/qwen3-next-80b-a3b-instruct",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_QWEN_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=16384,
                context_window=131072,
                supports_streaming=True,
                supports_functions=True,
            ),
            "meta/llama-3.2-11b-vision-instruct": ModelConfig(
                model_id="meta/llama-3.2-11b-vision-instruct",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_VISION_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=512,
                context_window=128000,
                supports_streaming=False,
                supports_functions=False,
                supports_vision=True,
            ),
            "parakeet-1.1b-rnnt-multilingual-asr": ModelConfig(
                model_id="parakeet-1.1b-rnnt-multilingual-asr",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_STT_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=0,
                context_window=0,
                supports_streaming=False,
                supports_functions=False,
            ),
            "chatterbox-multilingual-tts": ModelConfig(
                model_id="chatterbox-multilingual-tts",
                provider=ModelProvider.NVIDIA,
                api_key_env="NVIDIA_TTS_FALLBACK_API_KEY",
                endpoint=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                max_tokens=0,
                context_window=0,
                supports_streaming=False,
                supports_functions=False,
            ),
        }

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Get model configuration by ID."""
        return self.models.get(model_id)

    def get_default(self) -> ModelConfig | None:
        """Get default model configuration."""
        return self.models.get(self.default_model)

    def get_fallback(self) -> ModelConfig | None:
        """Get fallback model configuration."""
        return self.models.get(self.fallback_model)

    def add_model(self, config: ModelConfig) -> None:
        """Add or update a model configuration."""
        self.models[config.model_id] = config

    def list_models(self, provider: ModelProvider | None = None) -> list[str]:
        """List available model IDs."""
        if provider:
            return [
                m.model_id for m in self.models.values()
                if m.provider == provider
            ]
        return list(self.models.keys())


# =============================================================================
# BUDGET CONFIGURATION
# =============================================================================


@dataclass
class BudgetConfig:
    """
    Budget configuration for resource limits.

    Attributes:
        token_limit_daily: Daily token limit
        token_limit_monthly: Monthly token limit
        cost_limit_daily: Daily cost limit (USD)
        cost_limit_monthly: Monthly cost limit (USD)
        request_limit_hourly: Hourly request limit
        request_limit_daily: Daily request limit
        max_tokens_per_request: Maximum tokens per request
        max_steps_per_plan: Maximum steps per plan
        max_retries: Maximum retries per step
        warning_threshold_percent: Warning threshold (0-100)
        critical_threshold_percent: Critical threshold (0-100)
        auto_downgrade_on_limit: Auto-downgrade to cheaper model
        hard_limit_enforcement: Enforce hard limits
    """

    # Token limits
    token_limit_daily: int = 1000000
    token_limit_monthly: int = 30000000

    # Cost limits (USD)
    cost_limit_daily: Decimal = Decimal("50.00")
    cost_limit_monthly: Decimal = Decimal("1000.00")

    # Request limits
    request_limit_hourly: int = 1000
    request_limit_daily: int = 10000

    # Per-request limits
    max_tokens_per_request: int = 8192
    max_steps_per_plan: int = 50
    max_retries: int = 3

    # Thresholds
    warning_threshold_percent: float = 75.0
    critical_threshold_percent: float = 90.0

    # Behavior
    auto_downgrade_on_limit: bool = True
    hard_limit_enforcement: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_limit_daily": self.token_limit_daily,
            "token_limit_monthly": self.token_limit_monthly,
            "cost_limit_daily": str(self.cost_limit_daily),
            "cost_limit_monthly": str(self.cost_limit_monthly),
            "request_limit_hourly": self.request_limit_hourly,
            "request_limit_daily": self.request_limit_daily,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_steps_per_plan": self.max_steps_per_plan,
            "max_retries": self.max_retries,
            "warning_threshold_percent": self.warning_threshold_percent,
            "critical_threshold_percent": self.critical_threshold_percent,
            "auto_downgrade_on_limit": self.auto_downgrade_on_limit,
            "hard_limit_enforcement": self.hard_limit_enforcement,
        }


# =============================================================================
# EXECUTION CONFIGURATION
# =============================================================================


@dataclass
class ExecutionConfig:
    """
    Execution configuration for the agent controller.

    Attributes:
        mode: Execution mode
        max_concurrent_steps: Maximum concurrent steps
        max_concurrent_requests: Maximum concurrent LLM requests
        enable_parallel_execution: Enable parallel step execution
        enable_streaming: Enable streaming responses
        enable_caching: Enable response caching
        cache_ttl_seconds: Cache TTL in seconds
        checkpoint_interval: Steps between checkpoints
        enable_checkpointing: Enable automatic checkpointing
        enable_replanning: Enable automatic replanning
        max_replan_attempts: Maximum replan attempts
        enable_approval_workflow: Enable human approval
        approval_required_actions: Actions requiring approval
    """

    mode: ExecutionMode = ExecutionMode.ASYNC
    max_concurrent_steps: int = 5
    max_concurrent_requests: int = 10
    enable_parallel_execution: bool = True
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    checkpoint_interval: int = 5
    enable_checkpointing: bool = True
    enable_replanning: bool = True
    max_replan_attempts: int = 3
    enable_approval_workflow: bool = True
    approval_required_actions: set[str] = field(
        default_factory=lambda: {
            "file_delete",
            "terminal_command",
            "code_execution",
            "system_config",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "max_concurrent_steps": self.max_concurrent_steps,
            "max_concurrent_requests": self.max_concurrent_requests,
            "enable_parallel_execution": self.enable_parallel_execution,
            "enable_streaming": self.enable_streaming,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "checkpoint_interval": self.checkpoint_interval,
            "enable_checkpointing": self.enable_checkpointing,
            "enable_replanning": self.enable_replanning,
            "max_replan_attempts": self.max_replan_attempts,
            "enable_approval_workflow": self.enable_approval_workflow,
            "approval_required_actions": list(self.approval_required_actions),
        }


# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================


@dataclass
class StorageConfig:
    """
    Storage configuration.

    Attributes:
        backend: Storage backend type
        base_path: Base path for filesystem storage
        redis_url: Redis connection URL
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix
        database_url: Database connection URL
        max_size_mb: Maximum storage size in MB
        cleanup_interval_hours: Cleanup interval in hours
        retention_days: Data retention in days
    """

    backend: StorageBackend = StorageBackend.FILESYSTEM
    base_path: str = "./data"
    redis_url: str | None = None
    s3_bucket: str | None = None
    s3_prefix: str = "agent-platform/"
    database_url: str | None = None
    max_size_mb: int = 1024
    cleanup_interval_hours: int = 24
    retention_days: int = 30

    def __post_init__(self) -> None:
        """Load from environment if not set."""
        if self.redis_url is None:
            self.redis_url = os.environ.get("REDIS_URL")
        if self.s3_bucket is None:
            self.s3_bucket = os.environ.get("S3_BUCKET")
        if self.database_url is None:
            self.database_url = os.environ.get("DATABASE_URL")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "base_path": self.base_path,
            "redis_url": self.redis_url,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "database_url": self.database_url,
            "max_size_mb": self.max_size_mb,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "retention_days": self.retention_days,
        }


# =============================================================================
# FEATURE FLAGS
# =============================================================================


@dataclass
class FeatureFlags:
    """
    Feature flags for enabling/disabling functionality.

    Attributes:
        enable_memory: Enable memory system
        enable_rag: Enable RAG system
        enable_tools: Enable tool execution
        enable_web_search: Enable web search
        enable_code_execution: Enable code execution
        enable_file_operations: Enable file operations
        enable_multi_agent: Enable multi-agent support
        enable_voice: Enable voice input/output
        enable_vision: Enable vision capabilities
        enable_telemetry: Enable telemetry collection
        enable_debug_mode: Enable debug mode
        enable_experimental: Enable experimental features
        custom_flags: Custom feature flags
    """

    enable_memory: bool = True
    enable_rag: bool = True
    enable_tools: bool = True
    enable_web_search: bool = False
    enable_code_execution: bool = False
    enable_file_operations: bool = True
    enable_multi_agent: bool = False
    enable_voice: bool = False
    enable_vision: bool = False
    enable_telemetry: bool = True
    enable_debug_mode: bool = False
    enable_experimental: bool = False
    custom_flags: dict[str, bool] = field(default_factory=dict)

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a flag is enabled."""
        # Check custom flags first
        if flag_name in self.custom_flags:
            return self.custom_flags[flag_name]

        # Check standard flags
        attr_name = f"enable_{flag_name}" if not flag_name.startswith("enable_") else flag_name
        return getattr(self, attr_name, False)

    def set_flag(self, flag_name: str, value: bool) -> None:
        """Set a flag value."""
        attr_name = f"enable_{flag_name}" if not flag_name.startswith("enable_") else flag_name
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        else:
            self.custom_flags[flag_name] = value

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary."""
        result = {
            "enable_memory": self.enable_memory,
            "enable_rag": self.enable_rag,
            "enable_tools": self.enable_tools,
            "enable_web_search": self.enable_web_search,
            "enable_code_execution": self.enable_code_execution,
            "enable_file_operations": self.enable_file_operations,
            "enable_multi_agent": self.enable_multi_agent,
            "enable_voice": self.enable_voice,
            "enable_vision": self.enable_vision,
            "enable_telemetry": self.enable_telemetry,
            "enable_debug_mode": self.enable_debug_mode,
            "enable_experimental": self.enable_experimental,
        }
        result.update(self.custom_flags)
        return result


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


@dataclass
class LoggingConfig:
    """
    Logging configuration.

    Attributes:
        level: Log level
        format: Log format string
        date_format: Date format string
        log_file: Log file path
        max_file_size_mb: Maximum log file size
        backup_count: Number of backup files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON format
        enable_colors: Enable colored output
    """

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: str | None = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = False
    enable_json: bool = False
    enable_colors: bool = True

    def get_level(self) -> int:
        """Get logging level as int."""
        return getattr(logging, self.level.value, logging.INFO)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "format": self.format,
            "date_format": self.date_format,
            "log_file": self.log_file,
            "max_file_size_mb": self.max_file_size_mb,
            "backup_count": self.backup_count,
            "enable_console": self.enable_console,
            "enable_file": self.enable_file,
            "enable_json": self.enable_json,
            "enable_colors": self.enable_colors,
        }


# =============================================================================
# API CONFIGURATION
# =============================================================================


@dataclass
class APIConfig:
    """
    API server configuration.

    Attributes:
        host: Server host
        port: Server port
        workers: Number of workers
        enable_cors: Enable CORS
        cors_origins: Allowed CORS origins
        enable_docs: Enable API documentation
        docs_url: Documentation URL
        api_prefix: API route prefix
        rate_limit_enabled: Enable rate limiting
        rate_limit_requests: Requests per window
        rate_limit_window_seconds: Rate limit window
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_cors: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    docs_url: str = "/docs"
    api_prefix: str = "/api/v1"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origins,
            "enable_docs": self.enable_docs,
            "docs_url": self.docs_url,
            "api_prefix": self.api_prefix,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
        }


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================


@dataclass
class SecurityConfig:
    """
    Security configuration.

    Attributes:
        secret_key_env: Environment variable for secret key
        api_key_env: Environment variable for API key
        enable_authentication: Enable authentication
        enable_authorization: Enable authorization
        jwt_algorithm: JWT algorithm
        jwt_expiry_hours: JWT expiry in hours
        allowed_hosts: Allowed hosts
        blocked_actions: Blocked action types
        sandbox_enabled: Enable sandboxed execution
        max_file_size_mb: Maximum file upload size
    """

    secret_key_env: str = "SECRET_KEY"
    api_key_env: str = "API_KEY"
    enable_authentication: bool = True
    enable_authorization: bool = True
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    allowed_hosts: list[str] = field(default_factory=lambda: ["*"])
    blocked_actions: set[str] = field(default_factory=set)
    sandbox_enabled: bool = True
    max_file_size_mb: int = 100

    @property
    def secret_key(self) -> str | None:
        """Get secret key from environment."""
        return os.environ.get(self.secret_key_env)

    @property
    def api_key(self) -> str | None:
        """Get API key from environment."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "secret_key_env": self.secret_key_env,
            "api_key_env": self.api_key_env,
            "enable_authentication": self.enable_authentication,
            "enable_authorization": self.enable_authorization,
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expiry_hours": self.jwt_expiry_hours,
            "allowed_hosts": self.allowed_hosts,
            "blocked_actions": list(self.blocked_actions),
            "sandbox_enabled": self.sandbox_enabled,
            "max_file_size_mb": self.max_file_size_mb,
        }


# =============================================================================
# EXTENSIONS CONFIGURATION
# =============================================================================


@dataclass
class ExtensionConfig:
    """Configuration for a single extension."""

    name: str
    enabled: bool = True
    version: str = "1.0.0"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtensionsConfig:
    """
    Extensions configuration for future extensibility.

    Attributes:
        extensions_dir: Directory for extensions
        auto_load: Auto-load extensions on startup
        extensions: Dictionary of extension configurations
    """

    extensions_dir: str = "./extensions"
    auto_load: bool = True
    extensions: dict[str, ExtensionConfig] = field(default_factory=dict)

    def register_extension(
        self,
        name: str,
        enabled: bool = True,
        version: str = "1.0.0",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register an extension."""
        self.extensions[name] = ExtensionConfig(
            name=name,
            enabled=enabled,
            version=version,
            config=config or {},
        )

    def is_enabled(self, name: str) -> bool:
        """Check if extension is enabled."""
        ext = self.extensions.get(name)
        return ext.enabled if ext else False

    def get_config(self, name: str) -> dict[str, Any]:
        """Get extension configuration."""
        ext = self.extensions.get(name)
        return ext.config if ext else {}


# =============================================================================
# KERNEL CONFIGURATION
# =============================================================================


@dataclass
class KernelConfig:
    """
    Main kernel configuration aggregating all sub-configurations.

    Attributes:
        environment: Deployment environment
        app_name: Application name
        app_version: Application version
        timeouts: Timeout configuration
        models: Model configuration
        budget: Budget configuration
        execution: Execution configuration
        storage: Storage configuration
        features: Feature flags
        logging: Logging configuration
        api: API configuration
        security: Security configuration
        extensions: Extensions configuration
    """

    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "AI Agent Platform"
    app_version: str = "1.0.0"

    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)

    def __post_init__(self) -> None:
        """Load environment-specific overrides."""
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Environment
        env_str = os.environ.get("ENVIRONMENT", "development").lower()
        try:
            self.environment = Environment(env_str)
        except ValueError:
            pass

        # App info
        self.app_name = os.environ.get("APP_NAME", self.app_name)
        self.app_version = os.environ.get("APP_VERSION", self.app_version)

        # Logging level
        log_level = os.environ.get("LOG_LEVEL", "").upper()
        if log_level and log_level in LogLevel.__members__:
            self.logging.level = LogLevel[log_level]

        # Debug mode
        if os.environ.get("DEBUG", "").lower() in ("true", "1", "yes"):
            self.features.enable_debug_mode = True
            self.logging.level = LogLevel.DEBUG

        # API settings
        if port := os.environ.get("PORT"):
            try:
                self.api.port = int(port)
            except ValueError:
                pass

        if host := os.environ.get("HOST"):
            self.api.host = host

        # Storage
        if base_path := os.environ.get("STORAGE_PATH"):
            self.storage.base_path = base_path

        # Default model (NVIDIA NIM primary)
        if default_model := os.environ.get("NVIDIA_DEEPSEEK_MODEL"):
            self.models.default_model = default_model
        elif default_model := os.environ.get("DEFAULT_MODEL"):
            self.models.default_model = default_model

        # Fallback model
        if fallback_model := os.environ.get("NVIDIA_DEEPSEEK_FALLBACK_MODEL"):
            self.models.fallback_model = fallback_model

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.features.enable_debug_mode

    def get_model(self, model_id: str | None = None) -> ModelConfig | None:
        """Get model configuration."""
        if model_id:
            return self.models.get_model(model_id)
        return self.models.get_default()

    def to_dict(self) -> dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "timeouts": self.timeouts.to_dict(),
            "models": {
                "default_model": self.models.default_model,
                "fallback_model": self.models.fallback_model,
                "routing_strategy": self.models.routing_strategy,
                "models": {k: v.to_dict() for k, v in self.models.models.items()},
            },
            "budget": self.budget.to_dict(),
            "execution": self.execution.to_dict(),
            "storage": self.storage.to_dict(),
            "features": self.features.to_dict(),
            "logging": self.logging.to_dict(),
            "api": self.api.to_dict(),
            "security": self.security.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelConfig:
        """Create configuration from dictionary."""
        config = cls()

        if "environment" in data:
            config.environment = Environment(data["environment"])

        config.app_name = data.get("app_name", config.app_name)
        config.app_version = data.get("app_version", config.app_version)

        # Load sub-configurations
        if "timeouts" in data:
            config.timeouts = TimeoutConfig(**data["timeouts"])

        if "budget" in data:
            budget_data = data["budget"].copy()
            if "cost_limit_daily" in budget_data:
                budget_data["cost_limit_daily"] = Decimal(budget_data["cost_limit_daily"])
            if "cost_limit_monthly" in budget_data:
                budget_data["cost_limit_monthly"] = Decimal(budget_data["cost_limit_monthly"])
            config.budget = BudgetConfig(**budget_data)

        if "execution" in data:
            exec_data = data["execution"].copy()
            if "mode" in exec_data:
                exec_data["mode"] = ExecutionMode(exec_data["mode"])
            if "approval_required_actions" in exec_data:
                exec_data["approval_required_actions"] = set(exec_data["approval_required_actions"])
            config.execution = ExecutionConfig(**exec_data)

        if "storage" in data:
            storage_data = data["storage"].copy()
            if "backend" in storage_data:
                storage_data["backend"] = StorageBackend(storage_data["backend"])
            config.storage = StorageConfig(**storage_data)

        if "features" in data:
            config.features = FeatureFlags(**data["features"])

        if "logging" in data:
            log_data = data["logging"].copy()
            if "level" in log_data:
                log_data["level"] = LogLevel(log_data["level"])
            config.logging = LoggingConfig(**log_data)

        if "api" in data:
            config.api = APIConfig(**data["api"])

        if "security" in data:
            sec_data = data["security"].copy()
            if "blocked_actions" in sec_data:
                sec_data["blocked_actions"] = set(sec_data["blocked_actions"])
            config.security = SecurityConfig(**sec_data)

        if "models" in data:
            models_data = data["models"]
            config.models.default_model = models_data.get("default_model", config.models.default_model)
            config.models.fallback_model = models_data.get("fallback_model", config.models.fallback_model)
            config.models.routing_strategy = models_data.get("routing_strategy", config.models.routing_strategy)

            if "models" in models_data:
                for model_id, model_data in models_data["models"].items():
                    config.models.add_model(ModelConfig.from_dict(model_data))

        return config

    @classmethod
    def from_json(cls, json_str: str) -> KernelConfig:
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> KernelConfig:
        """Load configuration from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        content = path.read_text()

        if path.suffix == ".json":
            return cls.from_json(content)

        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
                data = yaml.safe_load(content)
                return cls.from_dict(data)
            except ImportError:
                raise ImportError("PyYAML required for YAML config: pip install pyyaml")

        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    def save_to_file(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            path.write_text(self.to_json())

        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
                path.write_text(yaml.dump(self.to_dict(), default_flow_style=False))
            except ImportError:
                raise ImportError("PyYAML required for YAML config: pip install pyyaml")

        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        # Validate model configuration
        if not self.models.get_default():
            errors.append(f"Default model '{self.models.default_model}' not found in models")

        # Validate budget
        if self.budget.cost_limit_daily > self.budget.cost_limit_monthly:
            errors.append("Daily cost limit cannot exceed monthly limit")

        if self.budget.token_limit_daily > self.budget.token_limit_monthly:
            errors.append("Daily token limit cannot exceed monthly limit")

        # Validate thresholds
        if not (0 <= self.budget.warning_threshold_percent <= 100):
            errors.append("Warning threshold must be between 0 and 100")

        if not (0 <= self.budget.critical_threshold_percent <= 100):
            errors.append("Critical threshold must be between 0 and 100")

        if self.budget.warning_threshold_percent >= self.budget.critical_threshold_percent:
            errors.append("Warning threshold must be less than critical threshold")

        # Validate API
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")

        # Validate security
        if self.is_production and not self.security.secret_key:
            errors.append("Secret key required in production")

        return errors


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================


class ConfigManager:
    """
    Configuration manager with hot-reload support.

    Manages configuration lifecycle and provides access
    to configuration values.
    """

    _instance: ConfigManager | None = None
    _config: KernelConfig | None = None

    def __new__(cls) -> ConfigManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize config manager."""
        if self._config is None:
            self._config = KernelConfig()
        self._watchers: list[Callable[[KernelConfig], None]] = []
        self._logger = logging.getLogger(__name__)

    @property
    def config(self) -> KernelConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = KernelConfig()
        return self._config

    def load(self, path: str | Path | None = None) -> KernelConfig:
        """
        Load configuration from file or environment.

        Args:
            path: Configuration file path

        Returns:
            Loaded configuration
        """
        if path:
            self._config = KernelConfig.from_file(path)
        else:
            self._config = KernelConfig()

        # Validate
        errors = self._config.validate()
        if errors:
            for error in errors:
                self._logger.warning(f"Config validation: {error}")

        # Notify watchers
        self._notify_watchers()

        return self._config

    def reload(self, path: str | Path | None = None) -> KernelConfig:
        """Reload configuration (alias for load)."""
        return self.load(path)

    def update(self, updates: dict[str, Any]) -> KernelConfig:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates

        Returns:
            Updated configuration
        """
        current_dict = self._config.to_dict() if self._config else {}
        self._deep_update(current_dict, updates)
        self._config = KernelConfig.from_dict(current_dict)

        self._notify_watchers()

        return self._config

    def _deep_update(self, base: dict, updates: dict) -> None:
        """Deep update dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def watch(self, callback: Callable[[KernelConfig], None]) -> None:
        """Register a configuration change watcher."""
        self._watchers.append(callback)

    def unwatch(self, callback: Callable[[KernelConfig], None]) -> None:
        """Unregister a configuration change watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)

    def _notify_watchers(self) -> None:
        """Notify all watchers of configuration change."""
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                self._logger.error(f"Config watcher error: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "models.default_model")
            default: Default value if not found

        Returns:
            Configuration value
        """
        parts = key.split(".")
        value: Any = self._config

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Dot-notation key
            value: Value to set
        """
        parts = key.split(".")
        target: Any = self._config

        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                return

        if hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)
            self._notify_watchers()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def get_config() -> KernelConfig:
    """Get the current configuration."""
    return ConfigManager().config


def load_config(path: str | Path | None = None) -> KernelConfig:
    """Load configuration from file or environment."""
    return ConfigManager().load(path)


def create_config(
    environment: Environment = Environment.DEVELOPMENT,
    default_model: str = "deepseek-ai/deepseek-v4-pro",
    enable_debug: bool = False,
    **kwargs: Any,
) -> KernelConfig:
    """
    Create a new configuration with custom settings.

    Args:
        environment: Deployment environment
        default_model: Default model ID
        enable_debug: Enable debug mode
        **kwargs: Additional configuration overrides

    Returns:
        New KernelConfig instance
    """
    config = KernelConfig(environment=environment)
    config.models.default_model = default_model
    config.features.enable_debug_mode = enable_debug

    if enable_debug:
        config.logging.level = LogLevel.DEBUG

    # Apply additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Environment",
    "LogLevel",
    "ModelProvider",
    "ExecutionMode",
    "StorageBackend",
    # Timeout Config
    "TimeoutConfig",
    # Model Config
    "ModelConfig",
    "ModelsConfig",
    # Budget Config
    "BudgetConfig",
    # Execution Config
    "ExecutionConfig",
    # Storage Config
    "StorageConfig",
    # Feature Flags
    "FeatureFlags",
    # Logging Config
    "LoggingConfig",
    # API Config
    "APIConfig",
    # Security Config
    "SecurityConfig",
    # Extensions Config
    "ExtensionConfig",
    "ExtensionsConfig",
    # Main Config
    "KernelConfig",
    # Manager
    "ConfigManager",
    # Factory Functions
    "get_config",
    "load_config",
    "create_config",
]
