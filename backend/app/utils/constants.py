"""
constants.py

Global, immutable constants for the application: defaults, limits,
timeouts, paths, versions, and static feature flags.
"""

from __future__ import annotations

from typing import Final, FrozenSet

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
APP_VERSION: Final[str] = "1.0.0"
API_VERSION: Final[str] = "v1"
CONFIG_SCHEMA_VERSION: Final[int] = 1

# ---------------------------------------------------------------------------
# Default models — NVIDIA NIM primary, third-party fallback
# ---------------------------------------------------------------------------
# NVIDIA NIM primary models
DEFAULT_NVIDIA_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
DEFAULT_DEEPSEEK_MODEL: Final[str] = "deepseek-ai/deepseek-v4-pro"
DEFAULT_FAST_MODEL: Final[str] = "deepseek-ai/deepseek-v4-flash"
DEFAULT_LLAMA_MODEL: Final[str] = "meta/llama-3.3-70b-instruct"
DEFAULT_QWEN_MODEL: Final[str] = "qwen/qwen3.5-397b-a17b"
DEFAULT_VISION_MODEL: Final[str] = "meta/llama-3.2-90b-vision-instruct"
DEFAULT_EMBED_MODEL: Final[str] = "nvidia/nv-embed-v1"
DEFAULT_STT_MODEL: Final[str] = "openai/whisper-large-v3"
DEFAULT_TTS_MODEL: Final[str] = "magpie-tts-multilingual"

# NVIDIA NIM fallback models
DEFAULT_DEEPSEEK_FALLBACK: Final[str] = "qwen/qwen3.5-122b-a10b"
DEFAULT_LLAMA_FALLBACK: Final[str] = "meta/llama-3.1-70b-instruct"
DEFAULT_QWEN_FALLBACK: Final[str] = "qwen/qwen3-next-80b-a3b-instruct"
DEFAULT_VISION_FALLBACK: Final[str] = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_STT_FALLBACK: Final[str] = "parakeet-1.1b-rnnt-multilingual-asr"
DEFAULT_TTS_FALLBACK: Final[str] = "chatterbox-multilingual-tts"

# Third-party defaults (kept for compatibility)
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-1.5-pro"
DEFAULT_EMBEDDING_MODEL: Final[str] = "nvidia/nv-embed-v1"

# All supported models
SUPPORTED_NVIDIA_MODELS: Final[FrozenSet[str]] = frozenset({
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-3.3-70b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/nv-embed-v1",
    "openai/whisper-large-v3",
    "magpie-tts-multilingual",
    "qwen/qwen3.5-122b-a10b",
    "meta/llama-3.1-70b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "parakeet-1.1b-rnnt-multilingual-asr",
    "chatterbox-multilingual-tts",
})
SUPPORTED_OPENAI_MODELS: Final[FrozenSet[str]] = frozenset(
    {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
)
SUPPORTED_ANTHROPIC_MODELS: Final[FrozenSet[str]] = frozenset(
    {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"}
)
SUPPORTED_GEMINI_MODELS: Final[FrozenSet[str]] = frozenset(
    {"gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"}
)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_TOKENS_DEFAULT: Final[int] = 4096
MAX_TOKENS_HARD_LIMIT: Final[int] = 200_000
MAX_PROMPT_CHARS: Final[int] = 1_000_000
MAX_FILE_SIZE_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MB
MAX_BATCH_SIZE: Final[int] = 100
MAX_CONCURRENT_REQUESTS: Final[int] = 50
MAX_RETRIES: Final[int] = 5
MAX_AGENT_DEPTH: Final[int] = 10
MAX_MEMORY_ITEMS: Final[int] = 10_000
MAX_TOOL_CALLS_PER_TASK: Final[int] = 25
MAX_CACHE_ENTRIES: Final[int] = 10_000

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------
DEFAULT_REQUEST_TIMEOUT: Final[float] = 60.0
DEFAULT_CONNECT_TIMEOUT: Final[float] = 10.0
DEFAULT_READ_TIMEOUT: Final[float] = 60.0
DEFAULT_TASK_TIMEOUT: Final[float] = 300.0
DEFAULT_TOOL_TIMEOUT: Final[float] = 30.0
DEFAULT_HEALTH_CHECK_TIMEOUT: Final[float] = 5.0
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600
DEFAULT_LOCK_TIMEOUT_SECONDS: Final[float] = 15.0
SHUTDOWN_GRACE_PERIOD_SECONDS: Final[float] = 30.0

# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------
RETRY_BASE_DELAY_SECONDS: Final[float] = 0.5
RETRY_MAX_DELAY_SECONDS: Final[float] = 60.0
RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0
RETRY_JITTER_FACTOR: Final[float] = 0.1

# ---------------------------------------------------------------------------
# Paths (relative defaults; absolute resolution lives in paths.py)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR: Final[str] = "data"
DEFAULT_LOGS_DIR: Final[str] = "logs"
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_ARTIFACTS_DIR: Final[str] = "artifacts"
DEFAULT_CACHE_DIR: Final[str] = ".cache"
DEFAULT_ENV_FILE: Final[str] = ".env"

# ---------------------------------------------------------------------------
# Database / Redis / VectorDB defaults
# ---------------------------------------------------------------------------
DEFAULT_DB_PORT: Final[int] = 5432
DEFAULT_REDIS_PORT: Final[int] = 6379
DEFAULT_VECTORDB_PORT: Final[int] = 6333
DEFAULT_DB_POOL_SIZE: Final[int] = 10
DEFAULT_REDIS_MAX_CONNECTIONS: Final[int] = 50
DEFAULT_VECTOR_EMBEDDING_DIM: Final[int] = 1536

# ---------------------------------------------------------------------------
# Monitoring defaults
# ---------------------------------------------------------------------------
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_PROMETHEUS_PORT: Final[int] = 9090
DEFAULT_TRACE_SAMPLE_RATE: Final[float] = 1.0
LOG_ROTATION_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_ROTATION_BACKUP_COUNT: Final[int] = 5

# ---------------------------------------------------------------------------
# Static feature flags (compile-time, not environment-overridable)
# ---------------------------------------------------------------------------
FEATURE_ASYNC_EXECUTION: Final[bool] = True
FEATURE_STRUCTURED_LOGGING: Final[bool] = True
FEATURE_TELEMETRY: Final[bool] = True
FEATURE_EXPERIMENTAL_AGENTS: Final[bool] = False

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
UTC_TIMEZONE: Final[str] = "UTC"
DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%fZ"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d"
SENSITIVE_KEY_MARKERS: Final[FrozenSet[str]] = frozenset(
    {"key", "secret", "token", "password", "credential"}
)
