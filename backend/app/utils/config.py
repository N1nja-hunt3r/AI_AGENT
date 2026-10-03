"""
config.py

Centralized, typed application configuration using Pydantic BaseSettings.
Supports environment variable loading, singleton access, runtime reload,
and health checks across LLM providers, databases, caches, vector stores,
monitoring, and feature flags.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: Optional[SecretStr] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.openai.com/v1")
    default_model: str = Field(default="gpt-4o")
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)

    def is_configured(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", extra="ignore")

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://api.anthropic.com")
    default_model: str = Field(default="claude-sonnet-4-6")
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)

    def is_configured(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI_", extra="ignore")

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://generativelanguage.googleapis.com")
    default_model: str = Field(default="gemini-1.5-pro")
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)

    def is_configured(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())


class NVIDIASettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NVIDIA_", extra="ignore")

    api_key: Optional[SecretStr] = Field(default=None)
    deepseek_api_key: Optional[SecretStr] = Field(default=None)
    fast_api_key: Optional[SecretStr] = Field(default=None)
    llama_api_key: Optional[SecretStr] = Field(default=None)
    qwen_api_key: Optional[SecretStr] = Field(default=None)
    vision_api_key: Optional[SecretStr] = Field(default=None)
    embed_api_key: Optional[SecretStr] = Field(default=None)
    stt_api_key: Optional[SecretStr] = Field(default=None)
    tts_api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    deepseek_model: str = Field(default="deepseek-ai/deepseek-v4-pro")
    fast_model: str = Field(default="deepseek-ai/deepseek-v4-flash")
    llama_model: str = Field(default="meta/llama-3.3-70b-instruct")
    qwen_model: str = Field(default="qwen/qwen3.5-397b-a17b")
    vision_model: str = Field(default="meta/llama-3.2-90b-vision-instruct")
    embed_model: str = Field(default="nvidia/nv-embed-v1")
    stt_model: str = Field(default="openai/whisper-large-v3")
    tts_model: str = Field(default="magpie-tts-multilingual")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1)
    request_timeout: float = Field(default=60.0, ge=1.0)
    retry_count: int = Field(default=3, ge=0)
    enable_streaming: bool = Field(default=True)
    enable_fallbacks: bool = Field(default=True)
    enable_cost_tracking: bool = Field(default=True)
    enable_provider_health_checks: bool = Field(default=True)
    enable_provider_metrics: bool = Field(default=True)
    enable_model_router: bool = Field(default=True)
    enable_reflection: bool = Field(default=True)
    enable_memory_compression: bool = Field(default=True)
    enable_embed_cache: bool = Field(default=True)
    enable_response_cache: bool = Field(default=True)
    max_concurrent_requests: int = Field(default=20, ge=1)
    stream_timeout: float = Field(default=120.0, ge=1.0)

    # ── Fallback models ──────────────────────────────────────────────────────
    deepseek_fallback_model: str = Field(default="qwen/qwen3.5-122b-a10b")
    llama_fallback_model: str = Field(default="meta/llama-3.1-70b-instruct")
    qwen_fallback_model: str = Field(default="qwen/qwen3-next-80b-a3b-instruct")
    vision_fallback_model: str = Field(default="meta/llama-3.2-11b-vision-instruct")
    stt_fallback_model: str = Field(default="parakeet-1.1b-rnnt-multilingual-asr")
    tts_fallback_model: str = Field(default="chatterbox-multilingual-tts")

    # ── Fallback API keys ────────────────────────────────────────────────────
    deepseek_fallback_api_key: Optional[SecretStr] = Field(default=None)
    llama_fallback_api_key: Optional[SecretStr] = Field(default=None)
    qwen_fallback_api_key: Optional[SecretStr] = Field(default=None)
    vision_fallback_api_key: Optional[SecretStr] = Field(default=None)
    stt_fallback_api_key: Optional[SecretStr] = Field(default=None)
    tts_fallback_api_key: Optional[SecretStr] = Field(default=None)

    def is_configured(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())

    def get_deepseek_key(self) -> str:
        return (self.deepseek_api_key or self.api_key).get_secret_value() if (self.deepseek_api_key or self.api_key) else ""

    def get_fast_key(self) -> str:
        return (self.fast_api_key or self.deepseek_api_key or self.api_key).get_secret_value() if (self.fast_api_key or self.deepseek_api_key or self.api_key) else ""

    def get_llama_key(self) -> str:
        return (self.llama_api_key or self.api_key).get_secret_value() if (self.llama_api_key or self.api_key) else ""

    def get_qwen_key(self) -> str:
        return (self.qwen_api_key or self.api_key).get_secret_value() if (self.qwen_api_key or self.api_key) else ""

    def get_vision_key(self) -> str:
        return (self.vision_api_key or self.api_key).get_secret_value() if (self.vision_api_key or self.api_key) else ""

    def get_embed_key(self) -> str:
        return (self.embed_api_key or self.api_key).get_secret_value() if (self.embed_api_key or self.api_key) else ""

    def get_stt_key(self) -> str:
        return (self.stt_api_key or self.api_key).get_secret_value() if (self.stt_api_key or self.api_key) else ""

    def get_tts_key(self) -> str:
        return (self.tts_api_key or self.api_key).get_secret_value() if (self.tts_api_key or self.api_key) else ""

    # ── Fallback key getters ─────────────────────────────────────────────────

    def get_deepseek_fallback_key(self) -> str:
        return (self.deepseek_fallback_api_key or self.api_key).get_secret_value() if (self.deepseek_fallback_api_key or self.api_key) else ""

    def get_llama_fallback_key(self) -> str:
        return (self.llama_fallback_api_key or self.llama_api_key or self.api_key).get_secret_value() if (self.llama_fallback_api_key or self.llama_api_key or self.api_key) else ""

    def get_qwen_fallback_key(self) -> str:
        return (self.qwen_fallback_api_key or self.qwen_api_key or self.api_key).get_secret_value() if (self.qwen_fallback_api_key or self.qwen_api_key or self.api_key) else ""

    def get_vision_fallback_key(self) -> str:
        return (self.vision_fallback_api_key or self.vision_api_key or self.api_key).get_secret_value() if (self.vision_fallback_api_key or self.vision_api_key or self.api_key) else ""

    def get_stt_fallback_key(self) -> str:
        return (self.stt_fallback_api_key or self.stt_api_key or self.api_key).get_secret_value() if (self.stt_fallback_api_key or self.stt_api_key or self.api_key) else ""

    def get_tts_fallback_key(self) -> str:
        return (self.tts_fallback_api_key or self.tts_api_key or self.api_key).get_secret_value() if (self.tts_fallback_api_key or self.tts_api_key or self.api_key) else ""


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = Field(default="app_db")
    user: str = Field(default="postgres")
    password: Optional[SecretStr] = Field(default=None)
    driver: str = Field(default="postgresql+asyncpg")
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout_seconds: float = Field(default=30.0, ge=1.0)
    echo: bool = Field(default=False)

    @property
    def dsn(self) -> str:
        pwd = self.password.get_secret_value() if self.password else ""
        auth = f"{self.user}:{pwd}" if pwd else self.user
        return f"{self.driver}://{auth}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: Optional[SecretStr] = Field(default=None)
    use_ssl: bool = Field(default=False)
    socket_timeout_seconds: float = Field(default=5.0, ge=0.1)
    max_connections: int = Field(default=50, ge=1)

    @property
    def url(self) -> str:
        scheme = "rediss" if self.use_ssl else "redis"
        pwd = self.password.get_secret_value() if self.password else None
        auth = f":{pwd}@" if pwd else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class VectorDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTORDB_", extra="ignore")

    provider: str = Field(default="chromadb")
    api_key: Optional[SecretStr] = Field(default=None)
    environment: Optional[str] = Field(default=None)
    host: Optional[str] = Field(default=None)
    port: int = Field(default=6333, ge=1, le=65535)
    index_name: str = Field(default="default-index")
    namespace: str = Field(default="default")
    embedding_dim: int = Field(default=1536, ge=1)
    metric: str = Field(default="cosine")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"pinecone", "qdrant", "weaviate", "chroma", "milvus", "pgvector"}
        if v.lower() not in allowed:
            raise ValueError(f"Unsupported vector db provider: {v}")
        return v.lower()


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONITORING_", extra="ignore")

    enabled: bool = Field(default=True)
    sentry_dsn: Optional[SecretStr] = Field(default=None)
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    tracing_enabled: bool = Field(default=False)
    otel_endpoint: Optional[str] = Field(default=None)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"Invalid log level: {v}")
        return upper


class FeatureFlags(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEATURE_", extra="ignore")

    enable_caching: bool = Field(default=True)
    enable_streaming: bool = Field(default=True)
    enable_function_calling: bool = Field(default=True)
    enable_memory: bool = Field(default=True)
    enable_vector_search: bool = Field(default=True)
    enable_rate_limiting: bool = Field(default=True)
    enable_async_execution: bool = Field(default=True)
    enable_multi_agent: bool = Field(default=False)
    enable_experimental: bool = Field(default=False)


class Settings(BaseSettings):
    """Root application settings aggregating all subsystem configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ai-agent-platform")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: SecretStr = Field(default=SecretStr("change-me"))
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    nvidia: NVIDIASettings = Field(default_factory=NVIDIASettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"Invalid app_env: {v}")
        return lower

    def is_production(self) -> bool:
        return self.app_env == "production"


class ConfigManager:
    """Thread-safe singleton manager for application Settings."""

    _instance: Optional["ConfigManager"] = None
    _lock: threading.Lock = threading.Lock()
    _settings: Optional[Settings] = None
    _init_lock: Optional[threading.RLock] = None
    _user_settings: Dict[str, Dict[str, Any]] = {}
    _user_settings_lock: threading.Lock = threading.Lock()

    async def get_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored settings for a user, or None if not set."""
        with self._user_settings_lock:
            return self._user_settings.get(user_id)

    async def update_settings(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings for a user, merging with any existing values."""
        with self._user_settings_lock:
            if user_id not in self._user_settings:
                self._user_settings[user_id] = {}
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(self._user_settings[user_id].get(key), dict):
                    self._user_settings[user_id][key].update(value)
                else:
                    self._user_settings[user_id][key] = value
            return self._user_settings[user_id]

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._settings = None
                    cls._instance._init_lock = threading.RLock()
        return cls._instance

    def load(self, **overrides: Any) -> Settings:
        """Load settings from environment, applying optional overrides."""
        with self._init_lock:
            if self._settings is None:
                self._settings = Settings(**overrides)
            return self._settings

    def reload(self, **overrides: Any) -> Settings:
        """Force a fresh reload of settings from environment."""
        with self._init_lock:
            get_cached_settings.cache_clear()
            self._settings = Settings(**overrides)
            return self._settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            return self.load()
        return self._settings

    def health_check(self) -> Dict[str, Any]:
        """Run lightweight health checks across configured subsystems."""
        s = self.settings
        report: Dict[str, Any] = {
            "app_env": s.app_env,
            "debug": s.debug,
            "components": {
                "openai": {
                    "configured": s.openai.is_configured(),
                    "model": s.openai.default_model,
                },
                "anthropic": {
                    "configured": s.anthropic.is_configured(),
                    "model": s.anthropic.default_model,
                },
                "gemini": {
                    "configured": s.gemini.is_configured(),
                    "model": s.gemini.default_model,
                },
                "nvidia": {
                    "configured": s.nvidia.is_configured(),
                    "deepseek_model": s.nvidia.deepseek_model,
                    "llama_model": s.nvidia.llama_model,
                    "qwen_model": s.nvidia.qwen_model,
                    "health_checks": s.nvidia.enable_provider_health_checks,
                    "metrics": s.nvidia.enable_provider_metrics,
                    "router": s.nvidia.enable_model_router,
                    "reflection": s.nvidia.enable_reflection,
                    "memory_compression": s.nvidia.enable_memory_compression,
                    "embed_cache": s.nvidia.enable_embed_cache,
                    "response_cache": s.nvidia.enable_response_cache,
                    "max_concurrent": s.nvidia.max_concurrent_requests,
                    "stream_timeout": s.nvidia.stream_timeout,
                },
                "database": {
                    "host": s.database.host,
                    "port": s.database.port,
                    "name": s.database.name,
                },
                "redis": {
                    "host": s.redis.host,
                    "port": s.redis.port,
                    "db": s.redis.db,
                },
                "vectordb": {
                    "provider": s.vectordb.provider,
                    "index_name": s.vectordb.index_name,
                },
                "monitoring": {
                    "enabled": s.monitoring.enabled,
                    "log_level": s.monitoring.log_level,
                },
            },
            "feature_flags": s.features.model_dump(),
        }
        missing: List[str] = []
        if not s.openai.is_configured():
            missing.append("openai_api_key")
        if not s.anthropic.is_configured():
            missing.append("anthropic_api_key")
        if not s.gemini.is_configured():
            missing.append("gemini_api_key")
        report["missing_credentials"] = missing
        report["status"] = "degraded" if missing else "healthy"
        return report


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    return Settings()


def get_config_manager() -> ConfigManager:
    """Module-level accessor for the ConfigManager singleton."""
    return ConfigManager()


def get_settings() -> Settings:
    """Convenience accessor returning the active Settings instance."""
    return get_config_manager().load()
