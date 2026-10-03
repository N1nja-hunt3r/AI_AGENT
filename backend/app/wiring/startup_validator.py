"""
startup_validator.py - Production startup validator for all services.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"

class ServiceCategory(str, Enum):
    LLM = "llm"
    MEMORY = "memory"
    RAG = "rag"
    TOOLS = "tools"
    AGENTS = "agents"
    COMPUTER = "computer"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    AUTOMATION = "automation"
    VECTORDB = "vectordb"
    PROMPTS = "prompts"
    CAPABILITIES = "capabilities"
    ENVIRONMENT = "environment"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    name: str
    category: ServiceCategory
    status: ValidationStatus = ValidationStatus.SKIP
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    required: bool = True
    error: Optional[str] = None

@dataclass
class StartupReport:
    timestamp: float = field(default_factory=time.time)
    platform: str = field(default_factory=platform.system)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    overall_status: ValidationStatus = ValidationStatus.PASS
    checks: List[ValidationCheck] = field(default_factory=list)
    total_duration_ms: float = 0.0
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    skip_count: int = 0
    ready_to_start: bool = False
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "platform": self.platform,
            "python_version": self.python_version,
            "overall_status": self.overall_status.value,
            "ready_to_start": self.ready_to_start,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "summary": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
                "skip": self.skip_count,
            },
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "checks": [
                {
                    "name": c.name,
                    "category": c.category.value,
                    "status": c.status.value,
                    "message": c.message,
                    "duration_ms": round(c.duration_ms, 2),
                    "required": c.required,
                    "error": c.error,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "STARTUP VALIDATION REPORT",
            f"Status: {self.overall_status.value.upper()}",
            f"Ready: {'YES' if self.ready_to_start else 'NO'}",
            f"Duration: {self.total_duration_ms:.0f}ms",
            f"Checks: {self.pass_count} pass | {self.fail_count} fail | "
            f"{self.warn_count} warn | {self.skip_count} skip",
            "=" * 60,
        ]
        if self.critical_failures:
            lines.append("CRITICAL FAILURES:")
            for f in self.critical_failures:
                lines.append(f"  ✗ {f}")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines.append("\nCHECKS:")
        for c in self.checks:
            icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "-"}.get(c.status.value, "?")
            lines.append(f"  {icon} [{c.category.value:12}] {c.name}: {c.message}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator helpers
# ---------------------------------------------------------------------------

async def _run_check(
    name: str,
    category: ServiceCategory,
    coro: Callable,
    required: bool = True,
    timeout: float = 10.0,
) -> ValidationCheck:
    check = ValidationCheck(name=name, category=category, required=required)
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(coro(), timeout=timeout)
        check.duration_ms = (time.perf_counter() - t0) * 1000
        if isinstance(result, dict):
            check.status = ValidationStatus.PASS if result.get("healthy", True) else (
                ValidationStatus.FAIL if required else ValidationStatus.WARN
            )
            check.message = result.get("message", "OK")
            check.details = result
        elif isinstance(result, bool):
            check.status = ValidationStatus.PASS if result else (
                ValidationStatus.FAIL if required else ValidationStatus.WARN
            )
            check.message = "OK" if result else "Failed"
        elif isinstance(result, str):
            check.status = ValidationStatus.PASS
            check.message = result
        else:
            check.status = ValidationStatus.PASS
            check.message = str(result) if result else "OK"
    except asyncio.TimeoutError:
        check.duration_ms = (time.perf_counter() - t0) * 1000
        check.status = ValidationStatus.FAIL if required else ValidationStatus.WARN
        check.message = f"Timeout after {timeout}s"
        check.error = "TimeoutError"
    except Exception as exc:
        check.duration_ms = (time.perf_counter() - t0) * 1000
        check.status = ValidationStatus.FAIL if required else ValidationStatus.WARN
        check.message = str(exc)[:200]
        check.error = type(exc).__name__
    return check


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

async def _check_environment() -> Dict[str, Any]:
    checks: dict[str, Any] = {}
    critical = []
    version = sys.version_info
    if version < (3, 9):
        critical.append(f"Python 3.9+ required, found {version.major}.{version.minor}")
    checks["python_version"] = f"{version.major}.{version.minor}.{version.micro}"
    checks["platform"] = platform.system()
    checks["pid"] = os.getpid()
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    found = [v for v in env_vars if os.environ.get(v)]
    checks["llm_api_keys"] = found
    if not found:
        critical.append("No LLM API keys configured")
    return {
        "healthy": len(critical) == 0,
        "message": f"Python {checks['python_version']} on {checks['platform']}",
        "details": checks,
        "critical": critical,
    }


async def _check_llm_service() -> Dict[str, Any]:
    try:
        from llm_connector import get_connector
        connector = get_connector()
        adapters = connector.list_adapters()
        if not adapters:
            return {"healthy": False, "message": "No LLM adapters configured"}
        return {
            "healthy": True,
            "message": f"LLM ready: {adapters}",
            "adapters": adapters,
            "primary": connector.get_primary_adapter_key(),
        }
    except ImportError:
        return {"healthy": False, "message": "llm_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_memory_service() -> Dict[str, Any]:
    try:
        from memory_connector import get_memory_connector
        connector = get_memory_connector()
        await connector._ensure_initialized()
        return await connector.health_check()
    except ImportError:
        return {"healthy": False, "message": "memory_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_rag_service() -> Dict[str, Any]:
    try:
        from rag_connector import get_rag_connector
        connector = get_rag_connector()
        return await connector.health_check()
    except ImportError:
        return {"healthy": False, "message": "rag_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_tool_service() -> Dict[str, Any]:
    try:
        from tool_connector import get_tool_connector
        connector = get_tool_connector()
        return await connector.health_check()
    except ImportError:
        return {"healthy": False, "message": "tool_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_agent_service() -> Dict[str, Any]:
    try:
        from agent_connector import get_agent_connector
        connector = get_agent_connector()
        return await connector.health_check()
    except ImportError:
        return {"healthy": False, "message": "agent_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_computer_service() -> Dict[str, Any]:
    try:
        from computer_connector import get_computer_connector
        connector = get_computer_connector()
        return await connector.health_check()
    except ImportError:
        return {"healthy": False, "message": "computer_connector not available"}
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}


async def _check_database() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    healthy = True
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(redis_url, socket_connect_timeout=3)
        await client.ping()
        await client.aclose()
        checks["redis"] = "connected"
    except ImportError:
        checks["redis"] = "redis package not installed"
    except Exception as exc:
        checks["redis"] = f"unavailable: {exc}"
        healthy = False

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        try:
            import asyncpg
            conn = await asyncio.wait_for(asyncpg.connect(db_url), timeout=5)
            await conn.close()
            checks["postgres"] = "connected"
        except ImportError:
            checks["postgres"] = "asyncpg not installed"
        except Exception as exc:
            checks["postgres"] = f"unavailable: {exc}"
    else:
        checks["postgres"] = "not configured"

    return {"healthy": healthy, "message": str(checks), **checks}


async def _check_vectordb() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    try:
        import chromadb
        client = chromadb.EphemeralClient()
        client.list_collections()
        checks["chroma"] = "available (in-memory)"
    except ImportError:
        checks["chroma"] = "not installed"
    except Exception as exc:
        checks["chroma"] = f"error: {exc}"

    chroma_url = os.environ.get("CHROMA_URL", "")
    if chroma_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{chroma_url}/api/v1/heartbeat")
                checks["chroma_server"] = "connected" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as exc:
            checks["chroma_server"] = f"unavailable: {exc}"

    pinecone_key = os.environ.get("PINECONE_API_KEY", "")
    checks["pinecone"] = "configured" if pinecone_key else "not configured"

    return {"healthy": True, "message": str(checks), **checks}


async def _check_security() -> Dict[str, Any]:
    issues = []
    info: Dict[str, Any] = {}
    secret = os.environ.get("SECRET_KEY", "")
    if not secret:
        issues.append("SECRET_KEY not set")
    elif len(secret) < 32:
        issues.append("SECRET_KEY too short (< 32 chars)")
    else:
        info["secret_key"] = "configured"

    debug = os.environ.get("DEBUG", "false").lower() == "true"
    if debug:
        issues.append("DEBUG mode enabled in production")
    info["debug_mode"] = debug

    env = os.environ.get("ENVIRONMENT", os.environ.get("ENV", "development"))
    info["environment"] = env
    if env.lower() in ("prod", "production") and debug:
        issues.append("DEBUG=true in production environment")

    cors_origins = os.environ.get("CORS_ORIGINS", "*")
    if cors_origins == "*":
        issues.append("CORS_ORIGINS is wildcard (*) - consider restricting")
    info["cors_origins"] = cors_origins

    return {
        "healthy": len([i for i in issues if "production" in i]) == 0,
        "message": f"{len(issues)} security issues" if issues else "Security OK",
        "issues": issues,
        **info,
    }


async def _check_monitoring() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    sentry_dsn = os.environ.get("SENTRY_DSN", "")
    checks["sentry"] = "configured" if sentry_dsn else "not configured"
    datadog = os.environ.get("DATADOG_API_KEY", "")
    checks["datadog"] = "configured" if datadog else "not configured"
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    checks["log_level"] = log_level
    configured = bool(sentry_dsn or datadog)
    return {
        "healthy": True,
        "message": "Monitoring configured" if configured else "No external monitoring",
        **checks,
    }


async def _check_automation() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    import importlib.util
    checks["pyautogui"] = "available" if importlib.util.find_spec("pyautogui") else "not installed"
    checks["selenium"] = "available" if importlib.util.find_spec("selenium") else "not installed"
    checks["playwright"] = "available" if importlib.util.find_spec("playwright") else "not installed"
    return {
        "healthy": True,
        "message": str(checks),
        **checks,
    }


async def _check_prompts() -> Dict[str, Any]:
    prompt_dirs = [
        "prompts",
        "templates",
        os.path.join(os.path.dirname(__file__), "prompts"),
    ]
    found_dirs = [d for d in prompt_dirs if os.path.isdir(d)]
    prompt_files: List[str] = []
    for d in found_dirs:
        try:
            prompt_files.extend(
                f for f in os.listdir(d)
                if f.endswith((".txt", ".md", ".json", ".yaml", ".yml", ".jinja2"))
            )
        except Exception:
            pass
    return {
        "healthy": True,
        "message": f"Found {len(prompt_files)} prompt files in {len(found_dirs)} directories",
        "prompt_dirs": found_dirs,
        "prompt_count": len(prompt_files),
    }


async def _check_capabilities() -> Dict[str, Any]:
    caps: Dict[str, bool] = {}
    packages = {
        "sentence_transformers": "Semantic embeddings",
        "chromadb": "Vector database",
        "redis": "Cache/memory storage",
        "openai": "OpenAI LLM",
        "anthropic": "Claude LLM",
        "google.generativeai": "Gemini LLM",
        "selenium": "Browser automation",
        "pyautogui": "Computer control",
        "PIL": "Image processing",
        "tiktoken": "Token counting",
        "httpx": "Async HTTP",
        "fastapi": "API server",
        "pydantic": "Data validation",
    }
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            caps[pkg] = True
        except ImportError:
            caps[pkg] = False

    available = sum(1 for v in caps.values() if v)
    return {
        "healthy": True,
        "message": f"{available}/{len(caps)} capability packages available",
        "capabilities": caps,
    }


# ---------------------------------------------------------------------------
# StartupValidator
# ---------------------------------------------------------------------------

class StartupValidator:
    _instance: Optional["StartupValidator"] = None

    def __init__(self) -> None:
        self._checks: List[Tuple[str, ServiceCategory, Callable, bool, float]] = []
        self._last_report: Optional[StartupReport] = None
        self._setup_default_checks()

    @classmethod
    def get_instance(cls) -> "StartupValidator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _setup_default_checks(self) -> None:
        self._checks = [
            ("Environment", ServiceCategory.ENVIRONMENT, _check_environment, True, 5.0),
            ("LLM Service", ServiceCategory.LLM, _check_llm_service, True, 10.0),
            ("Memory Service", ServiceCategory.MEMORY, _check_memory_service, False, 10.0),
            ("RAG Service", ServiceCategory.RAG, _check_rag_service, False, 10.0),
            ("Tool Service", ServiceCategory.TOOLS, _check_tool_service, False, 10.0),
            ("Agent Service", ServiceCategory.AGENTS, _check_agent_service, False, 15.0),
            ("Computer Service", ServiceCategory.COMPUTER, _check_computer_service, False, 10.0),
            ("Database", ServiceCategory.DATABASE, _check_database, False, 10.0),
            ("VectorDB", ServiceCategory.VECTORDB, _check_vectordb, False, 10.0),
            ("Security", ServiceCategory.SECURITY, _check_security, True, 5.0),
            ("Monitoring", ServiceCategory.MONITORING, _check_monitoring, False, 5.0),
            ("Automation", ServiceCategory.AUTOMATION, _check_automation, False, 5.0),
            ("Prompts", ServiceCategory.PROMPTS, _check_prompts, False, 5.0),
            ("Capabilities", ServiceCategory.CAPABILITIES, _check_capabilities, False, 10.0),
        ]

    def add_check(
        self,
        name: str,
        category: ServiceCategory,
        checker: Callable,
        required: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self._checks.append((name, category, checker, required, timeout))

    async def validate(
        self,
        categories: Optional[List[ServiceCategory]] = None,
        fail_fast: bool = False,
    ) -> StartupReport:
        t0 = time.perf_counter()
        report = StartupReport()
        checks_to_run = self._checks
        if categories:
            checks_to_run = [c for c in self._checks if c[1] in categories]

        tasks = [
            _run_check(name=name, category=cat, coro=fn, required=req, timeout=tout)
            for name, cat, fn, req, tout in checks_to_run
        ]

        if fail_fast:
            for task in tasks:
                check = await task
                report.checks.append(check)
                if check.status == ValidationStatus.FAIL and check.required:
                    report.critical_failures.append(f"{check.name}: {check.message}")
                    break
        else:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            report.checks = list(results)

        for check in report.checks:
            if check.status == ValidationStatus.PASS:
                report.pass_count += 1
            elif check.status == ValidationStatus.FAIL:
                report.fail_count += 1
                if check.required:
                    report.critical_failures.append(f"{check.name}: {check.message}")
            elif check.status == ValidationStatus.WARN:
                report.warn_count += 1
                report.warnings.append(f"{check.name}: {check.message}")
            else:
                report.skip_count += 1

        critical_fails = [c for c in report.checks if c.status == ValidationStatus.FAIL and c.required]
        if critical_fails:
            report.overall_status = ValidationStatus.FAIL
            report.ready_to_start = False
        elif report.warn_count > 0:
            report.overall_status = ValidationStatus.WARN
            report.ready_to_start = True
        else:
            report.overall_status = ValidationStatus.PASS
            report.ready_to_start = True

        report.total_duration_ms = (time.perf_counter() - t0) * 1000
        self._last_report = report
        return report

    async def validate_and_report(self, print_report: bool = True) -> StartupReport:
        report = await self.validate()
        if print_report:
            print(report.summary())
        if not report.ready_to_start:
            logger.error("Startup validation FAILED. Critical: %s", report.critical_failures)
        else:
            logger.info("Startup validation PASSED in %.0fms", report.total_duration_ms)
        return report

    async def quick_check(self) -> bool:
        report = await self.validate(
            categories=[ServiceCategory.ENVIRONMENT, ServiceCategory.LLM],
        )
        return report.ready_to_start

    @property
    def last_report(self) -> Optional[StartupReport]:
        return self._last_report

    def get_status_dict(self) -> Dict[str, Any]:
        if not self._last_report:
            return {"validated": False}
        return self._last_report.to_dict()


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def get_validator() -> StartupValidator:
    return StartupValidator.get_instance()


async def validate_startup(print_report: bool = True) -> StartupReport:
    return await get_validator().validate_and_report(print_report=print_report)


async def is_ready() -> bool:
    return await get_validator().quick_check()


if __name__ == "__main__":
    async def _main() -> None:
        logging.basicConfig(level=logging.WARNING)
        report = await validate_startup(print_report=True)
        sys.exit(0 if report.ready_to_start else 1)
    asyncio.run(_main())
