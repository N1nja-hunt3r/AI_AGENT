from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    INPUT = "input"
    PROMPT_INJECTION = "prompt_injection"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    URL = "url"
    FILE = "file"
    SCHEMA = "schema"


class ValidationError(Exception):
    def __init__(self, message: str, category: ValidationCategory, severity: Severity = Severity.MEDIUM) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity


@dataclass(frozen=True)
class Finding:
    category: ValidationCategory
    severity: Severity
    message: str
    pattern: Optional[str] = None
    span: Optional[tuple[int, int]] = None


@dataclass
class ValidationResult:
    valid: bool
    findings: list[Finding] = field(default_factory=list)
    sanitized: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def highest_severity(self) -> Optional[Severity]:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def raise_if_invalid(self) -> None:
        if not self.valid:
            msgs = "; ".join(f.message for f in self.findings)
            cat = self.findings[0].category if self.findings else ValidationCategory.INPUT
            sev = self.highest_severity or Severity.MEDIUM
            raise ValidationError(msgs or "validation failed", category=cat, severity=sev)


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


SQL_INJECTION_PATTERNS: tuple[str, ...] = (
    r"(?i)\b(union\s+select|select\s+.*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|drop\s+database|alter\s+table|truncate\s+table)\b",
    r"(?i)\b(or|and)\s+['\"]?\s*\d+\s*['\"]?\s*=\s*['\"]?\s*\d+",
    r"(?i)\b(or|and)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?\s*--",
    r"--\s*$",
    r"/\*.*?\*/",
    r"(?i)\bexec(\s|\+)+(s|x)p\w+",
    r"(?i)\bxp_cmdshell\b",
    r"(?i)\binformation_schema\b",
    r"(?i)\bsleep\s*\(\s*\d+\s*\)",
    r"(?i)\bbenchmark\s*\(",
    r"(?i)\bwaitfor\s+delay\b",
    r"(?i)\bcast\s*\(.*\bas\b",
    r"(?i)\bconvert\s*\(.*,.*\)",
    r"(?i)';.*--",
    r"(?i)\bhaving\s+\d+\s*=\s*\d+",
    r"(?i)\bunion\s+all\s+select\b",
    r"(?i)0x[0-9a-f]{4,}",
)

COMMAND_INJECTION_PATTERNS: tuple[str, ...] = (
    r"[;&|`]",
    r"\$\(",
    r"\$\{",
    r"(?i)\b(rm\s+-rf|wget\s+|curl\s+|nc\s+-e|bash\s+-i|sh\s+-i|chmod\s+\+x|mkfifo|/dev/tcp/)\b",
    r"(?i)\b(cat|less|more)\s+/etc/(passwd|shadow)\b",
    r">\s*/dev/null",
    r"\|\|",
    r"&&",
    r"(?i)\bnohup\b",
    r"(?i)\bpowershell\s+-(enc|e|c)\b",
    r"(?i)\bcmd(\.exe)?\s*/c\b",
    r"\n\s*(rm|del|format|shutdown|reboot)\b",
)

PATH_TRAVERSAL_PATTERNS: tuple[str, ...] = (
    r"\.\.[\\/]",
    r"%2e%2e[%2f%5c]",
    r"%252e%252e",
    r"\.\.%2f",
    r"\.\.%5c",
    r"/etc/passwd",
    r"\\windows\\system32",
    r"^[a-zA-Z]:[\\/]",
    r"\x00",
)

PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)\b",
    r"(?i)\byou\s+are\s+now\s+(in\s+)?(dan|jailbreak|developer\s+mode|unrestricted)\b",
    r"(?i)\bact\s+as\s+(if\s+you\s+(are|have)\s+)?no\s+(restrictions?|rules?|filters?)\b",
    r"(?i)\bsystem\s*[:>]\s*",
    r"(?i)\bnew\s+instructions?\s*[:>]",
    r"(?i)\boverride\s+(your\s+)?(system\s+)?prompt\b",
    r"(?i)\breveal\s+(your\s+)?(system\s+)?prompt\b",
    r"(?i)\bprint\s+(your\s+)?(system\s+)?prompt\b",
    r"(?i)\brepeat\s+(the\s+words?\s+)?above\b",
    r"(?i)\bpretend\s+(you\s+are|to\s+be)\s+",
    r"(?i)\bfrom\s+now\s+on\s*,?\s+you\s+(will|must|shall)\b",
    r"(?i)\bdo\s+anything\s+now\b",
    r"(?i)\bjailbreak\b",
    r"(?i)\bbypass\s+(your\s+)?(safety|content)\s+(filters?|guidelines?|restrictions?)\b",
    r"(?i)\[\[?system\]?\]",
    r"(?i)<\|?(system|im_start|im_end)\|?>",
    r"(?i)\bstop\s+being\s+(an?\s+)?(ai|assistant|claude)\b",
    r"(?i)\bdisable\s+(your\s+)?(safety|filters?|guardrails?)\b",
    r"(?i)\boutput\s+the\s+(text|words?|content)\s+above\b",
)

DANGEROUS_FILE_EXTENSIONS: frozenset[str] = frozenset({
    ".exe", ".dll", ".so", ".sh", ".bat", ".cmd", ".com", ".scr",
    ".msi", ".ps1", ".vbs", ".js", ".jar", ".app", ".deb", ".rpm",
    ".php", ".phtml", ".jsp", ".asp", ".aspx", ".cgi",
})

DEFAULT_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024
DEFAULT_MAX_STRING_LENGTH: int = 100_000

FILE_MAGIC_SIGNATURES: dict[str, bytes] = {
    "pdf": b"%PDF-",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "gif": b"GIF8",
    "zip": b"PK\x03\x04",
    "elf": b"\x7fELF",
    "exe": b"MZ",
}


class Validator:
    """Production input validator with injection, file, URL, and schema checks."""

    def __init__(
        self,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        allowed_url_schemes: Optional[frozenset[str]] = None,
        blocked_hosts: Optional[frozenset[str]] = None,
        allowed_file_extensions: Optional[frozenset[str]] = None,
        sql_patterns: tuple[str, ...] = SQL_INJECTION_PATTERNS,
        command_patterns: tuple[str, ...] = COMMAND_INJECTION_PATTERNS,
        path_patterns: tuple[str, ...] = PATH_TRAVERSAL_PATTERNS,
        prompt_patterns: tuple[str, ...] = PROMPT_INJECTION_PATTERNS,
    ) -> None:
        self.max_string_length = max_string_length
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_url_schemes = allowed_url_schemes or frozenset({"http", "https"})
        self.blocked_hosts = blocked_hosts or frozenset({"localhost", "0.0.0.0", "127.0.0.1", "::1"})
        self.allowed_file_extensions = allowed_file_extensions
        self._sql_re = [re.compile(p) for p in sql_patterns]
        self._cmd_re = [re.compile(p) for p in command_patterns]
        self._path_re = [re.compile(p, re.IGNORECASE) for p in path_patterns]
        self._prompt_re = [re.compile(p) for p in prompt_patterns]
        self._created_at = time.time()
        self._validation_count = 0
        self._failure_count = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        try:
            return unicodedata.normalize("NFKC", text)
        except Exception:
            return text

    def _scan(self, text: str, compiled: list[re.Pattern[str]], category: ValidationCategory, severity: Severity) -> list[Finding]:
        findings: list[Finding] = []
        normalized = self._normalize(text)
        for pattern in compiled:
            for match in pattern.finditer(normalized):
                findings.append(
                    Finding(
                        category=category,
                        severity=severity,
                        message=f"{category.value} pattern matched: {pattern.pattern[:60]}",
                        pattern=pattern.pattern,
                        span=match.span(),
                    )
                )
        return findings

    def validate_input(
        self,
        value: Any,
        *,
        max_length: Optional[int] = None,
        allow_empty: bool = True,
        check_sql: bool = True,
        check_command: bool = True,
        check_path_traversal: bool = True,
        check_prompt_injection: bool = False,
        pattern: Optional[str] = None,
    ) -> ValidationResult:
        self._validation_count += 1
        findings: list[Finding] = []
        limit = max_length if max_length is not None else self.max_string_length

        if value is None:
            if allow_empty:
                return ValidationResult(valid=True, sanitized=None)
            findings.append(Finding(ValidationCategory.INPUT, Severity.MEDIUM, "value is None but empty not allowed"))
            self._failure_count += 1
            return ValidationResult(valid=False, findings=findings)

        text = value if isinstance(value, str) else json.dumps(value, default=str)

        if not allow_empty and text.strip() == "":
            findings.append(Finding(ValidationCategory.INPUT, Severity.LOW, "empty input not allowed"))

        if len(text) > limit:
            findings.append(
                Finding(ValidationCategory.INPUT, Severity.MEDIUM, f"input length {len(text)} exceeds max {limit}")
            )

        if "\x00" in text:
            findings.append(Finding(ValidationCategory.INPUT, Severity.HIGH, "null byte detected in input"))

        if check_sql:
            findings.extend(self._scan(text, self._sql_re, ValidationCategory.SQL_INJECTION, Severity.HIGH))
        if check_command:
            findings.extend(self._scan(text, self._cmd_re, ValidationCategory.COMMAND_INJECTION, Severity.HIGH))
        if check_path_traversal:
            findings.extend(self._scan(text, self._path_re, ValidationCategory.PATH_TRAVERSAL, Severity.HIGH))
        if check_prompt_injection:
            findings.extend(self._scan(text, self._prompt_re, ValidationCategory.PROMPT_INJECTION, Severity.MEDIUM))

        if pattern is not None and not re.fullmatch(pattern, text):
            findings.append(Finding(ValidationCategory.INPUT, Severity.LOW, f"input does not match required pattern {pattern}"))

        valid = not any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)
        if not valid:
            self._failure_count += 1
        return ValidationResult(valid=valid, findings=findings, sanitized=text)

    def validate_prompt(
        self,
        prompt: str,
        *,
        max_length: int = 50_000,
        strict: bool = False,
    ) -> ValidationResult:
        self._validation_count += 1
        findings: list[Finding] = []

        if not isinstance(prompt, str):
            findings.append(Finding(ValidationCategory.PROMPT_INJECTION, Severity.HIGH, "prompt must be a string"))
            self._failure_count += 1
            return ValidationResult(valid=False, findings=findings)

        if len(prompt) > max_length:
            findings.append(
                Finding(ValidationCategory.PROMPT_INJECTION, Severity.MEDIUM, f"prompt length {len(prompt)} exceeds {max_length}")
            )

        findings.extend(self._scan(prompt, self._prompt_re, ValidationCategory.PROMPT_INJECTION, Severity.HIGH))

        zero_width = re.findall(r"[\u200b\u200c\u200d\u2060\ufeff]", prompt)
        if zero_width:
            findings.append(
                Finding(ValidationCategory.PROMPT_INJECTION, Severity.MEDIUM, f"{len(zero_width)} zero-width/invisible characters detected")
            )

        repeated_specials = re.findall(r"([^\w\s])\1{9,}", prompt)
        if repeated_specials:
            findings.append(Finding(ValidationCategory.PROMPT_INJECTION, Severity.LOW, "suspicious repeated special characters"))

        base64_blobs = re.findall(r"(?:[A-Za-z0-9+/]{4}){20,}={0,2}", prompt)
        if base64_blobs:
            findings.append(Finding(ValidationCategory.PROMPT_INJECTION, Severity.LOW, "large base64-like blob detected, possible obfuscation"))

        severity_threshold = (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL) if strict else (Severity.HIGH, Severity.CRITICAL)
        valid = not any(f.severity in severity_threshold for f in findings)
        if not valid:
            self._failure_count += 1
        return ValidationResult(valid=valid, findings=findings, sanitized=prompt)

    def validate_url(
        self,
        url: str,
        *,
        require_https: bool = False,
        allowed_hosts: Optional[frozenset[str]] = None,
        max_length: int = 2048,
    ) -> ValidationResult:
        self._validation_count += 1
        findings: list[Finding] = []

        if not isinstance(url, str) or not url:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, "url must be a non-empty string"))
            self._failure_count += 1
            return ValidationResult(valid=False, findings=findings)

        if len(url) > max_length:
            findings.append(Finding(ValidationCategory.URL, Severity.MEDIUM, f"url length exceeds {max_length}"))

        try:
            parsed = urlparse(url)
        except Exception as exc:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, f"url could not be parsed: {exc}"))
            self._failure_count += 1
            return ValidationResult(valid=False, findings=findings)

        if parsed.scheme.lower() not in self.allowed_url_schemes:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, f"scheme '{parsed.scheme}' not allowed"))

        if require_https and parsed.scheme.lower() != "https":
            findings.append(Finding(ValidationCategory.URL, Severity.MEDIUM, "https required"))

        host = (parsed.hostname or "").lower()
        if not host:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, "url missing host"))
        elif host in self.blocked_hosts:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, f"host '{host}' is blocked"))
        elif allowed_hosts is not None and host not in allowed_hosts:
            findings.append(Finding(ValidationCategory.URL, Severity.HIGH, f"host '{host}' not in allowlist"))

        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                findings.append(Finding(ValidationCategory.URL, Severity.HIGH, "url resolves to non-public/internal IP (SSRF risk)"))
        except ValueError:
            pass

        if "@" in url.split("//", 1)[-1].split("/", 1)[0]:
            findings.append(Finding(ValidationCategory.URL, Severity.MEDIUM, "url contains userinfo component, possible spoofing"))

        if re.search(r"(?i)\.(local|internal|corp)$", host):
            findings.append(Finding(ValidationCategory.URL, Severity.MEDIUM, "url targets internal-looking TLD"))

        valid = not any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)
        if not valid:
            self._failure_count += 1
        return ValidationResult(valid=valid, findings=findings, sanitized=url, metadata={"scheme": parsed.scheme, "host": host})

    def validate_file(
        self,
        path: str | Path,
        *,
        content: Optional[bytes] = None,
        allowed_extensions: Optional[frozenset[str]] = None,
        max_size_bytes: Optional[int] = None,
        check_magic_bytes: bool = True,
        base_directory: Optional[str | Path] = None,
    ) -> ValidationResult:
        self._validation_count += 1
        findings: list[Finding] = []
        file_path = Path(path)
        limit = max_size_bytes if max_size_bytes is not None else self.max_file_size_bytes
        extensions = allowed_extensions if allowed_extensions is not None else self.allowed_file_extensions

        raw_path_str = str(path)
        findings.extend(self._scan(raw_path_str, self._path_re, ValidationCategory.PATH_TRAVERSAL, Severity.HIGH))

        if base_directory is not None:
            try:
                resolved_base = Path(base_directory).resolve()
                resolved_target = (resolved_base / file_path).resolve()
                if resolved_base not in resolved_target.parents and resolved_target != resolved_base:
                    findings.append(
                        Finding(ValidationCategory.PATH_TRAVERSAL, Severity.CRITICAL, "resolved path escapes base directory")
                    )
            except Exception as exc:
                findings.append(Finding(ValidationCategory.PATH_TRAVERSAL, Severity.HIGH, f"path resolution failed: {exc}"))

        suffix = file_path.suffix.lower()
        if suffix in DANGEROUS_FILE_EXTENSIONS:
            findings.append(Finding(ValidationCategory.FILE, Severity.HIGH, f"dangerous file extension '{suffix}'"))

        if extensions is not None and suffix not in extensions:
            findings.append(Finding(ValidationCategory.FILE, Severity.MEDIUM, f"extension '{suffix}' not in allowlist"))

        if content is not None:
            size = len(content)
            if size > limit:
                findings.append(Finding(ValidationCategory.FILE, Severity.MEDIUM, f"file size {size} exceeds max {limit}"))
            if size == 0:
                findings.append(Finding(ValidationCategory.FILE, Severity.LOW, "file is empty"))

            if check_magic_bytes:
                detected = None
                for kind, sig in FILE_MAGIC_SIGNATURES.items():
                    if content.startswith(sig):
                        detected = kind
                        break
                if detected == "exe":
                    findings.append(Finding(ValidationCategory.FILE, Severity.CRITICAL, "executable magic bytes (MZ) detected"))
                if detected == "elf":
                    findings.append(Finding(ValidationCategory.FILE, Severity.CRITICAL, "ELF executable magic bytes detected"))
                if suffix in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}:
                    expected = {
                        ".png": "png", ".jpg": "jpg", ".jpeg": "jpg",
                        ".gif": "gif", ".pdf": "pdf", ".zip": "zip",
                    }[suffix]
                    if detected != expected:
                        findings.append(
                            Finding(ValidationCategory.FILE, Severity.HIGH, f"file content does not match extension '{suffix}' (magic byte mismatch)")
                        )

            if b"\x00" in content[:1024] and suffix not in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".bin"}:
                findings.append(Finding(ValidationCategory.FILE, Severity.MEDIUM, "null bytes detected in text-like file header"))

        valid = not any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)
        if not valid:
            self._failure_count += 1
        return ValidationResult(valid=valid, findings=findings, sanitized=str(file_path), metadata={"extension": suffix})

    def validate_schema(
        self,
        data: Any,
        schema: dict[str, Any],
        *,
        path: str = "$",
    ) -> ValidationResult:
        self._validation_count += 1
        findings: list[Finding] = []
        self._validate_schema_node(data, schema, path, findings)
        valid = not any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)
        if not valid:
            self._failure_count += 1
        return ValidationResult(valid=valid, findings=findings, sanitized=data)

    def _validate_schema_node(self, data: Any, schema: dict[str, Any], path: str, findings: list[Finding]) -> None:
        expected_type = schema.get("type")
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str, "number": (int, float), "integer": int,
            "boolean": bool, "array": list, "object": dict, "null": type(None),
        }
        if expected_type is not None:
            py_type = type_map.get(expected_type)
            if py_type is not None and not isinstance(data, py_type):
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.HIGH, f"{path}: expected type '{expected_type}', got '{type(data).__name__}'"))
                return

        if isinstance(data, str):
            min_len = schema.get("minLength")
            max_len = schema.get("maxLength")
            if min_len is not None and len(data) < min_len:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: string shorter than minLength {min_len}"))
            if max_len is not None and len(data) > max_len:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: string longer than maxLength {max_len}"))
            schema_pattern = schema.get("pattern")
            if schema_pattern and not re.search(schema_pattern, data):
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: string does not match pattern"))
            enum_vals = schema.get("enum")
            if enum_vals is not None and data not in enum_vals:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: value not in enum {enum_vals}"))

        if isinstance(data, (int, float)) and not isinstance(data, bool):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and data < minimum:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: value below minimum {minimum}"))
            if maximum is not None and data > maximum:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: value above maximum {maximum}"))

        if isinstance(data, dict):
            required = schema.get("required", [])
            for key in required:
                if key not in data:
                    findings.append(Finding(ValidationCategory.SCHEMA, Severity.HIGH, f"{path}: missing required property '{key}'"))
            properties = schema.get("properties", {})
            additional_allowed = schema.get("additionalProperties", True)
            for key, value in data.items():
                if key in properties:
                    self._validate_schema_node(value, properties[key], f"{path}.{key}", findings)
                elif not additional_allowed:
                    findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: additional property '{key}' not allowed"))

        if isinstance(data, list):
            item_schema = schema.get("items")
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if min_items is not None and len(data) < min_items:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: array shorter than minItems {min_items}"))
            if max_items is not None and len(data) > max_items:
                findings.append(Finding(ValidationCategory.SCHEMA, Severity.MEDIUM, f"{path}: array longer than maxItems {max_items}"))
            if item_schema is not None:
                for idx, item in enumerate(data):
                    self._validate_schema_node(item, item_schema, f"{path}[{idx}]", findings)

    async def validate_input_async(self, value: Any, **kwargs: Any) -> ValidationResult:
        async with self._lock:
            return await asyncio.to_thread(self.validate_input, value, **kwargs)

    async def validate_prompt_async(self, prompt: str, **kwargs: Any) -> ValidationResult:
        return await asyncio.to_thread(self.validate_prompt, prompt, **kwargs)

    async def validate_url_async(self, url: str, **kwargs: Any) -> ValidationResult:
        return await asyncio.to_thread(self.validate_url, url, **kwargs)

    async def validate_file_async(self, path: str | Path, **kwargs: Any) -> ValidationResult:
        return await asyncio.to_thread(self.validate_file, path, **kwargs)

    async def validate_schema_async(self, data: Any, schema: dict[str, Any], **kwargs: Any) -> ValidationResult:
        return await asyncio.to_thread(self.validate_schema, data, schema, **kwargs)

    def health_check(self) -> HealthStatus:
        try:
            test_result = self.validate_input("health-check-probe", check_sql=True, check_command=True)
            healthy = test_result.valid
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "validation_count": self._validation_count,
                "failure_count": self._failure_count,
                "sql_patterns_loaded": len(self._sql_re),
                "command_patterns_loaded": len(self._cmd_re),
                "path_patterns_loaded": len(self._path_re),
                "prompt_patterns_loaded": len(self._prompt_re),
            }
            return HealthStatus(healthy=healthy, component="validator", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="validator", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Validator",
    "ValidationResult",
    "ValidationError",
    "ValidationCategory",
    "Severity",
    "Finding",
    "HealthStatus",
]
