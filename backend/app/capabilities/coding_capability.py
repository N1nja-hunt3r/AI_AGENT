from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityType,
    ExecutionContext,
)


class CodingAction:
    GENERATE = "generate"
    REVIEW = "review"
    EXPLAIN = "explain"
    DEBUG = "debug"
    REFACTOR = "refactor"
    COMPLETE = "complete"
    CONVERT = "convert"
    FORMAT = "format"
    ANALYZE = "analyze"


LANGUAGE_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "react",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "cpp": "cpp",
    "c": "c",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin",
    "scala": "scala",
    "hs": "haskell",
    "ex": "elixir",
    "exs": "elixir",
    "sh": "bash",
    "bash": "bash",
    "ps1": "powershell",
    "sql": "sql",
    "r": "r",
    "lua": "lua",
    "pl": "perl",
}


SYSTEM_PROMPTS: dict[str, str] = {
    "generate": "You are an expert coding assistant. Generate clean, well-documented, production-ready code. Include appropriate error handling and follow best practices for the target language.",
    "review": "You are a senior code reviewer. Analyze the provided code for bugs, security issues, performance problems, and style violations. Be thorough and constructive.",
    "explain": "You are a technical educator. Explain the provided code in detail suitable for the requested depth level. Cover purpose, algorithms, data flow, and design decisions.",
    "debug": "You are a debugging expert. Analyze the code and error message to identify root causes and provide specific, actionable fixes.",
    "refactor": "You are a code quality expert. Refactor the provided code to improve the target aspect (readability, performance, maintainability, or security) while preserving behavior.",
    "complete": "You are a code completion engine. Given the context, provide the most likely and correct completion. Match the surrounding style, imports, and conventions.",
    "convert": "You are a polyglot programmer. Convert the source code from one language to another while preserving logic, behavior, and performance characteristics.",
    "format": "You are a code formatter. Apply consistent formatting following community standards for the detected language.",
    "analyze": "You are a static analysis tool. Analyze the code structure and provide metrics about complexity, size, and organization.",
}


class CodingCapability(Capability[dict[str, Any]]):
    METADATA = CapabilityMetadata(
        name="coding_model",
        version="1.0.0",
        capability_type=CapabilityType.CODING,
        description="Code generation, review, debugging, explanation, and transformation across multiple programming languages",
        supported_actions=(
            CodingAction.GENERATE,
            CodingAction.REVIEW,
            CodingAction.EXPLAIN,
            CodingAction.DEBUG,
            CodingAction.REFACTOR,
            CodingAction.COMPLETE,
            CodingAction.CONVERT,
            CodingAction.FORMAT,
            CodingAction.ANALYZE,
        ),
        tags=("coding", "code", "programming", "development"),
    )

    def __init__(self, llm_service: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm_service = llm_service

    async def _do_initialize(self) -> None:
        self._logger.info("Coding capability initialized")

    async def _do_shutdown(self) -> None:
        self._logger.info("Coding capability shut down")

    async def _do_execute(self, context: ExecutionContext) -> dict[str, Any]:
        action = context.action
        params = context.parameters

        if action == CodingAction.GENERATE:
            return await self._handle_generate(params)
        elif action == CodingAction.REVIEW:
            return await self._handle_review(params)
        elif action == CodingAction.EXPLAIN:
            return await self._handle_explain(params)
        elif action == CodingAction.DEBUG:
            return await self._handle_debug(params)
        elif action == CodingAction.REFACTOR:
            return await self._handle_refactor(params)
        elif action == CodingAction.COMPLETE:
            return await self._handle_complete(params)
        elif action == CodingAction.CONVERT:
            return await self._handle_convert(params)
        elif action == CodingAction.FORMAT:
            return await self._handle_format(params)
        elif action == CodingAction.ANALYZE:
            return await self._handle_analyze(params)
        else:
            raise ValueError(f"Unsupported coding action: {action}")

    async def _call_llm(
        self,
        prompt: str,
        system_key: str,
    ) -> str:
        if self._llm_service is not None:
            try:
                system_prompt = SYSTEM_PROMPTS.get(system_key, "")
                if hasattr(self._llm_service, "complete"):
                    result = await self._llm_service.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    return str(result)
                elif hasattr(self._llm_service, "generate"):
                    text, usage = await self._llm_service.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    return text
            except Exception as exc:
                self._logger.warning("LLM service failed for coding action '%s': %s", system_key, exc)

        return f"[Coding action '{system_key}' — LLM service not available] {prompt[:200]}"

    async def _handle_generate(self, params: dict[str, Any]) -> dict[str, Any]:
        description = params.get("description", params.get("task", ""))
        language = params.get("language", "python")
        framework = params.get("framework")
        test_framework = params.get("test_framework")

        parts = [f"Write {language} code for: {description}"]
        if framework:
            parts.append(f"Using framework: {framework}")
        if test_framework:
            parts.append(f"Include tests using: {test_framework}")
        prompt = "\n".join(parts)

        code = await self._call_llm(prompt, "generate")
        return {"success": True, "action": "generate", "language": language, "description": description, "code": code}

    async def _handle_review(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        language = params.get("language", self._detect_language(code))
        prompt = f"Review the following {language} code:\n\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "review")
        return {"success": True, "action": "review", "language": language, "review": result}

    async def _handle_explain(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        language = params.get("language", self._detect_language(code))
        detail_level = params.get("detail_level", "medium")
        prompt = f"Explain this {language} code at a {detail_level} detail level:\n\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "explain")
        return {"success": True, "action": "explain", "language": language, "detail_level": detail_level, "explanation": result}

    async def _handle_debug(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        error_message = params.get("error_message", "")
        language = params.get("language", self._detect_language(code))
        prompt = f"Debug this {language} code:\n\nError: {error_message}\n\nCode:\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "debug")
        return {"success": True, "action": "debug", "language": language, "error": error_message, "suggestions": [result]}

    async def _handle_refactor(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        language = params.get("language", self._detect_language(code))
        target = params.get("target", "readability")
        prompt = f"Refactor this {language} code to improve {target}. Preserve all behavior:\n\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "refactor")
        return {"success": True, "action": "refactor", "language": language, "target": target, "refactored": result}

    async def _handle_complete(self, params: dict[str, Any]) -> dict[str, Any]:
        context = params.get("context", "")
        cursor_position = params.get("cursor_position", len(context))
        language = params.get("language", self._detect_language(context))
        prompt = f"Complete this {language} code at cursor position {cursor_position}:\n\n```{language}\n{context}\n```"
        result = await self._call_llm(prompt, "complete")
        return {"success": True, "action": "complete", "language": language, "completion": result}

    async def _handle_convert(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        source_language = params.get("source_language", self._detect_language(code))
        target_language = params.get("target_language", "python")
        prompt = f"Convert this {source_language} code to {target_language}. Preserve all logic and behavior:\n\n```{source_language}\n{code}\n```"
        result = await self._call_llm(prompt, "convert")
        return {"success": True, "action": "convert", "source_language": source_language, "target_language": target_language, "converted": result}

    async def _handle_format(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        language = params.get("language", self._detect_language(code))
        prompt = f"Format this {language} code according to community best practices:\n\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "format")
        return {"success": True, "action": "format", "language": language, "formatted": result}

    async def _handle_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        code = params.get("code", "")
        language = params.get("language", self._detect_language(code))
        prompt = f"Analyze this {language} code. Provide metrics on structure, complexity, and organization:\n\n```{language}\n{code}\n```"
        result = await self._call_llm(prompt, "analyze")

        lines = code.split("\n")
        non_empty = [l for l in lines if l.strip()]
        return {
            "success": True,
            "action": "analyze",
            "language": language,
            "analysis": result,
            "stats": {"lines": len(lines), "code_lines": len(non_empty)},
        }

    def _detect_language(self, code: str) -> str:
        if not code.strip():
            return "unknown"
        first_line = code.strip().split("\n")[0].lower()
        for ext, lang in LANGUAGE_MAP.items():
            if first_line.endswith(ext):
                return lang
        if "import " in code or "def " in code or "class " in code:
            return "python"
        if "function " in code or "const " in code or "let " in code:
            return "javascript"
        if "fn " in code or "let mut " in code:
            return "rust"
        if "package " in code or "func " in code:
            return "go"
        return "unknown"
