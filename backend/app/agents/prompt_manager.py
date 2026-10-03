"""
Prompt Manager - Centralized prompt management for the AI Agent Platform.

This module provides comprehensive prompt management including:
- Loading prompts from files, databases, or code
- Template rendering with variable substitution
- Prompt versioning and A/B testing
- Dynamic prompt composition and merging
- Specialized prompts for system, memory, tools, and planning
- Prompt validation and optimization
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class PromptType(Enum):
    """Types of prompts in the system."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MEMORY = "memory"
    TOOL = "tool"
    PLANNER = "planner"
    ROUTER = "router"
    EXECUTOR = "executor"
    SUMMARIZER = "summarizer"
    ANALYZER = "analyzer"
    CUSTOM = "custom"


class PromptFormat(Enum):
    """Prompt template formats."""

    PLAIN = "plain"  # No templating
    PYTHON = "python"  # {variable} style
    JINJA2 = "jinja2"  # {{ variable }} style
    MUSTACHE = "mustache"  # {{ variable }} style (simplified)


class MergeStrategy(Enum):
    """Strategies for merging prompts."""

    APPEND = auto()  # Append sections
    PREPEND = auto()  # Prepend sections
    REPLACE = auto()  # Replace entirely
    INTERLEAVE = auto()  # Interleave sections
    SMART = auto()  # Context-aware merging


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


@dataclass
class PromptMetadata:
    """
    Metadata for a prompt template.

    Attributes:
        name: Unique prompt identifier
        version: Semantic version string
        prompt_type: Type classification
        description: Human-readable description
        author: Prompt author
        tags: Searchable tags
        created_at: Creation timestamp
        updated_at: Last update timestamp
        variables: Expected template variables
        examples: Example renderings
        deprecated: Whether prompt is deprecated
        replacement: Replacement prompt if deprecated
    """

    name: str
    version: str = "1.0.0"
    prompt_type: PromptType = PromptType.CUSTOM
    description: str = ""
    author: str = ""
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    variables: tuple[str, ...] = ()
    examples: tuple[dict[str, Any], ...] = ()
    deprecated: bool = False
    replacement: str | None = None

    @property
    def version_tuple(self) -> tuple[int, ...]:
        """Parse version into tuple for comparison."""
        try:
            return tuple(int(x) for x in self.version.split("."))
        except ValueError:
            return (0, 0, 0)


@dataclass
class PromptTemplate:
    """
    A prompt template with metadata and rendering capabilities.

    Attributes:
        content: The template content
        metadata: Template metadata
        format: Template format
        sections: Named sections within the template
    """

    content: str
    metadata: PromptMetadata
    format: PromptFormat = PromptFormat.PYTHON
    sections: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract sections from content if not provided."""
        if not self.sections and self.content:
            self.sections = self._extract_sections(self.content)

    @property
    def name(self) -> str:
        """Get template name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Get template version."""
        return self.metadata.version

    @property
    def prompt_type(self) -> PromptType:
        """Get prompt type."""
        return self.metadata.prompt_type

    @property
    def variables(self) -> set[str]:
        """Extract variables from template."""
        if self.format == PromptFormat.PYTHON:
            # Match {variable} but not {{escaped}}
            pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
            return set(re.findall(pattern, self.content))

        elif self.format in {PromptFormat.JINJA2, PromptFormat.MUSTACHE}:
            # Match {{ variable }}
            pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
            return set(re.findall(pattern, self.content))

        return set()

    @property
    def hash(self) -> str:
        """Get content hash for caching/comparison."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]

    def _extract_sections(self, content: str) -> dict[str, str]:
        """Extract named sections from content."""
        sections: dict[str, str] = {}

        # Match ## Section Name or # Section Name
        pattern = r"^#{1,2}\s+(.+?)$"
        lines = content.split("\n")

        current_section: str | None = None
        current_content: list[str] = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()

                current_section = match.group(1).strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def get_section(self, name: str) -> str | None:
        """Get a specific section by name."""
        return self.sections.get(name.lower().replace(" ", "_"))

    def validate(self, variables: dict[str, Any]) -> list[str]:
        """
        Validate that all required variables are provided.

        Returns:
            List of missing variable names
        """
        required = self.variables
        provided = set(variables.keys())
        return list(required - provided)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.prompt_type.value,
            "format": self.format.value,
            "content": self.content,
            "sections": self.sections,
            "variables": list(self.variables),
            "metadata": {
                "description": self.metadata.description,
                "author": self.metadata.author,
                "tags": list(self.metadata.tags),
                "deprecated": self.metadata.deprecated,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create from dictionary representation."""
        metadata = PromptMetadata(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            prompt_type=PromptType(data.get("type", "custom")),
            description=data.get("metadata", {}).get("description", ""),
            author=data.get("metadata", {}).get("author", ""),
            tags=tuple(data.get("metadata", {}).get("tags", [])),
            variables=tuple(data.get("variables", [])),
            deprecated=data.get("metadata", {}).get("deprecated", False),
        )

        return cls(
            content=data["content"],
            metadata=metadata,
            format=PromptFormat(data.get("format", "python")),
            sections=data.get("sections", {}),
        )


# =============================================================================
# PROMPT RENDERER
# =============================================================================


class PromptRenderer(Protocol):
    """Protocol for prompt rendering implementations."""

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render template with variables."""
        ...


class PythonRenderer:
    """
    Renders templates using Python string formatting.

    Supports {variable} syntax with safe fallbacks.
    """

    def __init__(self, strict: bool = False) -> None:
        """
        Initialize renderer.

        Args:
            strict: If True, raise on missing variables
        """
        self._strict = strict

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render template with Python format syntax."""
        if self._strict:
            return template.format(**variables)

        # Safe rendering with missing variable handling
        class SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return f"{{{key}}}"

        return template.format_map(SafeDict(variables))


class Jinja2Renderer:
    """
    Renders templates using Jinja2.

    Requires: pip install jinja2
    """

    def __init__(
        self,
        strict: bool = False,
        autoescape: bool = False,
    ) -> None:
        """
        Initialize Jinja2 renderer.

        Args:
            strict: If True, raise on undefined variables
            autoescape: Enable HTML autoescaping
        """
        try:
            from jinja2 import Environment, StrictUndefined, Undefined

            undefined = StrictUndefined if strict else Undefined
            self._env = Environment(
                autoescape=autoescape,
                undefined=undefined,
            )
        except ImportError:
            raise ImportError("jinja2 required: pip install jinja2")

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render template with Jinja2."""
        jinja_template = self._env.from_string(template)
        return jinja_template.render(**variables)


class MustacheRenderer:
    """
    Simple Mustache-style renderer.

    Supports {{ variable }} syntax without full Mustache features.
    """

    def __init__(self, strict: bool = False) -> None:
        self._strict = strict

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render template with Mustache-style syntax."""
        result = template

        # Replace {{ variable }} patterns
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            elif self._strict:
                raise KeyError(f"Missing variable: {var_name}")
            else:
                return match.group(0)

        return re.sub(pattern, replacer, result)


# =============================================================================
# PROMPT LOADER
# =============================================================================


class PromptLoader(ABC):
    """Abstract base for prompt loading implementations."""

    @abstractmethod
    def load(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load a prompt by name and optional version."""
        ...

    @abstractmethod
    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        ...


class FilePromptLoader(PromptLoader):
    """
    Loads prompts from filesystem.

    Supports .txt, .md, .json, and .yaml files.
    """

    def __init__(
        self,
        base_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize file loader.

        Args:
            base_path: Base directory for prompt files
            logger: Optional logger
        """
        self._base_path = Path(base_path)
        self._logger = logger or logging.getLogger(__name__)
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load prompt from file."""
        # Check cache
        cache_key = f"{name}:{version or 'latest'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find file
        file_path = self._find_file(name, version)
        if file_path is None:
            return None

        try:
            template = self._load_file(file_path, name)
            self._cache[cache_key] = template
            return template

        except Exception as e:
            self._logger.error(f"Failed to load prompt {name}: {e}")
            return None

    def _find_file(self, name: str, version: str | None) -> Path | None:
        """Find prompt file by name and version."""
        # Try versioned path first
        if version:
            versioned_paths = [
                self._base_path / f"{name}_v{version}.txt",
                self._base_path / f"{name}_v{version}.md",
                self._base_path / f"{name}_v{version}.json",
                self._base_path / name / f"v{version}.txt",
                self._base_path / name / f"v{version}.md",
            ]
            for path in versioned_paths:
                if path.exists():
                    return path

        # Try unversioned paths
        paths = [
            self._base_path / f"{name}.txt",
            self._base_path / f"{name}.md",
            self._base_path / f"{name}.json",
            self._base_path / name / "prompt.txt",
            self._base_path / name / "prompt.md",
        ]

        for path in paths:
            if path.exists():
                return path

        return None

    def _load_file(self, path: Path, name: str) -> PromptTemplate:
        """Load and parse prompt file."""
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".json":
            data = json.loads(content)
            return PromptTemplate.from_dict(data)

        # Parse frontmatter if present
        metadata, body = self._parse_frontmatter(content)

        prompt_metadata = PromptMetadata(
            name=metadata.get("name", name),
            version=metadata.get("version", "1.0.0"),
            prompt_type=PromptType(metadata.get("type", "custom")),
            description=metadata.get("description", ""),
            author=metadata.get("author", ""),
            tags=tuple(metadata.get("tags", [])),
            variables=tuple(metadata.get("variables", [])),
        )

        return PromptTemplate(
            content=body,
            metadata=prompt_metadata,
            format=PromptFormat(metadata.get("format", "python")),
        )

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from content."""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        try:
            import yaml
            metadata = yaml.safe_load(parts[1]) or {}
        except ImportError:
            # Simple key: value parsing
            metadata = {}
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
        except Exception:
            metadata = {}

        return metadata, parts[2].strip()

    def list_prompts(self) -> list[str]:
        """List all available prompts."""
        prompts: set[str] = set()

        if not self._base_path.exists():
            return []

        for path in self._base_path.rglob("*"):
            if path.suffix in {".txt", ".md", ".json"}:
                # Extract name from path
                relative = path.relative_to(self._base_path)
                name = relative.stem

                # Remove version suffix
                name = re.sub(r"_v\d+\.\d+\.\d+$", "", name)

                if name not in {"prompt", "template"}:
                    prompts.add(name)

        return sorted(prompts)

    def reload(self, name: str | None = None) -> None:
        """Reload prompts from disk."""
        if name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


class DictPromptLoader(PromptLoader):
    """
    Loads prompts from a dictionary.

    Useful for testing and embedded prompts.
    """

    def __init__(self, prompts: dict[str, str | PromptTemplate]) -> None:
        """
        Initialize with prompt dictionary.

        Args:
            prompts: Dictionary mapping names to content or templates
        """
        self._prompts: dict[str, PromptTemplate] = {}

        for name, value in prompts.items():
            if isinstance(value, str):
                self._prompts[name] = PromptTemplate(
                    content=value,
                    metadata=PromptMetadata(name=name),
                )
            else:
                self._prompts[name] = value

    def load(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load prompt from dictionary."""
        return self._prompts.get(name)

    def list_prompts(self) -> list[str]:
        """List all prompt names."""
        return list(self._prompts.keys())

    def add(self, name: str, content: str | PromptTemplate) -> None:
        """Add a prompt to the loader."""
        if isinstance(content, str):
            self._prompts[name] = PromptTemplate(
                content=content,
                metadata=PromptMetadata(name=name),
            )
        else:
            self._prompts[name] = content


# =============================================================================
# PROMPT MERGER
# =============================================================================


class PromptMerger:
    """
    Merges multiple prompts into a single prompt.

    Supports various merge strategies for combining prompts.
    """

    def __init__(
        self,
        default_strategy: MergeStrategy = MergeStrategy.APPEND,
        section_separator: str = "\n\n",
    ) -> None:
        """
        Initialize merger.

        Args:
            default_strategy: Default merge strategy
            section_separator: Separator between merged sections
        """
        self._default_strategy = default_strategy
        self._separator = section_separator

    def merge(
        self,
        prompts: list[PromptTemplate],
        strategy: MergeStrategy | None = None,
    ) -> PromptTemplate:
        """
        Merge multiple prompts into one.

        Args:
            prompts: Prompts to merge
            strategy: Merge strategy (None = default)

        Returns:
            Merged PromptTemplate
        """
        if not prompts:
            return PromptTemplate(
                content="",
                metadata=PromptMetadata(name="empty"),
            )

        if len(prompts) == 1:
            return prompts[0]

        strategy = strategy or self._default_strategy

        if strategy == MergeStrategy.APPEND:
            return self._merge_append(prompts)
        elif strategy == MergeStrategy.PREPEND:
            return self._merge_prepend(prompts)
        elif strategy == MergeStrategy.REPLACE:
            return prompts[-1]  # Last one wins
        elif strategy == MergeStrategy.INTERLEAVE:
            return self._merge_interleave(prompts)
        elif strategy == MergeStrategy.SMART:
            return self._merge_smart(prompts)
        else:
            return self._merge_append(prompts)

    def _merge_append(self, prompts: list[PromptTemplate]) -> PromptTemplate:
        """Append prompts in order."""
        contents = [p.content for p in prompts]
        merged_content = self._separator.join(contents)

        # Merge sections
        merged_sections: dict[str, str] = {}
        for prompt in prompts:
            for name, content in prompt.sections.items():
                if name in merged_sections:
                    merged_sections[name] += self._separator + content
                else:
                    merged_sections[name] = content

        # Merge metadata
        all_tags: set[str] = set()
        all_variables: set[str] = set()
        for prompt in prompts:
            all_tags.update(prompt.metadata.tags)
            all_variables.update(prompt.metadata.variables)

        metadata = PromptMetadata(
            name=f"merged_{prompts[0].name}",
            version="1.0.0",
            prompt_type=prompts[0].prompt_type,
            tags=tuple(all_tags),
            variables=tuple(all_variables),
        )

        return PromptTemplate(
            content=merged_content,
            metadata=metadata,
            format=prompts[0].format,
            sections=merged_sections,
        )

    def _merge_prepend(self, prompts: list[PromptTemplate]) -> PromptTemplate:
        """Prepend prompts (reverse order)."""
        return self._merge_append(list(reversed(prompts)))

    def _merge_interleave(self, prompts: list[PromptTemplate]) -> PromptTemplate:
        """Interleave sections from prompts."""
        # Collect all unique section names in order
        section_order: list[str] = []
        for prompt in prompts:
            for name in prompt.sections.keys():
                if name not in section_order:
                    section_order.append(name)

        # Build merged content by section
        merged_parts: list[str] = []
        merged_sections: dict[str, str] = {}

        for section_name in section_order:
            section_contents: list[str] = []
            for prompt in prompts:
                if section_name in prompt.sections:
                    section_contents.append(prompt.sections[section_name])

            if section_contents:
                merged_section = self._separator.join(section_contents)
                merged_sections[section_name] = merged_section
                merged_parts.append(f"## {section_name.replace('_', ' ').title()}\n\n{merged_section}")

        metadata = PromptMetadata(
            name=f"interleaved_{prompts[0].name}",
            version="1.0.0",
            prompt_type=prompts[0].prompt_type,
        )

        return PromptTemplate(
            content=self._separator.join(merged_parts),
            metadata=metadata,
            format=prompts[0].format,
            sections=merged_sections,
        )

    def _merge_smart(self, prompts: list[PromptTemplate]) -> PromptTemplate:
        """Smart merge based on prompt types and content."""
        # Group by type
        by_type: dict[PromptType, list[PromptTemplate]] = {}
        for prompt in prompts:
            if prompt.prompt_type not in by_type:
                by_type[prompt.prompt_type] = []
            by_type[prompt.prompt_type].append(prompt)

        # Merge each type group, then combine
        merged_groups: list[PromptTemplate] = []

        # Order: system first, then others
        type_order = [
            PromptType.SYSTEM,
            PromptType.MEMORY,
            PromptType.TOOL,
            PromptType.PLANNER,
            PromptType.USER,
        ]

        for prompt_type in type_order:
            if prompt_type in by_type:
                group_merged = self._merge_append(by_type[prompt_type])
                merged_groups.append(group_merged)
                del by_type[prompt_type]

        # Add remaining types
        for group in by_type.values():
            merged_groups.append(self._merge_append(group))

        return self._merge_append(merged_groups)

    def merge_sections(
        self,
        base: PromptTemplate,
        overrides: dict[str, str],
    ) -> PromptTemplate:
        """
        Merge section overrides into a base template.

        Args:
            base: Base template
            overrides: Section name -> content overrides

        Returns:
            New template with merged sections
        """
        merged_sections = dict(base.sections)
        merged_sections.update(overrides)

        # Rebuild content from sections
        parts: list[str] = []
        for name, content in merged_sections.items():
            header = name.replace("_", " ").title()
            parts.append(f"## {header}\n\n{content}")

        return PromptTemplate(
            content=self._separator.join(parts),
            metadata=base.metadata,
            format=base.format,
            sections=merged_sections,
        )


# =============================================================================
# SPECIALIZED PROMPT BUILDERS
# =============================================================================


class SystemPromptBuilder:
    """Builds system prompts with standard structure."""

    def __init__(
        self,
        base_persona: str = "",
        base_instructions: list[str] | None = None,
        base_constraints: list[str] | None = None,
    ) -> None:
        self._persona = base_persona
        self._instructions = list(base_instructions or [])
        self._constraints = list(base_constraints or [])

    def build(
        self,
        persona: str | None = None,
        instructions: list[str] | None = None,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
        output_format: str | None = None,
    ) -> PromptTemplate:
        """Build a system prompt."""
        sections: dict[str, str] = {}
        parts: list[str] = []

        # Persona
        final_persona = persona or self._persona
        if final_persona:
            sections["persona"] = final_persona
            parts.append(f"## Persona\n\n{final_persona}")

        # Instructions
        final_instructions = (instructions or []) + self._instructions
        if final_instructions:
            instruction_text = "\n".join(f"- {i}" for i in final_instructions)
            sections["instructions"] = instruction_text
            parts.append(f"## Instructions\n\n{instruction_text}")

        # Constraints
        final_constraints = (constraints or []) + self._constraints
        if final_constraints:
            constraint_text = "\n".join(f"- {c}" for c in final_constraints)
            sections["constraints"] = constraint_text
            parts.append(f"## Constraints\n\n{constraint_text}")

        # Context
        if context:
            context_text = "\n".join(f"- {k}: {v}" for k, v in context.items())
            sections["context"] = context_text
            parts.append(f"## Context\n\n{context_text}")

        # Output format
        if output_format:
            sections["output_format"] = output_format
            parts.append(f"## Output Format\n\n{output_format}")

        content = "\n\n".join(parts)

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="system_prompt",
                prompt_type=PromptType.SYSTEM,
            ),
            sections=sections,
        )


class MemoryPromptBuilder:
    """Builds prompts for memory operations."""

    RETRIEVAL_TEMPLATE = """## Memory Context

The following information has been retrieved from memory and may be relevant to the current request:

{memories}

Use this context to inform your response, but prioritize the current request."""

    STORAGE_TEMPLATE = """## Memory Storage

Analyze the following interaction and extract key information worth remembering:

User: {user_input}
Assistant: {assistant_response}

Extract:
1. Key facts or preferences mentioned
2. Important context for future interactions
3. Any explicit requests to remember something

Format as JSON:
{{"memories": [{{"content": "...", "type": "fact|preference|context", "importance": 1-10}}]}}"""

    def build_retrieval_prompt(
        self,
        memories: list[dict[str, Any]],
        max_memories: int = 10,
    ) -> PromptTemplate:
        """Build prompt for memory retrieval context."""
        memory_texts: list[str] = []

        for i, memory in enumerate(memories[:max_memories], 1):
            content = memory.get("content", "")
            relevance = memory.get("relevance", 0)
            memory_texts.append(f"[{i}] (relevance: {relevance:.2f}) {content}")

        memories_text = "\n".join(memory_texts) if memory_texts else "No relevant memories found."

        content = self.RETRIEVAL_TEMPLATE.format(memories=memories_text)

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="memory_retrieval",
                prompt_type=PromptType.MEMORY,
            ),
        )

    def build_storage_prompt(
        self,
        user_input: str,
        assistant_response: str,
    ) -> PromptTemplate:
        """Build prompt for memory storage analysis."""
        content = self.STORAGE_TEMPLATE.format(
            user_input=user_input,
            assistant_response=assistant_response,
        )

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="memory_storage",
                prompt_type=PromptType.MEMORY,
            ),
        )


class ToolPromptBuilder:
    """Builds prompts for tool usage."""

    TOOL_LIST_TEMPLATE = """## Available Tools

You have access to the following tools:

{tool_descriptions}

## Tool Usage

To use a tool, respond with a JSON object:
```json
{{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}
```(Only use tools when necessary. If you can answer directly, do so without using tools."""

    TOOL_RESULT_TEMPLATE = """## Tool Result

Tool: {tool_name}
Status: {status}
Result:

{result}

Use this result to continue with your response."""

    def build_tool_list_prompt(
        self,
        tools: list[dict[str, Any]],
    ) -> PromptTemplate:
        """Build prompt listing available tools."""
        tool_descriptions: list[str] = []

        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            parameters = tool.get("parameters", {})

            param_str = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in parameters.items()
            )

            tool_descriptions.append(
                f"- **{name}**({param_str}): {description}"
            )

        content = self.TOOL_LIST_TEMPLATE.format(
            tool_descriptions="\n".join(tool_descriptions) or "No tools available."
        )

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="tool_list",
                prompt_type=PromptType.TOOL,
            ),
        )

    def build_tool_result_prompt(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
    ) -> PromptTemplate:
        """Build prompt for tool execution result."""
        status = "Success" if success else "Failed"

        content = self.TOOL_RESULT_TEMPLATE.format(
            tool_name=tool_name,
            status=status,
            result=str(result),
        )

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="tool_result",
                prompt_type=PromptType.TOOL,
            ),
        )


class PlannerPromptBuilder:
    """Builds prompts for execution planning."""

    PLANNING_TEMPLATE = """## Execution Planning

Analyze the following request and create an execution plan.

### Request
{request}

### Available Capabilities
{capabilities}

### Instructions
Create a step-by-step execution plan. Consider:
- What capabilities are needed
- The optimal order of operations
- Dependencies between steps
- Potential failure points

### Output Format
Respond with a JSON execution plan:
```json
{{
  "goal": "High-level goal description",
  "steps": [
    {{
      "id": "step_1",
      "action": "action_name",
      "capability": "capability_type",
      "parameters": {{}},
      "dependencies": [],
      "description": "What this step does"
    }}
  ],
  "estimated_complexity": 1-10,
  "potential_issues": ["issue1", "issue2"]
}}
```"""

    REPLAN_TEMPLATE = """## Replanning Required

The original plan encountered an issue. Create a revised plan.

### Original Goal
{goal}

### Completed Steps
{completed_steps}

### Failed Step
{failed_step}

### Error
{error}

### Instructions
Create a new plan that:
1. Builds on completed work
2. Avoids the failure mode
3. Achieves the original goal

Respond with a revised JSON execution plan."""

    def build_planning_prompt(
        self,
        request: str,
        capabilities: list[dict[str, Any]],
    ) -> PromptTemplate:
        """Build prompt for initial planning."""
        cap_descriptions: list[str] = []
        for cap in capabilities:
            name = cap.get("name", "")
            cap_type = cap.get("type", "")
            actions = cap.get("actions", [])
            cap_descriptions.append(
                f"- {name} ({cap_type}): {', '.join(actions)}"
            )

        content = self.PLANNING_TEMPLATE.format(
            request=request,
            capabilities="\n".join(cap_descriptions) or "No capabilities available.",
        )

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="planner",
                prompt_type=PromptType.PLANNER,
            ),
        )

    def build_replan_prompt(
        self,
        goal: str,
        completed_steps: list[dict[str, Any]],
        failed_step: dict[str, Any],
        error: str,
    ) -> PromptTemplate:
        """Build prompt for replanning after failure."""
        completed_text = "\n".join(
            f"- {s.get('id', 'unknown')}: {s.get('description', 'No description')}"
            for s in completed_steps
        )

        failed_text = json.dumps(failed_step, indent=2)

        content = self.REPLAN_TEMPLATE.format(
            goal=goal,
            completed_steps=completed_text or "None",
            failed_step=failed_text,
            error=error,
        )

        return PromptTemplate(
            content=content,
            metadata=PromptMetadata(
                name="replanner",
                prompt_type=PromptType.PLANNER,
            ),
        )


# =============================================================================
# PROMPT MANAGER
# =============================================================================


@dataclass
class PromptManagerConfig:
    """Configuration for PromptManager."""

    default_format: PromptFormat = PromptFormat.PYTHON
    strict_rendering: bool = False
    enable_caching: bool = True
    enable_versioning: bool = True
    prompts_path: str | None = None


class PromptManager:
    """
    Central prompt management system.

    Provides loading, rendering, merging, and versioning
    of prompts across the agent platform.
    """

    def __init__(
        self,
        config: PromptManagerConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize prompt manager.

        Args:
            config: Manager configuration
            logger: Optional logger
        """
        self._config = config or PromptManagerConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Loaders
        self._loaders: list[PromptLoader] = []
        self._default_loader = DictPromptLoader({})

        # Add file loader if path configured
        if self._config.prompts_path:
            self._loaders.append(
                FilePromptLoader(self._config.prompts_path, self._logger)
            )

        # Renderers
        self._renderers: dict[PromptFormat, PromptRenderer] = {
            PromptFormat.PLAIN: PythonRenderer(strict=False),
            PromptFormat.PYTHON: PythonRenderer(strict=self._config.strict_rendering),
            PromptFormat.MUSTACHE: MustacheRenderer(strict=self._config.strict_rendering),
        }

        # Try to add Jinja2 renderer
        try:
            self._renderers[PromptFormat.JINJA2] = Jinja2Renderer(
                strict=self._config.strict_rendering
            )
        except ImportError:
            pass

        # Components
        self._merger = PromptMerger()

        # Specialized builders
        self._system_builder = SystemPromptBuilder()
        self._memory_builder = MemoryPromptBuilder()
        self._tool_builder = ToolPromptBuilder()
        self._planner_builder = PlannerPromptBuilder()

        # Cache
        self._render_cache: dict[str, str] = {}

        # Version tracking
        self._version_history: dict[str, list[PromptTemplate]] = {}

    # -------------------------------------------------------------------------
    # Loader Management
    # -------------------------------------------------------------------------

    def add_loader(self, loader: PromptLoader) -> None:
        """Add a prompt loader."""
        self._loaders.append(loader)

    def register_prompt(
        self,
        name: str,
        content: str | PromptTemplate,
        prompt_type: PromptType = PromptType.CUSTOM,
    ) -> None:
        """
        Register a prompt directly.

        Args:
            name: Prompt name
            content: Prompt content or template
            prompt_type: Type classification
        """
        if isinstance(content, str):
            template = PromptTemplate(
                content=content,
                metadata=PromptMetadata(name=name, prompt_type=prompt_type),
                format=self._config.default_format,
            )
        else:
            template = content

        self._default_loader.add(name, template)

        # Track version
        if self._config.enable_versioning:
            if name not in self._version_history:
                self._version_history[name] = []
            self._version_history[name].append(template)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptTemplate | None:
        """
        Load a prompt by name.

        Args:
            name: Prompt name
            version: Optional version

        Returns:
            PromptTemplate or None if not found
        """
        # Try default loader first
        template = self._default_loader.load(name, version)
        if template:
            return template

        # Try other loaders
        for loader in self._loaders:
            template = loader.load(name, version)
            if template:
                return template

        return None

    def load_or_default(
        self,
        name: str,
        default: str,
        version: str | None = None,
    ) -> PromptTemplate:
        """
        Load prompt or return default.

        Args:
            name: Prompt name
            default: Default content if not found
            version: Optional version

        Returns:
            PromptTemplate
        """
        template = self.load(name, version)
        if template:
            return template

        return PromptTemplate(
            content=default,
            metadata=PromptMetadata(name=name),
            format=self._config.default_format,
        )

    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        names: set[str] = set()

        names.update(self._default_loader.list_prompts())

        for loader in self._loaders:
            names.update(loader.list_prompts())

        return sorted(names)

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render(
        self,
        template: str | PromptTemplate,
        variables: dict[str, Any] | None = None,
        format: PromptFormat | None = None,
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            template: Template string or PromptTemplate
            variables: Variables for substitution
            format: Override template format

        Returns:
            Rendered prompt string
        """
        variables = variables or {}

        if isinstance(template, str):
            content = template
            fmt = format or self._config.default_format
        else:
            content = template.content
            fmt = format or template.format

        # Check cache
        if self._config.enable_caching:
            cache_key = f"{hash(content)}:{hash(frozenset(variables.items()))}"
            if cache_key in self._render_cache:
                return self._render_cache[cache_key]

        # Get renderer
        renderer = self._renderers.get(fmt)
        if renderer is None:
            renderer = self._renderers[PromptFormat.PYTHON]

        # Render
        try:
            result = renderer.render(content, variables)
        except Exception as e:
            self._logger.warning(f"Render failed, returning raw: {e}")
            result = content

        # Cache result
        if self._config.enable_caching:
            self._render_cache[cache_key] = result

        return result

    def render_prompt(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> str | None:
        """
        Load and render a prompt by name.

        Args:
            name: Prompt name
            variables: Variables for substitution
            version: Optional version

        Returns:
            Rendered string or None if not found
        """
        template = self.load(name, version)
        if template is None:
            return None

        return self.render(template, variables)

    # -------------------------------------------------------------------------
    # Merging
    # -------------------------------------------------------------------------

    def merge(
        self,
        prompts: list[str | PromptTemplate],
        strategy: MergeStrategy = MergeStrategy.APPEND,
    ) -> PromptTemplate:
        """
        Merge multiple prompts.

        Args:
            prompts: Prompt names or templates to merge
            strategy: Merge strategy

        Returns:
            Merged PromptTemplate
        """
        templates: list[PromptTemplate] = []

        for prompt in prompts:
            if isinstance(prompt, str):
                template = self.load(prompt)
                if template:
                    templates.append(template)
            else:
                templates.append(prompt)

        return self._merger.merge(templates, strategy)

    # -------------------------------------------------------------------------
    # Specialized Builders
    # -------------------------------------------------------------------------

    def build_system_prompt(
        self,
        persona: str | None = None,
        instructions: list[str] | None = None,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
        output_format: str | None = None,
    ) -> PromptTemplate:
        """Build a system prompt."""
        return self._system_builder.build(
            persona=persona,
            instructions=instructions,
            constraints=constraints,
            context=context,
            output_format=output_format,
        )

    def build_memory_retrieval_prompt(
        self,
        memories: list[dict[str, Any]],
    ) -> PromptTemplate:
        """Build memory retrieval context prompt."""
        return self._memory_builder.build_retrieval_prompt(memories)

    def build_memory_storage_prompt(
        self,
        user_input: str,
        assistant_response: str,
    ) -> PromptTemplate:
        """Build memory storage analysis prompt."""
        return self._memory_builder.build_storage_prompt(user_input, assistant_response)

    def build_tool_list_prompt(
        self,
        tools: list[dict[str, Any]],
    ) -> PromptTemplate:
        """Build tool list prompt."""
        return self._tool_builder.build_tool_list_prompt(tools)

    def build_tool_result_prompt(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
    ) -> PromptTemplate:
        """Build tool result prompt."""
        return self._tool_builder.build_tool_result_prompt(tool_name, result, success)

    def build_planning_prompt(
        self,
        request: str,
        capabilities: list[dict[str, Any]],
    ) -> PromptTemplate:
        """Build planning prompt."""
        return self._planner_builder.build_planning_prompt(request, capabilities)

    def build_replan_prompt(
        self,
        goal: str,
        completed_steps: list[dict[str, Any]],
        failed_step: dict[str, Any],
        error: str,
    ) -> PromptTemplate:
        """Build replanning prompt."""
        return self._planner_builder.build_replan_prompt(
            goal, completed_steps, failed_step, error
        )

    # -------------------------------------------------------------------------
    # Versioning
    # -------------------------------------------------------------------------

    def get_versions(self, name: str) -> list[str]:
        """Get all versions of a prompt."""
        if name in self._version_history:
            return [t.version for t in self._version_history[name]]
        return []

    def get_latest_version(self, name: str) -> str | None:
        """Get latest version of a prompt."""
        versions = self.get_versions(name)
        if not versions:
            return None

        # Sort by version tuple
        sorted_versions = sorted(
            versions,
            key=lambda v: tuple(int(x) for x in v.split(".")),
            reverse=True,
        )
        return sorted_versions[0]

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def validate_template(
        self,
        template: PromptTemplate,
        variables: dict[str, Any],
    ) -> list[str]:
        """
        Validate template variables.

        Returns:
            List of missing variable names
        """
        return template.validate(variables)

    def clear_cache(self) -> None:
        """Clear render cache."""
        self._render_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "registered_prompts": len(self._default_loader.list_prompts()),
            "loaders": len(self._loaders) + 1,
            "renderers": list(self._renderers.keys()),
            "cache_size": len(self._render_cache),
            "versioned_prompts": len(self._version_history),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_prompt_manager(
    prompts_path: str | None = None,
    default_prompts: dict[str, str] | None = None,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> PromptManager:
    """
    Factory function to create configured PromptManager.

    Args:
        prompts_path: Path to prompt files
        default_prompts: Default prompts to register
        strict: Enable strict rendering
        logger: Optional logger

    Returns:
        Configured PromptManager
    """
    config = PromptManagerConfig(
        prompts_path=prompts_path,
        strict_rendering=strict,
    )

    manager = PromptManager(config=config, logger=logger)

    if default_prompts:
        for name, content in default_prompts.items():
            manager.register_prompt(name, content)

    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PromptType",
    "PromptFormat",
    "MergeStrategy",
    # Templates
    "PromptMetadata",
    "PromptTemplate",
    # Renderers
    "PromptRenderer",
    "PythonRenderer",
    "Jinja2Renderer",
    "MustacheRenderer",
    # Loaders
    "PromptLoader",
    "FilePromptLoader",
    "DictPromptLoader",
    # Merger
    "PromptMerger",
    # Builders
    "SystemPromptBuilder",
    "MemoryPromptBuilder",
    "ToolPromptBuilder",
    "PlannerPromptBuilder",
    # Manager
    "PromptManagerConfig",
    "PromptManager",
    # Factory
    "create_prompt_manager",
]
