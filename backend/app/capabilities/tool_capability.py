"""
Tool Capability - Unified tool execution for the AI Agent Platform.

This module provides the ToolCapability class which handles:
- Calculator operations (math expressions, unit conversions)
- Weather lookups (current conditions, forecasts)
- Web search (query-based information retrieval)
- File operations (read, list, metadata)
- Custom tool registration and execution

Features:
- Tool registry for dynamic tool management
- Async execution with timeout support
- Health checks per tool
- Rate limiting and caching
- Structured input/output validation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Awaitable,
)

# Import from kernel modules
from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CapabilityType,
)


# =============================================================================
# ENUMS
# =============================================================================


class ToolCategory(Enum):
    """Categories of tools."""

    CALCULATOR = "calculator"
    WEATHER = "weather"
    WEB_SEARCH = "web_search"
    FILE = "file"
    HTTP = "http"
    DATABASE = "database"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """Status of a tool."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.

    Attributes:
        name: Parameter name
        param_type: Data type (string, number, boolean, array, object)
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not provided
        enum: Allowed values (if restricted)
        minimum: Minimum value (for numbers)
        maximum: Maximum value (for numbers)
        pattern: Regex pattern (for strings)
    """

    name: str
    param_type: str
    description: str = ""
    required: bool = False
    default: Any = None
    enum: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this parameter definition."""
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, ""

        # Type validation
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "integer": lambda v: isinstance(v, int),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(self.param_type)
        if validator and not validator(value):
            return False, f"Parameter '{self.name}' must be of type {self.param_type}"

        # Enum validation
        if self.enum and value not in self.enum:
            return False, f"Parameter '{self.name}' must be one of {self.enum}"

        # Range validation for numbers
        if self.param_type in ("number", "integer"):
            if self.minimum is not None and value < self.minimum:
                return False, f"Parameter '{self.name}' must be >= {self.minimum}"
            if self.maximum is not None and value > self.maximum:
                return False, f"Parameter '{self.name}' must be <= {self.maximum}"

        # Pattern validation for strings
        if self.param_type == "string" and self.pattern:
            if not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' must match pattern {self.pattern}"

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = list(self.enum)
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.pattern:
            result["pattern"] = self.pattern
        return result


@dataclass
class ToolDefinition:
    """
    Definition of a tool.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description
        category: Tool category
        parameters: Parameter definitions
        returns: Return type description
        examples: Usage examples
        requires_auth: Whether authentication is required
        rate_limit: Max calls per minute (0 = unlimited)
        timeout: Execution timeout in seconds
        cacheable: Whether results can be cached
        cache_ttl: Cache time-to-live in seconds
        enabled: Whether tool is enabled
        tags: Searchable tags
    """

    name: str
    description: str
    category: ToolCategory
    parameters: tuple[ToolParameter, ...] = ()
    returns: str = "any"
    examples: tuple[dict[str, Any], ...] = ()
    requires_auth: bool = False
    rate_limit: int = 0
    timeout: float = 30.0
    cacheable: bool = False
    cache_ttl: int = 300
    enabled: bool = True
    tags: tuple[str, ...] = ()

    def validate_parameters(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameters against definitions."""
        errors: list[str] = []

        for param_def in self.parameters:
            value = params.get(param_def.name, param_def.default)
            valid, error = param_def.validate(value)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": {p.name: p.to_dict() for p in self.parameters},
            "returns": self.returns,
            "examples": list(self.examples),
            "tags": list(self.tags),
        }


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Attributes:
        success: Whether execution succeeded
        data: Result data
        error: Error message if failed
        execution_time: Time taken in seconds
        cached: Whether result was from cache
        metadata: Additional metadata
    """

    success: bool
    data: Any = None
    error: str | None = None
    execution_time: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """Record of a tool call for rate limiting."""

    tool_name: str
    timestamp: datetime
    success: bool
    execution_time: float


# =============================================================================
# TOOL BASE CLASS
# =============================================================================


class Tool(ABC):
    """
    Abstract base class for tools.

    All tools must implement:
    - definition: Tool metadata
    - execute: Async execution logic
    - health_check: Health verification
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get tool definition."""
        ...

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    async def health_check(self) -> bool:
        """Check if tool is healthy and available."""
        return True

    def validate(self, parameters: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameters."""
        return self.definition.validate_parameters(parameters)


# =============================================================================
# CALCULATOR TOOL
# =============================================================================


class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.

    Supports:
    - Basic arithmetic (+, -, *, /, %, **)
    - Mathematical functions (sin, cos, tan, sqrt, log, etc.)
    - Unit conversions
    - Expression evaluation
    """

    # Safe math functions
    SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    # Unit conversion factors (to base unit)
    UNIT_CONVERSIONS: dict[str, dict[str, float | str]] = {
        "length": {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "mi": 1609.344,
            "yd": 0.9144,
            "ft": 0.3048,
            "in": 0.0254,
        },
        "weight": {
            "kg": 1.0,
            "g": 0.001,
            "mg": 0.000001,
            "lb": 0.453592,
            "oz": 0.0283495,
        },
        "temperature": {
            "c": "celsius",
            "f": "fahrenheit",
            "k": "kelvin",
        },
        "time": {
            "s": 1.0,
            "ms": 0.001,
            "min": 60.0,
            "h": 3600.0,
            "d": 86400.0,
        },
    }

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations, evaluate expressions, and convert units",
            category=ToolCategory.CALCULATOR,
            parameters=(
                ToolParameter(
                    name="expression",
                    param_type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                    required=False,
                ),
                ToolParameter(
                    name="operation",
                    param_type="string",
                    description="Specific operation: 'evaluate', 'convert'",
                    required=False,
                    default="evaluate",
                    enum=("evaluate", "convert"),
                ),
                ToolParameter(
                    name="value",
                    param_type="number",
                    description="Value for conversion",
                    required=False,
                ),
                ToolParameter(
                    name="from_unit",
                    param_type="string",
                    description="Source unit for conversion",
                    required=False,
                ),
                ToolParameter(
                    name="to_unit",
                    param_type="string",
                    description="Target unit for conversion",
                    required=False,
                ),
            ),
            returns="number or object with result",
            examples=(
                {"expression": "2 + 2", "result": 4},
                {"expression": "sqrt(16) * 2", "result": 8.0},
                {"operation": "convert", "value": 100, "from_unit": "cm", "to_unit": "m", "result": 1.0},
            ),
            cacheable=True,
            cache_ttl=3600,
            tags=("math", "calculation", "conversion"),
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute calculator operation."""
        start_time = time.time()

        try:
            operation = parameters.get("operation", "evaluate")

            if operation == "convert":
                result = self._convert_units(
                    value=parameters.get("value", 0),
                    from_unit=parameters.get("from_unit", ""),
                    to_unit=parameters.get("to_unit", ""),
                )
            else:
                expression = parameters.get("expression", "")
                if not expression:
                    return ToolResult(
                        success=False,
                        error="Expression is required for evaluation",
                        execution_time=time.time() - start_time,
                    )
                result = self._evaluate_expression(expression)

            return ToolResult(
                success=True,
                data={"result": result, "expression": parameters.get("expression", "")},
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _evaluate_expression(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Clean expression
        expression = expression.strip()

        # Replace common constants
        expression = expression.replace("pi", str(math.pi))
        expression = expression.replace("e", str(math.e))

        # Build safe namespace
        safe_dict: dict[str, Any] = {
            "__builtins__": {},
            **self.SAFE_FUNCTIONS,
        }

        # Validate expression (only allow safe characters)
        allowed_pattern = r"^[\d\s\+\-\*\/\%\(\)\.\,\w]+$"
        if not re.match(allowed_pattern, expression):
            raise ValueError(f"Invalid characters in expression: {expression}")

        # Evaluate
        try:
            result = eval(expression, safe_dict)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")

    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between units."""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Find unit category
        category = None
        for cat, cat_units in self.UNIT_CONVERSIONS.items():
            if from_unit in cat_units and to_unit in cat_units:
                category = cat
                break

        if category is None:
            raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")

        # Handle temperature specially
        if category == "temperature":
            return self._convert_temperature(value, from_unit, to_unit)

        # Standard conversion via base unit (temperature handled above)
        base_value = value * float(self.UNIT_CONVERSIONS[category][from_unit])
        return base_value / float(self.UNIT_CONVERSIONS[category][to_unit])

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature units."""
        # Convert to Celsius first
        if from_unit == "f":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "k":
            celsius = value - 273.15
        else:
            celsius = value

        # Convert from Celsius to target
        if to_unit == "f":
            return celsius * 9 / 5 + 32
        elif to_unit == "k":
            return celsius + 273.15
        else:
            return celsius


# =============================================================================
# WEATHER TOOL
# =============================================================================


class WeatherTool(Tool):
    """
    Weather information tool.

    Provides:
    - Current weather conditions
    - Weather forecasts
    - Weather alerts

    Note: This is a mock implementation. In production,
    integrate with a real weather API (OpenWeatherMap, etc.)
    """

    # Mock weather data for demonstration
    MOCK_WEATHER: dict[str, dict[str, Any]] = {
        "new york": {
            "temperature": 22,
            "humidity": 65,
            "conditions": "Partly Cloudy",
            "wind_speed": 12,
            "wind_direction": "NW",
        },
        "london": {
            "temperature": 15,
            "humidity": 80,
            "conditions": "Rainy",
            "wind_speed": 20,
            "wind_direction": "SW",
        },
        "tokyo": {
            "temperature": 28,
            "humidity": 70,
            "conditions": "Sunny",
            "wind_speed": 8,
            "wind_direction": "E",
        },
        "sydney": {
            "temperature": 18,
            "humidity": 55,
            "conditions": "Clear",
            "wind_speed": 15,
            "wind_direction": "S",
        },
        "default": {
            "temperature": 20,
            "humidity": 60,
            "conditions": "Clear",
            "wind_speed": 10,
            "wind_direction": "N",
        },
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        """
        Initialize weather tool.

        Args:
            api_key: API key for weather service
            api_url: Base URL for weather API
        """
        self._api_key = api_key
        self._api_url = api_url or "https://api.openweathermap.org/data/2.5"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="weather",
            description="Get current weather conditions and forecasts for a location",
            category=ToolCategory.WEATHER,
            parameters=(
                ToolParameter(
                    name="location",
                    param_type="string",
                    description="City name or location (e.g., 'New York', 'London, UK')",
                    required=True,
                ),
                ToolParameter(
                    name="units",
                    param_type="string",
                    description="Temperature units: 'metric' (Celsius) or 'imperial' (Fahrenheit)",
                    required=False,
                    default="metric",
                    enum=("metric", "imperial"),
                ),
                ToolParameter(
                    name="forecast_days",
                    param_type="integer",
                    description="Number of forecast days (0 for current only)",
                    required=False,
                    default=0,
                    minimum=0,
                    maximum=7,
                ),
            ),
            returns="object with weather data",
            examples=(
                {
                    "location": "New York",
                    "result": {
                        "temperature": 22,
                        "conditions": "Partly Cloudy",
                        "humidity": 65,
                    },
                },
            ),
            cacheable=True,
            cache_ttl=600,  # 10 minutes
            tags=("weather", "forecast", "temperature"),
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Get weather for location."""
        start_time = time.time()

        try:
            location = parameters.get("location", "").lower()
            units = parameters.get("units", "metric")
            forecast_days = parameters.get("forecast_days", 0)

            if not location:
                return ToolResult(
                    success=False,
                    error="Location is required",
                    execution_time=time.time() - start_time,
                )

            # Get weather data (mock or real API)
            if self._api_key:
                weather_data = await self._fetch_real_weather(location, units)
            else:
                weather_data = self._get_mock_weather(location, units)

            # Add forecast if requested
            if forecast_days > 0:
                weather_data["forecast"] = self._generate_mock_forecast(
                    weather_data, forecast_days
                )

            return ToolResult(
                success=True,
                data=weather_data,
                execution_time=time.time() - start_time,
                metadata={"location": location, "units": units},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _get_mock_weather(self, location: str, units: str) -> dict[str, Any]:
        """Get mock weather data."""
        # Find matching location or use default
        weather = None
        for city, data in self.MOCK_WEATHER.items():
            if city in location:
                weather = data.copy()
                break

        if weather is None:
            weather = self.MOCK_WEATHER["default"].copy()

        # Convert units if needed
        if units == "imperial":
            weather["temperature"] = weather["temperature"] * 9 / 5 + 32
            weather["temperature_unit"] = "°F"
            weather["wind_speed_unit"] = "mph"
        else:
            weather["temperature_unit"] = "°C"
            weather["wind_speed_unit"] = "km/h"

        weather["location"] = location.title()
        weather["timestamp"] = datetime.now(timezone.utc).isoformat()

        return weather

    async def _fetch_real_weather(self, location: str, units: str) -> dict[str, Any]:
        """Fetch weather from real API."""
        # Placeholder for real API integration
        # In production, use aiohttp to call the weather API
        return self._get_mock_weather(location, units)

    def _generate_mock_forecast(
        self, current: dict[str, Any], days: int
    ) -> list[dict[str, Any]]:
        """Generate mock forecast data."""
        import random

        forecast = []
        base_temp = current["temperature"]

        for i in range(1, days + 1):
            forecast.append({
                "day": i,
                "date": (datetime.now(timezone.utc).date().__add__(
                    __import__("datetime").timedelta(days=i)
                )).isoformat(),
                "high": base_temp + random.randint(-3, 5),
                "low": base_temp + random.randint(-8, -2),
                "conditions": random.choice([
                    "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Clear"
                ]),
                "precipitation_chance": random.randint(0, 100),
            })

        return forecast

    async def health_check(self) -> bool:
        """Check if weather service is available."""
        # In production, ping the API
        return True


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================


class WebSearchTool(Tool):
    """
    Web search tool.

    Provides:
    - Query-based web search
    - Result summarization
    - Source attribution

    Note: This is a mock implementation. In production,
    integrate with a real search API (Google, Bing, DuckDuckGo, etc.)
    """

    # Mock search results
    MOCK_RESULTS: dict[str, list[dict[str, str]]] = {
        "python": [
            {
                "title": "Python.org",
                "url": "https://www.python.org",
                "snippet": "The official home of the Python Programming Language.",
            },
            {
                "title": "Python Tutorial - W3Schools",
                "url": "https://www.w3schools.com/python/",
                "snippet": "Learn Python programming with our comprehensive tutorial.",
            },
        ],
        "weather": [
            {
                "title": "Weather.com",
                "url": "https://weather.com",
                "snippet": "Get the latest weather forecasts and conditions.",
            },
        ],
        "default": [
            {
                "title": "Wikipedia",
                "url": "https://www.wikipedia.org",
                "snippet": "The free encyclopedia that anyone can edit.",
            },
        ],
    }

    def __init__(
        self,
        api_key: str | None = None,
        search_engine: str = "duckduckgo",
    ) -> None:
        """
        Initialize web search tool.

        Args:
            api_key: API key for search service
            search_engine: Search engine to use
        """
        self._api_key = api_key
        self._search_engine = search_engine

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for information on any topic",
            category=ToolCategory.WEB_SEARCH,
            parameters=(
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    param_type="integer",
                    description="Number of results to return",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=20,
                ),
                ToolParameter(
                    name="safe_search",
                    param_type="boolean",
                    description="Enable safe search filtering",
                    required=False,
                    default=True,
                ),
            ),
            returns="array of search results",
            examples=(
                {
                    "query": "Python programming",
                    "result": [
                        {"title": "Python.org", "url": "...", "snippet": "..."},
                    ],
                },
            ),
            rate_limit=30,  # 30 requests per minute
            cacheable=True,
            cache_ttl=300,
            tags=("search", "web", "information"),
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute web search."""
        start_time = time.time()

        try:
            query = parameters.get("query", "")
            num_results = parameters.get("num_results", 5)

            if not query:
                return ToolResult(
                    success=False,
                    error="Query is required",
                    execution_time=time.time() - start_time,
                )

            # Get search results (mock or real API)
            if self._api_key:
                results = await self._fetch_real_results(query, num_results)
            else:
                results = self._get_mock_results(query, num_results)

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                },
                execution_time=time.time() - start_time,
                metadata={"search_engine": self._search_engine},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _get_mock_results(self, query: str, num_results: int) -> list[dict[str, str]]:
        """Get mock search results."""
        query_lower = query.lower()

        # Find matching results
        results = []
        for keyword, mock_results in self.MOCK_RESULTS.items():
            if keyword in query_lower:
                results.extend(mock_results)

        if not results:
            results = self.MOCK_RESULTS["default"].copy()

        # Add query-specific mock result
        results.insert(0, {
            "title": f"Search results for: {query}",
            "url": f"https://search.example.com/q={query.replace(' ', '+')}",
            "snippet": f"Information about {query} from various sources.",
        })

        return results[:num_results]

    async def _fetch_real_results(
        self, query: str, num_results: int
    ) -> list[dict[str, str]]:
        """Fetch results from real search API."""
        # Placeholder for real API integration
        return self._get_mock_results(query, num_results)

    async def health_check(self) -> bool:
        """Check if search service is available."""
        return True


# =============================================================================
# FILE READER TOOL
# =============================================================================


class FileReaderTool(Tool):
    """
    File reading tool.

    Provides:
    - Read file contents
    - List directory contents
    - Get file metadata
    - Support for text and JSON files

    Security:
    - Sandboxed to allowed directories
    - File size limits
    - Extension filtering
    """

    # Allowed file extensions
    ALLOWED_EXTENSIONS: set[str] = {
        ".txt", ".md", ".json", ".yaml", ".yml",
        ".py", ".js", ".ts", ".html", ".css",
        ".csv", ".xml", ".log", ".ini", ".cfg",
    }

    # Maximum file size (10 MB)
    MAX_FILE_SIZE: int = 10 * 1024 * 1024

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        max_file_size: int | None = None,
    ) -> None:
        """
        Initialize file reader tool.

        Args:
            allowed_paths: List of allowed directory paths
            max_file_size: Maximum file size in bytes
        """
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self._max_file_size = max_file_size or self.MAX_FILE_SIZE

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_reader",
            description="Read files, list directories, and get file metadata",
            category=ToolCategory.FILE,
            parameters=(
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="File or directory path",
                    required=True,
                ),
                ToolParameter(
                    name="operation",
                    param_type="string",
                    description="Operation: 'read', 'list', 'metadata'",
                    required=False,
                    default="read",
                    enum=("read", "list", "metadata"),
                ),
                ToolParameter(
                    name="encoding",
                    param_type="string",
                    description="File encoding for reading",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="max_lines",
                    param_type="integer",
                    description="Maximum lines to read (0 = all)",
                    required=False,
                    default=0,
                    minimum=0,
                ),
            ),
            returns="string (content) or object (metadata/listing)",
            examples=(
                {"path": "README.md", "operation": "read"},
                {"path": "./src", "operation": "list"},
            ),
            tags=("file", "read", "directory"),
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute file operation."""
        start_time = time.time()

        try:
            path_str = parameters.get("path", "")
            operation = parameters.get("operation", "read")

            if not path_str:
                return ToolResult(
                    success=False,
                    error="Path is required",
                    execution_time=time.time() - start_time,
                )

            # Resolve and validate path
            path = Path(path_str).resolve()
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    error=f"Access denied: {path_str}",
                    execution_time=time.time() - start_time,
                )

            # Execute operation
            if operation == "read":
                result = await self._read_file(
                    path,
                    encoding=parameters.get("encoding", "utf-8"),
                    max_lines=parameters.get("max_lines", 0),
                )
            elif operation == "list":
                result = await self._list_directory(path)
            elif operation == "metadata":
                result = await self._get_metadata(path)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                    execution_time=time.time() - start_time,
                )

            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start_time,
                metadata={"path": str(path), "operation": operation},
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"File not found: {parameters.get('path', '')}",
                execution_time=time.time() - start_time,
            )
        except PermissionError:
            return ToolResult(
                success=False,
                error=f"Permission denied: {parameters.get('path', '')}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories."""
        try:
            resolved = path.resolve()
            return any(
                resolved == allowed or allowed in resolved.parents
                for allowed in self._allowed_paths
            )
        except Exception:
            return False

    async def _read_file(
        self, path: Path, encoding: str, max_lines: int
    ) -> dict[str, Any]:
        """Read file contents."""
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

        # Check extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed: {path.suffix}")

        # Check size
        size = path.stat().st_size
        if size > self._max_file_size:
            raise ValueError(f"File too large: {size} bytes (max: {self._max_file_size})")

        # Read content
        content = path.read_text(encoding=encoding)

        # Limit lines if requested
        if max_lines > 0:
            lines = content.split("\n")
            content = "\n".join(lines[:max_lines])
            truncated = len(lines) > max_lines
        else:
            truncated = False

        # Parse JSON if applicable
        if path.suffix.lower() == ".json":
            try:
                parsed = json.loads(content)
                return {
                    "content": parsed,
                    "format": "json",
                    "size": size,
                    "truncated": truncated,
                }
            except json.JSONDecodeError:
                pass

        return {
            "content": content,
            "format": "text",
            "size": size,
            "lines": content.count("\n") + 1,
            "truncated": truncated,
        }

    async def _list_directory(self, path: Path) -> dict[str, Any]:
        """List directory contents."""
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries = []
        for entry in path.iterdir():
            entries.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
                "extension": entry.suffix if entry.is_file() else None,
            })

        # Sort: directories first, then files
        entries.sort(key=lambda e: (e["type"] != "directory", str(e["name"]).lower()))

        return {
            "path": str(path),
            "entries": entries,
            "total": len(entries),
        }

    async def _get_metadata(self, path: Path) -> dict[str, Any]:
        """Get file or directory metadata."""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat = path.stat()

        return {
            "path": str(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime, timezone.utc).isoformat(),
            "extension": path.suffix if path.is_file() else None,
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
        }

    async def health_check(self) -> bool:
        """Check if file system is accessible."""
        return all(p.exists() for p in self._allowed_paths)


# =============================================================================
# TOOL REGISTRY
# =============================================================================


class ToolRegistry:
    """
    Registry for managing tools.

    Features:
    - Tool registration and discovery
    - Tool lookup by name or category
    - Health monitoring
    - Rate limiting
    - Result caching
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize tool registry.

        Args:
            logger: Optional logger
        """
        self._tools: dict[str, Tool] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._call_history: list[ToolCallRecord] = []
        self._cache: dict[str, tuple[ToolResult, float]] = {}
        self._max_history = 1000

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        name = tool.definition.name
        if name in self._tools:
            self._logger.warning(f"Overwriting existing tool: {name}")
        self._tools[name] = tool
        self._logger.debug(f"Registered tool: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            self._logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """List all registered tools."""
        definitions = []
        for tool in self._tools.values():
            defn = tool.definition
            if enabled_only and not defn.enabled:
                continue
            if category and defn.category != category:
                continue
            definitions.append(defn)
        return definitions

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM consumption."""
        return [tool.definition.to_dict() for tool in self._tools.values()]

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            ToolResult
        """
        start_time = time.time()

        # Get tool
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
                execution_time=time.time() - start_time,
            )

        defn = tool.definition

        # Check if enabled
        if not defn.enabled:
            return ToolResult(
                success=False,
                error=f"Tool is disabled: {tool_name}",
                execution_time=time.time() - start_time,
            )

        # Check rate limit
        if defn.rate_limit > 0:
            if not self._check_rate_limit(tool_name, defn.rate_limit):
                return ToolResult(
                    success=False,
                    error=f"Rate limit exceeded for tool: {tool_name}",
                    execution_time=time.time() - start_time,
                )

        # Check cache
        if defn.cacheable:
            cache_key = self._get_cache_key(tool_name, parameters)
            cached = self._get_cached(cache_key, defn.cache_ttl)
            if cached:
                cached.cached = True
                return cached

        # Validate parameters
        valid, errors = tool.validate(parameters)
        if not valid:
            return ToolResult(
                success=False,
                error=f"Validation failed: {'; '.join(errors)}",
                execution_time=time.time() - start_time,
            )

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(parameters),
                timeout=defn.timeout,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                success=False,
                error=f"Tool execution timed out after {defn.timeout}s",
                execution_time=defn.timeout,
            )

        # Record call
        self._record_call(tool_name, result)

        # Cache result
        if defn.cacheable and result.success:
            cache_key = self._get_cache_key(tool_name, parameters)
            self._cache[cache_key] = (result, time.time())

        return result

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if tool is within rate limit."""
        now = datetime.now(timezone.utc)
        minute_ago = now.timestamp() - 60

        recent_calls = sum(
            1 for record in self._call_history
            if record.tool_name == tool_name
            and record.timestamp.timestamp() > minute_ago
        )

        return recent_calls < limit

    def _record_call(self, tool_name: str, result: ToolResult) -> None:
        """Record a tool call."""
        record = ToolCallRecord(
            tool_name=tool_name,
            timestamp=datetime.now(timezone.utc),
            success=result.success,
            execution_time=result.execution_time,
        )
        self._call_history.append(record)

        # Trim history
        if len(self._call_history) > self._max_history:
            self._call_history = self._call_history[-self._max_history:]

    def _get_cache_key(self, tool_name: str, parameters: dict[str, Any]) -> str:
        """Generate cache key for tool call."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()

    def _get_cached(self, cache_key: str, ttl: int) -> ToolResult | None:
        """Get cached result if valid."""
        if cache_key not in self._cache:
            return None

        result, cached_at = self._cache[cache_key]
        if time.time() - cached_at > ttl:
            del self._cache[cache_key]
            return None

        return result

    async def health_check(self) -> dict[str, bool]:
        """Check health of all tools."""
        results = {}
        for name, tool in self._tools.items():
            try:
                results[name] = await tool.health_check()
            except Exception:
                results[name] = False
        return results

    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "enabled_tools": sum(1 for t in self._tools.values() if t.definition.enabled),
            "cache_size": len(self._cache),
            "total_calls": len(self._call_history),
            "calls_by_tool": self._get_calls_by_tool(),
        }

    def _get_calls_by_tool(self) -> dict[str, int]:
        """Get call counts by tool."""
        counts: dict[str, int] = {}
        for record in self._call_history:
            counts[record.tool_name] = counts.get(record.tool_name, 0) + 1
        return counts


# =============================================================================
# TOOL CAPABILITY
# =============================================================================


class ToolCapability(Capability):
    """
    Tool execution capability for the AI Agent Platform.

    Provides unified interface for:
    - Executing registered tools
    - Tool discovery and listing
    - Health monitoring
    - Rate limiting and caching

    Compatible with:
    - CapabilityRegistry for registration
    - Executor for action execution
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize tool capability.

        Args:
            registry: Tool registry (creates default if None)
            logger: Optional logger
        """
        self._registry = registry or ToolRegistry(logger=logger)
        self._logger = logger or logging.getLogger(__name__)
        self._initialized = False

    # -------------------------------------------------------------------------
    # Capability Interface
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return CapabilityMetadata(
            name="tools",
            version="1.0.0",
            capability_type=CapabilityType.TOOL,  # type: ignore[attr-defined]
            description="Tool execution capability for calculator, weather, web search, and file operations",
            actions=(
                "execute",
                "list",
                "describe",
                "health",
                "stats",
            ),
            required_permissions=("tools:execute", "tools:read"),
            config_schema={
                "type": "object",
                "properties": {
                    "enabled_tools": {"type": "array", "items": {"type": "string"}},
                    "rate_limit_multiplier": {"type": "number", "minimum": 0.1},
                },
            },
            tags=("tools", "execution", "calculator", "weather", "search", "file"),
        )

    async def _do_initialize(self) -> None:
        self._logger.debug("ToolCapability._do_initialize")

    async def _do_shutdown(self) -> None:
        self._logger.debug("ToolCapability._do_shutdown")

    async def _do_execute(self, context) -> CapabilityResult:
        return CapabilityResult(success=True, status=CapabilityStatus.SUCCESS)  # type: ignore[attr-defined]

    async def initialize(self) -> None:
        """Initialize the capability with default tools."""
        if self._initialized:
            return

        # Register default tools
        self._registry.register(CalculatorTool())
        self._registry.register(WeatherTool())
        self._registry.register(WebSearchTool())
        self._registry.register(FileReaderTool())

        self._initialized = True
        self._logger.info("ToolCapability initialized with default tools")

    async def shutdown(self) -> None:
        """Shutdown the capability."""
        if not self._initialized:
            return

        self._registry.clear_cache()
        self._initialized = False
        self._logger.info("ToolCapability shutdown")

    async def health_check(self) -> bool:
        """Check capability health."""
        try:
            results = await self._registry.health_check()
            # Healthy if at least one tool is healthy
            return any(results.values())
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    async def execute(  # type: ignore[override]
        self,
        action: str,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> CapabilityResult:
        """
        Execute a tool action.

        Args:
            action: Action to perform (execute, list, describe, health, stats)
            parameters: Action parameters
            context: Execution context

        Returns:
            CapabilityResult with operation outcome
        """
        start_time = time.time()
        context = context or {}

        try:
            if action == "execute":
                return await self._handle_execute(parameters, start_time)
            elif action == "list":
                return await self._handle_list(parameters, start_time)
            elif action == "describe":
                return await self._handle_describe(parameters, start_time)
            elif action == "health":
                return await self._handle_health(start_time)
            elif action == "stats":
                return await self._handle_stats(start_time)
            else:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            self._logger.exception(f"Tool action failed: {e}")
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    async def _handle_execute(
        self,
        parameters: dict[str, Any],
        start_time: float,
    ) -> CapabilityResult:
        """Handle tool execution."""
        tool_name = parameters.get("tool")
        if not tool_name:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error="Tool name is required",
                execution_time=time.time() - start_time,
            )

        tool_params = parameters.get("parameters", {})

        result = await self._registry.execute(tool_name, tool_params)

        return CapabilityResult(
            success=result.success,
            status=CapabilityStatus.COMPLETED if result.success else CapabilityStatus.FAILED,
            data={
                "tool": tool_name,
                "result": result.data,
                "cached": result.cached,
                "metadata": result.metadata,
            },
            error=result.error,
            execution_time=time.time() - start_time,
        )

    async def _handle_list(
        self,
        parameters: dict[str, Any],
        start_time: float,
    ) -> CapabilityResult:
        """Handle tool listing."""
        category_str = parameters.get("category")
        category = ToolCategory(category_str) if category_str else None
        enabled_only = parameters.get("enabled_only", True)

        tools = self._registry.list_tools(category=category, enabled_only=enabled_only)

        return CapabilityResult(
            success=True,
            status=CapabilityStatus.COMPLETED,
            data={
                "tools": [t.to_dict() for t in tools],
                "total": len(tools),
            },
            execution_time=time.time() - start_time,
        )

    async def _handle_describe(
        self,
        parameters: dict[str, Any],
        start_time: float,
    ) -> CapabilityResult:
        """Handle tool description."""
        tool_name = parameters.get("tool")
        if not tool_name:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error="Tool name is required",
                execution_time=time.time() - start_time,
            )

        tool = self._registry.get(tool_name)
        if not tool:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=f"Tool not found: {tool_name}",
                execution_time=time.time() - start_time,
            )

        return CapabilityResult(
            success=True,
            status=CapabilityStatus.COMPLETED,
            data=tool.definition.to_dict(),
            execution_time=time.time() - start_time,
        )

    async def _handle_health(self, start_time: float) -> CapabilityResult:
        """Handle health check."""
        results = await self._registry.health_check()

        return CapabilityResult(
            success=True,
            status=CapabilityStatus.COMPLETED,
            data={
                "tools": results,
                "healthy_count": sum(1 for v in results.values() if v),
                "total_count": len(results),
            },
            execution_time=time.time() - start_time,
        )

    async def _handle_stats(self, start_time: float) -> CapabilityResult:
        """Handle stats retrieval."""
        stats = self._registry.get_stats()

        return CapabilityResult(
            success=True,
            status=CapabilityStatus.COMPLETED,
            data=stats,
            execution_time=time.time() - start_time,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a custom tool."""
        self._registry.register(tool)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        return self._registry.unregister(name)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._registry.get(name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM consumption."""
        return self._registry.get_tool_schemas()

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool directly.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            ToolResult
        """
        return await self._registry.execute(tool_name, parameters)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_tool_capability(
    include_calculator: bool = True,
    include_weather: bool = True,
    include_web_search: bool = True,
    include_file_reader: bool = True,
    weather_api_key: str | None = None,
    search_api_key: str | None = None,
    allowed_file_paths: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> ToolCapability:
    """
    Factory function to create ToolCapability with specified tools.

    Args:
        include_calculator: Include calculator tool
        include_weather: Include weather tool
        include_web_search: Include web search tool
        include_file_reader: Include file reader tool
        weather_api_key: API key for weather service
        search_api_key: API key for search service
        allowed_file_paths: Allowed paths for file reader
        logger: Optional logger

    Returns:
        Configured ToolCapability
    """
    logger = logger or logging.getLogger(__name__)
    registry = ToolRegistry(logger=logger)

    if include_calculator:
        registry.register(CalculatorTool())

    if include_weather:
        registry.register(WeatherTool(api_key=weather_api_key))

    if include_web_search:
        registry.register(WebSearchTool(api_key=search_api_key))

    if include_file_reader:
        registry.register(FileReaderTool(allowed_paths=allowed_file_paths))

    capability = ToolCapability(registry=registry, logger=logger)

    # Mark as initialized since we manually registered tools
    capability._initialized = True

    return capability


# =============================================================================
# CUSTOM TOOL HELPER
# =============================================================================


def create_custom_tool(
    name: str,
    description: str,
    handler: Callable[[dict[str, Any]], Awaitable[Any]],
    parameters: list[dict[str, Any]] | None = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    **kwargs: Any,
) -> Tool:
    """
    Create a custom tool from a handler function.

    Args:
        name: Tool name
        description: Tool description
        handler: Async handler function
        parameters: Parameter definitions
        category: Tool category
        **kwargs: Additional ToolDefinition arguments

    Returns:
        Custom Tool instance
    """

    # Convert parameter dicts to ToolParameter objects
    param_objects = tuple(
        ToolParameter(
            name=p["name"],
            param_type=p.get("type", "string"),
            description=p.get("description", ""),
            required=p.get("required", False),
            default=p.get("default"),
            enum=tuple(p["enum"]) if "enum" in p else None,
        )
        for p in (parameters or [])
    )

    class CustomTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name=name,
                description=description,
                category=category,
                parameters=param_objects,
                **kwargs,
            )

        async def execute(self, params: dict[str, Any]) -> ToolResult:
            start_time = time.time()
            try:
                result = await handler(params)
                return ToolResult(
                    success=True,
                    data=result,
                    execution_time=time.time() - start_time,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    execution_time=time.time() - start_time,
                )

    return CustomTool()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ToolCategory",
    "ToolStatus",
    # Data Classes
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "ToolCallRecord",
    # Base
    "Tool",
    # Built-in Tools
    "CalculatorTool",
    "WeatherTool",
    "WebSearchTool",
    "FileReaderTool",
    # Registry
    "ToolRegistry",
    # Capability
    "ToolCapability",
    # Factory
    "create_tool_capability",
    "create_custom_tool",
]
