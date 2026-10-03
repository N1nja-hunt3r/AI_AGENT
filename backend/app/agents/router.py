"""
Capability Router - Intelligent routing decisions for the AI Agent Platform.

This module analyzes requests and determines which capabilities should be
invoked to fulfill them. It makes routing decisions based on:
- Intent analysis results
- Session context and history
- Available capabilities
- Configurable routing rules
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from app.agents.models import (
    CapabilityDecision,
    CapabilityType,
    IntentAnalysis,
    IntentType,
    SessionState,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# ROUTING SIGNALS
# =============================================================================


class RoutingSignal(Enum):
    """
    Signals that influence routing decisions.

    These signals are extracted from requests and context
    to guide capability selection.
    """

    # Memory signals
    REFERENCES_PAST = auto()  # "as I mentioned", "earlier", "before"
    REQUESTS_REMEMBER = auto()  # "remember this", "save this"
    USES_PRONOUNS = auto()  # "it", "that", "they" without clear referent
    CONTINUES_TOPIC = auto()  # Continuation of previous conversation

    # RAG signals
    DOMAIN_SPECIFIC = auto()  # Technical or specialized terminology
    REQUIRES_FACTS = auto()  # Factual questions
    DOCUMENTATION_QUERY = auto()  # "how to", "what is", documentation-style

    # Web search signals
    CURRENT_EVENTS = auto()  # News, recent events
    REAL_TIME_DATA = auto()  # Prices, weather, live data
    EXPLICIT_SEARCH = auto()  # "search for", "look up", "find online"
    TIME_SENSITIVE = auto()  # "today", "latest", "current"

    # Tool signals
    ACTION_REQUIRED = auto()  # Verbs indicating action
    COMPUTATION_NEEDED = auto()  # Math, calculations
    EXTERNAL_SERVICE = auto()  # API calls, integrations
    FILE_OPERATION = auto()  # Read, write, create files


@dataclass
class SignalDetectionResult:
    """
    Result of signal detection analysis.

    Attributes:
        signals: Set of detected signals
        confidence_scores: Confidence for each signal (0.0-1.0)
        evidence: Supporting evidence for each signal
    """

    signals: set[RoutingSignal] = field(default_factory=set)
    confidence_scores: dict[RoutingSignal, float] = field(default_factory=dict)
    evidence: dict[RoutingSignal, list[str]] = field(default_factory=dict)

    def has_signal(self, signal: RoutingSignal, min_confidence: float = 0.5) -> bool:
        """Check if signal is present with minimum confidence."""
        if signal not in self.signals:
            return False
        return self.confidence_scores.get(signal, 0.0) >= min_confidence

    def add_signal(
        self,
        signal: RoutingSignal,
        confidence: float,
        evidence: str | None = None,
    ) -> None:
        """Add a detected signal."""
        self.signals.add(signal)
        self.confidence_scores[signal] = max(
            confidence,
            self.confidence_scores.get(signal, 0.0),
        )
        if evidence:
            if signal not in self.evidence:
                self.evidence[signal] = []
            self.evidence[signal].append(evidence)

    @property
    def memory_signals(self) -> set[RoutingSignal]:
        """Get memory-related signals."""
        return self.signals & {
            RoutingSignal.REFERENCES_PAST,
            RoutingSignal.REQUESTS_REMEMBER,
            RoutingSignal.USES_PRONOUNS,
            RoutingSignal.CONTINUES_TOPIC,
        }

    @property
    def rag_signals(self) -> set[RoutingSignal]:
        """Get RAG-related signals."""
        return self.signals & {
            RoutingSignal.DOMAIN_SPECIFIC,
            RoutingSignal.REQUIRES_FACTS,
            RoutingSignal.DOCUMENTATION_QUERY,
        }

    @property
    def web_signals(self) -> set[RoutingSignal]:
        """Get web search-related signals."""
        return self.signals & {
            RoutingSignal.CURRENT_EVENTS,
            RoutingSignal.REAL_TIME_DATA,
            RoutingSignal.EXPLICIT_SEARCH,
            RoutingSignal.TIME_SENSITIVE,
        }

    @property
    def tool_signals(self) -> set[RoutingSignal]:
        """Get tool-related signals."""
        return self.signals & {
            RoutingSignal.ACTION_REQUIRED,
            RoutingSignal.COMPUTATION_NEEDED,
            RoutingSignal.EXTERNAL_SERVICE,
            RoutingSignal.FILE_OPERATION,
        }


# =============================================================================
# SIGNAL DETECTORS
# =============================================================================


class SignalDetector(ABC):
    """Abstract base for signal detection strategies."""

    @abstractmethod
    def detect(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> SignalDetectionResult:
        """
        Detect routing signals from content and context.

        Args:
            content: Request content
            intent: Intent analysis result
            session: Current session state

        Returns:
            SignalDetectionResult with detected signals
        """
        ...


class PatternBasedDetector(SignalDetector):
    """
    Detects signals using regex patterns and keyword matching.

    Fast, deterministic detection suitable for common patterns.
    """

    # Pattern definitions
    MEMORY_PATTERNS: dict[RoutingSignal, list[re.Pattern[str]]] = {
        RoutingSignal.REFERENCES_PAST: [
            re.compile(r"\b(earlier|before|previously|last time|as (i|we) (said|mentioned|discussed))\b", re.I),
            re.compile(r"\b(remember when|you (said|told|mentioned))\b", re.I),
        ],
        RoutingSignal.REQUESTS_REMEMBER: [
            re.compile(r"\b(remember (this|that)|save (this|that)|don'?t forget)\b", re.I),
            re.compile(r"\b(keep (this|that) in mind|note (this|that))\b", re.I),
        ],
        RoutingSignal.USES_PRONOUNS: [
            re.compile(r"^(it|that|this|they|those|these)\b", re.I),
            re.compile(r"\b(do it|fix it|change it|update it)\b", re.I),
        ],
    }

    RAG_PATTERNS: dict[RoutingSignal, list[re.Pattern[str]]] = {
        RoutingSignal.DOCUMENTATION_QUERY: [
            re.compile(r"\b(how (do|can|to)|what is|explain|describe)\b", re.I),
            re.compile(r"\b(documentation|docs|guide|tutorial)\b", re.I),
        ],
        RoutingSignal.REQUIRES_FACTS: [
            re.compile(r"\b(what|who|when|where|why|which)\b.*\?", re.I),
            re.compile(r"\b(definition|meaning|difference between)\b", re.I),
        ],
    }

    WEB_PATTERNS: dict[RoutingSignal, list[re.Pattern[str]]] = {
        RoutingSignal.CURRENT_EVENTS: [
            re.compile(r"\b(news|latest|recent|happening|announced)\b", re.I),
            re.compile(r"\b(today'?s?|this (week|month|year))\b", re.I),
        ],
        RoutingSignal.REAL_TIME_DATA: [
            re.compile(r"\b(price|stock|weather|temperature|exchange rate)\b", re.I),
            re.compile(r"\b(live|real-?time|current)\b", re.I),
        ],
        RoutingSignal.EXPLICIT_SEARCH: [
            re.compile(r"\b(search|look up|find|google|browse)\b", re.I),
            re.compile(r"\b(on the (web|internet)|online)\b", re.I),
        ],
        RoutingSignal.TIME_SENSITIVE: [
            re.compile(r"\b(today|now|currently|at the moment|right now)\b", re.I),
            re.compile(r"\b(latest|newest|most recent|up-?to-?date)\b", re.I),
        ],
    }

    TOOL_PATTERNS: dict[RoutingSignal, list[re.Pattern[str]]] = {
        RoutingSignal.ACTION_REQUIRED: [
            re.compile(r"\b(create|make|build|generate|send|execute|run)\b", re.I),
            re.compile(r"\b(delete|remove|update|modify|change)\b", re.I),
        ],
        RoutingSignal.COMPUTATION_NEEDED: [
            re.compile(r"\b(calculate|compute|sum|average|total)\b", re.I),
            re.compile(r"\d+\s*[\+\-\*\/\%]\s*\d+", re.I),  # Math expressions
        ],
        RoutingSignal.FILE_OPERATION: [
            re.compile(r"\b(file|document|read|write|save|load|open)\b", re.I),
            re.compile(r"\b(upload|download|export|import)\b", re.I),
        ],
        RoutingSignal.EXTERNAL_SERVICE: [
            re.compile(r"\b(api|service|endpoint|webhook)\b", re.I),
            re.compile(r"\b(slack|email|github|jira|database)\b", re.I),
        ],
    }

    def detect(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> SignalDetectionResult:
        """Detect signals using pattern matching."""
        result = SignalDetectionResult()

        # Check all pattern categories
        all_patterns = {
            **self.MEMORY_PATTERNS,
            **self.RAG_PATTERNS,
            **self.WEB_PATTERNS,
            **self.TOOL_PATTERNS,
        }

        for signal, patterns in all_patterns.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    result.add_signal(
                        signal=signal,
                        confidence=0.8,
                        evidence=match.group(0),
                    )
                    break  # One match per signal is enough

        # Context-based detection
        self._detect_context_signals(result, content, session)

        return result

    def _detect_context_signals(
        self,
        result: SignalDetectionResult,
        content: str,
        session: SessionState,
    ) -> None:
        """Detect signals based on session context."""
        # Topic continuation detection
        if session.has_history:
            recent = session.get_recent_history(1)
            if recent:
                last_content = recent[0].get("content", "")
                # Simple similarity check (could be enhanced)
                common_words = set(content.lower().split()) & set(last_content.lower().split())
                if len(common_words) >= 3:
                    result.add_signal(
                        signal=RoutingSignal.CONTINUES_TOPIC,
                        confidence=0.7,
                        evidence=f"Common words: {', '.join(list(common_words)[:5])}",
                    )


class IntentBasedDetector(SignalDetector):
    """
    Detects signals based on intent analysis results.

    Uses the pre-computed intent analysis to inform routing.
    """

    def detect(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> SignalDetectionResult:
        """Detect signals from intent analysis."""
        result = SignalDetectionResult()

        # Map intent types to signals
        if intent.intent_type == IntentType.ACTION:
            result.add_signal(
                signal=RoutingSignal.ACTION_REQUIRED,
                confidence=intent.confidence,
                evidence=f"Intent type: {intent.intent_type.value}",
            )

        if intent.intent_type == IntentType.QUERY:
            result.add_signal(
                signal=RoutingSignal.REQUIRES_FACTS,
                confidence=intent.confidence * 0.8,
                evidence=f"Intent type: {intent.intent_type.value}",
            )

        # Use intent flags
        if intent.requires_context:
            result.add_signal(
                signal=RoutingSignal.REFERENCES_PAST,
                confidence=0.9,
                evidence="Intent requires context",
            )

        if intent.requires_current_info:
            result.add_signal(
                signal=RoutingSignal.TIME_SENSITIVE,
                confidence=0.9,
                evidence="Intent requires current info",
            )

        # Check entities for tool hints
        if "tool" in intent.entities or "action" in intent.entities:
            result.add_signal(
                signal=RoutingSignal.ACTION_REQUIRED,
                confidence=0.85,
                evidence=f"Entities: {list(intent.entities.keys())}",
            )

        return result


class CompositeDetector(SignalDetector):
    """
    Combines multiple detectors for comprehensive signal detection.

    Merges results from all child detectors, taking the highest
    confidence for each signal.
    """

    def __init__(self, detectors: list[SignalDetector]) -> None:
        """
        Initialize with list of detectors.

        Args:
            detectors: Child detectors to combine
        """
        self._detectors = detectors

    def detect(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> SignalDetectionResult:
        """Combine detection results from all detectors."""
        combined = SignalDetectionResult()

        for detector in self._detectors:
            result = detector.detect(content, intent, session)

            for signal in result.signals:
                combined.add_signal(
                    signal=signal,
                    confidence=result.confidence_scores.get(signal, 0.5),
                    evidence=", ".join(result.evidence.get(signal, [])),
                )

        return combined


# =============================================================================
# ROUTING RULES
# =============================================================================


@dataclass
class RoutingRule:
    """
    Single routing rule for capability decisions.

    Attributes:
        name: Rule identifier
        capability: Target capability
        condition: Function that evaluates if rule applies
        priority: Rule priority (higher = evaluated first)
        confidence_threshold: Minimum confidence to trigger
    """

    name: str
    capability: CapabilityType
    condition: Callable[[SignalDetectionResult, IntentAnalysis, SessionState], bool]
    priority: int = 0
    confidence_threshold: float = 0.5

    def evaluate(
        self,
        signals: SignalDetectionResult,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> bool:
        """Evaluate if this rule should trigger."""
        return self.condition(signals, intent, session)


class RoutingRuleEngine:
    """
    Engine for evaluating routing rules.

    Manages a collection of rules and evaluates them
    to produce capability decisions.
    """

    def __init__(self) -> None:
        self._rules: list[RoutingRule] = []
        self._setup_default_rules()

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < original_len

    def evaluate(
        self,
        signals: SignalDetectionResult,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> dict[CapabilityType, bool]:
        """
        Evaluate all rules and return capability decisions.

        Args:
            signals: Detected routing signals
            intent: Intent analysis result
            session: Current session state

        Returns:
            Dictionary mapping capabilities to activation status
        """
        decisions: dict[CapabilityType, bool] = {
            CapabilityType.MEMORY: False,
            CapabilityType.RAG: False,
            CapabilityType.WEB_SEARCH: False,
            CapabilityType.TOOLS: False,
            CapabilityType.CHAT: True,
            CapabilityType.CODING: False,
            CapabilityType.VISION: False,
            CapabilityType.FILES: False,
            CapabilityType.COMPUTER: False,
        }

        for rule in self._rules:
            if rule.evaluate(signals, intent, session):
                decisions[rule.capability] = True

        return decisions

    def _setup_default_rules(self) -> None:
        """Configure default routing rules."""

        # Memory rules
        self.add_rule(
            RoutingRule(
                name="memory_past_reference",
                capability=CapabilityType.MEMORY,
                priority=10,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.REFERENCES_PAST)
                    or s.has_signal(RoutingSignal.USES_PRONOUNS)
                ),
            )
        )

        self.add_rule(
            RoutingRule(
                name="memory_remember_request",
                capability=CapabilityType.MEMORY,
                priority=10,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.REQUESTS_REMEMBER),
            )
        )

        self.add_rule(
            RoutingRule(
                name="memory_has_refs",
                capability=CapabilityType.MEMORY,
                priority=5,
                condition=lambda s, i, sess: bool(sess.memory_refs),
            )
        )

        self.add_rule(
            RoutingRule(
                name="memory_topic_continuation",
                capability=CapabilityType.MEMORY,
                priority=5,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.CONTINUES_TOPIC)
                    and sess.has_history
                ),
            )
        )

        # RAG rules
        self.add_rule(
            RoutingRule(
                name="rag_documentation",
                capability=CapabilityType.RAG,
                priority=10,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.DOCUMENTATION_QUERY),
            )
        )

        self.add_rule(
            RoutingRule(
                name="rag_factual_query",
                capability=CapabilityType.RAG,
                priority=8,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.REQUIRES_FACTS)
                    and i.intent_type == IntentType.QUERY
                    and not i.requires_current_info
                ),
            )
        )

        self.add_rule(
            RoutingRule(
                name="rag_domain_specific",
                capability=CapabilityType.RAG,
                priority=7,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.DOMAIN_SPECIFIC)
                    and i.complexity_score >= 4
                ),
            )
        )

        # Web search rules
        self.add_rule(
            RoutingRule(
                name="web_explicit_search",
                capability=CapabilityType.WEB_SEARCH,
                priority=10,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.EXPLICIT_SEARCH),
            )
        )

        self.add_rule(
            RoutingRule(
                name="web_current_events",
                capability=CapabilityType.WEB_SEARCH,
                priority=9,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.CURRENT_EVENTS),
            )
        )

        self.add_rule(
            RoutingRule(
                name="web_realtime_data",
                capability=CapabilityType.WEB_SEARCH,
                priority=9,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.REAL_TIME_DATA),
            )
        )

        self.add_rule(
            RoutingRule(
                name="web_time_sensitive",
                capability=CapabilityType.WEB_SEARCH,
                priority=8,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.TIME_SENSITIVE)
                    and i.requires_current_info
                ),
            )
        )

        # Tool rules
        self.add_rule(
            RoutingRule(
                name="tools_action_intent",
                capability=CapabilityType.TOOLS,
                priority=10,
                condition=lambda s, i, sess: i.intent_type == IntentType.ACTION,
            )
        )

        self.add_rule(
            RoutingRule(
                name="tools_computation",
                capability=CapabilityType.TOOLS,
                priority=9,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.COMPUTATION_NEEDED),
            )
        )

        self.add_rule(
            RoutingRule(
                name="tools_file_operation",
                capability=CapabilityType.TOOLS,
                priority=8,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.FILE_OPERATION),
            )
        )

        self.add_rule(
            RoutingRule(
                name="tools_external_service",
                capability=CapabilityType.TOOLS,
                priority=8,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.EXTERNAL_SERVICE),
            )
        )

        # Coding rules
        self.add_rule(
            RoutingRule(
                name="coding_code_request",
                capability=CapabilityType.CODING,
                priority=10,
                condition=lambda s, i, sess: (
                    s.has_signal(RoutingSignal.ACTION_REQUIRED)
                    and i.intent_type in {IntentType.ACTION, IntentType.CREATION}
                    and any(kw in s.evidence.get(next(iter(s.tool_signals), None), [''])[0].lower() if s.tool_signals else False
                            for kw in ['code', 'function', 'script', 'implement', 'program', 'write'])
                ),
            )
        )

        self.add_rule(
            RoutingRule(
                name="coding_explicit",
                capability=CapabilityType.CODING,
                priority=9,
                condition=lambda s, i, sess: any(
                    kw in str(s.evidence).lower()
                    for kw in ['code', 'function', 'implement', 'program', 'algorithm']
                ) if s.signals else False,
            )
        )

        # Vision rules
        self.add_rule(
            RoutingRule(
                name="vision_image_request",
                capability=CapabilityType.VISION,
                priority=10,
                condition=lambda s, i, sess: i.intent_type in {IntentType.QUERY, IntentType.ANALYSIS}
                and any(kw in str(s.evidence).lower() for kw in ['image', 'picture', 'photo', 'screenshot', 'see', 'look at', 'visual']),
            )
        )

        # Files rules
        self.add_rule(
            RoutingRule(
                name="files_operation",
                capability=CapabilityType.FILES,
                priority=10,
                condition=lambda s, i, sess: s.has_signal(RoutingSignal.FILE_OPERATION),
            )
        )

        self.add_rule(
            RoutingRule(
                name="files_explicit",
                capability=CapabilityType.FILES,
                priority=8,
                condition=lambda s, i, sess: any(
                    kw in str(s.evidence).lower()
                    for kw in ['file', 'document', 'folder', 'directory', 'read', 'write', 'save']
                ) if s.signals else False,
            )
        )

        # Computer rules
        self.add_rule(
            RoutingRule(
                name="computer_explicit",
                capability=CapabilityType.COMPUTER,
                priority=10,
                condition=lambda s, i, sess: any(
                    kw in str(s.evidence).lower()
                    for kw in [
                        'computer', 'screen', 'mouse', 'keyboard', 'browser',
                        'terminal', 'click', 'open', 'navigate', 'launch',
                        'go to', 'run', 'execute', 'start', 'type', 'press',
                        'chrome', 'firefox', 'edge', 'explorer', 'notepad',
                        'calculator', 'paint', 'cmd', 'powershell', 'desktop',
                        'window', 'tab', 'download', 'install', 'screenshot',
                    ]
                ) if s.signals else False,
            )
        )


# =============================================================================
# TOOL MATCHER
# =============================================================================


@dataclass
class ToolDefinition:
    """
    Definition of an available tool.

    Attributes:
        name: Tool identifier
        description: What the tool does
        keywords: Keywords that suggest this tool
        intent_types: Intent types this tool handles
        required_entities: Entities that must be present
    """

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    intent_types: list[IntentType] = field(default_factory=list)
    required_entities: list[str] = field(default_factory=list)


class ToolMatcher:
    """
    Matches requests to appropriate tools.

    Uses keyword matching, intent analysis, and entity
    extraction to select relevant tools.
    """

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        """
        Initialize tool matcher.

        Args:
            tools: Available tool definitions
        """
        self._tools = {t.name: t for t in (tools or [])}

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    @property
    def available_tools(self) -> list[str]:
        """List of available tool names."""
        return list(self._tools.keys())

    def match(
        self,
        content: str,
        intent: IntentAnalysis,
    ) -> list[str]:
        """
        Match content and intent to appropriate tools.

        Args:
            content: Request content
            intent: Intent analysis result

        Returns:
            List of matched tool names
        """
        matched: list[tuple[str, float]] = []
        content_lower = content.lower()

        for name, tool in self._tools.items():
            score = self._calculate_match_score(content_lower, intent, tool)
            if score > 0:
                matched.append((name, score))

        # Sort by score and return names
        matched.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in matched]

    def _calculate_match_score(
        self,
        content: str,
        intent: IntentAnalysis,
        tool: ToolDefinition,
    ) -> float:
        """Calculate match score for a tool."""
        score = 0.0

        # Keyword matching
        for keyword in tool.keywords:
            if keyword.lower() in content:
                score += 0.3

        # Intent type matching
        if intent.intent_type in tool.intent_types:
            score += 0.4

        # Entity matching
        for entity in tool.required_entities:
            if intent.has_entity(entity):
                score += 0.3

        # Explicit tool mention
        if tool.name.lower() in content:
            score += 0.5

        return min(score, 1.0)


# =============================================================================
# ROUTER CONFIGURATION
# =============================================================================


@dataclass
class RouterConfig:
    """
    Configuration for CapabilityRouter.

    Attributes:
        available_capabilities: Capabilities that can be routed to
        default_to_memory: Always include memory for context
        complexity_threshold_for_rag: Min complexity to use RAG
        enable_tool_matching: Whether to match specific tools
        max_tools_per_request: Maximum tools to select
    """

    available_capabilities: set[CapabilityType] = field(
        default_factory=lambda: {
            CapabilityType.MEMORY,
            CapabilityType.RAG,
            CapabilityType.WEB_SEARCH,
            CapabilityType.TOOLS,
            CapabilityType.CHAT,
            CapabilityType.CODING,
            CapabilityType.VISION,
            CapabilityType.FILES,
            CapabilityType.COMPUTER,
        }
    )
    default_to_memory: bool = False
    complexity_threshold_for_rag: int = 3
    enable_tool_matching: bool = True
    max_tools_per_request: int = 5


# =============================================================================
# MAIN ROUTER
# =============================================================================


class CapabilityRouter:
    """
    Determines which capabilities should be invoked for a request.

    The router analyzes requests using signal detection, applies
    routing rules, and produces capability decisions.

    This is the main entry point for routing decisions.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        tools: list[ToolDefinition] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the capability router.

        Args:
            config: Router configuration
            tools: Available tool definitions
            logger: Optional logger instance
        """
        self._config = config or RouterConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Initialize components
        self._detector = CompositeDetector([
            PatternBasedDetector(),
            IntentBasedDetector(),
        ])
        self._rule_engine = RoutingRuleEngine()
        self._tool_matcher = ToolMatcher(tools)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool for matching."""
        self._tool_matcher.register_tool(tool)

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self._rule_engine.add_rule(rule)

    @property
    def available_capabilities(self) -> set[CapabilityType]:
        """Get available capabilities."""
        return self._config.available_capabilities

    @property
    def available_tools(self) -> list[str]:
        """Get available tool names."""
        return self._tool_matcher.available_tools

    # -------------------------------------------------------------------------
    # Main Routing
    # -------------------------------------------------------------------------

    def decide(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> CapabilityDecision:
        """
        Make routing decision for a request.

        Args:
            content: Request content
            intent: Intent analysis result
            session: Current session state

        Returns:
            CapabilityDecision indicating which capabilities to use
        """
        # Detect signals
        signals = self._detector.detect(content, intent, session)

        # Evaluate rules
        rule_decisions = self._rule_engine.evaluate(signals, intent, session)

        # Filter by available capabilities
        filtered_decisions = {
            cap: active
            for cap, active in rule_decisions.items()
            if cap in self._config.available_capabilities
        }

        # Match tools if needed
        tools_to_use: list[str] = []
        if (
            filtered_decisions.get(CapabilityType.TOOLS, False)
            and self._config.enable_tool_matching
        ):
            tools_to_use = self._tool_matcher.match(content, intent)
            tools_to_use = tools_to_use[: self._config.max_tools_per_request]

            # If no tools matched, disable tools capability
            if not tools_to_use:
                filtered_decisions[CapabilityType.TOOLS] = False

        # Apply configuration overrides
        if self._config.default_to_memory:
            if CapabilityType.MEMORY in self._config.available_capabilities:
                filtered_decisions[CapabilityType.MEMORY] = True

        # Build reasoning
        reasoning = self._build_reasoning(signals, filtered_decisions, tools_to_use)

        decision = CapabilityDecision(
            use_memory=filtered_decisions.get(CapabilityType.MEMORY, False),
            use_rag=filtered_decisions.get(CapabilityType.RAG, False),
            use_web_search=filtered_decisions.get(CapabilityType.WEB_SEARCH, False),
            use_chat=filtered_decisions.get(CapabilityType.CHAT, True),
            use_coding=filtered_decisions.get(CapabilityType.CODING, False),
            use_vision=filtered_decisions.get(CapabilityType.VISION, False),
            use_files=filtered_decisions.get(CapabilityType.FILES, False),
            use_computer=filtered_decisions.get(CapabilityType.COMPUTER, False),
            tools_to_use=tools_to_use,
            reasoning=reasoning,
        )

        self._logger.debug(
            "Routing decision made",
            extra={
                "capabilities": [c.value for c in decision.active_capabilities],
                "tools": tools_to_use,
                "signals": [s.name for s in signals.signals],
            },
        )

        return decision

    def decide_from_intent(
        self,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> CapabilityDecision:
        """
        Make routing decision using only intent analysis.

        Simplified routing when original content is not available.

        Args:
            intent: Intent analysis result
            session: Current session state

        Returns:
            CapabilityDecision
        """
        # Create minimal content from intent
        content = " ".join(str(v) for v in intent.entities.values())
        return self.decide(content, intent, session)

    # -------------------------------------------------------------------------
    # Individual Capability Checks
    # -------------------------------------------------------------------------

    def should_use_memory(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """
        Check if memory capability should be used.

        Args:
            content: Request content
            intent: Intent analysis
            session: Session state

        Returns:
            Tuple of (should_use, reason)
        """
        if CapabilityType.MEMORY not in self._config.available_capabilities:
            return False, "Memory capability not available"

        signals = self._detector.detect(content, intent, session)

        if signals.has_signal(RoutingSignal.REFERENCES_PAST):
            return True, "References past conversation"

        if signals.has_signal(RoutingSignal.REQUESTS_REMEMBER):
            return True, "Explicit remember request"

        if signals.has_signal(RoutingSignal.USES_PRONOUNS) and session.has_history:
            return True, "Ambiguous pronouns with history"

        if session.memory_refs:
            return True, "Session has memory references"

        if intent.requires_context:
            return True, "Intent requires context"

        return False, "No memory signals detected"

    def should_use_rag(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """
        Check if RAG capability should be used.

        Args:
            content: Request content
            intent: Intent analysis
            session: Session state

        Returns:
            Tuple of (should_use, reason)
        """
        if CapabilityType.RAG not in self._config.available_capabilities:
            return False, "RAG capability not available"

        signals = self._detector.detect(content, intent, session)

        if signals.has_signal(RoutingSignal.DOCUMENTATION_QUERY):
            return True, "Documentation-style query"

        if signals.has_signal(RoutingSignal.DOMAIN_SPECIFIC):
            return True, "Domain-specific terminology"

        if (
            signals.has_signal(RoutingSignal.REQUIRES_FACTS)
            and intent.intent_type == IntentType.QUERY
            and not intent.requires_current_info
        ):
            return True, "Factual query not requiring current info"

        if intent.complexity_score >= self._config.complexity_threshold_for_rag:
            if intent.intent_type in {IntentType.QUERY, IntentType.ANALYSIS}:
                return True, f"Complex {intent.intent_type.value} task"

        return False, "No RAG signals detected"

    def should_use_web_search(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """
        Check if web search capability should be used.

        Args:
            content: Request content
            intent: Intent analysis
            session: Session state

        Returns:
            Tuple of (should_use, reason)
        """
        if CapabilityType.WEB_SEARCH not in self._config.available_capabilities:
            return False, "Web search capability not available"

        signals = self._detector.detect(content, intent, session)

        if signals.has_signal(RoutingSignal.EXPLICIT_SEARCH):
            return True, "Explicit search request"

        if signals.has_signal(RoutingSignal.CURRENT_EVENTS):
            return True, "Current events query"

        if signals.has_signal(RoutingSignal.REAL_TIME_DATA):
            return True, "Real-time data request"

        if intent.requires_current_info:
            return True, "Intent requires current information"

        return False, "No web search signals detected"

    def should_use_tools(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, list[str], str]:
        """
        Check if tools capability should be used.

        Args:
            content: Request content
            intent: Intent analysis
            session: Session state

        Returns:
            Tuple of (should_use, tool_names, reason)
        """
        if CapabilityType.TOOLS not in self._config.available_capabilities:
            return False, [], "Tools capability not available"

        signals = self._detector.detect(content, intent, session)

        # Check for tool signals
        should_use = False
        reason = ""

        if intent.intent_type == IntentType.ACTION:
            should_use = True
            reason = "Action intent detected"
        elif signals.has_signal(RoutingSignal.COMPUTATION_NEEDED):
            should_use = True
            reason = "Computation required"
        elif signals.has_signal(RoutingSignal.FILE_OPERATION):
            should_use = True
            reason = "File operation requested"
        elif signals.has_signal(RoutingSignal.EXTERNAL_SERVICE):
            should_use = True
            reason = "External service interaction"

        if not should_use:
            return False, [], "No tool signals detected"

        # Match specific tools
        tools = self._tool_matcher.match(content, intent)
        tools = tools[: self._config.max_tools_per_request]

        if not tools:
            return False, [], "No matching tools found"

        return True, tools, reason

    def should_use_coding(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """Check if coding capability should be used."""
        if CapabilityType.CODING not in self._config.available_capabilities:
            return False, "Coding capability not available"

        signals = self._detector.detect(content, intent, session)

        code_keywords = ['code', 'function', 'script', 'implement', 'program', 'write a', 'develop', 'build']
        content_lower = content.lower()

        if any(kw in content_lower for kw in code_keywords):
            if intent.intent_type in {IntentType.ACTION, IntentType.CREATION}:
                return True, "Code generation requested"

        if intent.intent_type == IntentType.CREATION:
            return True, "Creation intent may require coding"

        return False, "No coding signals detected"

    def should_use_vision(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """Check if vision capability should be used."""
        if CapabilityType.VISION not in self._config.available_capabilities:
            return False, "Vision capability not available"

        vision_keywords = ['image', 'picture', 'photo', 'screenshot', 'see', 'look at', 'visual', 'diagram', 'chart']
        content_lower = content.lower()

        if any(kw in content_lower for kw in vision_keywords):
            return True, "Visual content referenced"

        return False, "No vision signals detected"

    def should_use_files(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """Check if files capability should be used."""
        if CapabilityType.FILES not in self._config.available_capabilities:
            return False, "Files capability not available"

        signals = self._detector.detect(content, intent, session)

        if signals.has_signal(RoutingSignal.FILE_OPERATION):
            return True, "File operation requested"

        file_keywords = ['file', 'document', 'folder', 'directory', 'read file', 'write file', 'save', 'upload', 'download']
        content_lower = content.lower()

        if any(kw in content_lower for kw in file_keywords):
            return True, "File operation referenced"

        return False, "No files signals detected"

    def should_use_computer(
        self,
        content: str,
        intent: IntentAnalysis,
        session: SessionState,
    ) -> tuple[bool, str]:
        """Check if computer capability should be used."""
        if CapabilityType.COMPUTER not in self._config.available_capabilities:
            return False, "Computer capability not available"

        computer_keywords = ['computer', 'screen', 'mouse', 'keyboard', 'browser', 'terminal', 'click', 'type', 'navigate']
        content_lower = content.lower()

        if any(kw in content_lower for kw in computer_keywords):
            if intent.intent_type == IntentType.ACTION:
                return True, "Computer interaction requested"

        return False, "No computer interaction signals detected"

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_reasoning(
        self,
        signals: SignalDetectionResult,
        decisions: dict[CapabilityType, bool],
        tools: list[str],
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts: list[str] = []

        if decisions.get(CapabilityType.MEMORY):
            evidence = signals.evidence.get(
                next(iter(signals.memory_signals), None), []  # type: ignore
            )
            parts.append(f"Memory: {evidence[0] if evidence else 'context needed'}")

        if decisions.get(CapabilityType.RAG):
            evidence = signals.evidence.get(
                next(iter(signals.rag_signals), None), []  # type: ignore
            )
            parts.append(f"RAG: {evidence[0] if evidence else 'knowledge needed'}")

        if decisions.get(CapabilityType.WEB_SEARCH):
            evidence = signals.evidence.get(
                next(iter(signals.web_signals), None), []  # type: ignore
            )
            parts.append(f"Web: {evidence[0] if evidence else 'current info needed'}")

        if decisions.get(CapabilityType.CODING):
            parts.append("Coding: code generation requested")

        if decisions.get(CapabilityType.VISION):
            parts.append("Vision: image understanding needed")

        if decisions.get(CapabilityType.FILES):
            parts.append("Files: file operation needed")

        if decisions.get(CapabilityType.COMPUTER):
            parts.append("Computer: computer interaction needed")

        if tools:
            parts.append(f"Tools: {', '.join(tools)}")

        return "; ".join(parts) if parts else "Direct response (no capabilities needed)"

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics and configuration."""
        return {
            "available_capabilities": [c.value for c in self._config.available_capabilities],
            "available_tools": self.available_tools,
            "default_to_memory": self._config.default_to_memory,
            "complexity_threshold_for_rag": self._config.complexity_threshold_for_rag,
            "rule_count": len(self._rule_engine._rules),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_router(
    capabilities: list[CapabilityType] | None = None,
    tools: list[ToolDefinition] | None = None,
    config: RouterConfig | None = None,
    logger: logging.Logger | None = None,
) -> CapabilityRouter:
    """
    Factory function to create a configured CapabilityRouter.

    Args:
        capabilities: Available capabilities (default: all)
        tools: Tool definitions to register
        config: Router configuration
        logger: Optional logger

    Returns:
        Configured CapabilityRouter instance
    """
    if config is None:
        config = RouterConfig()

    if capabilities is not None:
        config.available_capabilities = set(capabilities)

    return CapabilityRouter(
        config=config,
        tools=tools,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Signals
    "RoutingSignal",
    "SignalDetectionResult",
    # Detectors
    "SignalDetector",
    "PatternBasedDetector",
    "IntentBasedDetector",
    "CompositeDetector",
    # Rules
    "RoutingRule",
    "RoutingRuleEngine",
    # Tools
    "ToolDefinition",
    "ToolMatcher",
    # Router
    "RouterConfig",
    "CapabilityRouter",
    # Factory
    "create_router",
]
