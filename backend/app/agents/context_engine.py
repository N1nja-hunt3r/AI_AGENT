"""
Context Engine - Context assembly system for the AI Agent Platform.

This module builds optimized context windows for LLM calls by:
- Aggregating multiple context sources (history, memory, RAG, tools)
- Managing token budgets and model limits
- Ranking context by relevance and priority
- Trimming and compressing when over budget
- Supporting multimodal content (text, images, structured data)
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
)

from app.agents.models import SessionState

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================


class ContextSourceType(Enum):
    """Types of context sources."""

    SYSTEM = auto()  # System prompt
    HISTORY = auto()  # Conversation history
    MEMORY = auto()  # Long-term memory
    RAG = auto()  # Retrieved documents
    TOOL_OUTPUT = auto()  # Tool execution results
    USER_INPUT = auto()  # Current user input
    AGENT_SCRATCHPAD = auto()  # Agent reasoning
    METADATA = auto()  # Request metadata
    MULTIMODAL = auto()  # Images, audio, etc.


class ContentType(Enum):
    """Types of content in context."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # JSON, tables
    CODE = "code"
    EMBEDDING = "embedding"


class MessageRole(Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Legacy OpenAI


class TrimmingStrategy(Enum):
    """Strategies for trimming context."""

    TRUNCATE_START = auto()  # Remove from beginning
    TRUNCATE_END = auto()  # Remove from end
    TRUNCATE_MIDDLE = auto()  # Keep start and end
    SUMMARIZE = auto()  # Summarize content
    REMOVE_LOWEST_PRIORITY = auto()  # Remove by priority
    REMOVE_OLDEST = auto()  # Remove oldest items
    COMPRESS = auto()  # Compress/deduplicate


# =============================================================================
# TOKEN COUNTING
# =============================================================================


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in message list."""
        ...


class SimpleTokenCounter:
    """
    Simple token counter using character/word estimation.

    For production, use tiktoken or model-specific tokenizer.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """
        Initialize counter.

        Args:
            chars_per_token: Average characters per token
        """
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0
        return max(1, int(len(text) / self._chars_per_token))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count from messages."""
        total = 0
        for msg in messages:
            # Role overhead
            total += 4

            # Content
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Multimodal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += self.count(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            total += 85  # Base image token cost
            
            # Name/tool overhead
            if "name" in msg:
                total += self.count(msg["name"]) + 1
            if "tool_call_id" in msg:
                total += 10

        # Message formatting overhead
        total += 3

        return total


class TiktokenCounter:
    """
    Token counter using tiktoken library.

    Requires: pip install tiktoken
    """

    def __init__(self, model: str = "gpt-4") -> None:
        """
        Initialize with model encoding.

        Args:
            model: Model name for encoding selection
        """
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
        except ImportError:
            raise ImportError("tiktoken required: pip install tiktoken")
        except KeyError:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages using tiktoken."""
        total = 0

        for msg in messages:
            total += 4  # Message overhead

            for key, value in msg.items():
                if isinstance(value, str):
                    total += self.count(value)
                elif key == "content" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total += self.count(item.get("text", ""))

            if "name" in msg:
                total += -1  # Name adjustment

        total += 2  # Reply priming

        return total


# =============================================================================
# CONTEXT ITEMS
# =============================================================================


@dataclass
class ContextItem:
    """
    Single item of context.

    Represents a piece of context from any source with
    metadata for ranking and management.

    Attributes:
        content: The actual content
        source_type: Type of context source
        content_type: Type of content
        token_count: Number of tokens
        relevance_score: Relevance to current request (0-1)
        priority: Priority level (higher = more important)
        timestamp: When content was created/retrieved
        metadata: Additional metadata
        id: Unique identifier
    """

    content: str | dict[str, Any] | list[Any]
    source_type: ContextSourceType
    content_type: ContentType = ContentType.TEXT
    token_count: int = 0
    relevance_score: float = 1.0
    priority: int = 50
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            content_str = str(self.content)[:100]
            self.id = hashlib.md5(
                f"{self.source_type.name}:{content_str}".encode()
            ).hexdigest()[:12]

    @property
    def text_content(self) -> str:
        """Get content as text."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return str(self.content)
        elif isinstance(self.content, list):
            return "\n".join(str(item) for item in self.content)
        return str(self.content)

    @property
    def effective_priority(self) -> float:
        """Calculate effective priority (priority * relevance)."""
        return self.priority * self.relevance_score

    def truncate(self, max_tokens: int, counter: TokenCounter) -> ContextItem:
        """Create truncated copy of this item."""
        if self.token_count <= max_tokens:
            return self

        text = self.text_content
        # Binary search for truncation point
        low, high = 0, len(text)

        while low < high:
            mid = (low + high + 1) // 2
            if counter.count(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        truncated_text = text[:low] + "..."

        return ContextItem(
            content=truncated_text,
            source_type=self.source_type,
            content_type=self.content_type,
            token_count=counter.count(truncated_text),
            relevance_score=self.relevance_score,
            priority=self.priority,
            timestamp=self.timestamp,
            metadata={**self.metadata, "truncated": True},
            id=self.id,
        )


@dataclass
class MultimodalContent:
    """
    Multimodal content container.

    Supports text, images, and other media types
    for multimodal LLM calls.
    """

    type: ContentType
    data: str | bytes | dict[str, Any]
    mime_type: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI multimodal format."""
        if self.type == ContentType.TEXT:
            return {"type": "text", "text": str(self.data)}

        elif self.type == ContentType.IMAGE:
            if self.url:
                return {
                    "type": "image_url",
                    "image_url": {"url": self.url},
                }
            elif isinstance(self.data, bytes):
                import base64
                b64 = base64.b64encode(self.data).decode()
                mime = self.mime_type or "image/png"
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }

        return {"type": "text", "text": str(self.data)}


# =============================================================================
# CONTEXT MESSAGE
# =============================================================================


@dataclass
class ContextMessage:
    """
    Message in the context window.

    Represents a single message with role, content,
    and associated metadata.
    """

    role: MessageRole
    content: str | list[MultimodalContent]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        """Check if message contains multimodal content."""
        return isinstance(self.content, list)

    @property
    def text_content(self) -> str:
        """Get text content from message."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            item.data if item.type == ContentType.TEXT else f"[{item.type.value}]"
            for item in self.content
            if isinstance(item.data, str)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        msg: dict[str, Any] = {"role": self.role.value}

        if isinstance(self.content, str):
            msg["content"] = self.content
        else:
            msg["content"] = [item.to_openai_format() for item in self.content]

        if self.name:
            msg["name"] = self.name

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        return msg


# =============================================================================
# CONTEXT WINDOW
# =============================================================================


@dataclass
class ContextWindow:
    """
    Complete context window for LLM call.

    Contains all messages and metadata for a single
    LLM invocation.

    Attributes:
        messages: Ordered list of messages
        total_tokens: Total token count
        max_tokens: Maximum allowed tokens
        metadata: Window metadata
        created_at: Creation timestamp
    """

    messages: list[ContextMessage] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def token_usage_ratio(self) -> float:
        """Get ratio of used tokens to max."""
        if self.max_tokens == 0:
            return 0.0
        return self.total_tokens / self.max_tokens

    @property
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert to list of message dicts for API."""
        return [msg.to_dict() for msg in self.messages]

    def get_system_message(self) -> ContextMessage | None:
        """Get system message if present."""
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                return msg
        return None

    def get_last_user_message(self) -> ContextMessage | None:
        """Get last user message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None


# =============================================================================
# BUDGET MANAGER
# =============================================================================


@dataclass
class TokenBudget:
    """
    Token budget allocation.

    Attributes:
        total: Total available tokens
        system: Budget for system prompt
        history: Budget for conversation history
        memory: Budget for memory context
        rag: Budget for RAG chunks
        tools: Budget for tool outputs
        user_input: Budget for current input
        response: Reserved for response
    """

    total: int
    system: int = 0
    history: int = 0
    memory: int = 0
    rag: int = 0
    tools: int = 0
    user_input: int = 0
    response: int = 0

    @classmethod
    def create(
        cls,
        total: int,
        response_reserve: int = 1000,
        system_ratio: float = 0.1,
        history_ratio: float = 0.3,
        memory_ratio: float = 0.1,
        rag_ratio: float = 0.25,
        tools_ratio: float = 0.15,
        user_input_ratio: float = 0.1,
    ) -> TokenBudget:
        """
        Create budget with ratio-based allocation.

        Args:
            total: Total token limit
            response_reserve: Tokens reserved for response
            *_ratio: Ratio of remaining budget for each source
        """
        available = total - response_reserve

        return cls(
            total=total,
            system=int(available * system_ratio),
            history=int(available * history_ratio),
            memory=int(available * memory_ratio),
            rag=int(available * rag_ratio),
            tools=int(available * tools_ratio),
            user_input=int(available * user_input_ratio),
            response=response_reserve,
        )

    @property
    def allocated(self) -> int:
        """Total allocated tokens."""
        return (
            self.system
            + self.history
            + self.memory
            + self.rag
            + self.tools
            + self.user_input
            + self.response
        )

    @property
    def unallocated(self) -> int:
        """Unallocated tokens."""
        return self.total - self.allocated

    def get_budget(self, source_type: ContextSourceType) -> int:
        """Get budget for a source type."""
        mapping = {
            ContextSourceType.SYSTEM: self.system,
            ContextSourceType.HISTORY: self.history,
            ContextSourceType.MEMORY: self.memory,
            ContextSourceType.RAG: self.rag,
            ContextSourceType.TOOL_OUTPUT: self.tools,
            ContextSourceType.USER_INPUT: self.user_input,
        }
        return mapping.get(source_type, 0)


class BudgetManager:
    """
    Manages token budget allocation and tracking.

    Tracks usage across context sources and handles
    reallocation when sources are under/over budget.
    """

    def __init__(
        self,
        budget: TokenBudget,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize budget manager.

        Args:
            budget: Token budget allocation
            logger: Optional logger
        """
        self._budget = budget
        self._logger = logger or logging.getLogger(__name__)

        # Track actual usage
        self._usage: dict[ContextSourceType, int] = {
            source: 0 for source in ContextSourceType
        }

    @property
    def budget(self) -> TokenBudget:
        """Get current budget."""
        return self._budget

    @property
    def total_used(self) -> int:
        """Get total tokens used."""
        return sum(self._usage.values())

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self._budget.total - self._budget.response - self.total_used

    def get_remaining_for_source(self, source_type: ContextSourceType) -> int:
        """Get remaining budget for a source."""
        allocated = self._budget.get_budget(source_type)
        used = self._usage.get(source_type, 0)
        source_remaining = allocated - used

        # Can also use unallocated pool
        return source_remaining + max(0, self.remaining - source_remaining)

    def allocate(self, source_type: ContextSourceType, tokens: int) -> bool:
        """
        Allocate tokens for a source.

        Args:
            source_type: Source requesting allocation
            tokens: Number of tokens

        Returns:
            True if allocation successful
        """
        if tokens <= 0:
            return True

        remaining = self.get_remaining_for_source(source_type)

        if tokens > remaining:
            self._logger.warning(
                f"Budget exceeded for {source_type.name}: "
                f"requested {tokens}, available {remaining}"
            )
            return False

        self._usage[source_type] = self._usage.get(source_type, 0) + tokens
        return True

    def deallocate(self, source_type: ContextSourceType, tokens: int) -> None:
        """Return tokens to budget."""
        current = self._usage.get(source_type, 0)
        self._usage[source_type] = max(0, current - tokens)

    def get_usage_report(self) -> dict[str, Any]:
        """Get detailed usage report."""
        return {
            "total_budget": self._budget.total,
            "response_reserve": self._budget.response,
            "total_used": self.total_used,
            "remaining": self.remaining,
            "usage_by_source": {
                source.name: {
                    "allocated": self._budget.get_budget(source),
                    "used": self._usage.get(source, 0),
                }
                for source in ContextSourceType
            },
        }


# =============================================================================
# CONTEXT RANKER
# =============================================================================


class ContextRanker:
    """
    Ranks context items by relevance and priority.

    Uses multiple signals to determine the importance
    of each context item.
    """

    def __init__(
        self,
        recency_weight: float = 0.2,
        relevance_weight: float = 0.5,
        priority_weight: float = 0.3,
    ) -> None:
        """
        Initialize ranker.

        Args:
            recency_weight: Weight for recency score
            relevance_weight: Weight for relevance score
            priority_weight: Weight for priority score
        """
        self._recency_weight = recency_weight
        self._relevance_weight = relevance_weight
        self._priority_weight = priority_weight

    def rank(
        self,
        items: list[ContextItem],
        reference_time: datetime | None = None,
    ) -> list[ContextItem]:
        """
        Rank context items by combined score.

        Args:
            items: Items to rank
            reference_time: Reference time for recency (default: now)

        Returns:
            Items sorted by score (highest first)
        """
        if not items:
            return []

        reference_time = reference_time or datetime.now(timezone.utc)

        scored_items: list[tuple[float, ContextItem]] = []

        for item in items:
            score = self._calculate_score(item, reference_time)
            scored_items.append((score, item))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored_items]

    def _calculate_score(
        self,
        item: ContextItem,
        reference_time: datetime,
    ) -> float:
        """Calculate combined score for an item."""
        # Recency score (exponential decay)
        age_seconds = (reference_time - item.timestamp).total_seconds()
        recency_score = 1.0 / (1.0 + age_seconds / 3600)  # 1-hour half-life

        # Relevance score (already 0-1)
        relevance_score = item.relevance_score

        # Priority score (normalize to 0-1)
        priority_score = item.priority / 100.0

        # Combined score
        return (
            self._recency_weight * recency_score
            + self._relevance_weight * relevance_score
            + self._priority_weight * priority_score
        )

    def filter_by_threshold(
        self,
        items: list[ContextItem],
        min_relevance: float = 0.0,
        min_priority: int = 0,
    ) -> list[ContextItem]:
        """Filter items by minimum thresholds."""
        return [
            item
            for item in items
            if item.relevance_score >= min_relevance
            and item.priority >= min_priority
        ]


# =============================================================================
# CONTEXT TRIMMER
# =============================================================================


class ContextTrimmer:
    """
    Trims context to fit within token budget.

    Supports multiple trimming strategies.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize trimmer.

        Args:
            token_counter: Token counter implementation
            logger: Optional logger
        """
        self._counter = token_counter
        self._logger = logger or logging.getLogger(__name__)

    def trim(
        self,
        items: list[ContextItem],
        max_tokens: int,
        strategy: TrimmingStrategy = TrimmingStrategy.REMOVE_LOWEST_PRIORITY,
    ) -> list[ContextItem]:
        """
        Trim items to fit within token budget.

        Args:
            items: Items to trim
            max_tokens: Maximum total tokens
            strategy: Trimming strategy

        Returns:
            Trimmed list of items
        """
        if not items:
            return []

        total_tokens = sum(item.token_count for item in items)

        if total_tokens <= max_tokens:
            return items

        self._logger.debug(
            f"Trimming context: {total_tokens} -> {max_tokens} tokens"
        )

        if strategy == TrimmingStrategy.REMOVE_LOWEST_PRIORITY:
            return self._trim_by_priority(items, max_tokens)

        elif strategy == TrimmingStrategy.REMOVE_OLDEST:
            return self._trim_by_age(items, max_tokens)

        elif strategy == TrimmingStrategy.TRUNCATE_START:
            return self._truncate_items(items, max_tokens, from_start=True)

        elif strategy == TrimmingStrategy.TRUNCATE_END:
            return self._truncate_items(items, max_tokens, from_start=False)

        elif strategy == TrimmingStrategy.TRUNCATE_MIDDLE:
            return self._truncate_middle(items, max_tokens)

        else:
            # Default: remove lowest priority
            return self._trim_by_priority(items, max_tokens)

    def _trim_by_priority(
        self,
        items: list[ContextItem],
        max_tokens: int,
    ) -> list[ContextItem]:
        """Remove lowest priority items first."""
        # Sort by effective priority (ascending, so lowest first)
        sorted_items = sorted(items, key=lambda x: x.effective_priority)

        result = list(items)
        total = sum(item.token_count for item in result)

        for item in sorted_items:
            if total <= max_tokens:
                break

            if item in result:
                result.remove(item)
                total -= item.token_count

        return result

    def _trim_by_age(
        self,
        items: list[ContextItem],
        max_tokens: int,
    ) -> list[ContextItem]:
        """Remove oldest items first."""
        # Sort by timestamp (ascending, so oldest first)
        sorted_items = sorted(items, key=lambda x: x.timestamp)

        result = list(items)
        total = sum(item.token_count for item in result)

        for item in sorted_items:
            if total <= max_tokens:
                break

            if item in result:
                result.remove(item)
                total -= item.token_count

        return result

    def _truncate_items(
        self,
        items: list[ContextItem],
        max_tokens: int,
        from_start: bool,
    ) -> list[ContextItem]:
        """Truncate individual items."""
        result: list[ContextItem] = []
        remaining = max_tokens

        # Process in order (or reverse for from_start)
        process_order = items if not from_start else list(reversed(items))

        for item in process_order:
            if remaining <= 0:
                break

            if item.token_count <= remaining:
                result.append(item)
                remaining -= item.token_count
            else:
                # Truncate this item
                truncated = item.truncate(remaining, self._counter)
                result.append(truncated)
                remaining -= truncated.token_count

        if from_start:
            result.reverse()

        return result

    def _truncate_middle(
        self,
        items: list[ContextItem],
        max_tokens: int,
    ) -> list[ContextItem]:
        """Keep start and end, remove middle."""
        if len(items) <= 2:
            return self._truncate_items(items, max_tokens, from_start=False)

        # Allocate half to start, half to end
        half_budget = max_tokens // 2

        start_items = self._truncate_items(
            items[: len(items) // 2],
            half_budget,
            from_start=False,
        )

        end_items = self._truncate_items(
            items[len(items) // 2 :],
            max_tokens - sum(i.token_count for i in start_items),
            from_start=True,
        )

        return start_items + end_items


# =============================================================================
# CONTEXT PROVIDERS
# =============================================================================


class ContextProvider(ABC):
    """Abstract base for context providers."""

    @property
    @abstractmethod
    def source_type(self) -> ContextSourceType:
        """Get the source type this provider handles."""
        ...

    @abstractmethod
    async def get_context(
        self,
        session: SessionState,
        query: str,
        max_tokens: int,
    ) -> list[ContextItem]:
        """
        Get context items from this source.

        Args:
            session: Current session state
            query: Current query/request
            max_tokens: Maximum tokens to return

        Returns:
            List of context items
        """
        ...


class HistoryProvider(ContextProvider):
    """Provides conversation history context."""

    def __init__(
        self,
        max_turns: int = 10,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._max_turns = max_turns
        self._counter = token_counter or SimpleTokenCounter()

    @property
    def source_type(self) -> ContextSourceType:
        return ContextSourceType.HISTORY

    async def get_context(
        self,
        session: SessionState,
        query: str,
        max_tokens: int,
    ) -> list[ContextItem]:
        """Get conversation history as context."""
        items: list[ContextItem] = []

        history = session.get_recent_history(self._max_turns)

        for i, entry in enumerate(history):
            content = entry.get("content", "")
            response = entry.get("response", "")

            # User message
            user_tokens = self._counter.count(content)
            items.append(
                ContextItem(
                    content=content,
                    source_type=ContextSourceType.HISTORY,
                    content_type=ContentType.TEXT,
                    token_count=user_tokens,
                    priority=50 + i,  # More recent = higher priority
                    timestamp=datetime.fromisoformat(
                        entry.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    metadata={"role": "user", "turn": i},
                )
            )

            # Assistant message
            assistant_tokens = self._counter.count(response)
            items.append(
                ContextItem(
                    content=response,
                    source_type=ContextSourceType.HISTORY,
                    content_type=ContentType.TEXT,
                    token_count=assistant_tokens,
                    priority=50 + i,
                    timestamp=datetime.fromisoformat(
                        entry.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    metadata={"role": "assistant", "turn": i},
                )
            )

        return items


class MemoryProvider(ContextProvider):
    """Provides memory context."""

    def __init__(
        self,
        memory_retriever: Any = None,  # Memory capability
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._retriever = memory_retriever
        self._counter = token_counter or SimpleTokenCounter()

    @property
    def source_type(self) -> ContextSourceType:
        return ContextSourceType.MEMORY

    async def get_context(
        self,
        session: SessionState,
        query: str,
        max_tokens: int,
    ) -> list[ContextItem]:
        """Get memory context."""
        items: list[ContextItem] = []

        if self._retriever is None:
            return items

        # Retrieve relevant memories
        try:
            memories = await self._retriever.retrieve(
                query=query,
                session_id=session.session_id,
                limit=10,
            )

            for memory in memories:
                content = memory.get("content", "")
                tokens = self._counter.count(content)

                items.append(
                    ContextItem(
                        content=content,
                        source_type=ContextSourceType.MEMORY,
                        content_type=ContentType.TEXT,
                        token_count=tokens,
                        relevance_score=memory.get("relevance", 0.5),
                        priority=60,
                        metadata=memory.get("metadata", {}),
                    )
                )

        except Exception as e:
            logging.getLogger(__name__).warning(f"Memory retrieval failed: {e}")

        return items


class RAGProvider(ContextProvider):
    """Provides RAG context from retrieved documents."""

    def __init__(
        self,
        rag_retriever: Any = None,  # RAG capability
        token_counter: TokenCounter | None = None,
        chunk_overlap_tokens: int = 50,
    ) -> None:
        self._retriever = rag_retriever
        self._counter = token_counter or SimpleTokenCounter()
        self._chunk_overlap = chunk_overlap_tokens

    @property
    def source_type(self) -> ContextSourceType:
        return ContextSourceType.RAG

    async def get_context(
        self,
        session: SessionState,
        query: str,
        max_tokens: int,
    ) -> list[ContextItem]:
        """Get RAG context from retrieved documents."""
        items: list[ContextItem] = []

        if self._retriever is None:
            return items

        try:
            chunks = await self._retriever.search(
                query=query,
                top_k=10,
            )

            for chunk in chunks:
                content = chunk.get("content", chunk.get("text", ""))
                tokens = self._counter.count(content)

                items.append(
                    ContextItem(
                        content=content,
                        source_type=ContextSourceType.RAG,
                        content_type=ContentType.TEXT,
                        token_count=tokens,
                        relevance_score=chunk.get("score", chunk.get("relevance", 0.5)),
                        priority=70,
                        metadata={
                            "source": chunk.get("source", ""),
                            "chunk_id": chunk.get("id", ""),
                        },
                    )
                )

        except Exception as e:
            logging.getLogger(__name__).warning(f"RAG retrieval failed: {e}")

        return items


class ToolOutputProvider(ContextProvider):
    """Provides tool execution output context."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._counter = token_counter or SimpleTokenCounter()
        self._outputs: list[dict[str, Any]] = []

    @property
    def source_type(self) -> ContextSourceType:
        return ContextSourceType.TOOL_OUTPUT

    def add_output(
        self,
        tool_name: str,
        output: Any,
        call_id: str | None = None,
    ) -> None:
        """Add a tool output."""
        self._outputs.append({
            "tool_name": tool_name,
            "output": output,
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc),
        })

    def clear_outputs(self) -> None:
        """Clear all tool outputs."""
        self._outputs.clear()

    async def get_context(
        self,
        session: SessionState,
        query: str,
        max_tokens: int,
    ) -> list[ContextItem]:
        """Get tool outputs as context."""
        items: list[ContextItem] = []

        for output in self._outputs:
            content = str(output["output"])
            tokens = self._counter.count(content)

            items.append(
                ContextItem(
                    content=content,
                    source_type=ContextSourceType.TOOL_OUTPUT,
                    content_type=ContentType.STRUCTURED,
                    token_count=tokens,
                    priority=80,  # High priority for tool outputs
                    timestamp=output["timestamp"],
                    metadata={
                        "tool_name": output["tool_name"],
                        "call_id": output["call_id"],
                    },
                )
            )

        return items


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================


@dataclass
class SystemPromptConfig:
    """Configuration for system prompt building."""

    base_prompt: str = ""
    persona: str = ""
    instructions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    output_format: str = ""
    examples: list[str] = field(default_factory=list)
    dynamic_sections: dict[str, str] = field(default_factory=dict)


class SystemPromptBuilder:
    """
    Builds system prompts with dynamic injection.

    Supports templates, personas, instructions, and
    dynamic content injection.
    """

    def __init__(
        self,
        config: SystemPromptConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """
        Initialize builder.

        Args:
            config: System prompt configuration
            token_counter: Token counter
        """
        self._config = config or SystemPromptConfig()
        self._counter = token_counter or SimpleTokenCounter()

    def build(
        self,
        max_tokens: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> ContextItem:
        """
        Build system prompt.

        Args:
            max_tokens: Maximum tokens for prompt
            context: Dynamic context for injection

        Returns:
            ContextItem containing system prompt
        """
        sections: list[str] = []

        # Base prompt
        if self._config.base_prompt:
            sections.append(self._config.base_prompt)

        # Persona
        if self._config.persona:
            sections.append(f"## Persona\n{self._config.persona}")

        # Instructions
        if self._config.instructions:
            instructions = "\n".join(f"- {i}" for i in self._config.instructions)
            sections.append(f"## Instructions\n{instructions}")

        # Constraints
        if self._config.constraints:
            constraints = "\n".join(f"- {c}" for c in self._config.constraints)
            sections.append(f"## Constraints\n{constraints}")

        # Output format
        if self._config.output_format:
            sections.append(f"## Output Format\n{self._config.output_format}")

        # Examples
        if self._config.examples:
            examples = "\n\n".join(self._config.examples)
            sections.append(f"## Examples\n{examples}")

        # Dynamic sections
        for name, content in self._config.dynamic_sections.items():
            sections.append(f"## {name}\n{content}")

        # Inject context
        if context:
            for key, value in context.items():
                for i, section in enumerate(sections):
                    sections[i] = section.replace(f"{{{{{key}}}}}", str(value))

        prompt = "\n\n".join(sections)
        tokens = self._counter.count(prompt)

        # Truncate if needed
        if max_tokens and tokens > max_tokens:
            # Simple truncation
            ratio = max_tokens / tokens
            prompt = prompt[: int(len(prompt) * ratio * 0.95)] + "..."
            tokens = self._counter.count(prompt)

        return ContextItem(
            content=prompt,
            source_type=ContextSourceType.SYSTEM,
            content_type=ContentType.TEXT,
            token_count=tokens,
            priority=100,  # Highest priority
        )


# =============================================================================
# CONTEXT ENGINE
# =============================================================================


@dataclass
class ContextEngineConfig:
    """Configuration for ContextEngine."""

    max_context_tokens: int = 8000
    response_reserve_tokens: int = 1000
    default_trimming_strategy: TrimmingStrategy = TrimmingStrategy.REMOVE_LOWEST_PRIORITY
    enable_ranking: bool = True
    min_relevance_threshold: float = 0.1
    system_prompt_config: SystemPromptConfig | None = None


class ContextEngine:
    """
    Main context assembly engine.

    Orchestrates context gathering, ranking, trimming,
    and window construction for LLM calls.
    """

    def __init__(
        self,
        config: ContextEngineConfig | None = None,
        token_counter: TokenCounter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize context engine.

        Args:
            config: Engine configuration
            token_counter: Token counter implementation
            logger: Optional logger
        """
        self._config = config or ContextEngineConfig()
        self._counter = token_counter or SimpleTokenCounter()
        self._logger = logger or logging.getLogger(__name__)

        # Components
        self._ranker = ContextRanker()
        self._trimmer = ContextTrimmer(self._counter, logger)
        self._system_builder = SystemPromptBuilder(
            self._config.system_prompt_config,
            self._counter,
        )

        # Providers
        self._providers: dict[ContextSourceType, ContextProvider] = {}

        # Default providers
        self._providers[ContextSourceType.HISTORY] = HistoryProvider(
            token_counter=self._counter
        )
        self._providers[ContextSourceType.TOOL_OUTPUT] = ToolOutputProvider(
            token_counter=self._counter
        )

    # -------------------------------------------------------------------------
    # Provider Management
    # -------------------------------------------------------------------------

    def register_provider(self, provider: ContextProvider) -> None:
        """Register a context provider."""
        self._providers[provider.source_type] = provider

    def get_provider(
        self,
        source_type: ContextSourceType,
    ) -> ContextProvider | None:
        """Get provider for source type."""
        return self._providers.get(source_type)

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    async def build_context(
        self,
        session: SessionState,
        user_input: str,
        system_prompt: str | None = None,
        include_sources: set[ContextSourceType] | None = None,
        budget: TokenBudget | None = None,
    ) -> ContextWindow:
        """
        Build complete context window.

        Args:
            session: Current session state
            user_input: Current user input
            system_prompt: Optional system prompt override
            include_sources: Sources to include (None = all)
            budget: Token budget (None = default)

        Returns:
            Assembled ContextWindow
        """
        # Create budget
        if budget is None:
            budget = TokenBudget.create(
                total=self._config.max_context_tokens,
                response_reserve=self._config.response_reserve_tokens,
            )

        budget_manager = BudgetManager(budget, self._logger)

        # Collect context from all sources
        all_items: list[ContextItem] = []

        # System prompt
        if system_prompt:
            system_item = ContextItem(
                content=system_prompt,
                source_type=ContextSourceType.SYSTEM,
                content_type=ContentType.TEXT,
                token_count=self._counter.count(system_prompt),
                priority=100,
            )
        else:
            system_item = self._system_builder.build(
                max_tokens=budget.system,
            )

        all_items.append(system_item)
        budget_manager.allocate(ContextSourceType.SYSTEM, system_item.token_count)

        # User input
        user_tokens = self._counter.count(user_input)
        user_item = ContextItem(
            content=user_input,
            source_type=ContextSourceType.USER_INPUT,
            content_type=ContentType.TEXT,
            token_count=user_tokens,
            priority=95,
        )
        all_items.append(user_item)
        budget_manager.allocate(ContextSourceType.USER_INPUT, user_tokens)

        # Gather from providers
        sources_to_include = include_sources or set(self._providers.keys())

        for source_type, provider in self._providers.items():
            if source_type not in sources_to_include:
                continue

            if source_type in {ContextSourceType.SYSTEM, ContextSourceType.USER_INPUT}:
                continue

            source_budget = budget_manager.get_remaining_for_source(source_type)

            try:
                items = await provider.get_context(
                    session=session,
                    query=user_input,
                    max_tokens=source_budget,
                )

                for item in items:
                    if item.token_count == 0:
                        item.token_count = self._counter.count(item.text_content)

                all_items.extend(items)

            except Exception as e:
                self._logger.warning(
                    f"Failed to get context from {source_type.name}: {e}"
                )

        # Rank items
        if self._config.enable_ranking:
            # Separate system and user (always keep)
            fixed_items = [
                i for i in all_items
                if i.source_type in {ContextSourceType.SYSTEM, ContextSourceType.USER_INPUT}
            ]
            rankable_items = [
                i for i in all_items
                if i.source_type not in {ContextSourceType.SYSTEM, ContextSourceType.USER_INPUT}
            ]

            # Filter by relevance threshold
            rankable_items = self._ranker.filter_by_threshold(
                rankable_items,
                min_relevance=self._config.min_relevance_threshold,
            )

            # Rank
            ranked_items = self._ranker.rank(rankable_items)
            all_items = fixed_items + ranked_items

        # Trim to budget
        available_tokens = budget.total - budget.response
        all_items = self._trim_to_budget(all_items, available_tokens)

        # Build context window
        return self._build_window(all_items, budget)

    def _trim_to_budget(
        self,
        items: list[ContextItem],
        max_tokens: int,
    ) -> list[ContextItem]:
        """Trim items to fit budget."""
        total = sum(item.token_count for item in items)

        if total <= max_tokens:
            return items

        # Separate fixed items (system, user input)
        fixed = [
            i for i in items
            if i.source_type in {ContextSourceType.SYSTEM, ContextSourceType.USER_INPUT}
        ]
        trimmable = [
            i for i in items
            if i.source_type not in {ContextSourceType.SYSTEM, ContextSourceType.USER_INPUT}
        ]

        fixed_tokens = sum(i.token_count for i in fixed)
        remaining_budget = max_tokens - fixed_tokens

        # Trim trimmable items
        trimmed = self._trimmer.trim(
            trimmable,
            remaining_budget,
            self._config.default_trimming_strategy,
        )

        return fixed + trimmed

    def _build_window(
        self,
        items: list[ContextItem],
        budget: TokenBudget,
    ) -> ContextWindow:
        """Build context window from items."""
        messages: list[ContextMessage] = []

        # System message
        system_items = [i for i in items if i.source_type == ContextSourceType.SYSTEM]
        if system_items:
            messages.append(
                ContextMessage(
                    role=MessageRole.SYSTEM,
                    content=system_items[0].text_content,
                    token_count=system_items[0].token_count,
                )
            )

        # History messages
        history_items = [i for i in items if i.source_type == ContextSourceType.HISTORY]
        for item in history_items:
            role_str = item.metadata.get("role", "user")
            role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT

            messages.append(
                ContextMessage(
                    role=role,
                    content=item.text_content,
                    token_count=item.token_count,
                    metadata=item.metadata,
                )
            )

        # Inject RAG and memory into a context message
        context_items = [
            i for i in items
            if i.source_type in {ContextSourceType.RAG, ContextSourceType.MEMORY}
        ]
        if context_items:
            context_text = self._format_context_items(context_items)
            context_tokens = self._counter.count(context_text)

            messages.append(
                ContextMessage(
                    role=MessageRole.SYSTEM,
                    content=f"## Relevant Context\n\n{context_text}",
                    token_count=context_tokens,
                    metadata={"injected_context": True},
                )
            )

        # Tool outputs
        tool_items = [i for i in items if i.source_type == ContextSourceType.TOOL_OUTPUT]
        for item in tool_items:
            messages.append(
                ContextMessage(
                    role=MessageRole.TOOL,
                    content=item.text_content,
                    tool_call_id=item.metadata.get("call_id"),
                    name=item.metadata.get("tool_name"),
                    token_count=item.token_count,
                )
            )

        # User input (always last)
        user_items = [i for i in items if i.source_type == ContextSourceType.USER_INPUT]
        if user_items:
            messages.append(
                ContextMessage(
                    role=MessageRole.USER,
                    content=user_items[0].text_content,
                    token_count=user_items[0].token_count,
                )
            )

        total_tokens = sum(msg.token_count for msg in messages)

        return ContextWindow(
            messages=messages,
            total_tokens=total_tokens,
            max_tokens=budget.total,
            metadata={
                "source_counts": self._count_sources(items),
                "budget": budget,
            },
        )

    def _format_context_items(self, items: list[ContextItem]) -> str:
        """Format context items for injection."""
        sections: list[str] = []

        # Group by source
        by_source: dict[ContextSourceType, list[ContextItem]] = {}
        for item in items:
            if item.source_type not in by_source:
                by_source[item.source_type] = []
            by_source[item.source_type].append(item)

        # Format each source
        for source_type, source_items in by_source.items():
            if source_type == ContextSourceType.RAG:
                header = "### Retrieved Documents"
            elif source_type == ContextSourceType.MEMORY:
                header = "### Relevant Memories"
            else:
                header = f"### {source_type.name}"

            content_parts = []
            for i, item in enumerate(source_items, 1):
                source_info = item.metadata.get("source", "")
                if source_info:
                    content_parts.append(f"[{i}] ({source_info})\n{item.text_content}")
                else:
                    content_parts.append(f"[{i}] {item.text_content}")

            sections.append(f"{header}\n\n" + "\n\n".join(content_parts))

        return "\n\n".join(sections)

    def _count_sources(self, items: list[ContextItem]) -> dict[str, int]:
        """Count items by source type."""
        counts: dict[str, int] = {}
        for item in items:
            name = item.source_type.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def build_simple_context(
        self,
        user_input: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ContextWindow:
        """
        Build simple context without session.

        Args:
            user_input: User input
            system_prompt: Optional system prompt
            history: Optional conversation history

        Returns:
            ContextWindow
        """
        # Create temporary session
        session = SessionState.create()

        if history:
            for entry in history:
                session.history.append({
                    "content": entry.get("user", ""),
                    "response": entry.get("assistant", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return await self.build_context(
            session=session,
            user_input=user_input,
            system_prompt=system_prompt,
            include_sources={
                ContextSourceType.SYSTEM,
                ContextSourceType.HISTORY,
                ContextSourceType.USER_INPUT,
            },
        )

    def add_tool_output(
        self,
        tool_name: str,
        output: Any,
        call_id: str | None = None,
    ) -> None:
        """Add tool output for context."""
        provider = self._providers.get(ContextSourceType.TOOL_OUTPUT)
        if isinstance(provider, ToolOutputProvider):
            provider.add_output(tool_name, output, call_id)

    def clear_tool_outputs(self) -> None:
        """Clear tool outputs."""
        provider = self._providers.get(ContextSourceType.TOOL_OUTPUT)
        if isinstance(provider, ToolOutputProvider):
            provider.clear_outputs()

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self._counter.count(text)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "max_context_tokens": self._config.max_context_tokens,
            "response_reserve": self._config.response_reserve_tokens,
            "providers": list(self._providers.keys()),
            "ranking_enabled": self._config.enable_ranking,
            "min_relevance_threshold": self._config.min_relevance_threshold,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_context_engine(
    max_tokens: int = 8000,
    response_reserve: int = 1000,
    system_prompt: str | None = None,
    use_tiktoken: bool = False,
    model: str = "gpt-4",
    logger: logging.Logger | None = None,
) -> ContextEngine:
    """
    Factory function to create configured ContextEngine.

    Args:
        max_tokens: Maximum context tokens
        response_reserve: Tokens reserved for response
        system_prompt: Base system prompt
        use_tiktoken: Use tiktoken for counting
        model: Model name for tiktoken
        logger: Optional logger

    Returns:
        Configured ContextEngine
    """
    # Token counter
    if use_tiktoken:
        try:
            counter: TokenCounter = TiktokenCounter(model)
        except ImportError:
            counter = SimpleTokenCounter()
    else:
        counter = SimpleTokenCounter()

    # System prompt config
    system_config = None
    if system_prompt:
        system_config = SystemPromptConfig(base_prompt=system_prompt)

    # Engine config
    config = ContextEngineConfig(
        max_context_tokens=max_tokens,
        response_reserve_tokens=response_reserve,
        system_prompt_config=system_config,
    )

    return ContextEngine(
        config=config,
        token_counter=counter,
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

def create_agent() -> BudgetManager:
    """Factory function for agent_loader compatibility."""
    return BudgetManager(
        budget=TokenBudget.create(total=8000),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ContextSourceType",
    "ContentType",
    "MessageRole",
    "TrimmingStrategy",
    # Token Counting
    "TokenCounter",
    "SimpleTokenCounter",
    "TiktokenCounter",
    # Context Items
    "ContextItem",
    "MultimodalContent",
    "ContextMessage",
    "ContextWindow",
    # Budget
    "TokenBudget",
    "BudgetManager",
    # Ranking & Trimming
    "ContextRanker",
    "ContextTrimmer",
    # Providers
    "ContextProvider",
    "HistoryProvider",
    "MemoryProvider",
    "RAGProvider",
    "ToolOutputProvider",
    # System Prompt
    "SystemPromptConfig",
    "SystemPromptBuilder",
    # Engine
    "ContextEngineConfig",
    "ContextEngine",
    # Factory
    "create_context_engine",
]
