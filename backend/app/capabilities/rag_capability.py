"""
RAG Capability - Document intelligence for the AI Agent Platform.

This module provides the RAGCapability class which handles:
- Document ingestion and chunking
- Semantic search and retrieval
- Multiple retrieval strategies (top_k, MMR, reranking, hybrid)
- Context assembly for LLM consumption
- Citation tracking and source attribution
- Metadata filtering and namespace support

Supports multiple backends:
- InMemory (default, for development/testing)
- ChromaDB (future, for persistent vector storage)
- FAISS (future, for high-performance similarity search)
- Pinecone (future, for managed vector database)
- PostgreSQL pgvector (future, for SQL-based vector search)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

if TYPE_CHECKING:
    pass

# Import from kernel modules
from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CapabilityType,
)
from app.agents.exceptions import (
    ValidationError,
)


# =============================================================================
# ENUMS
# =============================================================================


class RetrievalStrategy(Enum):
    """Strategies for retrieving documents."""

    TOP_K = "top_k"  # Simple top-k by similarity
    MMR = "mmr"  # Maximal Marginal Relevance (diversity)
    RERANK = "rerank"  # Two-stage with reranking
    HYBRID = "hybrid"  # Combine semantic + keyword search


class DocumentStatus(Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class ChunkingStrategy(Enum):
    """Strategies for chunking documents."""

    FIXED_SIZE = "fixed_size"  # Fixed character/token count
    SENTENCE = "sentence"  # Sentence boundaries
    PARAGRAPH = "paragraph"  # Paragraph boundaries
    SEMANTIC = "semantic"  # Semantic similarity boundaries
    RECURSIVE = "recursive"  # Recursive character splitting


class RAGAction(Enum):
    """Actions supported by the RAG capability."""

    INGEST = "ingest"
    RETRIEVE = "retrieve"
    SEARCH = "search"
    DELETE = "delete"
    LIST = "list"
    STATS = "stats"
    ASSEMBLE_CONTEXT = "assemble_context"
    UPDATE_METADATA = "update_metadata"
    GET_DOCUMENT = "get_document"
    GET_CHUNK = "get_chunk"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Citation:
    """
    Source citation for retrieved content.

    Attributes:
        document_id: Source document ID
        chunk_id: Source chunk ID
        title: Document title
        source: Document source/URL
        page: Page number (if applicable)
        section: Section name (if applicable)
        relevance_score: Retrieval relevance score
        text_snippet: Relevant text snippet
    """

    document_id: str
    chunk_id: str
    title: str = ""
    source: str = ""
    page: int | None = None
    section: str | None = None
    relevance_score: float = 0.0
    text_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "relevance_score": self.relevance_score,
            "text_snippet": self.text_snippet,
        }

    def format_citation(self, style: str = "inline") -> str:
        """Format citation for display."""
        if style == "inline":
            return f"[{self.title}]" if self.title else f"[{self.document_id[:8]}]"
        elif style == "footnote":
            parts = [self.title or self.document_id[:8]]
            if self.source:
                parts.append(self.source)
            if self.page:
                parts.append(f"p. {self.page}")
            return ", ".join(parts)
        else:
            return f"[{self.document_id[:8]}]"


@dataclass
class Chunk:
    """
    A chunk of text from a document with embedding.

    Attributes:
        id: Unique chunk identifier
        document_id: Parent document ID
        content: Chunk text content
        embedding: Vector embedding
        metadata: Additional metadata
        chunk_index: Position in document
        start_char: Start character offset
        end_char: End character offset
        token_count: Estimated token count
        created_at: Creation timestamp
        namespace: Namespace for isolation
    """

    id: str
    document_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    namespace: str = "default"

    @property
    def content_hash(self) -> str:
        """Get hash of content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            chunk_index=data.get("chunk_index", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            token_count=data.get("token_count", 0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now(timezone.utc))
            ),
            namespace=data.get("namespace", "default"),
        )


@dataclass
class Document:
    """
    A source document for RAG.

    Attributes:
        id: Unique document identifier
        title: Document title
        content: Full document content
        source: Source URL or path
        metadata: Additional metadata
        status: Processing status
        chunk_ids: IDs of chunks created from this document
        created_at: Creation timestamp
        updated_at: Last update timestamp
        namespace: Namespace for isolation
        content_type: MIME type or content type
        language: Document language
        author: Document author
        tags: Searchable tags
    """

    id: str
    title: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    namespace: str = "default"
    content_type: str = "text/plain"
    language: str = "en"
    author: str = ""
    tags: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        """Get hash of content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()

    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self.chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "status": self.status.value,
            "chunk_ids": self.chunk_ids,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "namespace": self.namespace,
            "content_type": self.content_type,
            "language": self.language,
            "author": self.author,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
            status=DocumentStatus(data.get("status", "pending")),
            chunk_ids=data.get("chunk_ids", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now(timezone.utc))
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data.get("updated_at"), str)
                else data.get("updated_at", datetime.now(timezone.utc))
            ),
            namespace=data.get("namespace", "default"),
            content_type=data.get("content_type", "text/plain"),
            language=data.get("language", "en"),
            author=data.get("author", ""),
            tags=tuple(data.get("tags", [])),
        )


@dataclass
class RetrievalQuery:
    """
    Query parameters for retrieval.

    Attributes:
        query: Query text
        top_k: Number of results to return
        strategy: Retrieval strategy
        namespace: Namespace to search
        filters: Metadata filters
        min_score: Minimum similarity score
        include_metadata: Include chunk metadata
        include_content: Include chunk content
        include_embeddings: Include embeddings in response
        mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance)
        rerank_top_n: Number of candidates for reranking
        hybrid_alpha: Hybrid search weight (0=keyword, 1=semantic)
        document_ids: Filter by specific documents
        tags: Filter by tags
        date_range: Filter by date range
    """

    query: str
    top_k: int = 5
    strategy: RetrievalStrategy = RetrievalStrategy.TOP_K
    namespace: str = "default"
    filters: dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0
    include_metadata: bool = True
    include_content: bool = True
    include_embeddings: bool = False
    mmr_lambda: float = 0.5
    rerank_top_n: int = 20
    hybrid_alpha: float = 0.7
    document_ids: list[str] | None = None
    tags: list[str] | None = None
    date_range: tuple[datetime, datetime] | None = None


@dataclass
class RetrievalResult:
    """
    Result of a retrieval operation.

    Attributes:
        success: Whether retrieval succeeded
        chunks: Retrieved chunks
        citations: Source citations
        total_found: Total matching chunks
        query: Original query
        strategy_used: Strategy that was used
        execution_time: Time taken
        metadata: Additional result metadata
    """

    success: bool
    chunks: list[Chunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    total_found: int = 0
    query: str = ""
    strategy_used: RetrievalStrategy = RetrievalStrategy.TOP_K
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "chunks": [c.to_dict() for c in self.chunks],
            "citations": [c.to_dict() for c in self.citations],
            "total_found": self.total_found,
            "query": self.query,
            "strategy_used": self.strategy_used.value,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class RAGStats:
    """Statistics about the RAG system."""

    total_documents: int = 0
    total_chunks: int = 0
    namespaces: list[str] = field(default_factory=list)
    documents_by_status: dict[str, int] = field(default_factory=dict)
    avg_chunks_per_document: float = 0.0
    total_tokens: int = 0
    index_size_bytes: int = 0


# =============================================================================
# VECTOR STORE (Abstract)
# =============================================================================


class VectorStore(ABC):
    """
    Abstract base class for vector storage backends.

    Implementations:
    - InMemoryVectorStore: Default, for development/testing
    - ChromaDBVectorStore: Future, for persistent storage
    - FAISSVectorStore: Future, for high-performance search
    - PineconeVectorStore: Future, for managed service
    - PgVectorStore: Future, for PostgreSQL integration
    """

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk]) -> bool:
        """Add chunks to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        namespace: str = "default",
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        ...

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        ...

    @abstractmethod
    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by IDs."""
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        ...

    @abstractmethod
    async def delete_by_namespace(self, namespace: str) -> int:
        """Delete all chunks in a namespace."""
        ...

    @abstractmethod
    async def list_namespaces(self) -> list[str]:
        """List all namespaces."""
        ...

    @abstractmethod
    async def count(self, namespace: str | None = None) -> int:
        """Count chunks in store."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check store health."""
        ...


# =============================================================================
# IN-MEMORY VECTOR STORE
# =============================================================================


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for development and testing.

    Features:
    - Fast similarity search using cosine similarity
    - Namespace isolation
    - Metadata filtering
    - No external dependencies
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize in-memory store."""
        self._chunks: dict[str, Chunk] = {}
        self._by_document: dict[str, list[str]] = {}
        self._by_namespace: dict[str, list[str]] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()

    async def add_chunks(self, chunks: list[Chunk]) -> bool:
        """Add chunks to the store."""
        async with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

                # Index by document
                if chunk.document_id not in self._by_document:
                    self._by_document[chunk.document_id] = []
                if chunk.id not in self._by_document[chunk.document_id]:
                    self._by_document[chunk.document_id].append(chunk.id)

                # Index by namespace
                if chunk.namespace not in self._by_namespace:
                    self._by_namespace[chunk.namespace] = []
                if chunk.id not in self._by_namespace[chunk.namespace]:
                    self._by_namespace[chunk.namespace].append(chunk.id)

            return True

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        namespace: str = "default",
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using cosine similarity."""
        async with self._lock:
            results: list[tuple[Chunk, float]] = []

            # Get chunks in namespace
            chunk_ids = self._by_namespace.get(namespace, [])

            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is None or chunk.embedding is None:
                    continue

                # Apply filters
                if filters and not self._matches_filters(chunk, filters):
                    continue

                # Calculate similarity
                score = self._cosine_similarity(embedding, chunk.embedding)

                if score >= min_score:
                    results.append((chunk, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k]

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if isinstance(value, list):
                # List filter: chunk value must be in list
                if chunk_value not in value:
                    return False
            elif isinstance(value, dict):
                # Range filter
                if "$gte" in value and chunk_value < value["$gte"]:
                    return False
                if "$lte" in value and chunk_value > value["$lte"]:
                    return False
                if "$gt" in value and chunk_value <= value["$gt"]:
                    return False
                if "$lt" in value and chunk_value >= value["$lt"]:
                    return False
                if "$ne" in value and chunk_value == value["$ne"]:
                    return False
                if "$in" in value and chunk_value not in value["$in"]:
                    return False
            else:
                # Exact match
                if chunk_value != value:
                    return False

        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        async with self._lock:
            return self._chunks.get(chunk_id)

    async def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        async with self._lock:
            chunk_ids = self._by_document.get(document_id, [])
            return [
                self._chunks[cid]
                for cid in chunk_ids
                if cid in self._chunks
            ]

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by IDs."""
        async with self._lock:
            deleted = 0
            for chunk_id in chunk_ids:
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]

                    # Remove from document index
                    if chunk.document_id in self._by_document:
                        if chunk_id in self._by_document[chunk.document_id]:
                            self._by_document[chunk.document_id].remove(chunk_id)

                    # Remove from namespace index
                    if chunk.namespace in self._by_namespace:
                        if chunk_id in self._by_namespace[chunk.namespace]:
                            self._by_namespace[chunk.namespace].remove(chunk_id)

                    del self._chunks[chunk_id]
                    deleted += 1

            return deleted

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        chunk_ids = self._by_document.get(document_id, []).copy()
        return await self.delete_chunks(chunk_ids)

    async def delete_by_namespace(self, namespace: str) -> int:
        """Delete all chunks in a namespace."""
        chunk_ids = self._by_namespace.get(namespace, []).copy()
        return await self.delete_chunks(chunk_ids)

    async def list_namespaces(self) -> list[str]:
        """List all namespaces."""
        async with self._lock:
            return list(self._by_namespace.keys())

    async def count(self, namespace: str | None = None) -> int:
        """Count chunks in store."""
        async with self._lock:
            if namespace:
                return len(self._by_namespace.get(namespace, []))
            return len(self._chunks)

    async def health_check(self) -> bool:
        """Check store health."""
        return True


# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================


class DocumentProcessor:
    """
    Processes documents into chunks.

    Features:
    - Multiple chunking strategies
    - Overlap support
    - Metadata preservation
    - Token counting
    """

    # Approximate characters per token
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize document processor.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy
            logger: Optional logger
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._strategy = strategy
        self._logger = logger or logging.getLogger(__name__)

        # Convert to characters
        self._char_size = chunk_size * self.CHARS_PER_TOKEN
        self._char_overlap = chunk_overlap * self.CHARS_PER_TOKEN

    def process(
        self,
        document: Document,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> list[Chunk]:
        """
        Process a document into chunks.

        Args:
            document: Document to process
            embedding_fn: Function to generate embeddings

        Returns:
            List of chunks
        """
        if self._strategy == ChunkingStrategy.FIXED_SIZE:
            texts = self._chunk_fixed_size(document.content)
        elif self._strategy == ChunkingStrategy.SENTENCE:
            texts = self._chunk_by_sentence(document.content)
        elif self._strategy == ChunkingStrategy.PARAGRAPH:
            texts = self._chunk_by_paragraph(document.content)
        elif self._strategy == ChunkingStrategy.RECURSIVE:
            texts = self._chunk_recursive(document.content)
        else:
            texts = self._chunk_fixed_size(document.content)

        chunks: list[Chunk] = []
        char_offset = 0

        for i, text in enumerate(texts):
            # Find actual position in document
            start_char = document.content.find(text, char_offset)
            if start_char == -1:
                start_char = char_offset
            end_char = start_char + len(text)
            char_offset = start_char + 1

            # Generate embedding
            embedding = None
            if embedding_fn:
                try:
                    embedding = embedding_fn(text)
                except Exception as e:
                    self._logger.warning(f"Failed to generate embedding: {e}")

            # Estimate token count
            token_count = len(text) // self.CHARS_PER_TOKEN

            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=text,
                embedding=embedding,
                metadata={
                    "title": document.title,
                    "source": document.source,
                    "author": document.author,
                    "language": document.language,
                    **document.metadata,
                },
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                namespace=document.namespace,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_fixed_size(self, text: str) -> list[str]:
        """Chunk by fixed character size with overlap."""
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self._char_size

            # Try to break at word boundary
            if end < len(text):
                # Look for space within last 10% of chunk
                search_start = end - int(self._char_size * 0.1)
                space_pos = text.rfind(" ", search_start, end)
                if space_pos > start:
                    end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self._char_overlap
            if start < 0:
                start = 0

        return chunks

    def _chunk_by_sentence(self, text: str) -> list[str]:
        """Chunk by sentence boundaries."""
        # Simple sentence splitting
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > self._char_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_sentences: list[str] = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self._char_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> list[str]:
        """Chunk by paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If single paragraph exceeds size, split it
            if para_size > self._char_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sub_chunks = self._chunk_by_sentence(para)
                chunks.extend(sub_chunks)
                continue

            if current_size + para_size > self._char_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_recursive(self, text: str) -> list[str]:
        """Recursively chunk using multiple separators."""
        separators = ["\n\n", "\n", ". ", " ", ""]

        def split_recursive(text: str, sep_idx: int) -> list[str]:
            if sep_idx >= len(separators):
                return [text]

            separator = separators[sep_idx]

            if separator:
                parts = text.split(separator)
            else:
                # Character-level split
                parts = list(text)

            chunks: list[str] = []
            current = ""

            for part in parts:
                test = current + (separator if current else "") + part

                if len(test) <= self._char_size:
                    current = test
                else:
                    if current:
                        chunks.append(current)

                    if len(part) > self._char_size:
                        # Recursively split with next separator
                        sub_chunks = split_recursive(part, sep_idx + 1)
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part

            if current:
                chunks.append(current)

            return chunks

        return split_recursive(text, 0)


# =============================================================================
# RERANKER
# =============================================================================


class Reranker:
    """
    Reranks retrieved results for improved relevance.

    Supports:
    - Cross-encoder reranking (future)
    - BM25 reranking
    - Reciprocal Rank Fusion
    """

    def __init__(
        self,
        model_fn: Callable[[str, str], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize reranker.

        Args:
            model_fn: Function that scores (query, text) pairs
            logger: Optional logger
        """
        self._model_fn = model_fn
        self._logger = logger or logging.getLogger(__name__)

    def rerank(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Rerank chunks for a query.

        Args:
            query: Query text
            chunks: List of (chunk, initial_score) tuples
            top_k: Number of results to return

        Returns:
            Reranked list of (chunk, score) tuples
        """
        if self._model_fn:
            # Use provided model for reranking
            return self._rerank_with_model(query, chunks, top_k)
        else:
            # Use BM25-style reranking
            return self._rerank_bm25(query, chunks, top_k)

    def _rerank_with_model(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Rerank using provided model function."""
        scored: list[tuple[Chunk, float]] = []

        for chunk, _ in chunks:
            try:
                score = self._model_fn(query, chunk.content)
                scored.append((chunk, score))
            except Exception as e:
                self._logger.warning(f"Reranking failed for chunk: {e}")
                scored.append((chunk, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _rerank_bm25(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Rerank using BM25-style scoring."""
        query_terms = self._tokenize(query.lower())

        if not query_terms:
            return chunks[:top_k]

        # Calculate document frequencies
        doc_freq: dict[str, int] = {}
        for chunk, _ in chunks:
            terms = set(self._tokenize(chunk.content.lower()))
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(chunks)
        avg_doc_len = sum(len(c.content) for c, _ in chunks) / max(n_docs, 1)

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        scored: list[tuple[Chunk, float]] = []

        for chunk, semantic_score in chunks:
            content_lower = chunk.content.lower()
            doc_len = len(chunk.content)
            terms = set(self._tokenize(content_lower))
            term_freq: dict[str, int] = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1

            bm25_score = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue

                tf = term_freq[term]
                df = doc_freq.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                bm25_score += idf * numerator / denominator

            # Combine semantic and BM25 scores
            combined_score = 0.7 * semantic_score + 0.3 * (bm25_score / max(len(query_terms), 1))
            scored.append((chunk, combined_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\b\w+\b", text)

    def reciprocal_rank_fusion(
        self,
        result_lists: list[list[tuple[Chunk, float]]],
        k: int = 60,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.

        Args:
            result_lists: List of ranked result lists
            k: RRF parameter
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        for results in result_lists:
            for rank, (chunk, _) in enumerate(results):
                chunk_id = chunk.id
                chunks[chunk_id] = chunk
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [(chunks[cid], scores[cid]) for cid in sorted_ids[:top_k]]


# =============================================================================
# CONTEXT ASSEMBLER
# =============================================================================


class ContextAssembler:
    """
    Assembles retrieved chunks into context for LLM.

    Features:
    - Token budget management
    - Citation formatting
    - Context structuring
    - Deduplication
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        include_citations: bool = True,
        citation_style: str = "inline",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize context assembler.

        Args:
            max_tokens: Maximum tokens in assembled context
            include_citations: Include citation markers
            citation_style: Citation format (inline, footnote, none)
            logger: Optional logger
        """
        self._max_tokens = max_tokens
        self._include_citations = include_citations
        self._citation_style = citation_style
        self._logger = logger or logging.getLogger(__name__)

    def assemble(
        self,
        chunks: list[tuple[Chunk, float]],
        query: str | None = None,
        format: str = "markdown",
    ) -> dict[str, Any]:
        """
        Assemble chunks into context.

        Args:
            chunks: List of (chunk, score) tuples
            query: Original query (for context)
            format: Output format (markdown, plain, json)

        Returns:
            Dictionary with context and citations
        """
        # Deduplicate by content hash
        seen_hashes: set[str] = set()
        unique_chunks: list[tuple[Chunk, float]] = []

        for chunk, score in chunks:
            content_hash = chunk.content_hash
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append((chunk, score))

        # Build context within token budget
        context_parts: list[str] = []
        citations: list[Citation] = []
        total_tokens = 0

        for i, (chunk, score) in enumerate(unique_chunks):
            if total_tokens + chunk.token_count > self._max_tokens:
                break

            # Create citation
            citation = Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                title=chunk.metadata.get("title", ""),
                source=chunk.metadata.get("source", ""),
                page=chunk.metadata.get("page"),
                section=chunk.metadata.get("section"),
                relevance_score=score,
                text_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            )
            citations.append(citation)

            # Format chunk
            if format == "markdown":
                if self._include_citations:
                    citation_marker = f" [{i + 1}]"
                else:
                    citation_marker = ""

                context_parts.append(f"{chunk.content}{citation_marker}")
            elif format == "json":
                context_parts.append(chunk.content)
            else:
                context_parts.append(chunk.content)

            total_tokens += chunk.token_count

        # Assemble final context
        if format == "markdown":
            context_text = "\n\n---\n\n".join(context_parts)

            if self._include_citations and citations:
                context_text += "\n\n## Sources\n"
                for i, citation in enumerate(citations):
                    context_text += f"\n[{i + 1}] {citation.format_citation('footnote')}"
        elif format == "json":
            context_text = context_parts  # type: ignore[assignment]
        else:
            context_text = "\n\n".join(context_parts)

        return {
            "context": context_text,
            "citations": [c.to_dict() for c in citations],
            "total_chunks": len(unique_chunks),
            "total_tokens": total_tokens,
            "truncated": len(unique_chunks) < len(chunks),
        }


# =============================================================================
# RAG CAPABILITY
# =============================================================================


class RAGCapability(Capability):
    """
    RAG (Retrieval-Augmented Generation) capability for the AI Agent Platform.

    Provides unified interface for:
    - Document ingestion and chunking
    - Semantic search and retrieval
    - Multiple retrieval strategies
    - Context assembly for LLM consumption
    - Citation tracking

    Compatible with:
    - CapabilityRegistry for registration
    - Executor for action execution
    - ContextEngine for context preparation
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        rerank_fn: Callable[[str, str], float] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        max_context_tokens: int = 4000,
        default_namespace: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize RAG capability.

        Args:
            vector_store: Vector storage backend
            embedding_fn: Function to generate embeddings
            rerank_fn: Function for reranking (query, text) -> score
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking documents
            max_context_tokens: Maximum tokens in assembled context
            default_namespace: Default namespace for documents
            logger: Optional logger
        """
        self._vector_store = vector_store or InMemoryVectorStore(logger=logger)
        self._embedding_fn = embedding_fn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._default_namespace = default_namespace
        self._logger = logger or logging.getLogger(__name__)
        self._initialized = False

        # Components
        self._processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
            logger=logger,
        )
        self._reranker = Reranker(model_fn=rerank_fn, logger=logger)
        self._assembler = ContextAssembler(
            max_tokens=max_context_tokens,
            logger=logger,
        )

        # Document storage
        self._documents: dict[str, Document] = {}
        self._lock = asyncio.Lock()

        # Action handlers
        self._action_handlers: dict[RAGAction, Callable[..., Any]] = {
            RAGAction.INGEST: self._handle_ingest,
            RAGAction.RETRIEVE: self._handle_retrieve,
            RAGAction.SEARCH: self._handle_search,
            RAGAction.DELETE: self._handle_delete,
            RAGAction.LIST: self._handle_list,
            RAGAction.STATS: self._handle_stats,
            RAGAction.ASSEMBLE_CONTEXT: self._handle_assemble_context,
            RAGAction.UPDATE_METADATA: self._handle_update_metadata,
            RAGAction.GET_DOCUMENT: self._handle_get_document,
            RAGAction.GET_CHUNK: self._handle_get_chunk,
        }

    # -------------------------------------------------------------------------
    # Capability Interface
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata."""
        return CapabilityMetadata(
            name="rag",
            version="1.0.0",
            capability_type=CapabilityType.RAG,
            description="RAG capability for document retrieval, semantic search, and context assembly",
            actions=tuple(action.value for action in RAGAction),
            required_permissions=("rag:read", "rag:write"),
            config_schema={
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "minimum": 100},
                    "chunk_overlap": {"type": "integer", "minimum": 0},
                    "max_context_tokens": {"type": "integer", "minimum": 500},
                    "default_namespace": {"type": "string"},
                },
            },
            tags=("rag", "retrieval", "search", "documents", "embeddings"),
        )

    async def _do_initialize(self) -> None:
        self._logger.debug("RAGCapability._do_initialize")

    async def _do_shutdown(self) -> None:
        self._logger.debug("RAGCapability._do_shutdown")

    async def _do_execute(self, context) -> CapabilityResult:
        return CapabilityResult(success=True, status=CapabilityStatus.SUCCESS)  # type: ignore[attr-defined]

    async def initialize(self) -> None:
        """Initialize the capability."""
        if self._initialized:
            return

        self._initialized = True
        self._logger.info("RAGCapability initialized")

    async def shutdown(self) -> None:
        """Shutdown the capability."""
        if not self._initialized:
            return

        self._initialized = False
        self._logger.info("RAGCapability shutdown")

    async def health_check(self) -> bool:
        """Check capability health."""
        try:
            return await self._vector_store.health_check()
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
        Execute a RAG action.

        Args:
            action: Action to perform
            parameters: Action parameters
            context: Execution context

        Returns:
            CapabilityResult with operation outcome
        """
        start_time = time.time()
        context = context or {}

        try:
            # Validate action
            try:
                rag_action = RAGAction(action)
            except ValueError:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Get handler
            handler = self._action_handlers.get(rag_action)
            if handler is None:
                return CapabilityResult(
                    success=False,
                    status=CapabilityStatus.FAILED,
                    error=f"No handler for action: {action}",
                    execution_time=time.time() - start_time,
                )

            # Execute handler
            result = await handler(parameters, context)

            return CapabilityResult(
                success=True,
                status=CapabilityStatus.COMPLETED,
                data=result,
                execution_time=time.time() - start_time,
            )

        except ValidationError as e:
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=f"Validation error: {e}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            self._logger.exception(f"RAG action failed: {e}")
            return CapabilityResult(
                success=False,
                status=CapabilityStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    async def _handle_ingest(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle document ingestion."""
        content = parameters.get("content")
        if not content:
            raise ValidationError("content is required")

        title = parameters.get("title", "Untitled")
        source = parameters.get("source", "")
        namespace = parameters.get("namespace", self._default_namespace)
        metadata = parameters.get("metadata", {})
        tags = tuple(parameters.get("tags", []))

        document = await self.ingest(
            content=content,
            title=title,
            source=source,
            namespace=namespace,
            metadata=metadata,
            tags=tags,
        )

        return {
            "document_id": document.id,
            "title": document.title,
            "chunk_count": document.chunk_count,
            "status": document.status.value,
        }

    async def _handle_retrieve(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle retrieval."""
        query = parameters.get("query")
        if not query:
            raise ValidationError("query is required")

        # Build retrieval query
        strategy_str = parameters.get("strategy", "top_k")
        try:
            strategy = RetrievalStrategy(strategy_str)
        except ValueError:
            strategy = RetrievalStrategy.TOP_K

        retrieval_query = RetrievalQuery(
            query=query,
            top_k=parameters.get("top_k", 5),
            strategy=strategy,
            namespace=parameters.get("namespace", self._default_namespace),
            filters=parameters.get("filters", {}),
            min_score=parameters.get("min_score", 0.0),
            include_metadata=parameters.get("include_metadata", True),
            include_content=parameters.get("include_content", True),
            mmr_lambda=parameters.get("mmr_lambda", 0.5),
            rerank_top_n=parameters.get("rerank_top_n", 20),
            hybrid_alpha=parameters.get("hybrid_alpha", 0.7),
            document_ids=parameters.get("document_ids"),
            tags=parameters.get("tags"),
        )

        result = await self.retrieve(retrieval_query)

        return result.to_dict()

    async def _handle_search(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle semantic search."""
        query = parameters.get("query")
        if not query:
            raise ValidationError("query is required")

        top_k = parameters.get("top_k", 10)
        namespace = parameters.get("namespace", self._default_namespace)
        filters = parameters.get("filters", {})
        min_score = parameters.get("min_score", 0.0)

        chunks = await self.search(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
            min_score=min_score,
        )

        return {
            "results": [
                {
                    "chunk": chunk.to_dict(),
                    "score": score,
                }
                for chunk, score in chunks
            ],
            "total": len(chunks),
        }

    async def _handle_delete(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle deletion."""
        document_id = parameters.get("document_id")
        chunk_ids = parameters.get("chunk_ids")
        namespace = parameters.get("namespace")

        deleted = 0

        if document_id:
            deleted = await self.delete_document(document_id)
        elif chunk_ids:
            deleted = await self._vector_store.delete_chunks(chunk_ids)
        elif namespace:
            deleted = await self._vector_store.delete_by_namespace(namespace)
        else:
            raise ValidationError("document_id, chunk_ids, or namespace required")

        return {"deleted": deleted}

    async def _handle_list(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle document listing."""
        namespace = parameters.get("namespace")
        status_str = parameters.get("status")
        limit = parameters.get("limit", 100)
        offset = parameters.get("offset", 0)

        status = DocumentStatus(status_str) if status_str else None

        documents = await self.list_documents(
            namespace=namespace,
            status=status,
            limit=limit,
            offset=offset,
        )

        return {
            "documents": [d.to_dict() for d in documents],
            "total": len(documents),
        }

    async def _handle_stats(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle stats retrieval."""
        stats = await self.get_stats()

        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "namespaces": stats.namespaces,
            "documents_by_status": stats.documents_by_status,
            "avg_chunks_per_document": stats.avg_chunks_per_document,
            "total_tokens": stats.total_tokens,
        }

    async def _handle_assemble_context(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle context assembly."""
        query = parameters.get("query")
        if not query:
            raise ValidationError("query is required")

        top_k = parameters.get("top_k", 5)
        namespace = parameters.get("namespace", self._default_namespace)
        format = parameters.get("format", "markdown")
        max_tokens = parameters.get("max_tokens")

        assembled = await self.assemble_context(
            query=query,
            top_k=top_k,
            namespace=namespace,
            format=format,
            max_tokens=max_tokens,
        )

        return assembled

    async def _handle_update_metadata(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle metadata update."""
        document_id = parameters.get("document_id")
        if not document_id:
            raise ValidationError("document_id is required")

        metadata = parameters.get("metadata", {})
        tags = parameters.get("tags")

        async with self._lock:
            if document_id not in self._documents:
                raise ValidationError(f"Document not found: {document_id}")

            document = self._documents[document_id]
            document.metadata.update(metadata)
            if tags is not None:
                document.tags = tuple(tags)
            document.updated_at = datetime.now(timezone.utc)

        return {"updated": True, "document_id": document_id}

    async def _handle_get_document(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle document retrieval."""
        document_id = parameters.get("document_id")
        if not document_id:
            raise ValidationError("document_id is required")

        async with self._lock:
            document = self._documents.get(document_id)

        if not document:
            raise ValidationError(f"Document not found: {document_id}")

        return document.to_dict()

    async def _handle_get_chunk(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle chunk retrieval."""
        chunk_id = parameters.get("chunk_id")
        if not chunk_id:
            raise ValidationError("chunk_id is required")

        chunk = await self._vector_store.get_chunk(chunk_id)

        if not chunk:
            raise ValidationError(f"Chunk not found: {chunk_id}")

        return chunk.to_dict()

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    async def ingest(
        self,
        content: str,
        title: str = "Untitled",
        source: str = "",
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: tuple[str, ...] = (),
    ) -> Document:
        """
        Ingest a document into the RAG system.

        Args:
            content: Document content
            title: Document title
            source: Source URL or path
            namespace: Namespace for isolation
            metadata: Additional metadata
            tags: Searchable tags

        Returns:
            Ingested Document
        """
        namespace = namespace or self._default_namespace

        # Create document
        document = Document(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source=source,
            metadata=metadata or {},
            status=DocumentStatus.PROCESSING,
            namespace=namespace,
            tags=tags,
        )

        async with self._lock:
            self._documents[document.id] = document

        try:
            # Process into chunks
            chunks = self._processor.process(document, self._embedding_fn)

            # Store chunks
            await self._vector_store.add_chunks(chunks)

            # Update document
            async with self._lock:
                document.chunk_ids = [c.id for c in chunks]
                document.status = DocumentStatus.INDEXED
                document.updated_at = datetime.now(timezone.utc)

            self._logger.info(
                f"Ingested document {document.id} with {len(chunks)} chunks"
            )

        except Exception as e:
            async with self._lock:
                document.status = DocumentStatus.FAILED
                document.metadata["error"] = str(e)
            raise

        return document

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Retrieval query parameters

        Returns:
            RetrievalResult with chunks and citations
        """
        start_time = time.time()

        # Generate query embedding
        if not self._embedding_fn:
            return RetrievalResult(
                success=False,
                query=query.query,
                metadata={"error": "No embedding function configured"},
            )

        try:
            query_embedding = self._embedding_fn(query.query)
        except Exception as e:
            return RetrievalResult(
                success=False,
                query=query.query,
                metadata={"error": f"Embedding failed: {e}"},
            )

        # Build filters
        filters = dict(query.filters)
        if query.document_ids:
            filters["document_id"] = {"$in": query.document_ids}
        if query.tags:
            filters["tags"] = {"$in": query.tags}

        # Retrieve based on strategy
        if query.strategy == RetrievalStrategy.TOP_K:
            results = await self._retrieve_top_k(
                query_embedding, query.top_k, query.namespace, filters, query.min_score
            )
        elif query.strategy == RetrievalStrategy.MMR:
            results = await self._retrieve_mmr(
                query_embedding, query.top_k, query.namespace, filters,
                query.min_score, query.mmr_lambda
            )
        elif query.strategy == RetrievalStrategy.RERANK:
            results = await self._retrieve_rerank(
                query.query, query_embedding, query.top_k, query.namespace,
                filters, query.min_score, query.rerank_top_n
            )
        elif query.strategy == RetrievalStrategy.HYBRID:
            results = await self._retrieve_hybrid(
                query.query, query_embedding, query.top_k, query.namespace,
                filters, query.min_score, query.hybrid_alpha
            )
        else:
            results = await self._retrieve_top_k(
                query_embedding, query.top_k, query.namespace, filters, query.min_score
            )

        # Build citations
        citations = [
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                title=chunk.metadata.get("title", ""),
                source=chunk.metadata.get("source", ""),
                page=chunk.metadata.get("page"),
                section=chunk.metadata.get("section"),
                relevance_score=score,
                text_snippet=chunk.content[:200],
            )
            for chunk, score in results
        ]

        # Optionally strip embeddings
        chunks = []
        for chunk, score in results:
            if not query.include_embeddings:
                chunk.embedding = None
            chunks.append(chunk)

        return RetrievalResult(
            success=True,
            chunks=chunks,
            citations=citations,
            total_found=len(results),
            query=query.query,
            strategy_used=query.strategy,
            execution_time=time.time() - start_time,
        )

    async def _retrieve_top_k(
        self,
        embedding: list[float],
        top_k: int,
        namespace: str,
        filters: dict[str, Any],
        min_score: float,
    ) -> list[tuple[Chunk, float]]:
        """Simple top-k retrieval."""
        return await self._vector_store.search(
            embedding=embedding,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
            min_score=min_score,
        )

    async def _retrieve_mmr(
        self,
        embedding: list[float],
        top_k: int,
        namespace: str,
        filters: dict[str, Any],
        min_score: float,
        lambda_param: float,
    ) -> list[tuple[Chunk, float]]:
        """Maximal Marginal Relevance retrieval for diversity."""
        # Get more candidates
        candidates = await self._vector_store.search(
            embedding=embedding,
            top_k=top_k * 4,
            namespace=namespace,
            filters=filters,
            min_score=min_score,
        )

        if not candidates:
            return []

        # MMR selection
        selected: list[tuple[Chunk, float]] = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            best_score = -float("inf")
            best_idx = 0

            for i, (chunk, relevance) in enumerate(remaining):
                if chunk.embedding is None:
                    continue

                # Calculate diversity (min similarity to selected)
                if selected:
                    max_sim = max(
                        self._cosine_similarity(chunk.embedding, s[0].embedding or [])
                        for s in selected
                    )
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    async def _retrieve_rerank(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
        namespace: str,
        filters: dict[str, Any],
        min_score: float,
        rerank_top_n: int,
    ) -> list[tuple[Chunk, float]]:
        """Two-stage retrieval with reranking."""
        # First stage: get candidates
        candidates = await self._vector_store.search(
            embedding=embedding,
            top_k=rerank_top_n,
            namespace=namespace,
            filters=filters,
            min_score=min_score,
        )

        if not candidates:
            return []

        # Second stage: rerank
        return self._reranker.rerank(query, candidates, top_k)

    async def _retrieve_hybrid(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
        namespace: str,
        filters: dict[str, Any],
        min_score: float,
        alpha: float,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid semantic + keyword retrieval."""
        # Semantic search
        semantic_results = await self._vector_store.search(
            embedding=embedding,
            top_k=top_k * 2,
            namespace=namespace,
            filters=filters,
            min_score=min_score,
        )

        # Keyword search (using reranker's BM25)
        keyword_results = self._reranker._rerank_bm25(
            query, semantic_results, top_k * 2
        )

        # Combine using RRF
        return self._reranker.reciprocal_rank_fusion(
            [semantic_results, keyword_results],
            top_k=top_k,
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """
        Simple semantic search.

        Args:
            query: Search query
            top_k: Number of results
            namespace: Namespace to search
            filters: Metadata filters
            min_score: Minimum similarity score

        Returns:
            List of (chunk, score) tuples
        """
        namespace = namespace or self._default_namespace

        if not self._embedding_fn:
            return []

        embedding = self._embedding_fn(query)

        return await self._vector_store.search(
            embedding=embedding,
            top_k=top_k,
            namespace=namespace,
            filters=filters or {},
            min_score=min_score,
        )

    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document and its chunks.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        async with self._lock:
            if document_id in self._documents:
                document = self._documents[document_id]
                document.status = DocumentStatus.DELETED
                del self._documents[document_id]

        return await self._vector_store.delete_by_document(document_id)

    async def list_documents(
        self,
        namespace: str | None = None,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """
        List documents.

        Args:
            namespace: Filter by namespace
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of documents
        """
        async with self._lock:
            documents = list(self._documents.values())

        # Filter
        if namespace:
            documents = [d for d in documents if d.namespace == namespace]
        if status:
            documents = [d for d in documents if d.status == status]

        # Sort by created_at descending
        documents.sort(key=lambda d: d.created_at, reverse=True)

        # Paginate
        return documents[offset : offset + limit]

    async def assemble_context(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        format: str = "markdown",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve and assemble context for LLM.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            namespace: Namespace to search
            format: Output format
            max_tokens: Maximum tokens (overrides default)

        Returns:
            Assembled context with citations
        """
        namespace = namespace or self._default_namespace

        # Retrieve chunks
        retrieval_query = RetrievalQuery(
            query=query,
            top_k=top_k,
            strategy=RetrievalStrategy.RERANK,
            namespace=namespace,
        )

        result = await self.retrieve(retrieval_query)

        if not result.success:
            return {
                "context": "",
                "citations": [],
                "error": result.metadata.get("error", "Retrieval failed"),
            }

        # Assemble context
        chunks_with_scores = [
            (chunk, citation.relevance_score)
            for chunk, citation in zip(result.chunks, result.citations)
        ]

        if max_tokens:
            self._assembler._max_tokens = max_tokens

        return self._assembler.assemble(chunks_with_scores, query, format)

    async def get_stats(self) -> RAGStats:  # type: ignore[override]
        """Get RAG system statistics."""
        async with self._lock:
            documents = list(self._documents.values())

        total_docs = len(documents)
        total_chunks = await self._vector_store.count()
        namespaces = await self._vector_store.list_namespaces()

        # Count by status
        by_status: dict[str, int] = {}
        total_tokens = 0

        for doc in documents:
            status_key = doc.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

        # Get chunk token counts
        for doc in documents:
            for chunk_id in doc.chunk_ids:
                chunk = await self._vector_store.get_chunk(chunk_id)
                if chunk:
                    total_tokens += chunk.token_count

        avg_chunks = total_chunks / max(total_docs, 1)

        return RAGStats(
            total_documents=total_docs,
            total_chunks=total_chunks,
            namespaces=namespaces,
            documents_by_status=by_status,
            avg_chunks_per_document=avg_chunks,
            total_tokens=total_tokens,
        )

    async def get_context_for_engine(
        self,
        query: str,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Get context formatted for ContextEngine integration.

        Args:
            query: Query text
            namespace: Namespace to search
            top_k: Number of chunks

        Returns:
            Context dictionary for injection
        """
        assembled = await self.assemble_context(
            query=query,
            top_k=top_k,
            namespace=namespace,
            format="markdown",
        )

        return {
            "rag_context": assembled.get("context", ""),
            "rag_citations": assembled.get("citations", []),
            "rag_chunks": assembled.get("total_chunks", 0),
            "rag_tokens": assembled.get("total_tokens", 0),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_rag_capability(
    backend_type: str = "memory",
    embedding_fn: Callable[[str], list[float]] | None = None,
    rerank_fn: Callable[[str, str], float] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive",
    max_context_tokens: int = 4000,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> RAGCapability:
    """
    Factory function to create RAGCapability with specified backend.

    Args:
        backend_type: Backend type ("memory", "chromadb", "faiss", "pinecone", "pgvector")
        embedding_fn: Function to generate embeddings
        rerank_fn: Function for reranking
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        chunking_strategy: Chunking strategy name
        max_context_tokens: Maximum context tokens
        logger: Optional logger
        **kwargs: Additional backend configuration

    Returns:
        Configured RAGCapability
    """
    logger = logger or logging.getLogger(__name__)

    # Create backend
    if backend_type == "memory":
        vector_store = InMemoryVectorStore(logger=logger)
    elif backend_type == "chromadb":
        raise NotImplementedError("ChromaDB backend not yet implemented")
    elif backend_type == "faiss":
        raise NotImplementedError("FAISS backend not yet implemented")
    elif backend_type == "pinecone":
        raise NotImplementedError("Pinecone backend not yet implemented")
    elif backend_type == "pgvector":
        raise NotImplementedError("pgvector backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    # Parse chunking strategy
    try:
        strategy = ChunkingStrategy(chunking_strategy)
    except ValueError:
        strategy = ChunkingStrategy.RECURSIVE

    return RAGCapability(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        rerank_fn=rerank_fn,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=strategy,
        max_context_tokens=max_context_tokens,
        default_namespace=kwargs.get("default_namespace", "default"),
        logger=logger,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RetrievalStrategy",
    "DocumentStatus",
    "ChunkingStrategy",
    "RAGAction",
    # Data Classes
    "Citation",
    "Chunk",
    "Document",
    "RetrievalQuery",
    "RetrievalResult",
    "RAGStats",
    # Vector Store
    "VectorStore",
    "InMemoryVectorStore",
    # Components
    "DocumentProcessor",
    "Reranker",
    "ContextAssembler",
    # Capability
    "RAGCapability",
    # Factory
    "create_rag_capability",
]
