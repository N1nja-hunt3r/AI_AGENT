from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    doc_name: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0

    def __post_init__(self) -> None:
        self.token_estimate = max(1, len(self.content.split()) * 4 // 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section": self.section,
            "metadata": self.metadata,
            "token_estimate": self.token_estimate,
        }


@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 32
    max_chunk_size: int = 2048
    sentence_separators: List[str] = field(
        default_factory=lambda: [r"(?<=[.!?])\s+", r"(?<=。)\s*", r"(?<=！)\s*", r"(?<=？)\s*"]
    )
    paragraph_separator: str = r"\n\s*\n"
    recursive_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    preserve_section_headers: bool = True
    section_header_pattern: str = r"^(#{1,6}\s+.*|[A-Z][A-Z\s]{3,}:?\s*$)"
    page_break_pattern: str = r"\f|\[PAGE\s*\d+\]|---PAGE---"


class ChunkerError(Exception):
    pass


class Chunker:
    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self._config = config or ChunkerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def split(
        self,
        text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: str = "recursive",
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        if strategy == "sentence":
            chunks = await self.split_sentences(text, doc_id, doc_name, meta)
        elif strategy == "paragraph":
            chunks = await self.split_paragraphs(text, doc_id, doc_name, meta)
        else:
            chunks = await self.split_recursive(text, doc_id, doc_name, meta)
        return await self.validate_chunks(chunks)

    async def split_recursive(
        self,
        text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        pages = self._split_pages(text)
        all_chunks: List[Chunk] = []
        global_index = 0
        global_offset = 0

        for page_num, page_text in pages:
            sections = self._split_sections(page_text)
            for section_title, section_text in sections:
                raw = self._recursive_split(
                    section_text, self._config.recursive_separators
                )
                offset = global_offset
                for raw_chunk in raw:
                    if not raw_chunk.strip():
                        offset += len(raw_chunk)
                        continue
                    start = text.find(raw_chunk, max(0, offset - 50))
                    if start == -1:
                        start = offset
                    end = start + len(raw_chunk)
                    chunk_meta = dict(meta)
                    if section_title:
                        chunk_meta["section_title"] = section_title
                    all_chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        content=raw_chunk.strip(),
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_index=global_index,
                        start_char=start,
                        end_char=end,
                        page_number=page_num,
                        section=section_title,
                        metadata=chunk_meta,
                    ))
                    global_index += 1
                    offset = end
            global_offset += len(page_text)

        return self._apply_overlap(all_chunks, text)

    async def split_sentences(
        self,
        text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        pages = self._split_pages(text)
        all_chunks: List[Chunk] = []
        global_index = 0

        for page_num, page_text in pages:
            sentences = self._extract_sentences(page_text)
            current = ""
            current_start = 0
            pos = 0
            current_sentences: List[str] = []

            for sent in sentences:
                projected = len(current) + len(sent) + (1 if current else 0)
                if projected > self._config.chunk_size and current:
                    start = text.find(current[:40], max(0, current_start - 20))
                    if start == -1:
                        start = current_start
                    all_chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        content=current.strip(),
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_index=global_index,
                        start_char=start,
                        end_char=start + len(current),
                        page_number=page_num,
                        metadata=dict(meta),
                    ))
                    global_index += 1
                    overlap_sents = self._overlap_sentences(current_sentences)
                    current = " ".join(overlap_sents) + (" " + sent if overlap_sents else sent)
                    current_sentences = overlap_sents + [sent]
                    current_start = pos
                else:
                    current = (current + " " + sent).strip() if current else sent
                    current_sentences.append(sent)
                pos += len(sent) + 1

            if current.strip():
                start = text.find(current[:40], max(0, current_start - 20))
                if start == -1:
                    start = current_start
                all_chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    content=current.strip(),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=global_index,
                    start_char=start,
                    end_char=start + len(current),
                    page_number=page_num,
                    metadata=dict(meta),
                ))
                global_index += 1

        return all_chunks

    async def split_paragraphs(
        self,
        text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        pages = self._split_pages(text)
        all_chunks: List[Chunk] = []
        global_index = 0

        for page_num, page_text in pages:
            paragraphs = re.split(self._config.paragraph_separator, page_text)
            current = ""
            current_start = 0
            pos = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    pos += len(para) + 2
                    continue
                projected = len(current) + len(para) + (2 if current else 0)
                if projected > self._config.chunk_size and current:
                    start = text.find(current[:40], max(0, current_start - 20))
                    if start == -1:
                        start = current_start
                    section = self._detect_section(current)
                    all_chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        content=current.strip(),
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_index=global_index,
                        start_char=start,
                        end_char=start + len(current),
                        page_number=page_num,
                        section=section,
                        metadata=dict(meta),
                    ))
                    global_index += 1
                    overlap = current[-self._config.chunk_overlap:] if self._config.chunk_overlap else ""
                    current = (overlap + "\n\n" + para).strip() if overlap else para
                    current_start = pos
                else:
                    current = (current + "\n\n" + para).strip() if current else para
                    if not current_start:
                        current_start = pos
                pos += len(para) + 2

            if current.strip():
                start = text.find(current[:40], max(0, current_start - 20))
                if start == -1:
                    start = current_start
                all_chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    content=current.strip(),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=global_index,
                    start_char=start,
                    end_char=start + len(current),
                    page_number=page_num,
                    section=self._detect_section(current),
                    metadata=dict(meta),
                ))
                global_index += 1

        return all_chunks

    async def validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        valid: List[Chunk] = []
        seen_ids: set = set()
        for i, chunk in enumerate(chunks):
            if not chunk.content or not chunk.content.strip():
                continue
            if len(chunk.content) < self._config.min_chunk_size:
                if valid:
                    valid[-1].content += " " + chunk.content
                    valid[-1].end_char = chunk.end_char
                    valid[-1].token_estimate = max(1, len(valid[-1].content.split()) * 4 // 3)
                continue
            if len(chunk.content) > self._config.max_chunk_size:
                sub = self._hard_split(chunk)
                valid.extend(sub)
                continue
            if chunk.id in seen_ids:
                chunk.id = str(uuid.uuid4())
            seen_ids.add(chunk.id)
            chunk.chunk_index = len(valid)
            valid.append(chunk)
        return valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self._config.chunk_size or not separators:
            return [text] if text.strip() else []

        sep = separators[0]
        parts = re.split(re.escape(sep) if len(sep) > 1 else sep, text)

        if len(parts) == 1:
            return self._recursive_split(text, separators[1:])

        result: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self._config.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) > self._config.chunk_size:
                        result.extend(self._recursive_split(current, separators[1:]))
                    else:
                        result.append(current)
                current = part

        if current:
            if len(current) > self._config.chunk_size:
                result.extend(self._recursive_split(current, separators[1:]))
            else:
                result.append(current)

        return [r for r in result if r.strip()]

    def _extract_sentences(self, text: str) -> List[str]:
        pattern = "|".join(self._config.sentence_separators)
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_pages(self, text: str) -> List[tuple]:
        pages: List[tuple] = []
        pattern = self._config.page_break_pattern
        parts = re.split(pattern, text)
        for i, part in enumerate(parts):
            if part.strip():
                pages.append((i + 1, part))
        return pages if pages else [(1, text)]

    def _split_sections(self, text: str) -> List[tuple]:
        if not self._config.preserve_section_headers:
            return [(None, text)]
        lines = text.split("\n")
        sections: List[tuple] = []
        current_title: Optional[str] = None
        current_body: List[str] = []
        for line in lines:
            if re.match(self._config.section_header_pattern, line.strip()):
                if current_body:
                    sections.append((current_title, "\n".join(current_body)))
                current_title = line.strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body:
            sections.append((current_title, "\n".join(current_body)))
        return sections if sections else [(None, text)]

    def _detect_section(self, text: str) -> Optional[str]:
        for line in text.split("\n")[:3]:
            if re.match(self._config.section_header_pattern, line.strip()):
                return line.strip()
        return None

    def _apply_overlap(self, chunks: List[Chunk], full_text: str) -> List[Chunk]:
        if self._config.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks
        for i in range(1, len(chunks)):
            prev = chunks[i - 1].content
            overlap_text = prev[-self._config.chunk_overlap:]
            if not chunks[i].content.startswith(overlap_text):
                chunks[i].content = overlap_text + " " + chunks[i].content
                chunks[i].start_char = max(0, chunks[i].start_char - self._config.chunk_overlap)
                chunks[i].token_estimate = max(1, len(chunks[i].content.split()) * 4 // 3)
        return chunks

    def _overlap_sentences(self, sentences: List[str]) -> List[str]:
        total = 0
        overlap: List[str] = []
        for sent in reversed(sentences):
            total += len(sent)
            overlap.insert(0, sent)
            if total >= self._config.chunk_overlap:
                break
        return overlap

    def _hard_split(self, chunk: Chunk) -> List[Chunk]:
        text = chunk.content
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        parts: List[Chunk] = []
        i = 0
        idx = chunk.chunk_index
        while i < len(text):
            end = min(i + size, len(text))
            content = text[i:end].strip()
            if content:
                parts.append(Chunk(
                    id=str(uuid.uuid4()),
                    content=content,
                    doc_id=chunk.doc_id,
                    doc_name=chunk.doc_name,
                    chunk_index=idx,
                    start_char=chunk.start_char + i,
                    end_char=chunk.start_char + end,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    metadata=dict(chunk.metadata),
                ))
                idx += 1
            i += size - overlap
        return parts
