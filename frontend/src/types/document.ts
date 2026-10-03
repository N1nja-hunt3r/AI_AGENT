// frontend/src/types/document.ts

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum DocumentType {
  Pdf = "pdf",
  Word = "word",
  Excel = "excel",
  PowerPoint = "powerpoint",
  Markdown = "markdown",
  PlainText = "plain_text",
  Html = "html",
  Json = "json",
  Csv = "csv",
  Code = "code",
  Image = "image",
  Audio = "audio",
  Video = "video",
  Epub = "epub",
  Unknown = "unknown",
}

export enum DocumentStatus {
  Uploading = "uploading",
  Processing = "processing",
  Indexing = "indexing",
  Ready = "ready",
  Error = "error",
  Deleted = "deleted",
  Quarantined = "quarantined",
}

export enum DocumentVisibility {
  Private = "private",
  Shared = "shared",
  Public = "public",
  Workspace = "workspace",
}

export enum ChunkStrategy {
  FixedSize = "fixed_size",
  Semantic = "semantic",
  Paragraph = "paragraph",
  Sentence = "sentence",
  Page = "page",
  Custom = "custom",
}

export enum ProcessingStage {
  Upload = "upload",
  Validation = "validation",
  Extraction = "extraction",
  Chunking = "chunking",
  Embedding = "embedding",
  Indexing = "indexing",
  Complete = "complete",
}

// ─── Document chunk ───────────────────────────────────────────────────────────

export interface DocumentChunk {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  page_number?: number;
  section?: string;
  heading?: string;
  embedding_model?: string;
  similarity_score?: number;
  token_count: number;
  char_count: number;
  metadata?: Record<string, unknown>;
}

// ─── Processing info ──────────────────────────────────────────────────────────

export interface DocumentProcessingInfo {
  stage: ProcessingStage;
  progress_percent: number;
  started_at: number;
  completed_at?: number;
  duration_ms?: number;
  error?: string;
  pages_processed?: number;
  chunks_created?: number;
  tokens_processed?: number;
}

// ─── Document metadata ────────────────────────────────────────────────────────

export interface DocumentExtractedMetadata {
  title?: string;
  author?: string;
  subject?: string;
  keywords?: string[];
  created_date?: string;
  modified_date?: string;
  page_count?: number;
  word_count?: number;
  language?: string;
  encoding?: string;
}

// ─── Document ────────────────────────────────────────────────────────────────

export interface Document {
  id: string;
  name: string;
  original_name: string;
  type: DocumentType;
  mime_type: string;
  status: DocumentStatus;
  visibility: DocumentVisibility;
  size_bytes: number;
  url: string;
  preview_url?: string;
  thumbnail_url?: string;
  chunk_count: number;
  chunk_strategy: ChunkStrategy;
  chunk_size?: number;
  chunk_overlap?: number;
  extracted_metadata?: DocumentExtractedMetadata;
  processing?: DocumentProcessingInfo;
  tags: string[];
  description?: string;
  owner_id: string;
  workspace_id?: string;
  collection_id?: string;
  created_at: number;
  updated_at: number;
  indexed_at?: number;
  metadata?: Record<string, unknown>;
}

export type DocumentCreateInput = Pick<
  Document,
  | "name"
  | "type"
  | "visibility"
  | "chunk_strategy"
  | "chunk_size"
  | "chunk_overlap"
  | "tags"
  | "description"
  | "collection_id"
  | "metadata"
>;

export type DocumentUpdateInput = Partial<
  Pick<
    Document,
    | "name"
    | "visibility"
    | "tags"
    | "description"
    | "collection_id"
    | "metadata"
  >
>;

// ─── Collection ───────────────────────────────────────────────────────────────

export interface DocumentCollection {
  id: string;
  name: string;
  description?: string;
  document_count: number;
  total_size_bytes: number;
  visibility: DocumentVisibility;
  tags: string[];
  owner_id: string;
  created_at: number;
  updated_at: number;
  metadata?: Record<string, unknown>;
}

// ─── Search ───────────────────────────────────────────────────────────────────

export interface DocumentSearchRequest {
  query: string;
  types?: DocumentType[];
  collection_ids?: string[];
  tags?: string[];
  visibility?: DocumentVisibility[];
  semantic?: boolean;
  similarity_threshold?: number;
  include_chunks?: boolean;
  limit?: number;
  offset?: number;
}

export interface DocumentSearchResult {
  documents: Document[];
  chunks?: DocumentChunk[];
  total_count: number;
  query: string;
  search_time_ms: number;
  is_semantic: boolean;
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export interface DocumentUploadRequest {
  file: File;
  name?: string;
  visibility?: DocumentVisibility;
  chunk_strategy?: ChunkStrategy;
  chunk_size?: number;
  chunk_overlap?: number;
  collection_id?: string;
  tags?: string[];
  description?: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentUploadProgress {
  document_id: string;
  bytes_uploaded: number;
  total_bytes: number;
  percent: number;
  stage: ProcessingStage;
  is_complete: boolean;
  error?: string;
}