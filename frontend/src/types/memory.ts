// frontend/src/types/memory.ts

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum MemoryType {
  Episodic = "episodic",
  Semantic = "semantic",
  Procedural = "procedural",
  Working = "working",
  Autobiographical = "autobiographical",
  Declarative = "declarative",
}

export enum MemoryStatus {
  Active = "active",
  Archived = "archived",
  Deprecated = "deprecated",
  Pending = "pending",
  Deleted = "deleted",
}

export enum MemoryImportance {
  Low = "low",
  Medium = "medium",
  High = "high",
  Critical = "critical",
}

export enum MemorySourceType {
  Conversation = "conversation",
  Document = "document",
  Agent = "agent",
  User = "user",
  System = "system",
  Web = "web",
  Tool = "tool",
}

export enum MemoryRelationType {
  Supports = "supports",
  Contradicts = "contradicts",
  Extends = "extends",
  Supersedes = "supersedes",
  Related = "related",
  DerivedFrom = "derived_from",
  PartOf = "part_of",
}

export enum MemorySortField {
  CreatedAt = "created_at",
  UpdatedAt = "updated_at",
  AccessedAt = "accessed_at",
  Importance = "importance",
  AccessCount = "access_count",
  SimilarityScore = "similarity_score",
  RelevanceScore = "relevance_score",
}

export enum EmbeddingModel {
  NvEmbedV1 = "nvidia/nv-embed-v1",
}

// ─── Source ───────────────────────────────────────────────────────────────────

export interface MemorySource {
  type: MemorySourceType;
  id: string;
  label?: string;
  url?: string;
  timestamp?: number;
}

// ─── Relation ─────────────────────────────────────────────────────────────────

export interface MemoryRelation {
  memory_id: string;
  relation_type: MemoryRelationType;
  weight: number;
  description?: string;
  created_at: number;
}

// ─── Tag ─────────────────────────────────────────────────────────────────────

export interface MemoryTag {
  id: string;
  label: string;
  color?: string;
  count?: number;
}

// ─── Embedding ───────────────────────────────────────────────────────────────

export interface MemoryEmbedding {
  model: EmbeddingModel;
  vector: number[];
  dimensions: number;
  created_at: number;
}

// ─── Memory ───────────────────────────────────────────────────────────────────

export interface Memory {
  id: string;
  type: MemoryType;
  status: MemoryStatus;
  importance: MemoryImportance;
  content: string;
  summary?: string;
  tags: MemoryTag[];
  source: MemorySource;
  relations: MemoryRelation[];
  embedding?: MemoryEmbedding;
  similarity_score?: number;
  relevance_score?: number;
  access_count: number;
  agent_id?: string;
  conversation_id?: string;
  user_id?: string;
  expires_at?: number;
  created_at: number;
  updated_at: number;
  accessed_at: number;
  metadata?: Record<string, unknown>;
}

export type MemoryCreateInput = Omit<
  Memory,
  | "id"
  | "embedding"
  | "similarity_score"
  | "relevance_score"
  | "access_count"
  | "created_at"
  | "updated_at"
  | "accessed_at"
>;

export type MemoryUpdateInput = Partial<
  Pick<
    Memory,
    | "content"
    | "summary"
    | "tags"
    | "status"
    | "importance"
    | "expires_at"
    | "metadata"
  >
>;

// ─── Search ───────────────────────────────────────────────────────────────────

export interface MemorySearchRequest {
  query: string;
  type?: MemoryType[];
  status?: MemoryStatus[];
  importance?: MemoryImportance[];
  tags?: string[];
  agent_id?: string;
  conversation_id?: string;
  source_types?: MemorySourceType[];
  date_from?: number;
  date_to?: number;
  limit?: number;
  offset?: number;
  semantic?: boolean;
  similarity_threshold?: number;
  include_archived?: boolean;
}

export interface MemorySearchResult {
  memories: Memory[];
  total_count: number;
  query: string;
  is_semantic: boolean;
  search_time_ms: number;
  embedding_model?: EmbeddingModel;
}

// ─── Filters ─────────────────────────────────────────────────────────────────

export interface MemoryFilters {
  types: MemoryType[];
  statuses: MemoryStatus[];
  importance_levels: MemoryImportance[];
  tags: string[];
  source_types: MemorySourceType[];
  agent_id: string | null;
  conversation_id: string | null;
  date_from: number | null;
  date_to: number | null;
  has_relations: boolean | null;
  has_embedding: boolean | null;
}

// ─── Graph ────────────────────────────────────────────────────────────────────

export interface MemoryGraphNode {
  id: string;
  label: string;
  type: MemoryType;
  importance: MemoryImportance;
  access_count: number;
}

export interface MemoryGraphEdge {
  source_id: string;
  target_id: string;
  relation_type: MemoryRelationType;
  weight: number;
}

export interface MemoryGraph {
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
  computed_at: number;
}

// ─── Statistics ───────────────────────────────────────────────────────────────

export interface MemoryStatistics {
  total_count: number;
  by_type: Record<MemoryType, number>;
  by_status: Record<MemoryStatus, number>;
  by_importance: Record<MemoryImportance, number>;
  total_size_bytes: number;
  avg_access_count: number;
  oldest_memory_at: number;
  newest_memory_at: number;
  computed_at: number;
}