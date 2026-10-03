// frontend/src/types/registry.ts

// ─── Chat ─────────────────────────────────────────────────────────────────────

export type {
  TextContentPart,
  ImageUrlContentPart,
  ToolUseContentPart,
  ToolResultContentPart,
  AudioContentPart,
  FileContentPart,
  ContentPart,
  ToolFunction,
  Tool,
  ToolCall,
  ToolCallResult,
  TokenUsage,
  CostEstimate,
  MessageAttachment,
  Message,
  MessageCreateInput,
  ConversationParameters,
  ConversationSummary,
  Conversation,
  ConversationCreateInput,
  ConversationUpdateInput,
  StreamStartEvent,
  TokenChunkEvent,
  ThinkingChunkEvent,
  StreamEndEvent,
  StreamErrorEvent,
  ToolCallStartEvent,
  ToolCallResultEvent,
  ChatStreamEvent,
  SendMessageRequest,
  RegenerateMessageRequest,
} from "./chat";

export {
  MessageRole,
  MessageStatus,
  ConversationStatus,
  FinishReason,
  ContentPartType,
  ChatEventType,
} from "./chat";

// ─── Agent ────────────────────────────────────────────────────────────────────

export type {
  AgentTool,
  ToolExecution,
  AgentStep,
  AgentTask,
  AgentTaskCreateInput,
  HealthCheck,
  AgentHealth,
  LatencyPercentiles,
  LatencyMetrics,
  AgentMetrics,
  AgentResourceUsage,
  AgentConfig,
  Agent,
  AgentUpdateInput,
  AgentEvent,
  OrchestrationSession,
  AgentPlan,
  PlanStep,
} from "./agent";

export {
  AgentId,
  AgentStatus,
  TaskStatus,
  TaskPriority,
  AgentStepType,
  ToolExecutionStatus,
  HealthStatus,
  AgentEventType,
} from "./agent";

// ─── Memory ───────────────────────────────────────────────────────────────────

export type {
  MemorySource,
  MemoryRelation,
  MemoryTag,
  MemoryEmbedding,
  Memory,
  MemoryCreateInput,
  MemoryUpdateInput,
  MemorySearchRequest,
  MemorySearchResult,
  MemoryFilters,
  MemoryGraphNode,
  MemoryGraphEdge,
  MemoryGraph,
  MemoryStatistics,
} from "./memory";

export {
  MemoryType,
  MemoryStatus,
  MemoryImportance,
  MemorySourceType,
  MemoryRelationType,
  MemorySortField,
  EmbeddingModel,
} from "./memory";

// ─── Document ─────────────────────────────────────────────────────────────────

export type {
  DocumentChunk,
  DocumentProcessingInfo,
  DocumentExtractedMetadata,
  Document,
  DocumentCreateInput,
  DocumentUpdateInput,
  DocumentCollection,
  DocumentSearchRequest,
  DocumentSearchResult,
  DocumentUploadRequest,
  DocumentUploadProgress,
} from "./document";

export {
  DocumentType,
  DocumentStatus,
  DocumentVisibility,
  ChunkStrategy,
  ProcessingStage,
} from "./document";

// ─── Voice ────────────────────────────────────────────────────────────────────

export type {
  Voice,
  VoiceSessionConfig,
  VoiceSession,
  WordTimestamp,
  Transcription,
  SynthesisRequest,
  SynthesisResult,
  SynthesisChunk,
  VoiceActivityEvent,
  VoicePreferences,
} from "./voice";

export {
  AudioFormat,
  AudioSampleRate,
  VoiceProvider,
  VoiceGender,
  VoiceSessionStatus,
  VoiceActivityState,
  VoiceSessionEndReason,
  TranscriptionLanguage,
} from "./voice";

// ─── Monitoring ───────────────────────────────────────────────────────────────

export type {
  MetricDataPoint,
  MetricSeries,
  Metric,
  SystemMetricSnapshot,
  AgentMetricSnapshot,
  ProviderMetricSnapshot,
  CostEvent,
  CostBreakdown,
  CostSummary,
  AlertThreshold,
  MonitoringAlert,
  LogEntry,
  LogFilter,
  MonitoringDashboard,
} from "./monitoring";

export {
  MetricUnit,
  MetricTrend,
  AlertSeverity,
  AlertCategory,
  AlertStatus,
  LogLevel,
  TimeWindow,
  CostPeriod,
} from "./monitoring";

// ─── Settings ─────────────────────────────────────────────────────────────────

export type {
  ProviderConfig,
  ProviderCreateInput,
  ProviderUpdateInput,
  ModelParameters,
  ModelPricing,
  ModelConfig,
  ChatPreferences,
  MemoryPreferences,
  NotificationPreferences,
  AccessibilityPreferences,
  DeveloperSettings,
  CostLimits,
  AppSettings,
  AppSettingsUpdateInput,
} from "./settings";

export {
  Theme,
  Language,
  ProviderType,
  ProviderAuthType,
  ProviderStatus,
  ModelCapability,
  FontSize,
  FontFamily,
} from "./settings";

// ─── API ─────────────────────────────────────────────────────────────────────

export type {
  PaginationParams,
  SortParams,
  PaginationMeta,
  ApiResponse,
  ApiPaginatedResponse,
  ApiValidationError,
  ApiErrorResponse,
  ApiResult,
  RequestConfig,
  RateLimitInfo,
  SSEEvent,
  StreamChunk,
  WebSocketFrame,
  WebSocketError,
  UploadProgress,
  BatchRequest,
  BatchResponseItem,
  BatchResponse,
  ApiHealthCheck,
  ListRequest,
  MutationResult,
  OptimisticUpdate,
} from "./api";

export {
  HttpMethod,
  HttpStatus,
  ApiErrorCode,
  ContentType,
} from "./api";

export type {
  WebSocketFrameType,
  StreamChunkType,
} from "./api";

// ─── Utility types ────────────────────────────────────────────────────────────

/** Makes specified keys required on a type */
export type RequireKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

/** Makes specified keys optional on a type */
export type PartialKeys<T, K extends keyof T> = Omit<T, K> &
  Partial<Pick<T, K>>;

/** Deep partial — makes all nested properties optional */
export type DeepPartial<T> = T extends object
  ? { [K in keyof T]?: DeepPartial<T[K]> }
  : T;

/** Branded type for nominal typing */
export type Brand<TBase, TBrand extends string> = TBase & {
  readonly __brand: TBrand;
};

/** Nullable type alias */
export type Nullable<T> = T | null;

/** Optional type alias */
export type Optional<T> = T | undefined;

/** Extract the resolved type of a Promise */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/** Discriminated union helper — extracts a union member by type field */
export type ExtractByType<
  TUnion,
  TField extends keyof TUnion,
  TValue extends TUnion[TField],
> = Extract<TUnion, Record<TField, TValue>>;

/** Make all properties in T non-nullable */
export type NonNullableProperties<T> = {
  [K in keyof T]: NonNullable<T[K]>;
};

/** ID brand types for type-safe IDs */
export type ConversationId = Brand<string, "ConversationId">;
export type MessageId = Brand<string, "MessageId">;
export type MemoryId = Brand<string, "MemoryId">;
export type DocumentId = Brand<string, "DocumentId">;
export type AgentTaskId = Brand<string, "AgentTaskId">;
export type UserId = Brand<string, "UserId">;
export type SessionId = Brand<string, "SessionId">;
export type ProviderId = Brand<string, "ProviderId">;
export type ModelId = Brand<string, "ModelId">;
export type VoiceSessionId = Brand<string, "VoiceSessionId">;

/** Timestamp in milliseconds since epoch */
export type TimestampMs = Brand<number, "TimestampMs">;

/** Sort direction */
export type SortDirection = "asc" | "desc";

/** Generic key-value record */
export type KVRecord<TValue = unknown> = Record<string, TValue>;

/** Callback function types */
export type VoidCallback = () => void;
export type AsyncVoidCallback = () => Promise<void>;
export type ErrorCallback = (error: Error) => void;
export type AsyncErrorCallback = (error: Error) => Promise<void>;