// ── Service registry ──────────────────────────────────────────────────────────
// Single entry-point for all application services.
// Import from "@/services/registry" anywhere in the application.

// ── HTTP client ───────────────────────────────────────────────────────────────
export { default as api } from "@/services/api";
export type { ApiErrorPayload } from "@/services/api";
export type { ApiError } from "@/services/api";

// ── Chat service ──────────────────────────────────────────────────────────────
export { default as ChatService } from "@/services/chat.service";
export type {
  Message,
  Conversation,
  SendMessagePayload,
  SendMessageResponse,
  StreamMessagePayload,
  StreamChunk,
  GetConversationResponse,
  ClearConversationResponse,
  TokenUsage,
} from "@/services/chat.service";
export { MessageRole, MessageStatus } from "@/services/chat.service";

// ── Memory service ────────────────────────────────────────────────────────────
export { default as MemoryService } from "@/services/memory.service";
export type {
  Memory,
  CreateMemoryPayload,
  UpdateMemoryPayload,
  SearchMemoryPayload,
  MemorySearchResult,
  MemoryListResponse,
  MemorySearchResponse,
} from "@/services/memory.service";
export { MemoryType, MemoryStatus } from "@/services/memory.service";

// ── RAG service ───────────────────────────────────────────────────────────────
export { default as RagService } from "@/services/rag.service";
export type {
  Document,
  DocumentChunk,
  RetrievedChunk,
  Collection,
  UploadDocumentPayload,
  QueryPayload,
  QueryResponse,
  DocumentListResponse,
  CollectionListResponse,
} from "@/services/rag.service";
export {
  DocumentStatus,
  DocumentType,
  ChunkStrategy,
} from "@/services/rag.service";

// ── Agent service ─────────────────────────────────────────────────────────────
export { default as AgentService } from "@/services/agent.service";
export type {
  AgentInfo,
  SpawnAgentPayload,
  DelegateTaskPayload,
  ListAgentsParams,
  SpawnAgentResponse,
  AgentListResponse,
  ShutdownResponse,
  DelegateTaskResponse,
} from "@/services/agent.service";
export {
  AgentStatus,
  AgentType,
} from "@/services/agent.service";

// ── Voice service ─────────────────────────────────────────────────────────────
export { default as VoiceService } from "@/services/voice.service";
export type {
  Voice,
  Transcription,
  WordTimestamp,
  VoiceSession,
  TextToSpeechPayload,
  SpeechToTextPayload,
  TextToSpeechResponse,
  SpeechToTextResponse,
  VoiceListResponse,
} from "@/services/voice.service";
export {
  VoiceProvider,
  AudioFormat,
  TranscriptionStatus,
} from "@/services/voice.service";

// ── Computer service ──────────────────────────────────────────────────────────
export { default as ComputerService } from "@/services/computer.service";
export type {
  MouseClickRequest,
  MouseMoveRequest,
  KeyboardTypeRequest,
  KeyboardKeyRequest,
  BrowserNavigateRequest,
  BrowserActionRequest,
  TerminalExecRequest,
  ActionResponse,
  ScreenshotResponse,
  TerminalExecResponse,
  ApprovalRequiredResponse,
} from "@/services/computer.service";

// ── Automation service ────────────────────────────────────────────────────────
export { default as AutomationService } from "@/services/automation.service";
export type {
  TaskStep,
  TaskTrigger,
  Task,
  TaskRun,
  StepResult,
  CreateTaskPayload,
  ListTasksParams,
  TaskResponse,
  TaskListResponse,
  DeleteTaskResponse,
  TaskActionResponse,
} from "@/services/automation.service";
export {
  TaskStatus,
  TriggerType,
  StepType,
  RunStatus,
} from "@/services/automation.service";

// ── Monitoring service ────────────────────────────────────────────────────────
export { default as MonitoringService } from "@/services/monitoring.service";
export type {
  DashboardSummary,
  ProfilerSnapshot,
  Alert,
  AlertListResponse,
  AlertResolveResponse,
  LogEntry,
  LogListResponse,
} from "@/services/monitoring.service";
export {
  AlertSeverity,
} from "@/services/monitoring.service";

// ── Settings service ──────────────────────────────────────────────────────────
export { default as SettingsService } from "@/services/settings.service";
export type {
  ModelSettings,
  MemorySettings,
  RAGSettings,
  FeatureFlagSettings,
  ApiKey,
  CreateApiKeyRequest,
  ListApiKeysResponse,
  CreateApiKeyResponse,
  UserSettings,
  UpdateSettingsRequest,
} from "@/services/settings.service";