export { default as api, ApiError, BASE_URL, getAuthToken, clearAuthToken } from "@/api/client";
export type { ApiErrorPayload } from "@/api/client";

export {
  MessageRole, MessageStatus,
  sendMessage, streamMessage, getConversation, clearConversation,
} from "@/api/chat";
export type {
  Message, Conversation, SendMessagePayload, StreamMessagePayload,
  SendMessageResponse, StreamChunk, TokenUsage, GetConversationResponse,
  ClearConversationResponse,
} from "@/api/chat";

export {
  createMemory, listMemories, getMemory, updateMemory, deleteMemory,
  searchMemories, purgeMemories,
} from "@/api/memory";
export type {
  FrontendMemoryItem, CreateMemoryPayload, UpdateMemoryPayload,
  SearchMemoriesPayload, PurgeRequest, MemoryResponse, MemoryListResponse,
  SearchResult, SearchResponse, DeleteMemoryResponse, PurgeResponse,
} from "@/api/memory";

export {
  DocumentStatus, DocumentType, ChunkStrategy,
  uploadDocument, listDocuments, getDocument, deleteDocument,
  queryDocuments, createCollection, listCollections, deleteCollection,
} from "@/api/rag";
export type {
  Document, DocumentChunk, RetrievedChunk, Collection,
  UploadDocumentPayload, QueryPayload, CreateCollectionPayload,
  ListDocumentsParams, DocumentResponse, DocumentListResponse,
  QueryResponse, CollectionResponse, CollectionListResponse,
  DeleteDocumentResponse,
} from "@/api/rag";

export {
  listAgents, getAgent, spawnAgent, shutdownAgent, delegateTask,
} from "@/api/agents";
export type {
  AgentInfo, AgentListResponse, SpawnAgentPayload, SpawnAgentResponse,
  ShutdownResponse, DelegateTaskPayload, DelegateTaskResponse,
} from "@/api/agents";

export {
  listVoices, textToSpeech, speechToText,
  createVoiceSession, getVoiceSession, deleteVoiceSession,
} from "@/api/voice";
export type {
  VoiceItem, VoicesResponse, TTSRequest, TTSResponse, STTResponse,
  CreateSessionRequest, VoiceSession, CreateSessionResponse,
  GetSessionResponse, DeleteSessionResponse,
} from "@/api/voice";

export {
  mouseClick, mouseMove, keyboardType, keyboardKey,
  browserNavigate, browserAction,
  takeScreenshot, terminalExec, checkApproval,
} from "@/api/computer";
export type {
  MouseClickRequest, MouseMoveRequest, KeyboardTypeRequest,
  KeyboardKeyRequest, BrowserNavigateRequest, BrowserActionRequest,
  TerminalExecRequest, ActionResult, Screenshot, ScreenResolution,
  ActionResultResponse, ApprovalCheckResponse,
} from "@/api/computer";

export {
  TaskStatus,
  createTask, listTasks, getTask, deleteTask, pauseTask, resumeTask,
} from "@/api/automation";
export type {
  AutomationTaskInfo, CreateTaskPayload, CreateTaskResponse,
  TaskListResponse, DeleteTaskResponse, TaskActionResponse,
} from "@/api/automation";

export {
  AlertSeverity,
  getMetrics, getDashboard, getProfilerSnapshot,
  listAlerts, resolveAlert, queryLogs,
} from "@/api/monitoring";
export type {
  DashboardSummary, ProfilerSnapshot, Alert, AlertListResponse,
  AlertResolveResponse, LogEntry, LogQueryResponse, QueryLogsParams,
} from "@/api/monitoring";

export {
  getSettings, updateSettings, resetSettings, getFeatureFlags,
  listApiKeys, createApiKey, deleteApiKey,
} from "@/api/settings";
export type {
  ModelSettings, MemorySettings, RAGSettings, FeatureFlagSettings,
  UserSettings, UpdateSettingsRequest,
  ApiKey, ListApiKeysResponse, CreateApiKeyResponse, CreateApiKeyRequest,
} from "@/api/settings";
