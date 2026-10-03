// frontend/src/types/chat.ts

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum MessageRole {
  User = "user",
  Assistant = "assistant",
  System = "system",
  Tool = "tool",
}

export enum MessageStatus {
  Pending = "pending",
  Streaming = "streaming",
  Complete = "complete",
  Error = "error",
  Cancelled = "cancelled",
}

export enum ConversationStatus {
  Active = "active",
  Archived = "archived",
  Deleted = "deleted",
}

export enum FinishReason {
  Stop = "stop",
  Length = "length",
  ToolCalls = "tool_calls",
  ContentFilter = "content_filter",
  Error = "error",
  Cancelled = "cancelled",
}

export enum ContentPartType {
  Text = "text",
  ImageUrl = "image_url",
  ToolUse = "tool_use",
  ToolResult = "tool_result",
  Audio = "audio",
  File = "file",
}

export enum ChatEventType {
  TokenChunk = "token_chunk",
  StreamStart = "stream_start",
  StreamEnd = "stream_end",
  StreamError = "stream_error",
  ToolCallStart = "tool_call_start",
  ToolCallResult = "tool_call_result",
  ThinkingChunk = "thinking_chunk",
  MessageCreated = "message_created",
  MessageUpdated = "message_updated",
  ConversationUpdated = "conversation_updated",
}

// ─── Content parts ────────────────────────────────────────────────────────────

export interface TextContentPart {
  type: ContentPartType.Text;
  text: string;
}

export interface ImageUrlContentPart {
  type: ContentPartType.ImageUrl;
  image_url: {
    url: string;
    detail?: "low" | "high" | "auto";
    alt_text?: string;
  };
}

export interface ToolUseContentPart {
  type: ContentPartType.ToolUse;
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResultContentPart {
  type: ContentPartType.ToolResult;
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

export interface AudioContentPart {
  type: ContentPartType.Audio;
  audio_url: string;
  transcript?: string;
  duration_ms?: number;
  format?: string;
}

export interface FileContentPart {
  type: ContentPartType.File;
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  url?: string;
}

export type ContentPart =
  | TextContentPart
  | ImageUrlContentPart
  | ToolUseContentPart
  | ToolResultContentPart
  | AudioContentPart
  | FileContentPart;

// ─── Tool types ───────────────────────────────────────────────────────────────

export interface ToolFunction {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  strict?: boolean;
}

export interface Tool {
  type: "function";
  function: ToolFunction;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolCallResult {
  tool_call_id: string;
  role: MessageRole.Tool;
  content: string;
  is_error?: boolean;
}

// ─── Token usage ──────────────────────────────────────────────────────────────

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
}

export interface CostEstimate {
  input_cost_usd: number;
  output_cost_usd: number;
  total_cost_usd: number;
  currency: "USD";
}

// ─── Message ──────────────────────────────────────────────────────────────────

export interface MessageAttachment {
  id: string;
  type: "image" | "file" | "audio" | "video";
  name: string;
  size_bytes: number;
  mime_type: string;
  url: string;
  preview_url?: string;
  width?: number;
  height?: number;
  duration_ms?: number;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string | ContentPart[];
  status: MessageStatus;
  model?: string;
  provider?: string;
  token_usage?: TokenUsage;
  cost?: CostEstimate;
  finish_reason?: FinishReason;
  tool_calls?: ToolCall[];
  parent_message_id?: string;
  attachments?: MessageAttachment[];
  thinking?: string;
  created_at: number;
  updated_at: number;
  metadata?: Record<string, unknown>;
}

export type MessageCreateInput = Omit<
  Message,
  "id" | "created_at" | "updated_at" | "status" | "token_usage" | "cost" | "finish_reason"
> & {
  status?: MessageStatus;
};

// ─── Conversation ─────────────────────────────────────────────────────────────

export interface ConversationParameters {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop_sequences?: string[];
  seed?: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  model: string;
  provider: string;
  status: ConversationStatus;
  message_count: number;
  last_message_preview?: string;
  last_message_role?: MessageRole;
  total_tokens?: number;
  total_cost_usd?: number;
  pinned: boolean;
  tags: string[];
  agent_id?: string;
  created_at: number;
  updated_at: number;
  metadata?: Record<string, unknown>;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
  system_prompt?: string;
  parameters?: ConversationParameters;
  tools?: Tool[];
  context_window_used?: number;
  context_window_total?: number;
}

export type ConversationCreateInput = Pick<
  Conversation,
  | "title"
  | "model"
  | "provider"
  | "system_prompt"
  | "parameters"
  | "tools"
  | "tags"
  | "agent_id"
  | "metadata"
>;

export type ConversationUpdateInput = Partial<
  Pick<
    Conversation,
    | "title"
    | "status"
    | "system_prompt"
    | "model"
    | "provider"
    | "parameters"
    | "tools"
    | "pinned"
    | "tags"
    | "metadata"
  >
>;

// ─── Streaming ────────────────────────────────────────────────────────────────

export interface StreamStartEvent {
  type: ChatEventType.StreamStart;
  conversation_id: string;
  message_id: string;
  model: string;
  provider: string;
  started_at: number;
}

export interface TokenChunkEvent {
  type: ChatEventType.TokenChunk;
  conversation_id: string;
  message_id: string;
  token: string;
  index: number;
  is_final: boolean;
}

export interface ThinkingChunkEvent {
  type: ChatEventType.ThinkingChunk;
  conversation_id: string;
  message_id: string;
  thinking: string;
  index: number;
}

export interface StreamEndEvent {
  type: ChatEventType.StreamEnd;
  conversation_id: string;
  message_id: string;
  finish_reason: FinishReason;
  token_usage: TokenUsage;
  cost?: CostEstimate;
  duration_ms: number;
}

export interface StreamErrorEvent {
  type: ChatEventType.StreamError;
  conversation_id: string;
  message_id: string;
  code: string;
  message: string;
  recoverable: boolean;
  timestamp: number;
}

export interface ToolCallStartEvent {
  type: ChatEventType.ToolCallStart;
  conversation_id: string;
  message_id: string;
  tool_call_id: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
}

export interface ToolCallResultEvent {
  type: ChatEventType.ToolCallResult;
  conversation_id: string;
  message_id: string;
  tool_call_id: string;
  tool_name: string;
  result: unknown;
  is_error: boolean;
  duration_ms: number;
}

export type ChatStreamEvent =
  | StreamStartEvent
  | TokenChunkEvent
  | ThinkingChunkEvent
  | StreamEndEvent
  | StreamErrorEvent
  | ToolCallStartEvent
  | ToolCallResultEvent;

// ─── Send request ─────────────────────────────────────────────────────────────

export interface SendMessageRequest {
  conversation_id: string;
  content: string | ContentPart[];
  model: string;
  provider: string;
  system_prompt?: string;
  parameters?: ConversationParameters;
  tools?: Tool[];
  attachments?: MessageAttachment[];
  parent_message_id?: string;
  stream: boolean;
}

export interface RegenerateMessageRequest {
  conversation_id: string;
  message_id: string;
  model?: string;
  provider?: string;
  parameters?: ConversationParameters;
  stream?: boolean;
}