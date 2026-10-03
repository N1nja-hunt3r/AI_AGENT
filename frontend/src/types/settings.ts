// frontend/src/types/settings.ts

import type { MemoryType, EmbeddingModel } from "./memory";
import type { VoicePreferences } from "./voice";

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum Theme {
  Light = "light",
  Dark = "dark",
  System = "system",
}

export enum Language {
  En = "en",
  Es = "es",
  Fr = "fr",
  De = "de",
  Zh = "zh",
  Ja = "ja",
  Ko = "ko",
  Pt = "pt",
  Ar = "ar",
  Ru = "ru",
  Hi = "hi",
}

export enum ProviderType {
  OpenAI = "openai",
  Custom = "custom",
}

export enum ProviderAuthType {
  ApiKey = "api_key",
  OAuth = "oauth",
  ServiceAccount = "service_account",
  None = "none",
}

export enum ProviderStatus {
  Connected = "connected",
  Disconnected = "disconnected",
  Error = "error",
  Checking = "checking",
  RateLimited = "rate_limited",
}

export enum ModelCapability {
  Chat = "chat",
  Completion = "completion",
  Embedding = "embedding",
  Vision = "vision",
  Audio = "audio",
  FunctionCalling = "function_calling",
  Streaming = "streaming",
  FineTuning = "fine_tuning",
  CodeGeneration = "code_generation",
  Reasoning = "reasoning",
}

export enum FontSize {
  XSmall = "xs",
  Small = "sm",
  Medium = "md",
  Large = "lg",
  XLarge = "xl",
}

export enum FontFamily {
  System = "system",
  Serif = "serif",
  Mono = "mono",
  Inter = "inter",
}

// ─── Provider config ──────────────────────────────────────────────────────────

export interface ProviderConfig {
  id: string;
  name: string;
  type: ProviderType;
  auth_type: ProviderAuthType;
  api_key?: string;
  base_url?: string;
  organization_id?: string;
  project_id?: string;
  api_version?: string;
  region?: string;
  status: ProviderStatus;
  is_enabled: boolean;
  is_default: boolean;
  last_verified_at: number | null;
  error_message?: string;
  rate_limit_reset_at?: number;
  models_fetched_at?: number;
  metadata?: Record<string, unknown>;
}

export type ProviderCreateInput = Omit<
  ProviderConfig,
  "status" | "last_verified_at" | "error_message" | "rate_limit_reset_at" | "models_fetched_at"
>;

export type ProviderUpdateInput = Partial<
  Omit<ProviderConfig, "id" | "type" | "status" | "last_verified_at">
>;

// ─── Model config ─────────────────────────────────────────────────────────────

export interface ModelParameters {
  temperature: number;
  top_p: number;
  top_k?: number;
  max_tokens: number;
  frequency_penalty: number;
  presence_penalty: number;
  stop_sequences: string[];
  seed?: number;
}

export interface ModelPricing {
  input_cost_per_million_tokens: number;
  output_cost_per_million_tokens: number;
  cached_input_cost_per_million_tokens?: number;
  currency: "USD";
}

export interface ModelConfig {
  id: string;
  provider_id: string;
  display_name: string;
  description?: string;
  version?: string;
  context_window: number;
  max_output_tokens: number;
  capabilities: ModelCapability[];
  pricing?: ModelPricing;
  default_parameters?: Partial<ModelParameters>;
  is_enabled: boolean;
  is_default: boolean;
  is_deprecated?: boolean;
  deprecated_at?: number;
  metadata?: Record<string, unknown>;
}

// ─── Chat preferences ─────────────────────────────────────────────────────────

export interface ChatPreferences {
  send_on_enter: boolean;
  show_token_count: boolean;
  show_cost_estimate: boolean;
  show_model_badge: boolean;
  show_timestamps: boolean;
  auto_scroll: boolean;
  code_highlighting: boolean;
  render_markdown: boolean;
  compact_mode: boolean;
  enable_suggestions: boolean;
  show_thinking: boolean;
  default_system_prompt?: string;
}

// ─── Memory preferences ───────────────────────────────────────────────────────

export interface MemoryPreferences {
  auto_save: boolean;
  max_memories: number;
  default_memory_type: MemoryType;
  enable_embeddings: boolean;
  embedding_model: EmbeddingModel;
  retention_days: number | null;
  auto_archive_days: number | null;
  enable_memory_graph: boolean;
}

// ─── Notification preferences ─────────────────────────────────────────────────

export interface NotificationPreferences {
  desktop_enabled: boolean;
  sound_enabled: boolean;
  sound_volume: number;
  email_enabled: boolean;
  email_address?: string;
  agent_task_complete: boolean;
  agent_error: boolean;
  memory_limit_warning: boolean;
  cost_limit_warning: boolean;
  model_unavailable: boolean;
  stream_complete: boolean;
}

// ─── Accessibility preferences ────────────────────────────────────────────────

export interface AccessibilityPreferences {
  reduce_motion: boolean;
  high_contrast: boolean;
  font_size: FontSize;
  font_family: FontFamily;
  keyboard_shortcuts_enabled: boolean;
  screen_reader_optimized: boolean;
  focus_visible: boolean;
}

// ─── Developer settings ───────────────────────────────────────────────────────

export enum LogLevel {
  Debug = "debug",
  Info = "info",
  Warn = "warn",
  Error = "error",
}

export interface DeveloperSettings {
  enabled: boolean;
  show_token_counts: boolean;
  show_latency: boolean;
  show_raw_api_responses: boolean;
  enable_request_logging: boolean;
  log_level: LogLevel;
  enable_experimental_features: boolean;
  custom_system_prompt_prefix?: string;
  proxy_url?: string;
}

// ─── Cost limits ──────────────────────────────────────────────────────────────

export interface CostLimits {
  enabled: boolean;
  daily_limit_usd: number;
  monthly_limit_usd: number;
  per_request_limit_usd: number;
  alert_threshold_percent: number;
  current_daily_spend_usd: number;
  current_monthly_spend_usd: number;
  period_reset_at: number;
  hard_stop_on_limit: boolean;
}

// ─── Application settings ─────────────────────────────────────────────────────

export interface AppSettings {
  theme: Theme;
  language: Language;
  providers: Record<string, ProviderConfig>;
  models: Record<string, ModelConfig>;
  voice: VoicePreferences;
  chat: ChatPreferences;
  memory: MemoryPreferences;
  notifications: NotificationPreferences;
  accessibility: AccessibilityPreferences;
  developer: DeveloperSettings;
  cost_limits: CostLimits;
  updated_at: number;
  version: string;
}

export type AppSettingsUpdateInput = Partial<
  Omit<AppSettings, "updated_at" | "version">
>;