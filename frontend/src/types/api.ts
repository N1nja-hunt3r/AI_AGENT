// frontend/src/types/api.ts

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum HttpMethod {
  Get = "GET",
  Post = "POST",
  Put = "PUT",
  Patch = "PATCH",
  Delete = "DELETE",
  Head = "HEAD",
  Options = "OPTIONS",
}

export enum HttpStatus {
  Ok = 200,
  Created = 201,
  Accepted = 202,
  NoContent = 204,
  MovedPermanently = 301,
  NotModified = 304,
  BadRequest = 400,
  Unauthorized = 401,
  Forbidden = 403,
  NotFound = 404,
  MethodNotAllowed = 405,
  Conflict = 409,
  Gone = 410,
  UnprocessableEntity = 422,
  TooManyRequests = 429,
  InternalServerError = 500,
  BadGateway = 502,
  ServiceUnavailable = 503,
  GatewayTimeout = 504,
}

export enum ApiErrorCode {
  // Auth
  AuthRequired = "AUTH_REQUIRED",
  AuthInvalid = "AUTH_INVALID",
  AuthExpired = "AUTH_EXPIRED",
  AuthInsufficient = "AUTH_INSUFFICIENT",
  // Validation
  ValidationFailed = "VALIDATION_FAILED",
  InvalidInput = "INVALID_INPUT",
  MissingRequired = "MISSING_REQUIRED",
  // Resources
  NotFound = "NOT_FOUND",
  AlreadyExists = "ALREADY_EXISTS",
  Conflict = "CONFLICT",
  Gone = "GONE",
  // Limits
  RateLimited = "RATE_LIMITED",
  QuotaExceeded = "QUOTA_EXCEEDED",
  CostLimitExceeded = "COST_LIMIT_EXCEEDED",
  // Provider
  ProviderError = "PROVIDER_ERROR",
  ProviderUnavailable = "PROVIDER_UNAVAILABLE",
  ProviderAuthFailed = "PROVIDER_AUTH_FAILED",
  ModelUnavailable = "MODEL_UNAVAILABLE",
  // Network
  NetworkError = "NETWORK_ERROR",
  Timeout = "TIMEOUT",
  // Server
  InternalError = "INTERNAL_ERROR",
  NotImplemented = "NOT_IMPLEMENTED",
  ServiceUnavailable = "SERVICE_UNAVAILABLE",
  // Unknown
  Unknown = "UNKNOWN",
}

export enum ContentType {
  Json = "application/json",
  FormData = "multipart/form-data",
  Stream = "text/event-stream",
  OctetStream = "application/octet-stream",
  TextPlain = "text/plain",
}

// ─── Pagination ───────────────────────────────────────────────────────────────

export interface PaginationParams {
  page?: number;
  page_size?: number;
  cursor?: string;
}

export interface SortParams {
  sort_field?: string;
  sort_direction?: "asc" | "desc";
}

export interface PaginationMeta {
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
  has_next_page: boolean;
  has_prev_page: boolean;
  next_cursor?: string;
  prev_cursor?: string;
}

// ─── API Response ─────────────────────────────────────────────────────────────

export interface ApiResponse<TData = unknown> {
  success: true;
  data: TData;
  meta?: Record<string, unknown>;
  request_id: string;
  timestamp: number;
}

export interface ApiPaginatedResponse<TData = unknown> {
  success: true;
  data: TData[];
  pagination: PaginationMeta;
  meta?: Record<string, unknown>;
  request_id: string;
  timestamp: number;
}

// ─── API Error ────────────────────────────────────────────────────────────────

export interface ApiValidationError {
  field: string;
  message: string;
  code: string;
  value?: unknown;
}

export interface ApiErrorResponse {
  success: false;
  error: {
    code: ApiErrorCode;
    message: string;
    details?: Record<string, unknown>;
    validation_errors?: ApiValidationError[];
    stack?: string;
    provider_error?: {
      code: string;
      message: string;
      type?: string;
      param?: string;
    };
  };
  request_id: string;
  timestamp: number;
}

export type ApiResult<TData = unknown> =
  | ApiResponse<TData>
  | ApiErrorResponse;

// ─── Request config ───────────────────────────────────────────────────────────

export interface RequestConfig {
  method: HttpMethod;
  url: string;
  params?: Record<string, string | number | boolean | null | undefined>;
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  timeout_ms?: number;
  retry?: boolean;
  cache?: RequestCache;
}

// ─── Rate limit ───────────────────────────────────────────────────────────────

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_at: number;
  retry_after_ms?: number;
  policy?: string;
}

// ─── SSE / Streaming ─────────────────────────────────────────────────────────

export interface SSEEvent<TData = unknown> {
  id?: string;
  event?: string;
  data: TData;
  retry?: number;
}

export type StreamChunkType =
  | "text"
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "metadata"
  | "error"
  | "done";

export interface StreamChunk<TData = unknown> {
  type: StreamChunkType;
  data: TData;
  index: number;
  timestamp: number;
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

export type WebSocketFrameType =
  | "ping"
  | "pong"
  | "auth"
  | "auth_ack"
  | "subscribe"
  | "unsubscribe"
  | "event"
  | "error"
  | "ack";

export interface WebSocketFrame<TPayload = unknown> {
  id?: string;
  type: WebSocketFrameType;
  channel?: string;
  payload?: TPayload;
  timestamp: number;
  version: string;
}

export interface WebSocketError {
  code: string;
  message: string;
  recoverable: boolean;
  request_id?: string;
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export interface UploadProgress {
  loaded: number;
  total: number;
  percent: number;
  speed_bytes_per_sec?: number;
  estimated_remaining_ms?: number;
}

// ─── Batch ───────────────────────────────────────────────────────────────────

export interface BatchRequest<TInput = unknown> {
  id: string;
  input: TInput;
}

export interface BatchResponseItem<TOutput = unknown> {
  id: string;
  success: boolean;
  data?: TOutput;
  error?: ApiErrorResponse["error"];
}

export interface BatchResponse<TOutput = unknown> {
  results: BatchResponseItem<TOutput>[];
  total: number;
  succeeded: number;
  failed: number;
  request_id: string;
  timestamp: number;
}

// ─── Health check ─────────────────────────────────────────────────────────────

export interface ApiHealthCheck {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  uptime_ms: number;
  services: Record<
    string,
    { status: "ok" | "warn" | "error"; latency_ms?: number; message?: string }
  >;
  timestamp: number;
}

// ─── Generic list request ─────────────────────────────────────────────────────

export interface ListRequest extends PaginationParams, SortParams {
  search?: string;
  filters?: Record<string, string | string[] | number | boolean | null>;
}

// ─── Mutation result ──────────────────────────────────────────────────────────

export interface MutationResult<TData = unknown> {
  data: TData | null;
  error: ApiErrorResponse["error"] | null;
  is_loading: boolean;
  is_success: boolean;
  is_error: boolean;
  request_id: string | null;
}

// ─── Optimistic update ────────────────────────────────────────────────────────

export interface OptimisticUpdate<TData = unknown> {
  key: string;
  previous: TData;
  next: TData;
  timestamp: number;
}