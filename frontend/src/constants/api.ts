export const API_VERSION = "v1";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const WS_BASE_URL =
  import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";

export const API_ENDPOINTS = {
  AUTH: {
    SIGNUP: "/api/v1/auth/signup",
    LOGIN: "/api/v1/auth/login",
    LOGOUT: "/api/v1/auth/logout",
    REFRESH: "/api/v1/auth/refresh",
    ME: "/api/v1/auth/me",
    OAUTH: (provider: string) => `/api/v1/auth/oauth/${provider}`,
  },

  AGENTS: {
    LIST: "/api/v1/agents",
    GET: "/api/v1/agents/:agentId",
    SPAWN: "/api/v1/agents/spawn",
    SHUTDOWN: (id: string) => `/api/v1/agents/${id}/shutdown`,
    DELEGATE: "/api/v1/agents/delegate",
  },

  CHAT: {
    SEND: "/api/v1/chat/message",
    STREAM: "/api/v1/chat/stream",
    CONVERSATION: (id: string) => `/api/v1/chat/conversations/${id}`,
    CLEAR: (id: string) => `/api/v1/chat/conversations/${id}/clear`,
    PLAN: "/api/v1/chat/plan",
    EXECUTE: "/api/v1/chat/execute",
  },

  MEMORY: {
    BASE: "/api/v1/memories",
    BY_ID: (id: string) => `/api/v1/memories/${id}`,
    SEARCH: "/api/v1/memories/search",
    PURGE: "/api/v1/memories/purge",
  },

  SETTINGS: {
    GET: "/api/v1/settings",
    UPDATE: "/api/v1/settings",
    RESET: "/api/v1/settings/reset",
    FEATURES: "/api/v1/settings/features",
  },

  API_KEYS: {
    BASE: "/api/v1/api-keys",
    BY_ID: (id: string) => `/api/v1/api-keys/${id}`,
  },

  VOICE: {
    TTS: "/api/v1/voice/tts",
    STT: "/api/v1/voice/stt",
    VOICES: "/api/v1/voice/voices",
    SESSIONS: "/api/v1/voice/sessions",
    SESSION_BY_ID: (id: string) => `/api/v1/voice/sessions/${id}`,
  },

  COMPUTER: {
    MOUSE_CLICK: "/api/v1/computer/mouse/click",
    MOUSE_MOVE: "/api/v1/computer/mouse/move",
    KEYBOARD_TYPE: "/api/v1/computer/keyboard/type",
    KEYBOARD_KEY: "/api/v1/computer/keyboard/key",
    BROWSER_NAVIGATE: "/api/v1/computer/browser/navigate",
    BROWSER_ACTION: "/api/v1/computer/browser/action",
    SCREENSHOT: (id: string) => `/api/v1/computer/screenshot/${id}`,
    TERMINAL_EXEC: "/api/v1/computer/terminal/exec",
    APPROVALS_CHECK: "/api/v1/computer/approvals/check",
  },

  RAG: {
    DOCUMENTS: "/api/v1/rag/documents",
    DOCUMENT_BY_ID: (id: string) => `/api/v1/rag/documents/${id}`,
    QUERY: "/api/v1/rag/query",
    COLLECTIONS: "/api/v1/rag/collections",
    COLLECTION_BY_ID: (id: string) => `/api/v1/rag/collections/${id}`,
  },

  AUTOMATION: {
    TASKS: "/api/v1/automation/tasks",
    TASK_BY_ID: (id: string) => `/api/v1/automation/tasks/${id}`,
    PAUSE: (id: string) => `/api/v1/automation/tasks/${id}/pause`,
    RESUME: (id: string) => `/api/v1/automation/tasks/${id}/resume`,
  },

  MONITORING: {
    METRICS: "/api/v1/monitoring/metrics",
    DASHBOARD: "/api/v1/monitoring/dashboard",
    LOGS: "/api/v1/monitoring/logs",
    PROFILER: "/api/v1/monitoring/profiler",
    ALERTS: "/api/v1/monitoring/alerts",
    RESOLVE_ALERT: (id: string) => `/api/v1/monitoring/alerts/${id}/resolve`,
  },

  ARTIFACTS: {
    UPLOAD: "/api/v1/artifacts/upload",
    LIST: "/api/v1/artifacts",
    GET: (id: string) => `/api/v1/artifacts/${id}`,
    DOWNLOAD: (id: string) => `/api/v1/artifacts/${id}/download`,
    DELETE: (id: string) => `/api/v1/artifacts/${id}`,
  },

  SECURITY: {
    CHECK_PERMISSION: "/api/v1/security/permissions/check",
    APPROVALS: "/api/v1/security/approvals",
    APPROVAL_DECISION: (id: string) => `/api/v1/security/approvals/${id}/decision`,
    AUDIT: "/api/v1/security/audit",
    ROLES: "/api/v1/security/roles",
    USER_ROLE: (id: string) => `/api/v1/security/roles/${id}`,
  },

  TOOLS: {
    LIST: "/api/v1/tools",
    GET: (name: string) => `/api/v1/tools/${name}`,
    EXECUTE: "/api/v1/tools/execute",
    HEALTH: "/api/v1/tools/health/check",
  },

  SYSTEM: {
    HEALTH: "/health",
    LIVE: "/health/live",
    READY: "/health/ready",
    METRICS: "/metrics",
  },
} as const;

export const WS_ENDPOINTS = {
  CHAT: (sessionId?: string) => `${WS_BASE_URL}/ws/chat/${sessionId ?? "{session_id}"}`,
  AGENT: (agentId: string) => `${WS_BASE_URL}/ws/agents/${agentId}`,
  TASK: (taskId: string) => `${WS_BASE_URL}/ws/tasks/${taskId}`,
} as const;

export const REQUEST_DEFAULTS = {
  TIMEOUT_MS: 30_000,
  STREAM_TIMEOUT_MS: 120_000,
  MAX_RETRY_ATTEMPTS: 3,
  RETRY_DELAY_MS: 1_000,
  RETRY_MAX_DELAY_MS: 30_000,
  RETRY_BACKOFF_MULTIPLIER: 2,
} as const;

export const NON_RETRYABLE_STATUS_CODES: ReadonlySet<number> = new Set([
  400, 401, 403, 404, 405, 409, 410, 422,
]);

export const HTTP_HEADERS = {
  CONTENT_TYPE: "Content-Type",
  AUTHORIZATION: "Authorization",
  ACCEPT: "Accept",
  X_REQUEST_ID: "X-Request-ID",
  X_CORRELATION_ID: "X-Correlation-ID",
  X_API_VERSION: "X-API-Version",
  CACHE_CONTROL: "Cache-Control",
  IF_NONE_MATCH: "If-None-Match",
  ETAG: "ETag",
} as const;

export const CONTENT_TYPES = {
  JSON: "application/json",
  FORM_DATA: "multipart/form-data",
  STREAM: "text/event-stream",
  OCTET_STREAM: "application/octet-stream",
  TEXT_PLAIN: "text/plain",
  TEXT_HTML: "text/html",
  TEXT_CSV: "text/csv",
} as const;

export const UPLOAD_DEFAULTS = {
  MAX_FILE_SIZE_BYTES: 50 * 1024 * 1024,
  MAX_IMAGE_SIZE_BYTES: 20 * 1024 * 1024,
  MAX_AUDIO_SIZE_BYTES: 25 * 1024 * 1024,
  ALLOWED_IMAGE_TYPES: [
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml",
  ],
  ALLOWED_DOCUMENT_TYPES: [
    "application/pdf", "text/plain", "text/markdown", "text/csv",
    "application/json", "text/html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/x-python", "text/javascript", "application/typescript",
  ],
  ALLOWED_AUDIO_TYPES: [
    "audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/webm", "audio/flac",
  ],
  CHUNK_SIZE_BYTES: 1024 * 1024,
} as const;

export const PAGINATION_DEFAULTS = {
  PAGE_SIZE: 25,
  PAGE_SIZE_OPTIONS: [10, 25, 50, 100],
  MIN_PAGE_SIZE: 5,
  MAX_PAGE_SIZE: 200,
} as const;

export const QUERY_KEYS = {
  AUTH: { ME: ["auth", "me"], SESSION: ["auth", "session"] },
  CONVERSATIONS: {
    LIST: ["conversations"],
    DETAIL: (id: string) => ["conversations", id],
    MESSAGES: (id: string) => ["conversations", id, "messages"],
  },
  AGENTS: {
    LIST: ["agents"],
    DETAIL: (id: string) => ["agents", id],
    TASKS: (id: string) => ["agents", id, "tasks"],
    TASK: (id: string) => ["agents", "tasks", id],
  },
  MEMORY: {
    LIST: ["memory"],
    DETAIL: (id: string) => ["memory", id],
    SEARCH: ["memory", "search"],
  },
  DOCUMENTS: {
    LIST: ["documents"],
    DETAIL: (id: string) => ["documents", id],
    CHUNKS: (id: string) => ["documents", id, "chunks"],
    COLLECTIONS: ["documents", "collections"],
    COLLECTION: (id: string) => ["documents", "collections", id],
  },
  SETTINGS: { ALL: ["settings"] },
  SYSTEM: { HEALTH: ["system", "health"] },
} as const;

export const IS_SERVER = typeof window === "undefined";
export const IS_CLIENT = !IS_SERVER;
export const IS_DEVELOPMENT = import.meta.env.DEV;
export const IS_PRODUCTION = import.meta.env.PROD;
export const IS_TEST = import.meta.env.MODE === "test";
