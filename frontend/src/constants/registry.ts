// frontend/src/constants/registry.ts

// ─── Routes ───────────────────────────────────────────────────────────────────

export {
  ROUTES,
  ROUTE_PARAMS,
  ROUTE_PARAM_PATTERNS,
  SIDEBAR_NAVIGATION,
  ADMIN_NAVIGATION,
  PUBLIC_ROUTES,
  POST_LOGIN_REDIRECT,
  POST_LOGOUT_REDIRECT,
  buildRoute,
  extractParams,
  matchRoute,
  getParentRoute,
  generateBreadcrumbs,
  isPublicRoute,
} from "./routes";

export type {
  RoutePermission,
  RouteConfig,
} from "./routes";

// ─── Models ──────────────────────────────────────────────────────────────────

export {
  ASPIRE_MODELS,
  MODEL_ROLES,
  MODEL_DISPLAY_NAMES,
  MODEL_CAPABILITIES,
  MODEL_CONTEXT_WINDOWS,
  VISION_MODELS,
  EMBEDDING_MODELS,
  AUDIO_MODELS,
  REASONING_MODELS,
  CHAT_MODELS,
  DEFAULT_MODEL,
  DEFAULT_PROVIDER,
  DEFAULT_REASONING_MODEL,
  DEFAULT_CODE_MODEL,
  DEFAULT_VISION_MODEL,
  DEFAULT_AUDIO_MODEL,
  DEFAULT_EMBEDDING_MODEL,
} from "./models";

// ─── Providers ───────────────────────────────────────────────────────────────

export {
  PROVIDERS,
  PROVIDER_IDS,
  PROVIDER_LABELS,
  PROVIDER_ORDER,
  getProvider,
  getApiKeyProviders,
  getLocalProviders,
  getOrgSupportingProviders,
  getRegionSupportingProviders,
  isValidProviderUrl,
} from "./providers";

export type { ProviderMeta } from "./providers";

// ─── Theme ───────────────────────────────────────────────────────────────────

export {
  CSS_VAR,
  THEME_MODE,
  THEME_STORAGE_KEY,
  THEME_CLASSES,
  LIGHT_THEME_VALUES,
  DARK_THEME_VALUES,
  RADIUS_VALUES,
  Z_INDEX,
  BREAKPOINTS,
  BREAKPOINT_QUERIES,
  FONT_SIZE_SCALES,
  FONT_FAMILIES,
  TRANSITIONS,
  REDUCED_MOTION_QUERY,
  COLOR_SCHEME_MEDIA_QUERY,
} from "./theme";

// ─── API ─────────────────────────────────────────────────────────────────────

export {
  API_VERSION,
  API_BASE_URL,
  WS_BASE_URL,
  API_ENDPOINTS,
  WS_ENDPOINTS,
  REQUEST_DEFAULTS,
  NON_RETRYABLE_STATUS_CODES,
  HTTP_HEADERS,
  CONTENT_TYPES,
  UPLOAD_DEFAULTS,
  PAGINATION_DEFAULTS,
  QUERY_KEYS,
  IS_SERVER,
  IS_CLIENT,
  IS_DEVELOPMENT,
  IS_PRODUCTION,
  IS_TEST,
} from "./api";