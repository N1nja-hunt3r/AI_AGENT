// frontend/src/constants/routes.ts

// ─── Route paths ──────────────────────────────────────────────────────────────

export const ROUTES = {
  // Root
  HOME: "/",
  ROOT: "/",

  // Auth
  AUTH: {
    LOGIN: "/auth/login",
    REGISTER: "/auth/register",
    FORGOT_PASSWORD: "/auth/forgot-password",
    RESET_PASSWORD: "/auth/reset-password",
    VERIFY_EMAIL: "/auth/verify-email",
    MFA_SETUP: "/auth/mfa/setup",
    MFA_VERIFY: "/auth/mfa/verify",
  },

  // Chat
  CHAT: {
    BASE: "/chat",
    NEW: "/chat/new",
    CONVERSATION: "/chat/:conversationId",
    SHARED: "/chat/shared/:shareId",
  },

  // Agents
  AGENTS: {
    BASE: "/agents",
    AGENT: "/agents/:agentId",
    TASKS: "/agents/tasks",
    TASK: "/agents/tasks/:taskId",
    CONFIG: "/agents/:agentId/config",
    LOGS: "/agents/:agentId/logs",
  },

  // Memory
  MEMORY: {
    BASE: "/memory",
    LIST: "/memory/list",
    DETAIL: "/memory/:memoryId",
    SEARCH: "/memory/search",
    GRAPH: "/memory/graph",
    TAGS: "/memory/tags",
    IMPORT: "/memory/import",
  },

  // Documents
  DOCUMENTS: {
    BASE: "/documents",
    UPLOAD: "/documents/upload",
    DETAIL: "/documents/:documentId",
    COLLECTIONS: "/documents/collections",
    COLLECTION: "/documents/collections/:collectionId",
    SEARCH: "/documents/search",
  },

  // Voice
  VOICE: {
    BASE: "/voice",
    SESSION: "/voice/session",
    SETTINGS: "/voice/settings",
    HISTORY: "/voice/history",
  },

  // Analytics
  ANALYTICS: {
    BASE: "/analytics",
    OVERVIEW: "/analytics/overview",
    COSTS: "/analytics/costs",
    TOKENS: "/analytics/tokens",
    AGENTS: "/analytics/agents",
    MODELS: "/analytics/models",
    EXPORT: "/analytics/export",
  },

  // Settings
  SETTINGS: {
    BASE: "/settings",
    GENERAL: "/settings/general",
    PROVIDERS: "/settings/providers",
    PROVIDER_DETAIL: "/settings/providers/:providerId",
    MODELS: "/settings/models",
    VOICE: "/settings/voice",
    NOTIFICATIONS: "/settings/notifications",
    ACCESSIBILITY: "/settings/accessibility",
    SECURITY: "/settings/security",
    DATA: "/settings/data",
    DEVELOPER: "/settings/developer",
    ABOUT: "/settings/about",
  },

  // Playground
  PLAYGROUND: {
    BASE: "/playground",
    CHAT: "/playground/chat",
    COMPLETIONS: "/playground/completions",
    EMBEDDINGS: "/playground/embeddings",
    TOOLS: "/playground/tools",
  },

  // Admin
  ADMIN: {
    BASE: "/admin",
    USERS: "/admin/users",
    USER: "/admin/users/:userId",
    ROLES: "/admin/roles",
    SYSTEM: "/admin/system",
    AUDIT_LOG: "/admin/audit-log",
    KEYS: "/admin/api-keys",
  },

  // Error
  ERRORS: {
    FORBIDDEN: "/error/403",
    NOT_FOUND: "/error/404",
    SERVER: "/error/500",
    MAINTENANCE: "/error/maintenance",
  },
} as const;

// ─── Route parameter patterns ─────────────────────────────────────────────────

export const ROUTE_PARAMS = {
  CONVERSATION_ID: "conversationId",
  MESSAGE_ID: "messageId",
  AGENT_ID: "agentId",
  TASK_ID: "taskId",
  MEMORY_ID: "memoryId",
  DOCUMENT_ID: "documentId",
  COLLECTION_ID: "collectionId",
  SHARE_ID: "shareId",
  PROVIDER_ID: "providerId",
  USER_ID: "userId",
} as const;

// ─── Route parameter regex ────────────────────────────────────────────────────

export const ROUTE_PARAM_PATTERNS: Record<string, RegExp> = {
  [ROUTE_PARAMS.CONVERSATION_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.MESSAGE_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.AGENT_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.TASK_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.MEMORY_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.DOCUMENT_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.COLLECTION_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.SHARE_ID]: /^[a-zA-Z0-9_-]{8,128}$/,
  [ROUTE_PARAMS.PROVIDER_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
  [ROUTE_PARAMS.USER_ID]: /^[a-zA-Z0-9_-]{1,64}$/,
} as const;

// ─── Navigation ───────────────────────────────────────────────────────────────

export type RoutePermission = string | string[] | null;
export type RouteIcon = string;
export type RouteBadge = string | number | null;

export interface RouteConfig {
  path: string;
  label: string;
  labelKey?: string;
  icon: RouteIcon;
  badge?: RouteBadge;
  permission?: RoutePermission;
  children?: RouteConfig[];
  isExternal?: boolean;
  isHidden?: boolean;
  requiresAuth?: boolean;
}

export const SIDEBAR_NAVIGATION: RouteConfig[] = [
  {
    path: ROUTES.CHAT.BASE,
    label: "Chat",
    labelKey: "nav.chat",
    icon: "chat-bubble",
    requiresAuth: true,
  },
  {
    path: ROUTES.AGENTS.BASE,
    label: "Agents",
    labelKey: "nav.agents",
    icon: "smart_toy",
    permission: "agent:read",
    requiresAuth: true,
  },
  {
    path: ROUTES.MEMORY.BASE,
    label: "Memory",
    labelKey: "nav.memory",
    icon: "psychology",
    permission: "memory:read",
    requiresAuth: true,
  },
  {
    path: ROUTES.DOCUMENTS.BASE,
    label: "Documents",
    labelKey: "nav.documents",
    icon: "description",
    permission: "memory:read",
    requiresAuth: true,
  },
  {
    path: ROUTES.VOICE.BASE,
    label: "Voice",
    labelKey: "nav.voice",
    icon: "mic",
    requiresAuth: true,
  },
  {
    path: ROUTES.PLAYGROUND.BASE,
    label: "Playground",
    labelKey: "nav.playground",
    icon: "science",
    permission: "admin:system",
    requiresAuth: true,
  },
  {
    path: ROUTES.ANALYTICS.BASE,
    label: "Analytics",
    labelKey: "nav.analytics",
    icon: "analytics",
    permission: "analytics:read",
    requiresAuth: true,
  },
  {
    path: ROUTES.SETTINGS.BASE,
    label: "Settings",
    labelKey: "nav.settings",
    icon: "settings",
    requiresAuth: true,
  },
];

export const ADMIN_NAVIGATION: RouteConfig[] = [
  {
    path: ROUTES.ADMIN.BASE,
    label: "Admin",
    labelKey: "nav.admin",
    icon: "admin_panel_settings",
    permission: ["admin:users", "admin:system"],
    requiresAuth: true,
    children: [
      {
        path: ROUTES.ADMIN.USERS,
        label: "Users",
        labelKey: "nav.admin.users",
        icon: "group",
        permission: "admin:users",
      },
      {
        path: ROUTES.ADMIN.ROLES,
        label: "Roles",
        labelKey: "nav.admin.roles",
        icon: "shield",
        permission: "admin:users",
      },
      {
        path: ROUTES.ADMIN.SYSTEM,
        label: "System",
        labelKey: "nav.admin.system",
        icon: "tune",
        permission: "admin:system",
      },
      {
        path: ROUTES.ADMIN.AUDIT_LOG,
        label: "Audit Log",
        labelKey: "nav.admin.audit_log",
        icon: "history",
        permission: "admin:system",
      },
      {
        path: ROUTES.ADMIN.KEYS,
        label: "API Keys",
        labelKey: "nav.admin.api_keys",
        icon: "key",
        permission: "admin:system",
      },
    ],
  },
];

// ─── Route helpers ────────────────────────────────────────────────────────────

/**
 * Builds a path with route parameters.
 *
 * @example
 * buildRoute(ROUTES.CHAT.CONVERSATION, { conversationId: "abc-123" })
 * // → "/chat/abc-123"
 */
export const buildRoute = (
  path: string,
  params: Record<string, string> = {}
): string => {
  let result = path;
  for (const [key, value] of Object.entries(params)) {
    result = result.replace(`:${key}`, encodeURIComponent(value));
  }
  return result;
};

/**
 * Extracts route parameters from a pathname against a pattern.
 *
 * @example
 * extractParams("/chat/abc-123", ROUTES.CHAT.CONVERSATION)
 * // → { conversationId: "abc-123" }
 */
export const extractParams = (
  pathname: string,
  pattern: string
): Record<string, string> => {
  const patternParts = pattern.split("/");
  const pathnameParts = pathname.split("/");
  const params: Record<string, string> = {};

  for (let i = 0; i < patternParts.length; i++) {
    const patternPart = patternParts[i];
    const pathnamePart = pathnameParts[i];

    if (patternPart?.startsWith(":")) {
      const paramName = patternPart.slice(1);
      if (pathnamePart) {
        params[paramName] = decodeURIComponent(pathnamePart);
      }
    }
  }

  return params;
};

/**
 * Checks if a pathname matches a route pattern.
 */
export const matchRoute = (pathname: string, pattern: string): boolean => {
  const patternParts = pattern.split("/");
  const pathnameParts = pathname.split("/");

  if (patternParts.length !== pathnameParts.length) return false;

  return patternParts.every((part, i) => {
    if (part.startsWith(":")) return true;
    return part === pathnameParts[i];
  });
};

/**
 * Returns the parent route for a given path.
 */
export const getParentRoute = (path: string): string => {
  const segments = path.split("/").filter(Boolean);
  segments.pop();
  return segments.length > 0 ? `/${segments.join("/")}` : "/";
};

/**
 * Generates breadcrumbs from a pathname.
 */
export const generateBreadcrumbs = (
  pathname: string
): Array<{ label: string; path: string; is_current: boolean }> => {
  const segments = pathname.split("/").filter(Boolean);
  const breadcrumbs: Array<{ label: string; path: string; is_current: boolean }> = [];

  for (let i = 0; i < segments.length; i++) {
    const path = `/${segments.slice(0, i + 1).join("/")}`;
    const segment = segments[i]!;
    const label = segment
      .replace(/-/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

    breadcrumbs.push({
      label,
      path,
      is_current: i === segments.length - 1,
    });
  }

  return breadcrumbs;
};

// ─── Public routes (no auth required) ────────────────────────────────────────

export const PUBLIC_ROUTES: string[] = [
  ROUTES.HOME,
  ROUTES.AUTH.LOGIN,
  ROUTES.AUTH.REGISTER,
  ROUTES.AUTH.FORGOT_PASSWORD,
  ROUTES.AUTH.RESET_PASSWORD,
  ROUTES.AUTH.VERIFY_EMAIL,
  ROUTES.ERRORS.NOT_FOUND,
  ROUTES.ERRORS.SERVER,
  ROUTES.ERRORS.MAINTENANCE,
];

/**
 * Checks if a route is public (no authentication required).
 */
export const isPublicRoute = (pathname: string): boolean => {
  return PUBLIC_ROUTES.some(
    (route) => pathname === route || pathname.startsWith(route + "/")
  );
};

/**
 * Redirect path after login.
 */
export const POST_LOGIN_REDIRECT = ROUTES.CHAT.BASE;

/**
 * Redirect path after logout.
 */
export const POST_LOGOUT_REDIRECT = ROUTES.AUTH.LOGIN;