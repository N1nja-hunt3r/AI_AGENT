// frontend/src/auth/registry.ts

export {
  AuthContext,
  type AuthContextValue,
  type AuthStatus,
  type AuthError,
  type AuthErrorCode,
  type JWTPayload,
  type TokenPair,
  type LoginCredentials,
  type LoginResult,
  type RefreshResult,
  type LogoutReason,
  type Session,
  type UserProfile,
  type UserRole,
  type Permission,
} from "./AuthContext";

export {
  AuthProvider,
  type AuthProviderProps,
} from "./AuthProvider";

export {
  useAuth,
  useAuthStatus,
  useCurrentUser,
  useSession,
  useAuthError,
  useAuthActions,
  useAuthToken,
  usePermissions,
  useHasRole,
  useHasPermission,
  useRequireAllRoles,
  useRequireAnyPermission,
  useSessionId,
  useAuthHeaders,
  useIsTokenExpired,
} from "./useAuth";

export {
  ProtectedLayout,
  AuthGate,
  type ProtectedLayoutProps,
  type ProtectedLayoutSlots,
  type ProtectedLayoutCallbacks,
  type AccessGuard,
  type AuthGateProps,
} from "./ProtectedLayout";

export {
  withAuth,
  type WithAuthOptions,
} from "./withAuth";

// ─── Role constants ───────────────────────────────────────────────────────────

export const ROLES = {
  SUPERADMIN: "superadmin",
  ADMIN: "admin",
  DEVELOPER: "developer",
  ANALYST: "analyst",
  VIEWER: "viewer",
} as const satisfies Record<string, string>;

export type RoleConstant = (typeof ROLES)[keyof typeof ROLES];

// ─── Permission constants ─────────────────────────────────────────────────────

export const PERMISSIONS = {
  CHAT_READ: "chat:read",
  CHAT_WRITE: "chat:write",
  CHAT_DELETE: "chat:delete",
  MEMORY_READ: "memory:read",
  MEMORY_WRITE: "memory:write",
  MEMORY_DELETE: "memory:delete",
  AGENT_READ: "agent:read",
  AGENT_WRITE: "agent:write",
  AGENT_EXECUTE: "agent:execute",
  SETTINGS_READ: "settings:read",
  SETTINGS_WRITE: "settings:write",
  PROVIDER_READ: "provider:read",
  PROVIDER_WRITE: "provider:write",
  ANALYTICS_READ: "analytics:read",
  ADMIN_USERS: "admin:users",
  ADMIN_SYSTEM: "admin:system",
} as const satisfies Record<string, string>;

export type PermissionConstant =
  (typeof PERMISSIONS)[keyof typeof PERMISSIONS];

// ─── Preset guards ────────────────────────────────────────────────────────────

import type { AccessGuard } from "./ProtectedLayout";

export const GUARDS = {
  /** Any authenticated user. */
  authenticated: {} satisfies AccessGuard,

  /** Admin or superadmin. */
  adminOnly: {
    anyRole: [ROLES.ADMIN, ROLES.SUPERADMIN],
  } satisfies AccessGuard,

  /** Superadmin only. */
  superadminOnly: {
    requiredRoles: [ROLES.SUPERADMIN],
  } satisfies AccessGuard,

  /** Can read and write chat. */
  chatUser: {
    requiredPermissions: [PERMISSIONS.CHAT_READ, PERMISSIONS.CHAT_WRITE],
  } satisfies AccessGuard,

  /** Can execute agents. */
  agentExecutor: {
    requiredPermissions: [PERMISSIONS.AGENT_EXECUTE],
  } satisfies AccessGuard,

  /** Can manage providers and settings. */
  settingsManager: {
    requiredPermissions: [
      PERMISSIONS.SETTINGS_WRITE,
      PERMISSIONS.PROVIDER_WRITE,
    ],
  } satisfies AccessGuard,

  /** Can manage users. */
  userManager: {
    requiredPermissions: [PERMISSIONS.ADMIN_USERS],
  } satisfies AccessGuard,

  /** Developer or above. */
  developer: {
    anyRole: [ROLES.DEVELOPER, ROLES.ADMIN, ROLES.SUPERADMIN],
  } satisfies AccessGuard,

  /** Analytics read access. */
  analyticsViewer: {
    requiredPermissions: [PERMISSIONS.ANALYTICS_READ],
  } satisfies AccessGuard,
} as const;

export type GuardPreset = keyof typeof GUARDS;