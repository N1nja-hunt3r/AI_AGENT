// frontend/src/auth/AuthContext.ts

import { createContext } from "react";

// ─── Enums & Literals ─────────────────────────────────────────────────────────

export type UserRole =
  | "superadmin"
  | "admin"
  | "developer"
  | "analyst"
  | "viewer"
  | (string & NonNullable<unknown>);

export type Permission =
  | "chat:read"
  | "chat:write"
  | "chat:delete"
  | "memory:read"
  | "memory:write"
  | "memory:delete"
  | "agent:read"
  | "agent:write"
  | "agent:execute"
  | "settings:read"
  | "settings:write"
  | "provider:read"
  | "provider:write"
  | "analytics:read"
  | "admin:users"
  | "admin:system"
  | (string & NonNullable<unknown>);

export type AuthStatus =
  | "idle"
  | "loading"
  | "authenticated"
  | "unauthenticated"
  | "refreshing"
  | "error";

// ─── Token types ──────────────────────────────────────────────────────────────

export interface JWTPayload {
  sub: string;
  iat: number;
  exp: number;
  jti: string;
  roles: UserRole[];
  permissions: Permission[];
  email: string;
  session_id: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: "Bearer";
  expires_in: number;
  expires_at: number;
}

// ─── User types ───────────────────────────────────────────────────────────────

export interface UserProfile {
  id: string;
  email: string;
  display_name: string;
  avatar_url?: string;
  roles: UserRole[];
  permissions: Permission[];
  created_at: number;
  last_login_at: number;
  metadata?: Record<string, unknown>;
}

// ─── Session types ────────────────────────────────────────────────────────────

export interface Session {
  id: string;
  user: UserProfile;
  tokens: TokenPair;
  created_at: number;
  last_active_at: number;
  expires_at: number;
  device_fingerprint?: string;
}

// ─── Auth error types ─────────────────────────────────────────────────────────

export type AuthErrorCode =
  | "INVALID_CREDENTIALS"
  | "TOKEN_EXPIRED"
  | "TOKEN_INVALID"
  | "TOKEN_REFRESH_FAILED"
  | "SESSION_EXPIRED"
  | "SESSION_NOT_FOUND"
  | "INSUFFICIENT_PERMISSIONS"
  | "ACCOUNT_LOCKED"
  | "ACCOUNT_DISABLED"
  | "NETWORK_ERROR"
  | "UNKNOWN_ERROR";

export interface AuthError {
  code: AuthErrorCode;
  message: string;
  timestamp: number;
  recoverable: boolean;
}

// ─── Auth operations ──────────────────────────────────────────────────────────

export interface LoginCredentials {
  email: string;
  password: string;
  remember_me?: boolean;
  mfa_code?: string;
}

export interface LoginResult {
  session: Session;
  requires_mfa?: boolean;
  mfa_methods?: string[];
}

export interface RefreshResult {
  tokens: TokenPair;
  session: Session;
}

// ─── Context shape ────────────────────────────────────────────────────────────

export interface AuthContextValue {
  // State
  status: AuthStatus;
  session: Session | null;
  user: UserProfile | null;
  error: AuthError | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isRefreshing: boolean;

  // Auth operations
  login: (credentials: LoginCredentials) => Promise<LoginResult>;
  logout: (reason?: LogoutReason) => Promise<void>;
  refreshSession: () => Promise<RefreshResult>;
  clearError: () => void;

  // Token utilities
  getAccessToken: () => string | null;
  getDecodedToken: () => JWTPayload | null;
  isTokenExpired: () => boolean;
  getTokenExpiresInMs: () => number | null;

  // Permission utilities
  hasRole: (role: UserRole) => boolean;
  hasAnyRole: (roles: UserRole[]) => boolean;
  hasAllRoles: (roles: UserRole[]) => boolean;
  hasPermission: (permission: Permission) => boolean;
  hasAnyPermission: (permissions: Permission[]) => boolean;
  hasAllPermissions: (permissions: Permission[]) => boolean;

  // Session utilities
  getSessionId: () => string | null;
  updateLastActive: () => void;
}

export type LogoutReason =
  | "user_initiated"
  | "session_expired"
  | "token_expired"
  | "token_refresh_failed"
  | "forced"
  | "idle_timeout"
  | "concurrent_session";

export const AuthContext = createContext<AuthContextValue | null>(null);

AuthContext.displayName = "AuthContext";