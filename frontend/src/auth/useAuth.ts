// frontend/src/auth/useAuth.ts

import { useContext, useCallback } from "react";
import {
  AuthContext,
  type AuthContextValue,
  type Permission,
  type UserRole,
  type LoginCredentials,
  type LoginResult,
  type LogoutReason,
  type RefreshResult,
  type UserProfile,
  type Session,
  type AuthStatus,
  type AuthError,
  type JWTPayload,
} from "./AuthContext";

// ─── Base hook ────────────────────────────────────────────────────────────────

export const useAuth = (): AuthContextValue => {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error(
      "[useAuth] must be used within <AuthProvider>. " +
        "Ensure AuthProvider wraps your component tree."
    );
  }
  return context;
};

// ─── Derived hooks ────────────────────────────────────────────────────────────

/**
 * Returns only the current auth status and boolean flags.
 * Use when you only need status without triggering re-renders
 * from other auth state changes.
 */
export const useAuthStatus = (): {
  status: AuthStatus;
  isAuthenticated: boolean;
  isLoading: boolean;
  isRefreshing: boolean;
} => {
  const { status, isAuthenticated, isLoading, isRefreshing } = useAuth();
  return { status, isAuthenticated, isLoading, isRefreshing };
};

/**
 * Returns the current authenticated user profile.
 * Returns null when unauthenticated.
 */
export const useCurrentUser = (): UserProfile | null => {
  const { user } = useAuth();
  return user;
};

/**
 * Returns the current active session or null.
 */
export const useSession = (): Session | null => {
  const { session } = useAuth();
  return session;
};

/**
 * Returns the current auth error state and a clear function.
 */
export const useAuthError = (): {
  error: AuthError | null;
  clearError: () => void;
} => {
  const { error, clearError } = useAuth();
  return { error, clearError };
};

/**
 * Returns auth action functions only.
 * Stable references — safe to use in effects and callbacks.
 */
export const useAuthActions = (): {
  login: (credentials: LoginCredentials) => Promise<LoginResult>;
  logout: (reason?: LogoutReason) => Promise<void>;
  refreshSession: () => Promise<RefreshResult>;
  updateLastActive: () => void;
} => {
  const { login, logout, refreshSession, updateLastActive } = useAuth();
  return { login, logout, refreshSession, updateLastActive };
};

/**
 * Returns token utilities.
 */
export const useAuthToken = (): {
  getAccessToken: () => string | null;
  getDecodedToken: () => JWTPayload | null;
  isTokenExpired: () => boolean;
  getTokenExpiresInMs: () => number | null;
} => {
  const {
    getAccessToken,
    getDecodedToken,
    isTokenExpired,
    getTokenExpiresInMs,
  } = useAuth();
  return { getAccessToken, getDecodedToken, isTokenExpired, getTokenExpiresInMs };
};

/**
 * Permission and role check hook.
 * All checks are performed against the current session.
 */
export const usePermissions = (): {
  hasRole: (role: UserRole) => boolean;
  hasAnyRole: (roles: UserRole[]) => boolean;
  hasAllRoles: (roles: UserRole[]) => boolean;
  hasPermission: (permission: Permission) => boolean;
  hasAnyPermission: (permissions: Permission[]) => boolean;
  hasAllPermissions: (permissions: Permission[]) => boolean;
  roles: UserRole[];
  permissions: Permission[];
} => {
  const {
    hasRole,
    hasAnyRole,
    hasAllRoles,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    user,
  } = useAuth();

  return {
    hasRole,
    hasAnyRole,
    hasAllRoles,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    roles: user?.roles ?? [],
    permissions: user?.permissions ?? [],
  };
};

/**
 * Convenience hook: returns whether the user has a specific role.
 */
export const useHasRole = (role: UserRole): boolean => {
  const { hasRole } = useAuth();
  return hasRole(role);
};

/**
 * Convenience hook: returns whether the user has a specific permission.
 */
export const useHasPermission = (permission: Permission): boolean => {
  const { hasPermission } = useAuth();
  return hasPermission(permission);
};

/**
 * Returns true if the user has ALL provided roles.
 */
export const useRequireAllRoles = (roles: UserRole[]): boolean => {
  const { hasAllRoles } = useAuth();
  return hasAllRoles(roles);
};

/**
 * Returns true if the user has ANY of the provided permissions.
 */
export const useRequireAnyPermission = (permissions: Permission[]): boolean => {
  const { hasAnyPermission } = useAuth();
  return hasAnyPermission(permissions);
};

/**
 * Returns the session ID string or null.
 */
export const useSessionId = (): string | null => {
  const { getSessionId } = useAuth();
  return getSessionId();
};

/**
 * Returns a stable callback that appends the current Bearer token
 * to a headers object. Useful for configuring API clients.
 */
export const useAuthHeaders = (): (() => Record<string, string>) => {
  const { getAccessToken } = useAuth();

  return useCallback((): Record<string, string> => {
    const token = getAccessToken();
    if (!token) return {};
    return { Authorization: `Bearer ${token}` };
  }, [getAccessToken]);
};

/**
 * Returns whether the current access token is expired.
 * Useful for pre-flight checks before API calls.
 */
export const useIsTokenExpired = (): boolean => {
  const { isTokenExpired } = useAuth();
  return isTokenExpired();
};