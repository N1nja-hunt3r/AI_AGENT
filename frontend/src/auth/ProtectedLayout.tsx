// frontend/src/auth/ProtectedLayout.tsx

"use client";

import React, {
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { useAuth } from "./useAuth";
import type { Permission, UserRole, LogoutReason } from "./AuthContext";

// ─── Guard configuration ──────────────────────────────────────────────────────

export interface AccessGuard {
  /**
   * Require ALL of these roles to be present.
   */
  requiredRoles?: UserRole[];
  /**
   * Require at least ONE of these roles.
   */
  anyRole?: UserRole[];
  /**
   * Require ALL of these permissions.
   */
  requiredPermissions?: Permission[];
  /**
   * Require at least ONE of these permissions.
   */
  anyPermission?: Permission[];
  /**
   * Custom predicate for complex access logic.
   * Receives the current user roles and permissions.
   */
  custom?: (roles: UserRole[], permissions: Permission[]) => boolean;
}

// ─── Render slot types ────────────────────────────────────────────────────────

export interface ProtectedLayoutSlots {
  /**
   * Rendered while auth state is being resolved.
   */
  loading?: ReactNode;
  /**
   * Rendered when the user is not authenticated.
   * Should trigger redirect to login in production.
   */
  unauthenticated?: ReactNode;
  /**
   * Rendered when the user is authenticated but lacks access.
   */
  forbidden?: ReactNode;
}

// ─── Callback types ───────────────────────────────────────────────────────────

export interface ProtectedLayoutCallbacks {
  onUnauthenticated?: (reason?: LogoutReason) => void;
  onForbidden?: (
    user_roles: UserRole[],
    required: Partial<AccessGuard>
  ) => void;
  onAccessGranted?: () => void;
}

// ─── Props ────────────────────────────────────────────────────────────────────

export interface ProtectedLayoutProps
  extends ProtectedLayoutSlots,
    ProtectedLayoutCallbacks {
  children: ReactNode;
  guard?: AccessGuard;
}

// ─── Access evaluation ────────────────────────────────────────────────────────

const evaluateAccess = (
  guard: AccessGuard,
  roles: UserRole[],
  permissions: Permission[]
): boolean => {
  if (
    guard.requiredRoles &&
    !guard.requiredRoles.every((r) => roles.includes(r))
  ) {
    return false;
  }

  if (guard.anyRole && !guard.anyRole.some((r) => roles.includes(r))) {
    return false;
  }

  if (
    guard.requiredPermissions &&
    !guard.requiredPermissions.every((p) => permissions.includes(p))
  ) {
    return false;
  }

  if (
    guard.anyPermission &&
    !guard.anyPermission.some((p) => permissions.includes(p))
  ) {
    return false;
  }

  if (guard.custom && !guard.custom(roles, permissions)) {
    return false;
  }

  return true;
};

// ─── Default slot components ──────────────────────────────────────────────────

const DefaultLoadingSlot = (): React.JSX.Element => (
  <div
    role="status"
    aria-label="Verifying authentication"
    aria-live="polite"
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
    }}
  >
    <span className="sr-only">Verifying authentication…</span>
  </div>
);

const DefaultUnauthenticatedSlot = (): React.JSX.Element => (
  <div
    role="alert"
    aria-label="Authentication required"
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
    }}
  >
    <span className="sr-only">Authentication required. Redirecting…</span>
  </div>
);

const DefaultForbiddenSlot = (): React.JSX.Element => (
  <div
    role="alert"
    aria-label="Access denied"
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
    }}
  >
    <span className="sr-only">Access denied.</span>
  </div>
);

// ─── ProtectedLayout ──────────────────────────────────────────────────────────

export const ProtectedLayout = ({
  children,
  guard,
  loading = <DefaultLoadingSlot />,
  unauthenticated = <DefaultUnauthenticatedSlot />,
  forbidden = <DefaultForbiddenSlot />,
  onUnauthenticated,
  onForbidden,
  onAccessGranted,
}: ProtectedLayoutProps): React.JSX.Element => {
  const {
    isAuthenticated,
    isLoading,
    isRefreshing,
    user,
  } = useAuth();

  const accessGrantedFiredRef = useRef(false);

  const roles: UserRole[] = user?.roles ?? [];
  const permissions: Permission[] = user?.permissions ?? [];

  const isAccessGranted =
    isAuthenticated &&
    (guard ? evaluateAccess(guard, roles, permissions) : true);

  useEffect(() => {
    if (isLoading || isRefreshing) return;

    if (!isAuthenticated) {
      accessGrantedFiredRef.current = false;
      onUnauthenticated?.();
      return;
    }

    if (!isAccessGranted) {
      accessGrantedFiredRef.current = false;
      onForbidden?.(roles, guard ?? {});
      return;
    }

    if (!accessGrantedFiredRef.current) {
      accessGrantedFiredRef.current = true;
      onAccessGranted?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isAccessGranted, isLoading, isRefreshing]);

  if (isLoading || isRefreshing) {
    return <>{loading}</>;
  }

  if (!isAuthenticated) {
    return <>{unauthenticated}</>;
  }

  if (!isAccessGranted) {
    return <>{forbidden}</>;
  }

  return <>{children}</>;
};

// ─── Conditional render helper ────────────────────────────────────────────────

export interface AuthGateProps {
  /**
   * Access guard configuration.
   */
  guard: AccessGuard;
  /**
   * Rendered when access is granted.
   */
  children: ReactNode;
  /**
   * Rendered when access is denied. Optional.
   */
  fallback?: ReactNode;
}

/**
 * Lightweight inline component that conditionally renders children
 * based on the user's roles and permissions.
 * Does NOT handle unauthenticated state — use within ProtectedLayout.
 *
 * @example
 * <AuthGate guard={{ requiredPermissions: ["admin:users"] }}>
 *   <DeleteUserButton />
 * </AuthGate>
 */
export const AuthGate = ({
  guard,
  children,
  fallback = null,
}: AuthGateProps): React.JSX.Element => {
  const { user, isAuthenticated } = useAuth();

  const roles: UserRole[] = user?.roles ?? [];
  const permissions: Permission[] = user?.permissions ?? [];

  const granted =
    isAuthenticated && evaluateAccess(guard, roles, permissions);

  return <>{granted ? children : fallback}</>;
};