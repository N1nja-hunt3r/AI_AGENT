import React, { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";

// ── Types ─────────────────────────────────────────────────────────────────────
interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
}

interface ProtectedRouteProps {
  children: React.ReactNode;
  /**
   * Path to redirect unauthenticated users.
   * Defaults to "/login".
   */
  redirectTo?: string;
  /**
   * Optional override for authentication check.
   * Useful for testing or custom auth strategies.
   */
  authCheck?: () => Promise<boolean>;
}

// ── Placeholder auth helpers ──────────────────────────────────────────────────

/**
 * Reads the stored JWT token from localStorage.
 * Replace this with a proper token validation service when ready.
 */
const getStoredToken = (): string | null =>
  localStorage.getItem("auth_token");

/**
 * Validates the current session.
 *
 * TODO: Replace with a real API call, e.g.:
 *   const response = await fetch("/api/auth/verify", {
 *     headers: { Authorization: `Bearer ${token}` },
 *   });
 *   return response.ok;
 */
const defaultAuthCheck = async (): Promise<boolean> => {
  const token = getStoredToken();

  if (!token) {
    return false;
  }

  // ── Placeholder: accept any non-empty token ──────────────────────────────
  // Swap this block for real JWT verification or API validation.
  return token.length > 0;
};

// ── Loading indicator ─────────────────────────────────────────────────────────
const AuthLoadingSpinner: React.FC = () => (
  <div
    role="status"
    aria-label="Verifying authentication"
    className="flex h-screen w-full items-center justify-center bg-background"
  >
    <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
  </div>
);

// ── ProtectedRoute component ──────────────────────────────────────────────────
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  redirectTo = "/login",
  authCheck = defaultAuthCheck,
}) => {
  const location = useLocation();
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    isLoading: true,
  });

  useEffect(() => {
    let cancelled = false;

    const verify = async (): Promise<void> => {
      try {
        const isAuthenticated = await authCheck();

        if (!cancelled) {
          setAuthState({ isAuthenticated, isLoading: false });
        }
      } catch {
        if (!cancelled) {
          setAuthState({ isAuthenticated: false, isLoading: false });
        }
      }
    };

    void verify();

    return () => {
      cancelled = true;
    };
  }, [authCheck]);

  if (authState.isLoading) {
    return <AuthLoadingSpinner />;
  }

  if (!authState.isAuthenticated) {
    return (
      <Navigate
        to={redirectTo}
        state={{ from: location }}
        replace
      />
    );
  }

  return <>{children}</>;
};

export default ProtectedRoute;