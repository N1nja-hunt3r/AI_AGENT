// frontend/src/auth/AuthProvider.tsx

"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  AuthContext,
  type AuthContextValue,
  type AuthError,
  type AuthErrorCode,
  type AuthStatus,
  type JWTPayload,
  type LoginCredentials,
  type LoginResult,
  type LogoutReason,
  type Permission,
  type RefreshResult,
  type Session,
  type TokenPair,
  type UserRole,
} from "./AuthContext";

// ─── Constants ────────────────────────────────────────────────────────────────

const ACCESS_TOKEN_KEY = "ai_os_access_token";
const REFRESH_TOKEN_KEY = "ai_os_refresh_token";
const SESSION_KEY = "ai_os_session";

const TOKEN_REFRESH_THRESHOLD_MS = 60 * 1000; // refresh 60s before expiry
const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const SESSION_SYNC_CHANNEL = "ai_os_auth_sync";
const ACTIVITY_EVENTS: (keyof WindowEventMap)[] = [
  "mousemove",
  "mousedown",
  "keydown",
  "touchstart",
  "scroll",
  "click",
];

// ─── JWT utilities ────────────────────────────────────────────────────────────

const base64UrlDecode = (str: string): string => {
  const padded = str.replace(/-/g, "+").replace(/_/g, "/");
  const padding = "=".repeat((4 - (padded.length % 4)) % 4);
  return atob(padded + padding);
};

const decodeJWT = (token: string): JWTPayload | null => {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const payload = base64UrlDecode(parts[1]!);
    return JSON.parse(payload) as JWTPayload;
  } catch {
    return null;
  }
};

const isJWTExpired = (payload: JWTPayload, bufferMs = 0): boolean => {
  return Date.now() >= payload.exp * 1000 - bufferMs;
};

// ─── Storage utilities ────────────────────────────────────────────────────────

const safeStorage = {
  get: (key: string): string | null => {
    try {
      return sessionStorage.getItem(key) ?? localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set: (key: string, value: string, persistent = false): void => {
    try {
      if (persistent) {
        localStorage.setItem(key, value);
      } else {
        sessionStorage.setItem(key, value);
      }
    } catch {
      // storage unavailable — fail silently in production
    }
  },
  remove: (key: string): void => {
    try {
      sessionStorage.removeItem(key);
      localStorage.removeItem(key);
    } catch {
      // ignore
    }
  },
  clear: (): void => {
    [ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, SESSION_KEY].forEach(
      safeStorage.remove
    );
  },
};

// ─── Error factory ────────────────────────────────────────────────────────────

const makeAuthError = (
  code: AuthErrorCode,
  message: string,
  recoverable = false
): AuthError => ({
  code,
  message,
  timestamp: Date.now(),
  recoverable,
});

// ─── Props ────────────────────────────────────────────────────────────────────

export interface AuthProviderProps {
  children: ReactNode;
  /**
   * Adapter that handles token refresh against your backend.
   * Receives the current refresh token and returns new tokens.
   */
  onRefreshTokens: (refresh_token: string) => Promise<TokenPair>;
  /**
   * Adapter that handles login against your backend.
   */
  onLogin: (credentials: LoginCredentials) => Promise<LoginResult>;
  /**
   * Optional adapter called on logout to invalidate server-side session.
   */
  onLogout?: (session_id: string | null) => Promise<void>;
  /**
   * Optional: called when auth encounters an unrecoverable error.
   */
  onAuthError?: (error: AuthError) => void;
  /**
   * Optional: redirect path after logout. Consumed externally.
   */
  onSessionExpired?: (reason: LogoutReason) => void;
  /**
   * Idle timeout override in milliseconds.
   */
  idleTimeoutMs?: number;
  /**
   * Disable idle timeout watcher.
   */
  disableIdleTimeout?: boolean;
  /**
   * Disable cross-tab session sync.
   */
  disableCrossTabSync?: boolean;
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const AuthProvider = ({
  children,
  onRefreshTokens,
  onLogin,
  onLogout,
  onAuthError,
  onSessionExpired,
  idleTimeoutMs = IDLE_TIMEOUT_MS,
  disableIdleTimeout = false,
  disableCrossTabSync = false,
}: AuthProviderProps): React.JSX.Element => {
  const [session, setSession] = useState<Session | null>(() => {
    try {
      const raw = safeStorage.get(SESSION_KEY);
      if (!raw) return null;
      const sess = JSON.parse(raw) as Session;
      const payload = decodeJWT(sess.tokens.access_token);
      if (!payload) return null;
      return sess;
    } catch {
      return null;
    }
  });

  const [status, setStatus] = useState<AuthStatus>(() => {
    if (!session) return "unauthenticated";
    const payload = decodeJWT(session.tokens.access_token);
    if (!payload) return "unauthenticated";
    if (isJWTExpired(payload)) {
      const refreshToken = safeStorage.get(REFRESH_TOKEN_KEY);
      if (!refreshToken) return "unauthenticated";
      return "loading";
    }
    return "authenticated";
  });

  const [error, setError] = useState<AuthError | null>(null);

  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isRefreshingRef = useRef(false);
  const broadcastRef = useRef<BroadcastChannel | null>(null);
  const lastActiveRef = useRef<number>(0);
  const scheduleRefreshRef = useRef<(tokens: TokenPair) => void>(() => {});

  // ── Internal helpers ───────────────────────────────────────────────────────

  const clearTimers = useCallback(() => {
    if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    refreshTimerRef.current = null;
    idleTimerRef.current = null;
  }, []);

  const persistSession = useCallback(
    (sess: Session, persistent: boolean): void => {
      safeStorage.set(ACCESS_TOKEN_KEY, sess.tokens.access_token, persistent);
      safeStorage.set(
        REFRESH_TOKEN_KEY,
        sess.tokens.refresh_token,
        persistent
      );
      safeStorage.set(SESSION_KEY, JSON.stringify(sess), persistent);
    },
    []
  );

  const hydrateSession = useCallback((): Session | null => {
    try {
      const raw = safeStorage.get(SESSION_KEY);
      if (!raw) return null;
      const sess = JSON.parse(raw) as Session;
      // Validate access token integrity
      const payload = decodeJWT(sess.tokens.access_token);
      if (!payload) return null;
      return sess;
    } catch {
      return null;
    }
  }, []);

  const applySession = useCallback((sess: Session): void => {
    setSession(sess);
    setStatus("authenticated");
    setError(null);
  }, []);

  const terminateSession = useCallback(
    async (reason: LogoutReason): Promise<void> => {
      clearTimers();
      const sessionId = session?.id ?? null;
      safeStorage.clear();
      setSession(null);
      setStatus("unauthenticated");
      setError(null);

      if (!disableCrossTabSync && broadcastRef.current) {
        broadcastRef.current.postMessage({ type: "LOGOUT", reason });
      }

      try {
        await onLogout?.(sessionId);
      } catch {
        // Server-side cleanup failure should not block client logout
      }

      onSessionExpired?.(reason);
    },
    [clearTimers, disableCrossTabSync, onLogout, onSessionExpired, session?.id]
  );

  // ── Token refresh ──────────────────────────────────────────────────────────

  const scheduleRefresh = useCallback(
    (tokens: TokenPair): void => {
      if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);

      const msUntilExpiry = tokens.expires_at - Date.now();
      const refreshAt = Math.max(
        msUntilExpiry - TOKEN_REFRESH_THRESHOLD_MS,
        0
      );

      refreshTimerRef.current = setTimeout(async () => {
        if (isRefreshingRef.current) return;
        isRefreshingRef.current = true;
        setStatus("refreshing");

        try {
          const refreshToken = safeStorage.get(REFRESH_TOKEN_KEY);
          if (!refreshToken) {
            throw new Error("No refresh token available");
          }

          const newTokens = await onRefreshTokens(refreshToken);
          const payload = decodeJWT(newTokens.access_token);
          if (!payload) throw new Error("Invalid access token received");

          setSession((prev) => {
            if (!prev) return null;
            const updated: Session = {
              ...prev,
              tokens: newTokens,
              last_active_at: Date.now(),
            };
            persistSession(updated, !!localStorage.getItem(ACCESS_TOKEN_KEY));
            scheduleRefreshRef.current(newTokens);
            return updated;
          });

          setStatus("authenticated");
        } catch (err) {
          const authError = makeAuthError(
            "TOKEN_REFRESH_FAILED",
            err instanceof Error ? err.message : "Token refresh failed",
            false
          );
          setError(authError);
          onAuthError?.(authError);
          await terminateSession("token_refresh_failed");
        } finally {
          isRefreshingRef.current = false;
        }
      }, refreshAt);
    },
    [onRefreshTokens, persistSession, terminateSession, onAuthError]
  );

  useEffect(() => {
    scheduleRefreshRef.current = scheduleRefresh;
  }, [scheduleRefresh]);

  // ── Idle timeout ───────────────────────────────────────────────────────────

  const resetIdleTimer = useCallback(() => {
    if (disableIdleTimeout) return;
    lastActiveRef.current = Date.now();

    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => {
      void terminateSession("idle_timeout");
    }, idleTimeoutMs);
  }, [disableIdleTimeout, idleTimeoutMs, terminateSession]);

  // ── Cross-tab sync ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (disableCrossTabSync || typeof BroadcastChannel === "undefined") return;

    const channel = new BroadcastChannel(SESSION_SYNC_CHANNEL);
    broadcastRef.current = channel;

    channel.onmessage = (event: MessageEvent<{ type: string; reason?: LogoutReason }>) => {
      if (event.data.type === "LOGOUT") {
        clearTimers();
        safeStorage.clear();
        setSession(null);
        setStatus("unauthenticated");
        onSessionExpired?.(event.data.reason ?? "forced");
      }
      if (event.data.type === "SESSION_UPDATED") {
        const hydrated = hydrateSession();
        if (hydrated) applySession(hydrated);
      }
    };

    return () => {
      channel.close();
      broadcastRef.current = null;
    };
  }, [
    disableCrossTabSync,
    clearTimers,
    hydrateSession,
    applySession,
    onSessionExpired,
  ]);

  // ── Activity listeners ─────────────────────────────────────────────────────

  useEffect(() => {
    if (disableIdleTimeout) return;

    const handler = () => resetIdleTimer();
    ACTIVITY_EVENTS.forEach((evt) =>
      window.addEventListener(evt, handler, { passive: true })
    );

    return () => {
      ACTIVITY_EVENTS.forEach((evt) => window.removeEventListener(evt, handler));
    };
  }, [disableIdleTimeout, resetIdleTimer]);

  // ── Hydration on mount ─────────────────────────────────────────────────────

  useEffect(() => {
    if (!session) return;

    const payload = decodeJWT(session.tokens.access_token);
    if (!payload) {
      safeStorage.clear();
      return;
    }

    if (!isJWTExpired(payload)) {
      scheduleRefresh(session.tokens);
      resetIdleTimer();
      return;
    }

    // Attempt silent refresh using persisted refresh token
    const refreshToken = safeStorage.get(REFRESH_TOKEN_KEY);
    if (!refreshToken) return;

    isRefreshingRef.current = true;

    onRefreshTokens(refreshToken)
      .then((newTokens) => {
        const newPayload = decodeJWT(newTokens.access_token);
        if (!newPayload) throw new Error("Invalid token");

        const updated: Session = {
          ...session,
          tokens: newTokens,
          last_active_at: Date.now(),
        };

        persistSession(updated, !!localStorage.getItem(ACCESS_TOKEN_KEY));
        applySession(updated);
        scheduleRefresh(newTokens);
        resetIdleTimer();
      })
      .catch(async () => {
        safeStorage.clear();
        setSession(null);
        setStatus("unauthenticated");
        onSessionExpired?.("token_expired");
      })
      .finally(() => {
        isRefreshingRef.current = false;
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Cleanup on unmount ─────────────────────────────────────────────────────

  useEffect(() => () => clearTimers(), [clearTimers]);

  // ── Public API ─────────────────────────────────────────────────────────────

  const login = useCallback(
    async (credentials: LoginCredentials): Promise<LoginResult> => {
      setStatus("loading");
      setError(null);

      try {
        const result = await onLogin(credentials);

        if (!result.requires_mfa) {
          const { session: newSession } = result;
          persistSession(newSession, credentials.remember_me ?? false);
          applySession(newSession);
          scheduleRefresh(newSession.tokens);
          resetIdleTimer();

          if (!disableCrossTabSync && broadcastRef.current) {
            broadcastRef.current.postMessage({ type: "SESSION_UPDATED" });
          }
        }

        return result;
      } catch (err) {
        const authError = makeAuthError(
          "INVALID_CREDENTIALS",
          err instanceof Error ? err.message : "Login failed",
          true
        );
        setError(authError);
        setStatus("unauthenticated");
        onAuthError?.(authError);
        throw authError;
      }
    },
    [
      onLogin,
      persistSession,
      applySession,
      scheduleRefresh,
      resetIdleTimer,
      disableCrossTabSync,
      onAuthError,
    ]
  );

  const logout = useCallback(
    async (reason: LogoutReason = "user_initiated"): Promise<void> => {
      await terminateSession(reason);
    },
    [terminateSession]
  );

  const refreshSession = useCallback(async (): Promise<RefreshResult> => {
    const refreshToken = safeStorage.get(REFRESH_TOKEN_KEY);
    if (!refreshToken) {
      throw makeAuthError(
        "TOKEN_REFRESH_FAILED",
        "No refresh token available",
        false
      );
    }

    setStatus("refreshing");

    try {
      const newTokens = await onRefreshTokens(refreshToken);
      const payload = decodeJWT(newTokens.access_token);
      if (!payload) throw new Error("Invalid access token");

      let updatedSession!: Session;

      setSession((prev) => {
        if (!prev) return null;
        updatedSession = {
          ...prev,
          tokens: newTokens,
          last_active_at: Date.now(),
        };
        persistSession(updatedSession, !!localStorage.getItem(ACCESS_TOKEN_KEY));
        scheduleRefresh(newTokens);
        return updatedSession;
      });

      setStatus("authenticated");
      return { tokens: newTokens, session: updatedSession };
    } catch (err) {
      const authError = makeAuthError(
        "TOKEN_REFRESH_FAILED",
        err instanceof Error ? err.message : "Refresh failed",
        false
      );
      setError(authError);
      onAuthError?.(authError);
      await terminateSession("token_refresh_failed");
      throw authError;
    }
  }, [
    onRefreshTokens,
    persistSession,
    scheduleRefresh,
    terminateSession,
    onAuthError,
  ]);

  const clearError = useCallback(() => setError(null), []);

  const getAccessToken = useCallback(
    (): string | null => session?.tokens.access_token ?? null,
    [session]
  );

  const getDecodedToken = useCallback((): JWTPayload | null => {
    const token = session?.tokens.access_token;
    return token ? decodeJWT(token) : null;
  }, [session]);

  const isTokenExpired = useCallback((): boolean => {
    const payload = getDecodedToken();
    if (!payload) return true;
    return isJWTExpired(payload);
  }, [getDecodedToken]);

  const getTokenExpiresInMs = useCallback((): number | null => {
    const payload = getDecodedToken();
    if (!payload) return null;
    return Math.max(0, payload.exp * 1000 - Date.now());
  }, [getDecodedToken]);

  const hasRole = useCallback(
    (role: UserRole): boolean => session?.user.roles.includes(role) ?? false,
    [session]
  );

  const hasAnyRole = useCallback(
    (roles: UserRole[]): boolean =>
      roles.some((r) => session?.user.roles.includes(r) ?? false),
    [session]
  );

  const hasAllRoles = useCallback(
    (roles: UserRole[]): boolean =>
      roles.every((r) => session?.user.roles.includes(r) ?? false),
    [session]
  );

  const hasPermission = useCallback(
    (permission: Permission): boolean =>
      session?.user.permissions.includes(permission) ?? false,
    [session]
  );

  const hasAnyPermission = useCallback(
    (permissions: Permission[]): boolean =>
      permissions.some(
        (p) => session?.user.permissions.includes(p) ?? false
      ),
    [session]
  );

  const hasAllPermissions = useCallback(
    (permissions: Permission[]): boolean =>
      permissions.every(
        (p) => session?.user.permissions.includes(p) ?? false
      ),
    [session]
  );

  const getSessionId = useCallback(
    (): string | null => session?.id ?? null,
    [session]
  );

  const updateLastActive = useCallback((): void => {
    lastActiveRef.current = Date.now();
    setSession((prev) =>
      prev ? { ...prev, last_active_at: Date.now() } : null
    );
    resetIdleTimer();
  }, [resetIdleTimer]);

  // ── Context value ──────────────────────────────────────────────────────────

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      session,
      user: session?.user ?? null,
      error,
      isAuthenticated: status === "authenticated",
      isLoading: status === "loading" || status === "idle",
      isRefreshing: status === "refreshing",
      login,
      logout,
      refreshSession,
      clearError,
      getAccessToken,
      getDecodedToken,
      isTokenExpired,
      getTokenExpiresInMs,
      hasRole,
      hasAnyRole,
      hasAllRoles,
      hasPermission,
      hasAnyPermission,
      hasAllPermissions,
      getSessionId,
      updateLastActive,
    }),
    [
      status,
      session,
      error,
      login,
      logout,
      refreshSession,
      clearError,
      getAccessToken,
      getDecodedToken,
      isTokenExpired,
      getTokenExpiresInMs,
      hasRole,
      hasAnyRole,
      hasAllRoles,
      hasPermission,
      hasAnyPermission,
      hasAllPermissions,
      getSessionId,
      updateLastActive,
    ]
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};