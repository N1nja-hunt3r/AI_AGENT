// frontend/src/providers/registry.ts

"use client";

import React, { type ReactNode } from "react";
import { QueryProvider, type QueryProviderProps } from "./QueryProvider";
import { ThemeProvider, type ThemeProviderProps } from "./ThemeProvider";
import { ChatProvider, type ChatProviderProps } from "@/context/ChatContext";
import { VoiceProvider } from "@/context/VoiceContext";
import { AuthProvider } from "@/auth/AuthProvider";
import api from "@/api/client";
import type {
  LoginCredentials,
  LoginResult,
  TokenPair,
} from "@/auth/AuthContext";

// ─── Re-exports ───────────────────────────────────────────────────────────────

export { QueryProvider } from "./QueryProvider";
export type { QueryProviderProps } from "./QueryProvider";

export { ThemeProvider } from "./ThemeProvider";
export type {
  ThemeProviderProps,
  ColorTheme,
  ResolvedTheme,
} from "./ThemeProvider";

// ─── Default auth callbacks ───────────────────────────────────────────────────

const defaultOnRefreshTokens = async (
  refresh_token: string
): Promise<TokenPair> => {
  const { data } = await api.post("/api/v1/auth/refresh", { refresh_token });
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    token_type: data.token_type ?? "Bearer",
    expires_in: data.expires_in,
    expires_at: Date.now() + data.expires_in * 1000,
  };
};

const defaultOnLogin = async (
  credentials: LoginCredentials
): Promise<LoginResult> => {
  const { data: tokens } = await api.post("/api/v1/auth/login", {
    email: credentials.email,
    password: credentials.password,
  });
  const { data: user } = await api.get("/api/v1/auth/me");
  const now = Date.now();
  return {
    session: {
      id: crypto.randomUUID(),
      user: {
        id: user.id,
        email: user.email,
        display_name: user.name ?? user.email,
        roles: [user.role ?? "viewer"],
        permissions: [],
        created_at: new Date(user.created_at).getTime(),
        last_login_at: now,
      },
      tokens: {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        token_type: "Bearer",
        expires_in: tokens.expires_in,
        expires_at: now + tokens.expires_in * 1000,
      },
      created_at: now,
      last_active_at: now,
      expires_at: now + tokens.expires_in * 1000,
    },
  };
};

const defaultOnLogout = async (session_id: string | null) => {
  try {
    await api.post("/api/v1/auth/logout", { refresh_token: session_id });
  } catch {
    // Ignore server-side logout failure
  }
};

// ─── Composed provider props ──────────────────────────────────────────────────

export interface AppProvidersProps
  extends Pick<
      QueryProviderProps,
      "showDevtools" | "devtoolsPosition" | "devtoolsInitialIsOpen"
    >,
    Pick<
      ThemeProviderProps,
      | "defaultTheme"
      | "storageKey"
      | "forcedTheme"
      | "disableTransitionOnChange"
      | "themes"
    >,
    Partial<ChatProviderProps> {
  children: ReactNode;
  onRefreshTokens?: (refresh_token: string) => Promise<TokenPair>;
  onLogin?: (credentials: LoginCredentials) => Promise<LoginResult>;
  onLogout?: (session_id: string | null) => Promise<void>;
}

// ─── Composed root provider ───────────────────────────────────────────────────

export const AppProviders = ({
  children,
  // AuthProvider options
  onRefreshTokens = defaultOnRefreshTokens,
  onLogin = defaultOnLogin,
  onLogout = defaultOnLogout,
  // QueryProvider options
  showDevtools,
  devtoolsPosition,
  devtoolsInitialIsOpen,
  // ThemeProvider options
  defaultTheme = "system",
  storageKey = "ai-os-theme",
  forcedTheme,
  disableTransitionOnChange = false,
  themes,
  // ChatProvider options (partial)
  initial_mode,
  initial_conversation,
  max_message_history,
}: AppProvidersProps): React.JSX.Element => {
  return (
    <AuthProvider
      onRefreshTokens={onRefreshTokens}
      onLogin={onLogin}
      onLogout={onLogout}
    >
      <QueryProvider
        showDevtools={showDevtools}
        devtoolsPosition={devtoolsPosition}
        devtoolsInitialIsOpen={devtoolsInitialIsOpen}
      >
        <ThemeProvider
          defaultTheme={defaultTheme}
          storageKey={storageKey}
          forcedTheme={forcedTheme}
          disableTransitionOnChange={disableTransitionOnChange}
          themes={themes}
        >
          <VoiceProvider>
            <ChatProvider
              initial_mode={initial_mode}
              initial_conversation={initial_conversation}
              max_message_history={max_message_history}
            >
              {children}
            </ChatProvider>
          </VoiceProvider>
        </ThemeProvider>
      </QueryProvider>
    </AuthProvider>
  );
};

