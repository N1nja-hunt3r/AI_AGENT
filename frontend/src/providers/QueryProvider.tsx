// frontend/src/providers/QueryProvider.tsx

"use client";

import React, { useState, type ReactNode } from "react";
import {
  QueryClient,
  QueryClientProvider,
  defaultShouldDehydrateQuery,
  isServer,
} from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

// ─── Constants ────────────────────────────────────────────────────────────────

const STALE_TIME_MS = 60 * 1000; // 1 minute
const GC_TIME_MS = 5 * 60 * 1000; // 5 minutes
const MAX_RETRY_COUNT = 3;
const RETRY_DELAY_BASE_MS = 1_000;
const RETRY_DELAY_MAX_MS = 30_000;

// ─── Non-retryable HTTP status codes ─────────────────────────────────────────

const NON_RETRYABLE_STATUS_CODES = new Set([
  400, // Bad Request
  401, // Unauthorized
  403, // Forbidden
  404, // Not Found
  409, // Conflict
  410, // Gone
  422, // Unprocessable Entity
]);

// ─── Error shape guard ────────────────────────────────────────────────────────

interface HttpError {
  status?: number;
  statusCode?: number;
  response?: { status?: number };
}

const getHttpStatus = (error: unknown): number | null => {
  if (error === null || typeof error !== "object") return null;
  const err = error as HttpError;
  return (
    err.status ??
    err.statusCode ??
    err.response?.status ??
    null
  );
};

const isNonRetryableError = (error: unknown): boolean => {
  const status = getHttpStatus(error);
  return status !== null && NON_RETRYABLE_STATUS_CODES.has(status);
};

// ─── Retry policy ─────────────────────────────────────────────────────────────

const retryPolicy = (
  failureCount: number,
  error: unknown
): boolean => {
  if (isNonRetryableError(error)) return false;
  return failureCount < MAX_RETRY_COUNT;
};

const retryDelayPolicy = (attemptIndex: number): number => {
  const exponential = RETRY_DELAY_BASE_MS * 2 ** attemptIndex;
  const jitter = Math.random() * RETRY_DELAY_BASE_MS;
  return Math.min(exponential + jitter, RETRY_DELAY_MAX_MS);
};

// ─── QueryClient factory ──────────────────────────────────────────────────────

const makeQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: STALE_TIME_MS,
        gcTime: GC_TIME_MS,
        retry: retryPolicy,
        retryDelay: retryDelayPolicy,
        refetchOnWindowFocus: true,
        refetchOnReconnect: true,
        refetchOnMount: true,
        networkMode: "online",
      },
      mutations: {
        retry: (failureCount, error) => {
          if (isNonRetryableError(error)) return false;
          return failureCount < 1;
        },
        retryDelay: retryDelayPolicy,
        networkMode: "online",
        onError: (error) => {
          if (import.meta.env.DEV) {
            console.error("[Mutation Error]", error);
          }
        },
      },
      dehydrate: {
        shouldDehydrateQuery: (query) =>
          defaultShouldDehydrateQuery(query) ||
          query.state.status === "pending",
      },
    },
  });

// ─── SSR-safe singleton ───────────────────────────────────────────────────────

let browserQueryClient: QueryClient | undefined;

const getQueryClient = (): QueryClient => {
  if (isServer) {
    return makeQueryClient();
  }
  if (!browserQueryClient) {
    browserQueryClient = makeQueryClient();
  }
  return browserQueryClient;
};

// ─── Props ────────────────────────────────────────────────────────────────────

export interface QueryProviderProps {
  children: ReactNode;
  showDevtools?: boolean;
  devtoolsPosition?: "left" | "right" | "top" | "bottom";
  devtoolsInitialIsOpen?: boolean;
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const QueryProvider = ({
  children,
  showDevtools = import.meta.env.DEV,
  devtoolsPosition = "bottom",
  devtoolsInitialIsOpen = false,
}: QueryProviderProps): React.JSX.Element => {
  const [queryClient] = useState<QueryClient>(getQueryClient);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {showDevtools && (
        <ReactQueryDevtools
          initialIsOpen={devtoolsInitialIsOpen}
          position={devtoolsPosition}
          buttonPosition={`${devtoolsPosition === "top" ? "top" : "bottom"}-right` as const}
        />
      )}
    </QueryClientProvider>
  );
};

