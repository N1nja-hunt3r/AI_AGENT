import type { ComponentType, ReactNode } from "react";
import { QueryProvider } from "./QueryProvider";
import { ThemeProvider } from "./ThemeProvider";

interface ProviderWithProps {
  Component: ComponentType<{ children: ReactNode }>;
  props?: Record<string, unknown>;
}

export const providerRegistry: Record<string, ProviderWithProps> = {
  query: { Component: QueryProvider },
  theme: { Component: ThemeProvider },
} as const;
