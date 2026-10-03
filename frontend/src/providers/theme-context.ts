"use client";

import { createContext, useContext } from "react";
import type { ColorTheme, ResolvedTheme } from "./ThemeProvider";

export interface ThemeProviderState {
  theme: ColorTheme;
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: ColorTheme) => void;
}

export const ThemeProviderContext = createContext<ThemeProviderState | undefined>(
  undefined
);

export const useTheme = (): ThemeProviderState => {
  const context = useContext(ThemeProviderContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};
