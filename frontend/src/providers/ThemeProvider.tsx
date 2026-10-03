"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ThemeProviderContext } from "./theme-context";

export type ColorTheme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: ColorTheme;
  storageKey?: string;
  forcedTheme?: ColorTheme;
  disableTransitionOnChange?: boolean;
  themes?: ColorTheme[];
}

const getThemeFromStorage = (storageKey: string): ColorTheme | null => {
  if (typeof window === "undefined") return null;
  try {
    return localStorage.getItem(storageKey) as ColorTheme | null;
  } catch {
    return null;
  }
};

const setThemeInStorage = (storageKey: string, theme: ColorTheme): void => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(storageKey, theme);
  } catch { /* noop */ }
};

const getResolvedTheme = (
  theme: ColorTheme,
  prefersDark: boolean
): ResolvedTheme => {
  if (theme === "system") return prefersDark ? "dark" : "light";
  return theme;
};

const applyThemeToDocument = (resolved: ResolvedTheme): void => {
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(resolved);
};

export const ThemeProvider = ({
  children,
  defaultTheme = "system",
  storageKey = "ai-os-theme",
  forcedTheme,
  disableTransitionOnChange = false,
}: ThemeProviderProps): React.JSX.Element => {
  const [theme, setThemeState] = useState<ColorTheme>(() => {
    if (forcedTheme) return forcedTheme;
    return getThemeFromStorage(storageKey) ?? defaultTheme;
  });

  const [prefersDark, setPrefersDark] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => setPrefersDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const resolvedTheme: ResolvedTheme = forcedTheme !== undefined
    ? (forcedTheme === "system" ? (prefersDark ? "dark" : "light") : forcedTheme)
    : getResolvedTheme(theme, prefersDark);

  useEffect(() => {
    if (disableTransitionOnChange) {
      const css = document.createElement("style");
      css.appendChild(
        document.createTextNode(
          "*, *::before, *::after { transition: none !important; }"
        )
      );
      document.head.appendChild(css);

      const timeout = setTimeout(() => {
        document.head.removeChild(css);
      }, 0);

      return () => {
        clearTimeout(timeout);
        document.head.removeChild(css);
      };
    }
  }, [disableTransitionOnChange, resolvedTheme]);

  useEffect(() => {
    applyThemeToDocument(resolvedTheme);
  }, [resolvedTheme]);

  const setTheme = useCallback(
    (newTheme: ColorTheme) => {
      setThemeState(newTheme);
      if (!forcedTheme) {
        setThemeInStorage(storageKey, newTheme);
      }
    },
    [forcedTheme, storageKey]
  );

  const value = useMemo(
    () => ({ theme, resolvedTheme, setTheme }),
    [theme, resolvedTheme, setTheme]
  );

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
};


