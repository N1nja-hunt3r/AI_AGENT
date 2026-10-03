// frontend/src/context/ThemeContext.tsx
/* eslint-disable react-refresh/only-export-components */

"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ThemeMode = "light" | "dark" | "system";

export type ResolvedThemeMode = "light" | "dark";

export type ColorScheme = "default" | "ocean" | "forest" | "sunset" | "mono";

export type FontSize = "xs" | "sm" | "md" | "lg" | "xl";

export type FontFamily = "system" | "serif" | "mono" | "inter";

export type MotionPreference = "full" | "reduced" | "none";

export interface ThemeColors {
  bg_primary: string;
  bg_secondary: string;
  text_primary: string;
  text_secondary: string;
  accent: string;
  border: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

export interface ThemeTypography {
  font_size: FontSize;
  font_family: FontFamily;
  line_height: number;
  letter_spacing: number;
}

export interface ThemeContextValue {
  // Mode
  mode: ThemeMode;
  resolved_mode: ResolvedThemeMode;
  is_dark: boolean;
  is_light: boolean;
  is_system: boolean;
  setMode: (mode: ThemeMode) => void;
  toggleMode: () => void;

  // Color scheme
  color_scheme: ColorScheme;
  setColorScheme: (scheme: ColorScheme) => void;

  // Typography
  typography: ThemeTypography;
  setFontSize: (size: FontSize) => void;
  setFontFamily: (family: FontFamily) => void;
  setLineHeight: (value: number) => void;
  setLetterSpacing: (value: number) => void;

  // Motion
  motion_preference: MotionPreference;
  setMotionPreference: (pref: MotionPreference) => void;
  prefers_reduced_motion: boolean;

  // High contrast
  high_contrast: boolean;
  setHighContrast: (enabled: boolean) => void;

  // Computed CSS helpers
  getCSSVar: (variable: string) => string;
  applyThemeClass: (element?: HTMLElement) => void;
  getThemeDataAttributes: () => Record<string, string>;

  // Sidebar / layout
  sidebar_width: number;
  compact_mode: boolean;
  setSidebarWidth: (width: number) => void;
  setCompactMode: (compact: boolean) => void;

  // System preference
  system_preference: ResolvedThemeMode;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const THEME_STORAGE_KEY = "ai_os_theme";
const COLOR_SCHEME_STORAGE_KEY = "ai_os_color_scheme";
const TYPOGRAPHY_STORAGE_KEY = "ai_os_typography";
const MOTION_STORAGE_KEY = "ai_os_motion";
const LAYOUT_STORAGE_KEY = "ai_os_layout";

const DARK_MEDIA_QUERY = "(prefers-color-scheme: dark)";
const REDUCED_MOTION_MEDIA_QUERY = "(prefers-reduced-motion: reduce)";

const DEFAULT_TYPOGRAPHY: ThemeTypography = {
  font_size: "md",
  font_family: "system",
  line_height: 1.6,
  letter_spacing: 0,
};

const DEFAULT_LAYOUT = {
  sidebar_width: 280,
  compact_mode: false,
};

// ─── Storage helpers ──────────────────────────────────────────────────────────

const readStorage = <T,>(key: string, fallback: T): T => {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: unknown): void => {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage write failed
  }
};

// ─── CSS variable reader ──────────────────────────────────────────────────────

const readCSSVar = (variable: string): string => {
  if (typeof window === "undefined") return "";
  const style = getComputedStyle(document.documentElement);
  return style.getPropertyValue(variable).trim();
};

// ─── System preference detection ─────────────────────────────────────────────

const getSystemPreference = (): ResolvedThemeMode => {
  if (typeof window === "undefined") return "light";
  return window.matchMedia(DARK_MEDIA_QUERY).matches ? "dark" : "light";
};

const getSystemReducedMotion = (): boolean => {
  if (typeof window === "undefined") return false;
  return window.matchMedia(REDUCED_MOTION_MEDIA_QUERY).matches;
};

// ─── Context ─────────────────────────────────────────────────────────────────

export const ThemeContext = createContext<ThemeContextValue | null>(null);
ThemeContext.displayName = "ThemeContext";

// ─── Props ────────────────────────────────────────────────────────────────────

export interface ThemeProviderProps {
  children: ReactNode;
  default_mode?: ThemeMode;
  default_color_scheme?: ColorScheme;
  default_typography?: Partial<ThemeTypography>;
  storage_key?: string;
  disable_system_sync?: boolean;
  attribute?: "class" | "data-theme";
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const ThemeContextProvider = ({
  children,
  default_mode = "system",
  default_color_scheme = "default",
  default_typography,
  storage_key = THEME_STORAGE_KEY,
  disable_system_sync = false,
  attribute = "class",
}: ThemeProviderProps): React.JSX.Element => {
  // ── State initialisation ──────────────────────────────────────────────────

  const [mode, setModeState] = useState<ThemeMode>(() =>
    readStorage(storage_key, default_mode)
  );

  const [system_preference, setSystemPreference] =
    useState<ResolvedThemeMode>(getSystemPreference);

  const [color_scheme, setColorSchemeState] = useState<ColorScheme>(() =>
    readStorage(COLOR_SCHEME_STORAGE_KEY, default_color_scheme)
  );

  const [typography, setTypographyState] = useState<ThemeTypography>(() =>
    readStorage(TYPOGRAPHY_STORAGE_KEY, {
      ...DEFAULT_TYPOGRAPHY,
      ...default_typography,
    })
  );

  const [motion_preference, setMotionPreferenceState] =
    useState<MotionPreference>(() =>
      readStorage(
        MOTION_STORAGE_KEY,
        getSystemReducedMotion() ? "reduced" : "full"
      )
    );

  const [high_contrast, setHighContrastState] = useState(false);

  const [sidebar_width, setSidebarWidthState] = useState<number>(() =>
    readStorage<typeof DEFAULT_LAYOUT>(
      LAYOUT_STORAGE_KEY,
      DEFAULT_LAYOUT
    ).sidebar_width
  );

  const [compact_mode, setCompactModeState] = useState<boolean>(() =>
    readStorage<typeof DEFAULT_LAYOUT>(
      LAYOUT_STORAGE_KEY,
      DEFAULT_LAYOUT
    ).compact_mode
  );

  // ── Resolved mode ─────────────────────────────────────────────────────────

  const resolved_mode: ResolvedThemeMode =
    mode === "system" ? system_preference : mode;

  const is_dark = resolved_mode === "dark";
  const is_light = resolved_mode === "light";
  const is_system = mode === "system";

  // ── Apply theme to DOM ────────────────────────────────────────────────────

  const applyThemeClass = useCallback(
    (element: HTMLElement = document.documentElement): void => {
      if (attribute === "class") {
        element.classList.remove("light", "dark");
        element.classList.add(resolved_mode);
      } else {
        element.setAttribute("data-theme", resolved_mode);
      }

      element.setAttribute("data-color-scheme", color_scheme);
      element.setAttribute("data-font-size", typography.font_size);
      element.setAttribute("data-font-family", typography.font_family);
      element.setAttribute("data-motion", motion_preference);
      element.setAttribute(
        "data-high-contrast",
        high_contrast ? "true" : "false"
      );

      element.style.setProperty(
        "--font-size-base",
        {
          xs: "12px",
          sm: "14px",
          md: "16px",
          lg: "18px",
          xl: "20px",
        }[typography.font_size]
      );

      element.style.setProperty(
        "--line-height-base",
        String(typography.line_height)
      );

      element.style.setProperty(
        "--letter-spacing-base",
        `${typography.letter_spacing}em`
      );

      element.style.setProperty(
        "--sidebar-width",
        `${sidebar_width}px`
      );
    },
    [
      attribute,
      resolved_mode,
      color_scheme,
      typography,
      motion_preference,
      high_contrast,
      sidebar_width,
    ]
  );

  useEffect(() => {
    if (typeof document === "undefined") return;
    applyThemeClass();
  }, [applyThemeClass]);

  // ── System preference sync ────────────────────────────────────────────────

  useEffect(() => {
    if (disable_system_sync || typeof window === "undefined") return;

    const media = window.matchMedia(DARK_MEDIA_QUERY);
    const handleChange = (e: MediaQueryListEvent): void => {
      setSystemPreference(e.matches ? "dark" : "light");
    };

    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, [disable_system_sync]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia(REDUCED_MOTION_MEDIA_QUERY);
    const handleChange = (e: MediaQueryListEvent): void => {
      if (readStorage(MOTION_STORAGE_KEY, null) === null) {
        setMotionPreferenceState(e.matches ? "reduced" : "full");
      }
    };
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  // ── Mode setters ──────────────────────────────────────────────────────────

  const setMode = useCallback(
    (next_mode: ThemeMode): void => {
      setModeState(next_mode);
      writeStorage(storage_key, next_mode);
    },
    [storage_key]
  );

  const toggleMode = useCallback((): void => {
    const next: ResolvedThemeMode = resolved_mode === "dark" ? "light" : "dark";
    setMode(next);
  }, [resolved_mode, setMode]);

  // ── Color scheme ──────────────────────────────────────────────────────────

  const setColorScheme = useCallback((scheme: ColorScheme): void => {
    setColorSchemeState(scheme);
    writeStorage(COLOR_SCHEME_STORAGE_KEY, scheme);
  }, []);

  // ── Typography ────────────────────────────────────────────────────────────

  const updateTypography = useCallback(
    (patch: Partial<ThemeTypography>): void => {
      setTypographyState((prev) => {
        const next = { ...prev, ...patch };
        writeStorage(TYPOGRAPHY_STORAGE_KEY, next);
        return next;
      });
    },
    []
  );

  const setFontSize = useCallback(
    (size: FontSize) => updateTypography({ font_size: size }),
    [updateTypography]
  );

  const setFontFamily = useCallback(
    (family: FontFamily) => updateTypography({ font_family: family }),
    [updateTypography]
  );

  const setLineHeight = useCallback(
    (value: number) =>
      updateTypography({ line_height: Math.max(1, Math.min(3, value)) }),
    [updateTypography]
  );

  const setLetterSpacing = useCallback(
    (value: number) =>
      updateTypography({
        letter_spacing: Math.max(-0.1, Math.min(0.5, value)),
      }),
    [updateTypography]
  );

  // ── Motion ────────────────────────────────────────────────────────────────

  const setMotionPreference = useCallback(
    (pref: MotionPreference): void => {
      setMotionPreferenceState(pref);
      writeStorage(MOTION_STORAGE_KEY, pref);
    },
    []
  );

  // ── High contrast ─────────────────────────────────────────────────────────

  const setHighContrast = useCallback((enabled: boolean): void => {
    setHighContrastState(enabled);
  }, []);

  // ── Layout ────────────────────────────────────────────────────────────────

  const setSidebarWidth = useCallback((width: number): void => {
    const clamped = Math.max(200, Math.min(600, width));
    setSidebarWidthState(clamped);
    writeStorage(LAYOUT_STORAGE_KEY, { sidebar_width: clamped, compact_mode });
  }, [compact_mode]);

  const setCompactMode = useCallback(
    (compact: boolean): void => {
      setCompactModeState(compact);
      writeStorage(LAYOUT_STORAGE_KEY, {
        sidebar_width,
        compact_mode: compact,
      });
    },
    [sidebar_width]
  );

  // ── CSS helpers ───────────────────────────────────────────────────────────

  const getCSSVar = useCallback(
    (variable: string): string => readCSSVar(variable),
    []
  );

  const getThemeDataAttributes = useCallback(
    (): Record<string, string> => ({
      "data-theme": resolved_mode,
      "data-color-scheme": color_scheme,
      "data-font-size": typography.font_size,
      "data-font-family": typography.font_family,
      "data-motion": motion_preference,
      "data-high-contrast": high_contrast ? "true" : "false",
      "data-compact": compact_mode ? "true" : "false",
    }),
    [
      resolved_mode,
      color_scheme,
      typography.font_size,
      typography.font_family,
      motion_preference,
      high_contrast,
      compact_mode,
    ]
  );

  // ── Context value ─────────────────────────────────────────────────────────

  const value = useMemo<ThemeContextValue>(
    () => ({
      mode,
      resolved_mode,
      is_dark,
      is_light,
      is_system,
      setMode,
      toggleMode,
      color_scheme,
      setColorScheme,
      typography,
      setFontSize,
      setFontFamily,
      setLineHeight,
      setLetterSpacing,
      motion_preference,
      setMotionPreference,
      prefers_reduced_motion:
        motion_preference === "reduced" || motion_preference === "none",
      high_contrast,
      setHighContrast,
      getCSSVar,
      applyThemeClass,
      getThemeDataAttributes,
      sidebar_width,
      compact_mode,
      setSidebarWidth,
      setCompactMode,
      system_preference,
    }),
    [
      mode,
      resolved_mode,
      is_dark,
      is_light,
      is_system,
      setMode,
      toggleMode,
      color_scheme,
      setColorScheme,
      typography,
      setFontSize,
      setFontFamily,
      setLineHeight,
      setLetterSpacing,
      motion_preference,
      setMotionPreference,
      high_contrast,
      setHighContrast,
      getCSSVar,
      applyThemeClass,
      getThemeDataAttributes,
      sidebar_width,
      compact_mode,
      setSidebarWidth,
      setCompactMode,
      system_preference,
    ]
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export const useThemeContext = (): ThemeContextValue => {
  const context = useContext(ThemeContext);
  if (context === null) {
    throw new Error(
      "[useThemeContext] must be used within a <ThemeContextProvider>. " +
        "Ensure ThemeContextProvider wraps your component tree."
    );
  }
  return context;
};

// ─── Derived hooks ────────────────────────────────────────────────────────────

export const useIsDark = (): boolean => useThemeContext().is_dark;

export const useIsLight = (): boolean => useThemeContext().is_light;

export const useResolvedTheme = (): ResolvedThemeMode =>
  useThemeContext().resolved_mode;

export const usePrefersReducedMotion = (): boolean =>
  useThemeContext().prefers_reduced_motion;

export const useTypography = (): ThemeTypography =>
  useThemeContext().typography;