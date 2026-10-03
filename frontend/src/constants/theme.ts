// frontend/src/constants/theme.ts

import { Theme } from "../types/settings";

// ─── CSS custom property keys ────────────────────────────────────────────────

export const CSS_VAR = {
  // Colors
  BG_PRIMARY: "--color-bg-primary",
  BG_SECONDARY: "--color-bg-secondary",
  BG_TERTIARY: "--color-bg-tertiary",
  BG_INVERTED: "--color-bg-inverted",
  BG_OVERLAY: "--color-bg-overlay",
  BG_MUTE: "--color-bg-muted",
  BG_HOVER: "--color-bg-hover",
  BG_ACTIVE: "--color-bg-active",
  BG_SELECTED: "--color-bg-selected",
  BG_SOLID: "--color-bg-solid",

  TEXT_PRIMARY: "--color-text-primary",
  TEXT_SECONDARY: "--color-text-secondary",
  TEXT_TERTIARY: "--color-text-tertiary",
  TEXT_INVERTED: "--color-text-inverted",
  TEXT_MUTE: "--color-text-muted",
  TEXT_LINK: "--color-text-link",
  TEXT_LINK_HOVER: "--color-text-link-hover",

  BORDER_PRIMARY: "--color-border-primary",
  BORDER_SECONDARY: "--color-border-secondary",
  BORDER_FOCUS: "--color-border-focus",
  BORDER_ERROR: "--color-border-error",

  ACCENT: "--color-accent",
  ACCENT_HOVER: "--color-accent-hover",
  ACCENT_ACTIVE: "--color-accent-active",
  ACCENT_FG: "--color-accent-fg",

  SUCCESS: "--color-success",
  WARNING: "--color-warning",
  ERROR: "--color-error",
  INFO: "--color-info",

  // Shadows
  SHADOW_SM: "--shadow-sm",
  SHADOW_MD: "--shadow-md",
  SHADOW_LG: "--shadow-lg",
  SHADOW_XL: "--shadow-xl",

  // Radii
  RADIUS_SM: "--radius-sm",
  RADIUS_MD: "--radius-md",
  RADIUS_LG: "--radius-lg",
  RADIUS_XL: "--radius-xl",
  RADIUS_FULL: "--radius-full",

  // Spacing
  SPACING_XS: "--spacing-xs",
  SPACING_SM: "--spacing-sm",
  SPACING_MD: "--spacing-md",
  SPACING_LG: "--spacing-lg",
  SPACING_XL: "--spacing-xl",

  // Font
  FONT_SANS: "--font-sans",
  FONT_SERIF: "--font-serif",
  FONT_MONO: "--font-mono",
  FONT_SIZE: "--font-size",

  // Z-index
  Z_DROPDOWN: "--z-dropdown",
  Z_STICKY: "--z-sticky",
  Z_FIXED: "--z-fixed",
  Z_OVERLAY: "--z-overlay",
  Z_MODAL: "--z-modal",
  Z_POPOVER: "--z-popover",
  Z_TOAST: "--z-toast",
  Z_TOOLTIP: "--z-tooltip",
} as const;

// ─── Theme mode values ───────────────────────────────────────────────────────

export const THEME_MODE: Record<Theme, Theme> = {
  [Theme.Light]: Theme.Light,
  [Theme.Dark]: Theme.Dark,
  [Theme.System]: Theme.System,
} as const;

export const THEME_STORAGE_KEY = "ai-os-theme";

// ─── Theme class names ───────────────────────────────────────────────────────

export const THEME_CLASSES = {
  LIGHT: "theme-light",
  DARK: "theme-dark",
  SYSTEM: "theme-system",
} as const;

// ─── Light theme values ──────────────────────────────────────────────────────

export const LIGHT_THEME_VALUES: Record<string, string> = {
  [CSS_VAR.BG_PRIMARY]: "#ffffff",
  [CSS_VAR.BG_SECONDARY]: "#f8f9fa",
  [CSS_VAR.BG_TERTIARY]: "#f1f3f5",
  [CSS_VAR.BG_INVERTED]: "#1a1b1e",
  [CSS_VAR.BG_OVERLAY]: "rgba(0, 0, 0, 0.5)",
  [CSS_VAR.BG_MUTE]: "#f5f5f5",
  [CSS_VAR.BG_HOVER]: "#e9ecef",
  [CSS_VAR.BG_ACTIVE]: "#dee2e6",
  [CSS_VAR.BG_SELECTED]: "#e7f5ff",
  [CSS_VAR.BG_SOLID]: "#212529",

  [CSS_VAR.TEXT_PRIMARY]: "#1a1b1e",
  [CSS_VAR.TEXT_SECONDARY]: "#495057",
  [CSS_VAR.TEXT_TERTIARY]: "#868e96",
  [CSS_VAR.TEXT_INVERTED]: "#ffffff",
  [CSS_VAR.TEXT_MUTE]: "#adb5bd",
  [CSS_VAR.TEXT_LINK]: "#228be6",
  [CSS_VAR.TEXT_LINK_HOVER]: "#1c7ed6",

  [CSS_VAR.BORDER_PRIMARY]: "#dee2e6",
  [CSS_VAR.BORDER_SECONDARY]: "#e9ecef",
  [CSS_VAR.BORDER_FOCUS]: "#339af0",
  [CSS_VAR.BORDER_ERROR]: "#fa5252",

  [CSS_VAR.ACCENT]: "#228be6",
  [CSS_VAR.ACCENT_HOVER]: "#1c7ed6",
  [CSS_VAR.ACCENT_ACTIVE]: "#1971c2",
  [CSS_VAR.ACCENT_FG]: "#ffffff",

  [CSS_VAR.SUCCESS]: "#40c057",
  [CSS_VAR.WARNING]: "#fab005",
  [CSS_VAR.ERROR]: "#fa5252",
  [CSS_VAR.INFO]: "#339af0",

  [CSS_VAR.SHADOW_SM]: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
  [CSS_VAR.SHADOW_MD]: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
  [CSS_VAR.SHADOW_LG]: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
  [CSS_VAR.SHADOW_XL]: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
} as const;

// ─── Dark theme values ───────────────────────────────────────────────────────

export const DARK_THEME_VALUES: Record<string, string> = {
  [CSS_VAR.BG_PRIMARY]: "#0d1117",
  [CSS_VAR.BG_SECONDARY]: "#161b22",
  [CSS_VAR.BG_TERTIARY]: "#1c2129",
  [CSS_VAR.BG_INVERTED]: "#ffffff",
  [CSS_VAR.BG_OVERLAY]: "rgba(0, 0, 0, 0.7)",
  [CSS_VAR.BG_MUTE]: "#1a1f27",
  [CSS_VAR.BG_HOVER]: "#21262d",
  [CSS_VAR.BG_ACTIVE]: "#30363d",
  [CSS_VAR.BG_SELECTED]: "#1a2332",
  [CSS_VAR.BG_SOLID]: "#f0f6fc",

  [CSS_VAR.TEXT_PRIMARY]: "#e6edf3",
  [CSS_VAR.TEXT_SECONDARY]: "#8b949e",
  [CSS_VAR.TEXT_TERTIARY]: "#6e7681",
  [CSS_VAR.TEXT_INVERTED]: "#0d1117",
  [CSS_VAR.TEXT_MUTE]: "#484f58",
  [CSS_VAR.TEXT_LINK]: "#58a6ff",
  [CSS_VAR.TEXT_LINK_HOVER]: "#79c0ff",

  [CSS_VAR.BORDER_PRIMARY]: "#30363d",
  [CSS_VAR.BORDER_SECONDARY]: "#21262d",
  [CSS_VAR.BORDER_FOCUS]: "#58a6ff",
  [CSS_VAR.BORDER_ERROR]: "#f85149",

  [CSS_VAR.ACCENT]: "#58a6ff",
  [CSS_VAR.ACCENT_HOVER]: "#79c0ff",
  [CSS_VAR.ACCENT_ACTIVE]: "#a5d6ff",
  [CSS_VAR.ACCENT_FG]: "#0d1117",

  [CSS_VAR.SUCCESS]: "#3fb950",
  [CSS_VAR.WARNING]: "#d29922",
  [CSS_VAR.ERROR]: "#f85149",
  [CSS_VAR.INFO]: "#58a6ff",

  [CSS_VAR.SHADOW_SM]: "0 0 0 1px rgba(48, 54, 61, 0.5)",
  [CSS_VAR.SHADOW_MD]: "0 3px 6px rgba(0, 0, 0, 0.3)",
  [CSS_VAR.SHADOW_LG]: "0 8px 24px rgba(0, 0, 0, 0.4)",
  [CSS_VAR.SHADOW_XL]: "0 12px 48px rgba(0, 0, 0, 0.5)",
} as const;

// ─── Shared radius values ────────────────────────────────────────────────────

export const RADIUS_VALUES: Record<string, string> = {
  [CSS_VAR.RADIUS_SM]: "4px",
  [CSS_VAR.RADIUS_MD]: "8px",
  [CSS_VAR.RADIUS_LG]: "12px",
  [CSS_VAR.RADIUS_XL]: "16px",
  [CSS_VAR.RADIUS_FULL]: "9999px",
};

// ─── Z-index layers ──────────────────────────────────────────────────────────

export const Z_INDEX: Record<string, string> = {
  [CSS_VAR.Z_DROPDOWN]: "1000",
  [CSS_VAR.Z_STICKY]: "1020",
  [CSS_VAR.Z_FIXED]: "1030",
  [CSS_VAR.Z_OVERLAY]: "1040",
  [CSS_VAR.Z_MODAL]: "1050",
  [CSS_VAR.Z_POPOVER]: "1060",
  [CSS_VAR.Z_TOAST]: "1070",
  [CSS_VAR.Z_TOOLTIP]: "1080",
};

// ─── Breakpoint values ──────────────────────────────────────────────────────

export const BREAKPOINTS = {
  xs: "480px",
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px",
  "2xl": "1536px",
} as const;

export const BREAKPOINT_QUERIES = {
  xs: `(max-width: ${BREAKPOINTS.xs})`,
  sm: `(max-width: ${BREAKPOINTS.sm})`,
  md: `(max-width: ${BREAKPOINTS.md})`,
  lg: `(max-width: ${BREAKPOINTS.lg})`,
  xl: `(max-width: ${BREAKPOINTS.xl})`,
  "2xl": `(max-width: ${BREAKPOINTS["2xl"]})`,
} as const;

// ─── Font sizes ─────────────────────────────────────────────────────────────

export const FONT_SIZE_SCALES: Record<string, string> = {
  xs: "12px",
  sm: "14px",
  md: "16px",
  lg: "18px",
  xl: "20px",
};

export const FONT_FAMILIES = {
  system:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  serif: 'Georgia, "Times New Roman", Times, serif',
  mono: '"SF Mono", "Fira Code", "Fira Mono", "Roboto Mono", monospace',
  inter: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
} as const;

// ─── Transition durations ────────────────────────────────────────────────────

export const TRANSITIONS = {
  fast: "100ms",
  normal: "200ms",
  slow: "300ms",
  slower: "500ms",
} as const;

// ─── Animation reduced motion media query ────────────────────────────────────

export const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

// ─── Color mode detection ────────────────────────────────────────────────────

export const COLOR_SCHEME_MEDIA_QUERY = "(prefers-color-scheme: dark)";