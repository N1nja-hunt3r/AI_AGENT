// frontend/src/context/registry.ts

// ─── Chat context ─────────────────────────────────────────────────────────────

export {
  ChatContext,
  ChatProvider,
  useChatContext,
  type ChatContextValue,
  type ChatContextError,
  type TypingIndicatorState,
  type ConversationContextMode,
  type ScrollBehavior,
  type MessageDraft,
  type ConversationSession,
  type ChatProviderProps,
} from "./ChatContext";

// ─── Voice context ────────────────────────────────────────────────────────────

export {
  VoiceContext,
  VoiceProvider,
  useVoiceContext,
  type VoiceContextValue,
  type VoicePermissionState,
  type RecordingState,
  type PlaybackState,
  type AudioFormat,
  type AudioDeviceInfo,
  type RecordingSession,
  type AudioLevelSample,
  type VoiceContextError,
  type StartRecordingOptions,
  type PlaybackOptions,
  type VoiceProviderProps,
} from "./VoiceContext";

// ─── Theme context ────────────────────────────────────────────────────────────

export {
  ThemeContext,
  ThemeContextProvider,
  useThemeContext,
  useIsDark,
  useIsLight,
  useResolvedTheme,
  usePrefersReducedMotion,
  useTypography,
  type ThemeContextValue,
  type ThemeMode,
  type ResolvedThemeMode,
  type ColorScheme,
  type FontSize,
  type FontFamily,
  type MotionPreference,
  type ThemeColors,
  type ThemeTypography,
  type ThemeProviderProps,
} from "./ThemeContext";

// ─── Composed application context provider ────────────────────────────────────

import React, { type ReactNode } from "react";
import { ChatProvider, type ChatProviderProps } from "./ChatContext";
import { VoiceProvider, type VoiceProviderProps } from "./VoiceContext";
import {
  ThemeContextProvider,
  type ThemeProviderProps,
} from "./ThemeContext";

export interface AppContextProviderProps
  extends Partial<ChatProviderProps>,
    Partial<VoiceProviderProps>,
    Partial<ThemeProviderProps> {
  children: ReactNode;
}

/**
 * Composes all application-level React contexts into a single provider.
 *
 * @example
 * <AppContextProvider default_mode="dark">
 *   <App />
 * </AppContextProvider>
 */
export const AppContextProvider = ({
  children,
  // ChatProvider props
  initial_mode,
  initial_conversation,
  max_message_history,
  // VoiceProvider props
  default_volume,
  default_vad_enabled,
  default_vad_threshold,
  default_noise_cancellation,
  max_sessions,
  // ThemeContextProvider props
  default_mode,
  default_color_scheme,
  default_typography,
  storage_key,
  disable_system_sync,
  attribute,
}: AppContextProviderProps): React.JSX.Element => {
  return React.createElement(ThemeContextProvider, {
    default_mode,
    default_color_scheme,
    default_typography,
    storage_key,
    disable_system_sync,
    attribute,
    children: React.createElement(VoiceProvider, {
      default_volume,
      default_vad_enabled,
      default_vad_threshold,
      default_noise_cancellation,
      max_sessions,
      children: React.createElement(ChatProvider, {
        initial_mode,
        initial_conversation,
        max_message_history,
        children,
      }),
    }),
  });
};