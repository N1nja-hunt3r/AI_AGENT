/**
 * Store Registry
 *
 * Central export point for all Zustand stores.
 * Import stores from here to maintain a single source of truth
 * and enable consistent usage patterns across the application.
 */

// ─── Chat Store ───────────────────────────────────────────────────────────────

export {
  useChatStore,
  type ChatStore,
  type ChatState,
  type ChatActions,
  type Message,
  type MessageRole,
  type MessageStatus,
  type ContentPart,
  type Conversation,
  type StreamingState,
  type ModelConfig,
} from "./chatStore";

// ─── Memory Store ─────────────────────────────────────────────────────────────

export {
  useMemoryStore,
  type MemoryStore,
  type MemoryState,
  type MemoryActions,
  type Memory,
  type MemoryType,
  type MemoryStatus,
  type MemoryTag,
  type MemoryFilters,
  type MemorySearchState,
  type SortField,
  type SortOrder,
} from "./memoryStore";

// ─── Settings Store ───────────────────────────────────────────────────────────

export {
  useSettingsStore,
  type SettingsStore,
  type SettingsState,
  type SettingsActions,
  type Theme,
  type Language,
  type ProviderConfig,
  type ProviderStatus,
  type ModelPreference,
  type NotificationPreferences,
  type AccessibilityPreferences,
  type ChatPreferences,
  type MemoryPreferences,
} from "./settingsStore";

// ─── UI Store ─────────────────────────────────────────────────────────────────

export {
  useUIStore,
  type UIStore,
  type UIState,
  type UIActions,
  type UINotification,
  type NotificationSeverity,
  type DialogId,
  type DialogState,
} from "./uiStore";

// ─── Agent Store ──────────────────────────────────────────────────────────────

export {
  useAgentStore,
  type AgentStore,
  type AgentState,
  type AgentActions,
  type Agent,
  type AgentStatus,
  type AgentTask,
  type TaskStatus,
  type TaskPriority,
  type ToolCall,
  type ToolCallStatus,
  type HealthCheckResult,
  type LatencyMetrics,
  type ResourceUsage,
} from "./agentStore";

// ─── Voice Store ──────────────────────────────────────────────────────────────

export {
  useVoiceStore,
  type VoiceState,
  type VoiceActions,
  type AssistantState,
  type VoicePersonality,
} from "./voiceStore";

// ─── Composite selectors ──────────────────────────────────────────────────────

import { useChatStore } from "./chatStore";
import { useMemoryStore } from "./memoryStore";
import { useSettingsStore } from "./settingsStore";
import { useUIStore } from "./uiStore";
import { useAgentStore } from "./agentStore";
import { useVoiceStore } from "./voiceStore";

/**
 * Returns a snapshot of all store states for debugging / devtools.
 * Not intended for use in components.
 */
export const getAllStoreStates = () => ({
  chat: useChatStore.getState(),
  memory: useMemoryStore.getState(),
  settings: useSettingsStore.getState(),
  ui: useUIStore.getState(),
  agent: useAgentStore.getState(),
  voice: useVoiceStore.getState(),
});

/**
 * Reset all non-persisted stores to their initial state.
 * Useful for logout flows or test teardown.
 */
export const resetAllTransientStores = (): void => {
  useUIStore.getState().closeAllDialogs();
  useUIStore.getState().dismissAllNotifications();
  useAgentStore.setState({
    tasks: {},
    taskQueue: [],
    globalHealth: "unknown",
    isPolling: false,
    error: null,
  });
  useMemoryStore.getState().clearSearch();
  useMemoryStore.getState().clearFilters();
};

/**
 * Typed store hook map for dynamic store access patterns.
 */
export const storeHooks = {
  chat: useChatStore,
  memory: useMemoryStore,
  settings: useSettingsStore,
  ui: useUIStore,
  agent: useAgentStore,
} as const;

export type StoreKey = keyof typeof storeHooks;