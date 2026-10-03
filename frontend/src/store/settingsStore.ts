import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ─── Types ────────────────────────────────────────────────────────────────────

export type Theme = "light" | "dark" | "system";

export type Language = "en" | "es" | "fr" | "de" | "ja" | "zh";

export type ProviderStatus = "connected" | "disconnected" | "error" | "checking";

export interface ProviderConfig {
  id: string;
  name: string;
  type: "nvidia_nim" | "openai" | "custom";
  apiKey?: string;
  baseUrl?: string;
  organizationId?: string;
  status: ProviderStatus;
  isEnabled: boolean;
  lastChecked: number | null;
  errorMessage?: string;
  metadata?: Record<string, unknown>;
}

export interface ModelPreference {
  providerId: string;
  modelId: string;
  displayName: string;
  isDefault: boolean;
  parameters: {
    temperature: number;
    maxTokens: number;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;
    stopSequences: string[];
  };
}

export interface NotificationPreferences {
  enableDesktop: boolean;
  enableSound: boolean;
  enableEmail: boolean;
  emailAddress?: string;
  streamingComplete: boolean;
  agentErrors: boolean;
  memoryLimitWarning: boolean;
}

export interface AccessibilityPreferences {
  reduceMotion: boolean;
  highContrast: boolean;
  fontSize: "sm" | "md" | "lg" | "xl";
  keyboardNavigation: boolean;
}

export interface ChatPreferences {
  sendOnEnter: boolean;
  showTokenCount: boolean;
  showModelBadge: boolean;
  autoScroll: boolean;
  codeHighlighting: boolean;
  renderMarkdown: boolean;
  compactMode: boolean;
  showTimestamps: boolean;
}

export interface MemoryPreferences {
  autoSave: boolean;
  maxMemories: number;
  defaultMemoryType: "episodic" | "semantic";
  enableEmbeddings: boolean;
  retentionDays: number | null;
}

export interface SettingsState {
  // Theme
  theme: Theme;
  language: Language;

  // Providers
  providers: Record<string, ProviderConfig>;

  // Models
  modelPreferences: ModelPreference[];
  defaultModelId: string | null;

  // Preferences
  notifications: NotificationPreferences;
  accessibility: AccessibilityPreferences;
  chat: ChatPreferences;
  memory: MemoryPreferences;

  // State
  isDirty: boolean;
  isSaving: boolean;
  lastSaved: number | null;
  error: string | null;
}

export interface SettingsActions {
  // Theme
  setTheme: (theme: Theme) => void;
  setLanguage: (language: Language) => void;

  // Providers
  addProvider: (provider: Omit<ProviderConfig, "status" | "lastChecked">) => void;
  updateProvider: (id: string, patch: Partial<Omit<ProviderConfig, "id">>) => void;
  removeProvider: (id: string) => void;
  setProviderStatus: (id: string, status: ProviderStatus, errorMessage?: string) => void;
  toggleProvider: (id: string) => void;
  setProviders: (providers: ProviderConfig[]) => void;

  // Models
  addModelPreference: (model: ModelPreference) => void;
  updateModelPreference: (modelId: string, patch: Partial<Omit<ModelPreference, "modelId" | "providerId">>) => void;
  removeModelPreference: (modelId: string) => void;
  setDefaultModel: (modelId: string | null) => void;

  // Preferences
  updateNotifications: (patch: Partial<NotificationPreferences>) => void;
  updateAccessibility: (patch: Partial<AccessibilityPreferences>) => void;
  updateChatPreferences: (patch: Partial<ChatPreferences>) => void;
  updateMemoryPreferences: (patch: Partial<MemoryPreferences>) => void;

  // Persistence
  markSaved: () => void;
  setIsSaving: (isSaving: boolean) => void;
  setError: (error: string | null) => void;
  resetToDefaults: () => void;

  // Derived
  getProviderById: (id: string) => ProviderConfig | null;
  getConnectedProviders: () => ProviderConfig[];
  getDefaultModelPreference: () => ModelPreference | null;
}

export type SettingsStore = SettingsState & SettingsActions;

// ─── Defaults ─────────────────────────────────────────────────────────────────

const DEFAULT_NOTIFICATIONS: NotificationPreferences = {
  enableDesktop: true,
  enableSound: false,
  enableEmail: false,
  streamingComplete: true,
  agentErrors: true,
  memoryLimitWarning: true,
};

const DEFAULT_ACCESSIBILITY: AccessibilityPreferences = {
  reduceMotion: false,
  highContrast: false,
  fontSize: "md",
  keyboardNavigation: true,
};

const DEFAULT_CHAT: ChatPreferences = {
  sendOnEnter: true,
  showTokenCount: true,
  showModelBadge: true,
  autoScroll: true,
  codeHighlighting: true,
  renderMarkdown: true,
  compactMode: false,
  showTimestamps: false,
};

const DEFAULT_MEMORY: MemoryPreferences = {
  autoSave: true,
  maxMemories: 10_000,
  defaultMemoryType: "episodic",
  enableEmbeddings: true,
  retentionDays: null,
};

const DEFAULT_STATE: Omit<SettingsState, "isDirty" | "isSaving" | "lastSaved" | "error"> = {
  theme: "system",
  language: "en",
  providers: {},
  modelPreferences: [],
  defaultModelId: null,
  notifications: DEFAULT_NOTIFICATIONS,
  accessibility: DEFAULT_ACCESSIBILITY,
  chat: DEFAULT_CHAT,
  memory: DEFAULT_MEMORY,
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useSettingsStore = create<SettingsStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // ── Initial state ────────────────────────────────────────────────────
        ...DEFAULT_STATE,
        isDirty: false,
        isSaving: false,
        lastSaved: null,
        error: null,

        // ── Theme ────────────────────────────────────────────────────────────
        setTheme: (theme) => {
          set((state) => {
            state.theme = theme;
            state.isDirty = true;
          });
        },

        setLanguage: (language) => {
          set((state) => {
            state.language = language;
            state.isDirty = true;
          });
        },

        // ── Providers ────────────────────────────────────────────────────────
        addProvider: (provider) => {
          set((state) => {
            state.providers[provider.id] = {
              ...provider,
              status: "disconnected",
              lastChecked: null,
            };
            state.isDirty = true;
          });
        },

        updateProvider: (id, patch) => {
          set((state) => {
            const provider = state.providers[id];
            if (provider) {
              Object.assign(provider, patch);
              state.isDirty = true;
            }
          });
        },

        removeProvider: (id) => {
          set((state) => {
            delete state.providers[id];
            state.isDirty = true;
          });
        },

        setProviderStatus: (id, status, errorMessage) => {
          set((state) => {
            const provider = state.providers[id];
            if (provider) {
              provider.status = status;
              provider.lastChecked = Date.now();
              provider.errorMessage = errorMessage;
            }
          });
        },

        toggleProvider: (id) => {
          set((state) => {
            const provider = state.providers[id];
            if (provider) {
              provider.isEnabled = !provider.isEnabled;
              state.isDirty = true;
            }
          });
        },

        setProviders: (providers) => {
          set((state) => {
            state.providers = Object.fromEntries(providers.map((p) => [p.id, p]));
          });
        },

        // ── Models ────────────────────────────────────────────────────────────
        addModelPreference: (model) => {
          set((state) => {
            const existing = state.modelPreferences.findIndex(
              (m) => m.modelId === model.modelId
            );
            if (existing >= 0) {
              state.modelPreferences[existing] = model;
            } else {
              state.modelPreferences.push(model);
            }
            if (model.isDefault) {
              state.defaultModelId = model.modelId;
            }
            state.isDirty = true;
          });
        },

        updateModelPreference: (modelId, patch) => {
          set((state) => {
            const model = state.modelPreferences.find((m) => m.modelId === modelId);
            if (model) {
              Object.assign(model, patch);
              state.isDirty = true;
            }
          });
        },

        removeModelPreference: (modelId) => {
          set((state) => {
            state.modelPreferences = state.modelPreferences.filter(
              (m) => m.modelId !== modelId
            );
            if (state.defaultModelId === modelId) {
              state.defaultModelId = null;
            }
            state.isDirty = true;
          });
        },

        setDefaultModel: (modelId) => {
          set((state) => {
            state.defaultModelId = modelId;
            state.modelPreferences.forEach((m) => {
              m.isDefault = m.modelId === modelId;
            });
            state.isDirty = true;
          });
        },

        // ── Preferences ───────────────────────────────────────────────────────
        updateNotifications: (patch) => {
          set((state) => {
            Object.assign(state.notifications, patch);
            state.isDirty = true;
          });
        },

        updateAccessibility: (patch) => {
          set((state) => {
            Object.assign(state.accessibility, patch);
            state.isDirty = true;
          });
        },

        updateChatPreferences: (patch) => {
          set((state) => {
            Object.assign(state.chat, patch);
            state.isDirty = true;
          });
        },

        updateMemoryPreferences: (patch) => {
          set((state) => {
            Object.assign(state.memory, patch);
            state.isDirty = true;
          });
        },

        // ── Persistence ───────────────────────────────────────────────────────
        markSaved: () => {
          set((state) => {
            state.isDirty = false;
            state.isSaving = false;
            state.lastSaved = Date.now();
            state.error = null;
          });
        },

        setIsSaving: (isSaving) => {
          set((state) => {
            state.isSaving = isSaving;
          });
        },

        setError: (error) => {
          set((state) => {
            state.error = error;
            state.isSaving = false;
          });
        },

        resetToDefaults: () => {
          set((state) => {
            Object.assign(state, DEFAULT_STATE);
            state.isDirty = true;
          });
        },

        // ── Derived ───────────────────────────────────────────────────────────
        getProviderById: (id) => get().providers[id] ?? null,

        getConnectedProviders: () =>
          Object.values(get().providers).filter(
            (p) => p.isEnabled && p.status === "connected"
          ),

        getDefaultModelPreference: () => {
          const { modelPreferences, defaultModelId } = get();
          return (
            modelPreferences.find((m) => m.modelId === defaultModelId) ??
            modelPreferences.find((m) => m.isDefault) ??
            null
          );
        },
      })),
      {
        name: "settings-store",
        partialize: (state) => ({
          theme: state.theme,
          language: state.language,
          providers: state.providers,
          modelPreferences: state.modelPreferences,
          defaultModelId: state.defaultModelId,
          notifications: state.notifications,
          accessibility: state.accessibility,
          chat: state.chat,
          memory: state.memory,
        }),
      }
    ),
    { name: "SettingsStore" }
  )
);