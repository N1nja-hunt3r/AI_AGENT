import { create } from "zustand";

export type AssistantState =
  | "off"
  | "starting"
  | "initializing"
  | "ready"
  | "listening"
  | "thinking"
  | "researching"
  | "coding"
  | "speaking"
  | "idle"
  | "sleep"
  | "error";

export type VoicePersonality =
  | "professional"
  | "friendly"
  | "jarvis"
  | "minimal"
  | "custom";

export interface VoiceState {
  assistantState: AssistantState;
  wakeWordEnabled: boolean;
  wakeWordText: string;
  personality: VoicePersonality;
  userName: string | null;
  classicMode: boolean;
  isActivated: boolean;
  idleTimeoutSeconds: number;
  showGreeting: boolean;
  greetingText: string;
  audioLevel: number;
  isVoiceActive: boolean;
}

export interface VoiceActions {
  setAssistantState: (state: AssistantState) => void;
  setWakeWordEnabled: (enabled: boolean) => void;
  setWakeWordText: (text: string) => void;
  setPersonality: (personality: VoicePersonality) => void;
  setUserName: (name: string | null) => void;
  setClassicMode: (enabled: boolean) => void;
  setActivated: (activated: boolean) => void;
  setIdleTimeout: (seconds: number) => void;
  setShowGreeting: (show: boolean) => void;
  setGreetingText: (text: string) => void;
  setAudioLevel: (level: number) => void;
  setIsVoiceActive: (active: boolean) => void;
  reset: () => void;
}

const initialState: VoiceState = {
  assistantState: "off",
  wakeWordEnabled: true,
  wakeWordText: "hey aspire",
  personality: "professional",
  userName: null,
  classicMode: false,
  isActivated: false,
  idleTimeoutSeconds: 10,
  showGreeting: false,
  greetingText: "",
  audioLevel: 0,
  isVoiceActive: false,
};

export const useVoiceStore = create<VoiceState & VoiceActions>((set) => ({
  ...initialState,

  setAssistantState: (state) => set({ assistantState: state }),

  setWakeWordEnabled: (enabled) => set({ wakeWordEnabled: enabled }),

  setWakeWordText: (text) => set({ wakeWordText: text }),

  setPersonality: (personality) => set({ personality }),

  setUserName: (name) => set({ userName: name }),

  setClassicMode: (enabled) => set({ classicMode: enabled }),

  setActivated: (activated) => set({ isActivated: activated }),

  setIdleTimeout: (seconds) => set({ idleTimeoutSeconds: seconds }),

  setShowGreeting: (show) => set({ showGreeting: show }),

  setGreetingText: (text) => set({ greetingText: text }),

  setAudioLevel: (level) => set({ audioLevel: level }),

  setIsVoiceActive: (active) => set({ isVoiceActive: active }),

  reset: () => set(initialState),
}));
