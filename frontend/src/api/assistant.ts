import api from "./client";

export interface WakeWordStatus {
  enabled: boolean;
  state: string;
  wake_word: string;
  activation_count: number;
  cooldown_seconds: number;
}

export interface WakeWordResponse {
  success: boolean;
  activated: boolean;
  state: string;
  message: string;
}

export interface AssistantStateResponse {
  state: string;
  is_active: boolean;
  is_available: boolean;
  idle_timeout_seconds: number;
}

export interface GreetingResponse {
  greeting: string;
  personality: string;
  time_of_day: string;
  tone: string;
}

export interface PersonalityItem {
  id: string;
  name: string;
  description: string;
}

export const assistantApi = {
  getWakeWordStatus: async (): Promise<WakeWordStatus> => {
    const { data } = await api.get("/api/v1/voice/wake-word");
    return data;
  },

  toggleWakeWord: async (enabled: boolean): Promise<WakeWordStatus> => {
    const { data } = await api.put("/api/v1/voice/wake-word", { enabled });
    return data;
  },

  activateWakeWord: async (
    confidence?: number
  ): Promise<WakeWordResponse> => {
    const { data } = await api.post("/api/v1/voice/wake-word/activate", {
      confidence: confidence ?? 1.0,
    });
    return data;
  },

  deactivateWakeWord: async (): Promise<WakeWordResponse> => {
    const { data } = await api.post("/api/v1/voice/wake-word/deactivate");
    return data;
  },

  getAssistantState: async (): Promise<AssistantStateResponse> => {
    const { data } = await api.get("/api/v1/voice/state");
    return data;
  },

  transitionState: async (
    target: string,
    reason?: string
  ): Promise<AssistantStateResponse> => {
    const { data } = await api.post("/api/v1/voice/state/transition", {
      target,
      reason,
    });
    return data;
  },

  getGreeting: async (
    personality?: string,
    userName?: string | null,
    classicMode?: boolean,
    greetingType?: string
  ): Promise<GreetingResponse> => {
    const params = new URLSearchParams();
    if (personality) params.set("personality", personality);
    if (userName) params.set("user_name", userName);
    if (classicMode) params.set("classic_mode", "true");
    if (greetingType) params.set("greeting_type", greetingType ?? "greeting");
    const { data } = await api.get(
      `/api/v1/voice/greeting?${params.toString()}`
    );
    return data;
  },

  getPersonalities: async (): Promise<PersonalityItem[]> => {
    const { data } = await api.get("/api/v1/voice/personalities");
    return data;
  },
};
