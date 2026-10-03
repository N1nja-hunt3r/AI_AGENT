import api from "@/api/client";

export interface VoiceItem {
  id: string;
  name: string;
  gender?: string;
  language: string;
  category: string;
}

export interface VoicesResponse {
  voices: VoiceItem[];
}

export interface TTSRequest {
  text: string;
  voice?: string;
  language?: string;
  speed?: number;
  format?: string;
  options?: Record<string, unknown>;
}

export interface TTSResponse {
  audio_url: string;
  format: string;
  duration_seconds: number;
  model: string;
  latency_ms: number;
}

export interface STTResponse {
  text: string;
  model: string;
  language?: string;
  latency_ms: number;
  confidence: number;
}

export interface CreateSessionRequest {
  title?: string;
  voice?: string;
  language?: string;
  settings?: Record<string, unknown>;
}

export interface VoiceSession {
  id: string;
  title: string;
  voice?: string;
  language: string;
  settings: Record<string, unknown>;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface CreateSessionResponse {
  session: VoiceSession;
}

export interface GetSessionResponse {
  session: VoiceSession;
}

export interface DeleteSessionResponse {
  success: boolean;
  id: string;
  deleted_at: string;
}

const ENDPOINTS = {
  TTS: "/api/v1/voice/tts",
  STT: "/api/v1/voice/stt",
  VOICES: "/api/v1/voice/voices",
  SESSIONS: "/api/v1/voice/sessions",
  SESSION_BY_ID: (id: string) => `/api/v1/voice/sessions/${id}`,
} as const;

export const listVoices = async (): Promise<VoicesResponse> => {
  const { data } = await api.get<VoicesResponse>(ENDPOINTS.VOICES);
  return data;
};

export const textToSpeech = async (payload: TTSRequest): Promise<TTSResponse> => {
  const { data } = await api.post<TTSResponse>(ENDPOINTS.TTS, payload);
  return data;
};

export const speechToText = async (file: File, language?: string): Promise<STTResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  if (language) formData.append("language", language);
  const { data } = await api.post<STTResponse>(ENDPOINTS.STT, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const createVoiceSession = async (payload: CreateSessionRequest): Promise<CreateSessionResponse> => {
  const { data } = await api.post<CreateSessionResponse>(ENDPOINTS.SESSIONS, payload);
  return data;
};

export const getVoiceSession = async (id: string): Promise<GetSessionResponse> => {
  const { data } = await api.get<GetSessionResponse>(ENDPOINTS.SESSION_BY_ID(id));
  return data;
};

export const deleteVoiceSession = async (id: string): Promise<DeleteSessionResponse> => {
  const { data } = await api.delete<DeleteSessionResponse>(ENDPOINTS.SESSION_BY_ID(id));
  return data;
};
