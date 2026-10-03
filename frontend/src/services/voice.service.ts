import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const VoiceProvider = {
  OpenAI: "openai",
} as const;

export type VoiceProvider = (typeof VoiceProvider)[keyof typeof VoiceProvider];

export const AudioFormat = {
  MP3: "mp3",
  WAV: "wav",
  OGG: "ogg",
  FLAC: "flac",
  WEBM: "webm",
} as const;

export type AudioFormat = (typeof AudioFormat)[keyof typeof AudioFormat];

export const TranscriptionStatus = {
  Pending: "pending",
  Processing: "processing",
  Completed: "completed",
  Failed: "failed",
} as const;

export type TranscriptionStatus = (typeof TranscriptionStatus)[keyof typeof TranscriptionStatus];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface Voice {
  id: string;
  name: string;
  provider: VoiceProvider;
  language: string;
  gender?: string;
  previewUrl?: string;
  metadata?: Record<string, unknown>;
}

export interface Transcription {
  id: string;
  text: string;
  language: string;
  status: TranscriptionStatus;
  confidence: number;
  durationSeconds: number;
  wordTimestamps?: WordTimestamp[];
  createdAt: string;
  completedAt?: string;
}

export interface WordTimestamp {
  word: string;
  startTime: number;
  endTime: number;
  confidence: number;
}

export interface VoiceSession {
  id: string;
  provider: VoiceProvider;
  voiceId: string;
  createdAt: string;
  expiresAt: string;
  metadata?: Record<string, unknown>;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface TextToSpeechPayload {
  text: string;
  voiceId: string;
  provider?: VoiceProvider;
  format?: AudioFormat;
  speed?: number;
  pitch?: number;
}

export interface SpeechToTextPayload {
  audio: Blob;
  language?: string;
  format?: AudioFormat;
  includeTimestamps?: boolean;
}

export interface CreateVoiceSessionPayload {
  provider?: VoiceProvider;
  voiceId: string;
  metadata?: Record<string, unknown>;
}

export interface ListVoicesParams {
  provider?: VoiceProvider;
  language?: string;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface TextToSpeechResponse {
  audioUrl: string;
  format: AudioFormat;
  durationSeconds: number;
  characterCount: number;
}

export interface SpeechToTextResponse {
  transcription: Transcription;
}

export interface VoiceListResponse {
  voices: Voice[];
  total: number;
}

export interface VoiceSessionResponse {
  session: VoiceSession;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  TTS: "/api/v1/voice/tts",
  STT: "/api/v1/voice/stt",
  VOICES: "/api/v1/voice/voices",
  SESSIONS: "/api/v1/voice/sessions",
  SESSION_BY_ID: (id: string) => `/api/v1/voice/sessions/${id}`,
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const textToSpeech = async (
  payload: TextToSpeechPayload
): Promise<TextToSpeechResponse> => {
  const response: AxiosResponse<TextToSpeechResponse> =
    await api.post<TextToSpeechResponse>(ENDPOINTS.TTS, payload);
  return response.data;
};

const speechToText = async (
  payload: SpeechToTextPayload
): Promise<SpeechToTextResponse> => {
  const formData = new FormData();
  formData.append("audio", payload.audio, `audio.${payload.format ?? "webm"}`);
  if (payload.language) formData.append("language", payload.language);
  if (payload.format) formData.append("format", payload.format);
  if (payload.includeTimestamps !== undefined)
    formData.append("includeTimestamps", String(payload.includeTimestamps));

  const response: AxiosResponse<SpeechToTextResponse> =
    await api.post<SpeechToTextResponse>(ENDPOINTS.STT, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  return response.data;
};

const listVoices = async (
  params?: ListVoicesParams
): Promise<VoiceListResponse> => {
  const response: AxiosResponse<VoiceListResponse> =
    await api.get<VoiceListResponse>(ENDPOINTS.VOICES, { params });
  return response.data;
};

const createVoiceSession = async (
  payload: CreateVoiceSessionPayload
): Promise<VoiceSessionResponse> => {
  const response: AxiosResponse<VoiceSessionResponse> =
    await api.post<VoiceSessionResponse>(ENDPOINTS.SESSIONS, payload);
  return response.data;
};

const getVoiceSession = async (id: string): Promise<VoiceSessionResponse> => {
  const response: AxiosResponse<VoiceSessionResponse> =
    await api.get<VoiceSessionResponse>(ENDPOINTS.SESSION_BY_ID(id));
  return response.data;
};

const deleteVoiceSession = async (
  id: string
): Promise<{ success: boolean }> => {
  const response: AxiosResponse<{ success: boolean }> = await api.delete<{
    success: boolean;
  }>(ENDPOINTS.SESSION_BY_ID(id));
  return response.data;
};

export const VoiceService = {
  textToSpeech,
  speechToText,
  listVoices,
  createVoiceSession,
  getVoiceSession,
  deleteVoiceSession,
} as const;

export default VoiceService;