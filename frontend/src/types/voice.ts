// frontend/src/types/voice.ts

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum AudioFormat {
  PCM16 = "pcm16",
  PCM32 = "pcm32",
  Opus = "opus",
  MP3 = "mp3",
  WAV = "wav",
  FLAC = "flac",
  AAC = "aac",
  OGG = "ogg",
}

export enum AudioSampleRate {
  Rate8000 = 8000,
  Rate16000 = 16000,
  Rate22050 = 22050,
  Rate24000 = 24000,
  Rate44100 = 44100,
  Rate48000 = 48000,
}

export enum VoiceProvider {
  OpenAI = "openai",
  System = "system",
}

export enum VoiceGender {
  Male = "male",
  Female = "female",
  Neutral = "neutral",
}

export enum VoiceSessionStatus {
  Initializing = "initializing",
  Active = "active",
  Paused = "paused",
  Ending = "ending",
  Ended = "ended",
  Error = "error",
}

export enum VoiceActivityState {
  Active = "active",
  Inactive = "inactive",
  Transitioning = "transitioning",
}

export enum VoiceSessionEndReason {
  UserEnded = "user_ended",
  Timeout = "timeout",
  Error = "error",
  ServerEnded = "server_ended",
  NetworkLost = "network_lost",
  AuthExpired = "auth_expired",
}

export enum TranscriptionLanguage {
  Auto = "auto",
  En = "en",
  Es = "es",
  Fr = "fr",
  De = "de",
  Zh = "zh",
  Ja = "ja",
  Ko = "ko",
  Pt = "pt",
  Ar = "ar",
  Ru = "ru",
  Hi = "hi",
}

// ─── Voice ────────────────────────────────────────────────────────────────────

export interface Voice {
  id: string;
  name: string;
  provider: VoiceProvider;
  gender: VoiceGender;
  language: string;
  accent?: string;
  description?: string;
  preview_url?: string;
  is_custom: boolean;
  is_available: boolean;
  supported_formats: AudioFormat[];
  supported_sample_rates: AudioSampleRate[];
  metadata?: Record<string, unknown>;
}

// ─── Session config ───────────────────────────────────────────────────────────

export interface VoiceSessionConfig {
  session_id: string;
  voice_id: string;
  provider: VoiceProvider;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  channels: 1 | 2;
  chunk_size_ms: number;
  language: TranscriptionLanguage;
  enable_vad: boolean;
  enable_transcription: boolean;
  enable_noise_cancellation: boolean;
  enable_echo_cancellation: boolean;
  vad_threshold?: number;
  silence_timeout_ms?: number;
  max_duration_ms?: number;
  speed?: number;
  pitch?: number;
}

// ─── Session ─────────────────────────────────────────────────────────────────

export interface VoiceSession {
  id: string;
  status: VoiceSessionStatus;
  config: VoiceSessionConfig;
  conversation_id?: string;
  started_at: number;
  ended_at?: number;
  duration_ms?: number;
  total_audio_bytes: number;
  total_transcription_chars: number;
  end_reason?: VoiceSessionEndReason;
  error?: string;
  metadata?: Record<string, unknown>;
}

// ─── Transcription ────────────────────────────────────────────────────────────

export interface WordTimestamp {
  word: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
}

export interface Transcription {
  id: string;
  session_id: string;
  text: string;
  is_partial: boolean;
  confidence: number;
  language: string;
  words?: WordTimestamp[];
  started_at_ms: number;
  ended_at_ms: number;
  duration_ms: number;
  audio_bytes?: number;
}

// ─── Synthesis ────────────────────────────────────────────────────────────────

export interface SynthesisRequest {
  text: string;
  voice_id: string;
  provider: VoiceProvider;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  speed?: number;
  pitch?: number;
  volume?: number;
  stream?: boolean;
  emotion?: string;
  style?: string;
}

export interface SynthesisResult {
  request_id: string;
  audio_url?: string;
  audio_data?: ArrayBuffer;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  duration_ms: number;
  size_bytes: number;
  text: string;
  voice_id: string;
  created_at: number;
}

export interface SynthesisChunk {
  request_id: string;
  session_id: string;
  chunk_index: number;
  is_final: boolean;
  duration_ms: number;
  text_offset: number;
  word_boundary?: string;
  size_bytes: number;
}

// ─── VAD ─────────────────────────────────────────────────────────────────────

export interface VoiceActivityEvent {
  session_id: string;
  state: VoiceActivityState;
  timestamp_ms: number;
  confidence: number;
  duration_ms?: number;
}

// ─── Preferences ─────────────────────────────────────────────────────────────

export interface VoicePreferences {
  enabled: boolean;
  provider: VoiceProvider;
  voice_id: string;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  language: TranscriptionLanguage;
  speed: number;
  pitch: number;
  volume: number;
  auto_play_responses: boolean;
  wake_word_enabled: boolean;
  wake_word: string;
  enable_vad: boolean;
  enable_noise_cancellation: boolean;
  enable_echo_cancellation: boolean;
}