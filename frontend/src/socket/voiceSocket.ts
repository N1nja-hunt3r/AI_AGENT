// frontend/src/socket/voiceSocket.ts

import { SocketManager, type SocketManagerConfig } from "./socket";

// ─── Audio format types ───────────────────────────────────────────────────────

export type AudioFormat = "pcm16" | "pcm32" | "opus" | "mp3" | "wav";

export type AudioSampleRate = 8000 | 16000 | 22050 | 24000 | 44100 | 48000;

export type VoiceActivityState = "active" | "inactive" | "transitioning";

// ─── Event payload types ──────────────────────────────────────────────────────

export interface VoiceSessionConfig {
  session_id: string;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  channels: 1 | 2;
  chunk_size_bytes: number;
  voice_id: string;
  language: string;
  enable_vad: boolean;
  enable_transcription: boolean;
  enable_timestamps: boolean;
}

export interface VoiceSessionStarted {
  session_id: string;
  config: VoiceSessionConfig;
  started_at: number;
}

export interface VoiceSessionEnded {
  session_id: string;
  duration_ms: number;
  total_audio_bytes: number;
  ended_at: number;
  reason: "user_ended" | "timeout" | "error" | "server_ended";
}

export interface VoiceAudioChunkMeta {
  session_id: string;
  chunk_index: number;
  chunk_size_bytes: number;
  timestamp_ms: number;
  is_final: boolean;
  duration_ms: number;
}

export interface VoiceTranscription {
  session_id: string;
  text: string;
  is_partial: boolean;
  confidence: number;
  language: string;
  words?: VoiceWordTimestamp[];
  started_at_ms: number;
  ended_at_ms: number;
}

export interface VoiceWordTimestamp {
  word: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
}

export interface VoiceActivityEvent {
  session_id: string;
  state: VoiceActivityState;
  timestamp_ms: number;
  confidence: number;
}

export interface VoiceSynthesisRequest {
  session_id: string;
  text: string;
  voice_id: string;
  format: AudioFormat;
  sample_rate: AudioSampleRate;
  speed: number;
  pitch: number;
  stream: boolean;
}

export interface VoiceSynthesisChunkMeta {
  session_id: string;
  request_id: string;
  chunk_index: number;
  is_final: boolean;
  duration_ms: number;
  text_offset: number;
  word_boundary?: string;
}

export interface VoiceSynthesisComplete {
  session_id: string;
  request_id: string;
  total_duration_ms: number;
  total_bytes: number;
  text: string;
}

export interface VoiceError {
  session_id: string;
  code: string;
  message: string;
  recoverable: boolean;
  timestamp: number;
}

// ─── Channel names ────────────────────────────────────────────────────────────

export const VOICE_CHANNELS = {
  session: (session_id: string) => `voice:session:${session_id}`,
  transcript: (session_id: string) => `voice:transcript:${session_id}`,
  synthesis: (session_id: string) => `voice:synthesis:${session_id}`,
  vad: (session_id: string) => `voice:vad:${session_id}`,
} as const;

// ─── Listener types ───────────────────────────────────────────────────────────

export interface VoiceSocketListeners {
  onSessionStarted?: (event: VoiceSessionStarted) => void;
  onSessionEnded?: (event: VoiceSessionEnded) => void;
  onAudioChunkMeta?: (meta: VoiceAudioChunkMeta) => void;
  onAudioData?: (data: ArrayBuffer, meta?: VoiceAudioChunkMeta) => void;
  onTranscription?: (event: VoiceTranscription) => void;
  onVoiceActivity?: (event: VoiceActivityEvent) => void;
  onSynthesisChunkMeta?: (meta: VoiceSynthesisChunkMeta) => void;
  onSynthesisAudio?: (data: ArrayBuffer, meta?: VoiceSynthesisChunkMeta) => void;
  onSynthesisComplete?: (event: VoiceSynthesisComplete) => void;
  onError?: (error: VoiceError) => void;
}

// ─── Voice Socket ─────────────────────────────────────────────────────────────

export class VoiceSocket {
  private readonly manager: SocketManager;
  private readonly activeSubscriptions = new Map<string, () => void>();

  /** Last received synthesis chunk meta — used to correlate binary frames */
  private pendingSynthesisMeta: VoiceSynthesisChunkMeta | null = null;

  /** Last received audio chunk meta — used to correlate binary frames */
  private pendingAudioMeta: VoiceAudioChunkMeta | null = null;

  constructor(config: SocketManagerConfig) {
    this.manager = new SocketManager({
      ...config,
      binaryType: "arraybuffer",
    });
  }

  getManager(): SocketManager {
    return this.manager;
  }

  // ── Session control ────────────────────────────────────────────────────────

  async startSession(config: VoiceSessionConfig): Promise<void> {
    await this.manager.send("event", config, {
      channel: VOICE_CHANNELS.session(config.session_id),
      waitForAck: true,
      timeoutMs: 15_000,
    });
  }

  async endSession(session_id: string): Promise<void> {
    await this.manager.send(
      "event",
      { session_id, action: "end" },
      { channel: VOICE_CHANNELS.session(session_id) }
    );
  }

  // ── Audio streaming ────────────────────────────────────────────────────────

  /**
   * Send a raw PCM audio chunk to the server.
   * The binary frame is preceded by a JSON meta frame.
   */
  sendAudioChunk(
    meta: Omit<VoiceAudioChunkMeta, "chunk_size_bytes">,
    audioData: ArrayBuffer
  ): boolean {
    const fullMeta: VoiceAudioChunkMeta = {
      ...meta,
      chunk_size_bytes: audioData.byteLength,
    };

    // Send metadata frame first
    void this.manager.send("event", fullMeta, {
      channel: VOICE_CHANNELS.session(meta.session_id),
    });

    // Send binary audio data
    return this.manager.sendBinary(audioData);
  }

  // ── Synthesis ──────────────────────────────────────────────────────────────

  async requestSynthesis(request: VoiceSynthesisRequest): Promise<string> {
    const frame = await this.manager.send("event", request, {
      channel: VOICE_CHANNELS.synthesis(request.session_id),
      waitForAck: true,
      timeoutMs: 10_000,
    });
    return frame.id ?? "";
  }

  // ── Subscriptions ──────────────────────────────────────────────────────────

  subscribeToSession(
    session_id: string,
    listeners: VoiceSocketListeners
  ): () => void {
    const unsubscribers: Array<() => void> = [];

    // Session events channel
    const sessionUnsub = this.manager.subscribe<
      | VoiceSessionStarted
      | VoiceSessionEnded
      | VoiceAudioChunkMeta
      | VoiceError
    >(VOICE_CHANNELS.session(session_id), (payload) => {
      const p = payload as unknown as Record<string, unknown>;

      if ("started_at" in p && "config" in p) {
        listeners.onSessionStarted?.(payload as VoiceSessionStarted);
      } else if ("duration_ms" in p && "reason" in p) {
        listeners.onSessionEnded?.(payload as VoiceSessionEnded);
      } else if ("chunk_index" in p && "chunk_size_bytes" in p) {
        const chunkMeta = payload as VoiceAudioChunkMeta;
        this.pendingAudioMeta = chunkMeta;
        listeners.onAudioChunkMeta?.(chunkMeta);
      } else if ("code" in p && "recoverable" in p) {
        listeners.onError?.(payload as VoiceError);
      }
    });
    unsubscribers.push(sessionUnsub);

    // Transcript channel
    if (listeners.onTranscription) {
      const transcriptUnsub = this.manager.subscribe<VoiceTranscription>(
        VOICE_CHANNELS.transcript(session_id),
        (payload) => {
          listeners.onTranscription?.(payload);
        }
      );
      unsubscribers.push(transcriptUnsub);
    }

    // VAD channel
    if (listeners.onVoiceActivity) {
      const vadUnsub = this.manager.subscribe<VoiceActivityEvent>(
        VOICE_CHANNELS.vad(session_id),
        (payload) => {
          listeners.onVoiceActivity?.(payload);
        }
      );
      unsubscribers.push(vadUnsub);
    }

    // Synthesis channel
    if (
      listeners.onSynthesisChunkMeta ||
      listeners.onSynthesisAudio ||
      listeners.onSynthesisComplete
    ) {
      const synthUnsub = this.manager.subscribe<
        VoiceSynthesisChunkMeta | VoiceSynthesisComplete
      >(VOICE_CHANNELS.synthesis(session_id), (payload) => {
        const p = payload as unknown as Record<string, unknown>;
        if ("total_duration_ms" in p) {
          listeners.onSynthesisComplete?.(payload as VoiceSynthesisComplete);
        } else if ("chunk_index" in p) {
          const meta = payload as VoiceSynthesisChunkMeta;
          this.pendingSynthesisMeta = meta;
          listeners.onSynthesisChunkMeta?.(meta);
        }
      });
      unsubscribers.push(synthUnsub);
    }

    // Binary frame handler — correlate with pending meta
    if (listeners.onAudioData || listeners.onSynthesisAudio) {
      const binaryUnsub = this.manager.on<ArrayBuffer>(
        `binary:${session_id}`,
        (data) => {
          if (this.pendingSynthesisMeta) {
            listeners.onSynthesisAudio?.(data, this.pendingSynthesisMeta);
            if (this.pendingSynthesisMeta.is_final) {
              this.pendingSynthesisMeta = null;
            }
          } else if (this.pendingAudioMeta) {
            listeners.onAudioData?.(data, this.pendingAudioMeta);
            if (this.pendingAudioMeta.is_final) {
              this.pendingAudioMeta = null;
            }
          } else {
            listeners.onAudioData?.(data);
          }
        }
      );
      unsubscribers.push(binaryUnsub);
    }

    const key = `session:${session_id}`;
    const cleanup = () => unsubscribers.forEach((fn) => fn());
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, cleanup);

    return () => {
      cleanup();
      this.activeSubscriptions.delete(key);
    };
  }

  subscribeToTranscription(
    session_id: string,
    onTranscription: (event: VoiceTranscription) => void
  ): () => void {
    return this.manager.subscribe<VoiceTranscription>(
      VOICE_CHANNELS.transcript(session_id),
      onTranscription
    );
  }

  subscribeToVoiceActivity(
    session_id: string,
    onActivity: (event: VoiceActivityEvent) => void
  ): () => void {
    return this.manager.subscribe<VoiceActivityEvent>(
      VOICE_CHANNELS.vad(session_id),
      onActivity
    );
  }

  unsubscribeFromSession(session_id: string): void {
    const key = `session:${session_id}`;
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.delete(key);
  }

  destroy(): void {
    this.activeSubscriptions.forEach((fn) => fn());
    this.activeSubscriptions.clear();
    this.manager.destroy();
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export const createVoiceSocket = (
  config: SocketManagerConfig
): VoiceSocket => new VoiceSocket(config);