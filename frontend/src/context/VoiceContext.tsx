// frontend/src/context/VoiceContext.tsx
/* eslint-disable react-refresh/only-export-components */

"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type VoicePermissionState =
  | "unknown"
  | "granted"
  | "denied"
  | "prompt"
  | "unavailable";

export type RecordingState =
  | "idle"
  | "requesting"
  | "recording"
  | "paused"
  | "processing"
  | "error";

export type PlaybackState =
  | "idle"
  | "loading"
  | "playing"
  | "paused"
  | "ended"
  | "error";

export type AudioFormat = "webm" | "ogg" | "mp4" | "wav";

export interface AudioDeviceInfo {
  device_id: string;
  label: string;
  group_id: string;
  kind: "audioinput" | "audiooutput";
}

export interface RecordingSession {
  id: string;
  started_at: number;
  duration_ms: number;
  blob: Blob | null;
  url: string | null;
  format: AudioFormat;
  size_bytes: number;
}

export interface AudioLevelSample {
  level: number; // 0–1
  timestamp: number;
}

export interface VoiceContextError {
  code:
    | "PERMISSION_DENIED"
    | "NOT_SUPPORTED"
    | "DEVICE_NOT_FOUND"
    | "RECORDING_FAILED"
    | "PLAYBACK_FAILED"
    | "NETWORK_ERROR"
    | "UNKNOWN";
  message: string;
  timestamp: number;
  recoverable: boolean;
}

export interface VoiceContextValue {
  // Permissions
  permission: VoicePermissionState;
  requestPermission: () => Promise<VoicePermissionState>;
  checkPermission: () => Promise<VoicePermissionState>;

  // Devices
  input_devices: AudioDeviceInfo[];
  output_devices: AudioDeviceInfo[];
  selected_input_device_id: string | null;
  selected_output_device_id: string | null;
  selectInputDevice: (device_id: string) => void;
  selectOutputDevice: (device_id: string) => void;
  refreshDevices: () => Promise<void>;

  // Recording
  recording_state: RecordingState;
  current_session: RecordingSession | null;
  sessions: RecordingSession[];
  startRecording: (options?: StartRecordingOptions) => Promise<void>;
  stopRecording: () => Promise<RecordingSession | null>;
  pauseRecording: () => void;
  resumeRecording: () => void;
  cancelRecording: () => void;
  deleteSession: (id: string) => void;
  clearSessions: () => void;

  // Audio levels
  audio_level: number;
  audio_level_history: AudioLevelSample[];
  is_voice_active: boolean;

  // Playback
  playback_state: PlaybackState;
  current_playback_url: string | null;
  playback_progress_ms: number;
  playback_duration_ms: number;
  playback_volume: number;
  play: (url: string, options?: PlaybackOptions) => Promise<void>;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  seek: (position_ms: number) => void;
  setVolume: (volume: number) => void;

  // Synthesis playback
  playSynthesized: (audio_data: ArrayBuffer, format?: AudioFormat) => Promise<void>;

  // VAD
  vad_enabled: boolean;
  vad_threshold: number;
  setVADEnabled: (enabled: boolean) => void;
  setVADThreshold: (threshold: number) => void;

  // Noise cancellation
  noise_cancellation_enabled: boolean;
  setNoiseCancellationEnabled: (enabled: boolean) => void;

  // State flags
  is_supported: boolean;
  is_recording: boolean;
  is_playing: boolean;
  is_loading: boolean;

  // Error
  error: VoiceContextError | null;
  clearError: () => void;
}

export interface StartRecordingOptions {
  format?: AudioFormat;
  timeslice_ms?: number;
  max_duration_ms?: number;
  sample_rate?: number;
  channel_count?: 1 | 2;
  noise_suppression?: boolean;
  echo_cancellation?: boolean;
  auto_gain_control?: boolean;
}

export interface PlaybackOptions {
  volume?: number;
  auto_play?: boolean;
  on_end?: () => void;
  on_error?: (error: Error) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const nanoid = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

const isBrowserSupported = (): boolean =>
  typeof window !== "undefined" &&
  typeof navigator !== "undefined" &&
  "mediaDevices" in navigator &&
  "getUserMedia" in navigator.mediaDevices &&
  typeof MediaRecorder !== "undefined" &&
  typeof AudioContext !== "undefined";

const getSupportedMimeType = (): string => {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  return types.find((t) => MediaRecorder.isTypeSupported(t)) ?? "audio/webm";
};

const getAudioFormat = (mime: string): AudioFormat => {
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4")) return "mp4";
  if (mime.includes("wav")) return "wav";
  return "webm";
};

const VAD_HISTORY_SIZE = 30;
const LEVEL_HISTORY_MAX = 100;
const DEFAULT_VAD_THRESHOLD = 0.01;

// ─── Context ─────────────────────────────────────────────────────────────────

export const VoiceContext = createContext<VoiceContextValue | null>(null);
VoiceContext.displayName = "VoiceContext";

// ─── Props ────────────────────────────────────────────────────────────────────

export interface VoiceProviderProps {
  children: ReactNode;
  default_volume?: number;
  default_vad_enabled?: boolean;
  default_vad_threshold?: number;
  default_noise_cancellation?: boolean;
  max_sessions?: number;
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const VoiceProvider = ({
  children,
  default_volume = 0.8,
  default_vad_enabled = true,
  default_vad_threshold = DEFAULT_VAD_THRESHOLD,
  default_noise_cancellation = true,
  max_sessions = 50,
}: VoiceProviderProps): React.JSX.Element => {
  const is_supported = isBrowserSupported();

  // ── Permission ────────────────────────────────────────────────────────────
  const [permission, setPermission] = useState<VoicePermissionState>("unknown");

  // ── Devices ───────────────────────────────────────────────────────────────
  const [input_devices, setInputDevices] = useState<AudioDeviceInfo[]>([]);
  const [output_devices, setOutputDevices] = useState<AudioDeviceInfo[]>([]);
  const [selected_input_device_id, setSelectedInputDeviceId] = useState<
    string | null
  >(null);
  const [selected_output_device_id, setSelectedOutputDeviceId] = useState<
    string | null
  >(null);

  // ── Recording ─────────────────────────────────────────────────────────────
  const [recording_state, setRecordingState] = useState<RecordingState>("idle");
  const [current_session, setCurrentSession] =
    useState<RecordingSession | null>(null);
  const [sessions, setSessions] = useState<RecordingSession[]>([]);

  // ── Audio levels ──────────────────────────────────────────────────────────
  const [audio_level, setAudioLevel] = useState(0);
  const [audio_level_history, setAudioLevelHistory] = useState<
    AudioLevelSample[]
  >([]);
  const [is_voice_active, setIsVoiceActive] = useState(false);

  // ── Playback ──────────────────────────────────────────────────────────────
  const [playback_state, setPlaybackState] = useState<PlaybackState>("idle");
  const [current_playback_url, setCurrentPlaybackUrl] = useState<
    string | null
  >(null);
  const [playback_progress_ms, setPlaybackProgressMs] = useState(0);
  const [playback_duration_ms, setPlaybackDurationMs] = useState(0);
  const [playback_volume, setPlaybackVolumeState] =
    useState(default_volume);

  // ── VAD & noise cancellation ──────────────────────────────────────────────
  const [vad_enabled, setVADEnabledState] = useState(default_vad_enabled);
  const [vad_threshold, setVADThresholdState] = useState(
    default_vad_threshold
  );
  const [noise_cancellation_enabled, setNoiseCancellationEnabledState] =
    useState(default_noise_cancellation);

  // ── Error ─────────────────────────────────────────────────────────────────
  const [error, setErrorState] = useState<VoiceContextError | null>(null);

  // ── Refs ──────────────────────────────────────────────────────────────────
  const media_recorder_ref = useRef<MediaRecorder | null>(null);
  const media_stream_ref = useRef<MediaStream | null>(null);
  const audio_context_ref = useRef<AudioContext | null>(null);
  const analyser_ref = useRef<AnalyserNode | null>(null);
  const level_raf_ref = useRef<number | null>(null);
  const chunks_ref = useRef<Blob[]>([]);
  const session_start_ref = useRef<number>(0);
  const max_duration_timer_ref = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const vad_history_ref = useRef<number[]>([]);
  const audio_element_ref = useRef<HTMLAudioElement | null>(null);
  const playback_raf_ref = useRef<number | null>(null);
  const created_urls_ref = useRef<string[]>([]);

  // ── Permission ────────────────────────────────────────────────────────────

  const checkPermission = useCallback(async (): Promise<VoicePermissionState> => {
    if (!is_supported) {
      setPermission("unavailable");
      return "unavailable";
    }
    try {
      const result = await navigator.permissions.query({
        name: "microphone" as PermissionName,
      });
      const state = result.state as VoicePermissionState;
      setPermission(state);
      result.onchange = () => setPermission(result.state as VoicePermissionState);
      return state;
    } catch {
      // Permissions API may not support microphone query
      setPermission("unknown");
      return "unknown";
    }
  }, [is_supported]);

  const requestPermission = useCallback(async (): Promise<VoicePermissionState> => {
    if (!is_supported) {
      setPermission("unavailable");
      return "unavailable";
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      stream.getTracks().forEach((t) => t.stop());
      setPermission("granted");
      return "granted";
    } catch (err) {
      const state: VoicePermissionState =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "denied"
          : "unknown";
      setPermission(state);
      return state;
    }
  }, [is_supported]);

  // ── Devices ───────────────────────────────────────────────────────────────

  const refreshDevices = useCallback(async (): Promise<void> => {
    if (!is_supported) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs: AudioDeviceInfo[] = [];
      const outputs: AudioDeviceInfo[] = [];

      for (const d of devices) {
        if (d.kind === "audioinput") {
          inputs.push({
            device_id: d.deviceId,
            label: d.label || `Microphone ${inputs.length + 1}`,
            group_id: d.groupId,
            kind: "audioinput",
          });
        } else if (d.kind === "audiooutput") {
          outputs.push({
            device_id: d.deviceId,
            label: d.label || `Speaker ${outputs.length + 1}`,
            group_id: d.groupId,
            kind: "audiooutput",
          });
        }
      }

      setInputDevices(inputs);
      setOutputDevices(outputs);

      if (inputs.length > 0 && !selected_input_device_id) {
        setSelectedInputDeviceId(inputs[0]!.device_id);
      }
    } catch {
      // Device enumeration failed silently
    }
  }, [is_supported, selected_input_device_id]);

  const selectInputDevice = useCallback(
    (device_id: string) => setSelectedInputDeviceId(device_id),
    []
  );

  const selectOutputDevice = useCallback(
    (device_id: string) => setSelectedOutputDeviceId(device_id),
    []
  );

  // ── Audio level analysis ──────────────────────────────────────────────────

  const startLevelAnalysis = useCallback(
    (stream: MediaStream): void => {
      try {
        const ctx = new AudioContext();
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;

        const source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);

        audio_context_ref.current = ctx;
        analyser_ref.current = analyser;

        const buffer = new Float32Array(analyser.fftSize);

        const tick = (): void => {
          analyser.getFloatTimeDomainData(buffer);
          let sum = 0;
          for (let i = 0; i < buffer.length; i++) {
            sum += buffer[i]! * buffer[i]!;
          }
          const rms = Math.sqrt(sum / buffer.length);
          const level = Math.min(1, rms * 10);

          setAudioLevel(level);
          setAudioLevelHistory((prev) => {
            const next = [
              ...prev,
              { level, timestamp: Date.now() },
            ].slice(-LEVEL_HISTORY_MAX);
            return next;
          });

          if (vad_enabled) {
            vad_history_ref.current = [
              ...vad_history_ref.current.slice(-(VAD_HISTORY_SIZE - 1)),
              level,
            ];
            const avg =
              vad_history_ref.current.reduce((a, b) => a + b, 0) /
              vad_history_ref.current.length;
            setIsVoiceActive(avg > vad_threshold);
          }

          level_raf_ref.current = requestAnimationFrame(tick);
        };

        level_raf_ref.current = requestAnimationFrame(tick);
      } catch {
        // Audio analysis unavailable
      }
    },
    [vad_enabled, vad_threshold]
  );

  const stopLevelAnalysis = useCallback((): void => {
    if (level_raf_ref.current) {
      cancelAnimationFrame(level_raf_ref.current);
      level_raf_ref.current = null;
    }
    audio_context_ref.current?.close().catch(() => null);
    audio_context_ref.current = null;
    analyser_ref.current = null;
    setAudioLevel(0);
    setIsVoiceActive(false);
  }, []);

  // ── Recording ─────────────────────────────────────────────────────────────

  const stopRecording = useCallback(async (): Promise<RecordingSession | null> => {
    if (max_duration_timer_ref.current) {
      clearTimeout(max_duration_timer_ref.current);
      max_duration_timer_ref.current = null;
    }

    const recorder = media_recorder_ref.current;
    if (!recorder || recorder.state === "inactive") {
      setRecordingState("idle");
      return null;
    }

    setRecordingState("processing");
    stopLevelAnalysis();

    return new Promise<RecordingSession | null>((resolve) => {
      const onStop = (): void => {
        const blob = new Blob(chunks_ref.current, {
          type: recorder.mimeType,
        });
        const url = URL.createObjectURL(blob);
        created_urls_ref.current.push(url);

        const format = getAudioFormat(recorder.mimeType);
        const duration_ms = Date.now() - session_start_ref.current;

        const session: RecordingSession = {
          id: current_session?.id ?? nanoid(),
          started_at: session_start_ref.current,
          duration_ms,
          blob,
          url,
          format,
          size_bytes: blob.size,
        };

        setSessions((prev) => [session, ...prev].slice(0, max_sessions));
        setCurrentSession(session);
        setRecordingState("idle");

        media_stream_ref.current?.getTracks().forEach((t) => t.stop());
        media_stream_ref.current = null;
        media_recorder_ref.current = null;
        chunks_ref.current = [];

        resolve(session);
      };

      recorder.addEventListener("stop", onStop, { once: true });
      recorder.stop();
    });
  }, [current_session, max_sessions, stopLevelAnalysis]);

  const startRecording = useCallback(
    async (options: StartRecordingOptions = {}): Promise<void> => {
      if (!is_supported) {
        setErrorState({
          code: "NOT_SUPPORTED",
          message: "Audio recording is not supported in this browser",
          timestamp: Date.now(),
          recoverable: false,
        });
        return;
      }

      if (recording_state === "recording" || recording_state === "paused") {
        return;
      }

      setRecordingState("requesting");

      try {
        const constraints: MediaStreamConstraints = {
          audio: {
            deviceId: selected_input_device_id
              ? { ideal: selected_input_device_id }
              : undefined,
            sampleRate: options.sample_rate ?? 16000,
            channelCount: options.channel_count ?? 1,
            noiseSuppression:
              options.noise_suppression ?? noise_cancellation_enabled,
            echoCancellation:
              options.echo_cancellation ?? noise_cancellation_enabled,
            autoGainControl: options.auto_gain_control ?? true,
          },
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        media_stream_ref.current = stream;
        setPermission("granted");

        const mimeType = getSupportedMimeType();
        const recorder = new MediaRecorder(stream, { mimeType });
        media_recorder_ref.current = recorder;
        chunks_ref.current = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks_ref.current.push(e.data);
        };

        const format = getAudioFormat(mimeType);
        const session_id = nanoid();
        session_start_ref.current = Date.now();

        setCurrentSession({
          id: session_id,
          started_at: session_start_ref.current,
          duration_ms: 0,
          blob: null,
          url: null,
          format,
          size_bytes: 0,
        });

        recorder.start(options.timeslice_ms ?? 250);
        setRecordingState("recording");
        startLevelAnalysis(stream);

        if (options.max_duration_ms) {
          max_duration_timer_ref.current = setTimeout(async () => {
            await stopRecording();
          }, options.max_duration_ms);
        }
      } catch (err) {
        const code =
          err instanceof DOMException && err.name === "NotAllowedError"
            ? "PERMISSION_DENIED"
            : err instanceof DOMException &&
                err.name === "NotFoundError"
              ? "DEVICE_NOT_FOUND"
              : "RECORDING_FAILED";

        setPermission(code === "PERMISSION_DENIED" ? "denied" : permission);
        setErrorState({
          code,
          message:
            err instanceof Error ? err.message : "Recording failed",
          timestamp: Date.now(),
          recoverable: code !== "PERMISSION_DENIED",
        });
        setRecordingState("error");
      }
    },
    [
      is_supported,
      recording_state,
      selected_input_device_id,
      noise_cancellation_enabled,
      startLevelAnalysis,
      stopRecording,
      permission,
    ]
  );

  const pauseRecording = useCallback((): void => {
    if (media_recorder_ref.current?.state === "recording") {
      media_recorder_ref.current.pause();
      setRecordingState("paused");
    }
  }, []);

  const resumeRecording = useCallback((): void => {
    if (media_recorder_ref.current?.state === "paused") {
      media_recorder_ref.current.resume();
      setRecordingState("recording");
    }
  }, []);

  const cancelRecording = useCallback((): void => {
    if (max_duration_timer_ref.current) {
      clearTimeout(max_duration_timer_ref.current);
    }

    const recorder = media_recorder_ref.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.stop();
    }

    media_stream_ref.current?.getTracks().forEach((t) => t.stop());
    media_stream_ref.current = null;
    media_recorder_ref.current = null;
    chunks_ref.current = [];

    stopLevelAnalysis();
    setCurrentSession(null);
    setRecordingState("idle");
  }, [stopLevelAnalysis]);

  const deleteSession = useCallback((id: string): void => {
    setSessions((prev) => {
      const session = prev.find((s) => s.id === id);
      if (session?.url) {
        URL.revokeObjectURL(session.url);
        created_urls_ref.current = created_urls_ref.current.filter(
          (u) => u !== session.url
        );
      }
      return prev.filter((s) => s.id !== id);
    });
  }, []);

  const clearSessions = useCallback((): void => {
    setSessions((prev) => {
      prev.forEach((s) => {
        if (s.url) URL.revokeObjectURL(s.url);
      });
      return [];
    });
    created_urls_ref.current = [];
  }, []);

  // ── Playback ──────────────────────────────────────────────────────────────

  const stopPlaybackProgress = useCallback((): void => {
    if (playback_raf_ref.current) {
      cancelAnimationFrame(playback_raf_ref.current);
      playback_raf_ref.current = null;
    }
  }, []);

  const startPlaybackProgress = useCallback((): void => {
    const tick = (): void => {
      const audio = audio_element_ref.current;
      if (!audio) return;
      setPlaybackProgressMs(audio.currentTime * 1000);
      if (!audio.paused && !audio.ended) {
        playback_raf_ref.current = requestAnimationFrame(tick);
      }
    };
    playback_raf_ref.current = requestAnimationFrame(tick);
  }, []);

  const play = useCallback(
    async (
      url: string,
      options: PlaybackOptions = {}
    ): Promise<void> => {
      stopPlaybackProgress();

      if (!audio_element_ref.current) {
        audio_element_ref.current = new Audio();
      }

      const audio = audio_element_ref.current;
      audio.pause();
      audio.src = url;
      audio.volume = options.volume ?? playback_volume;

      setCurrentPlaybackUrl(url);
      setPlaybackState("loading");
      setPlaybackProgressMs(0);

      audio.onloadedmetadata = () => {
        setPlaybackDurationMs(audio.duration * 1000);
      };

      audio.onended = () => {
        setPlaybackState("ended");
        setPlaybackProgressMs(audio.duration * 1000);
        stopPlaybackProgress();
        options.on_end?.();
      };

      audio.onerror = () => {
        const err = new Error("Playback error");
        setPlaybackState("error");
        setErrorState({
          code: "PLAYBACK_FAILED",
          message: "Audio playback failed",
          timestamp: Date.now(),
          recoverable: true,
        });
        options.on_error?.(err);
      };

      try {
        await audio.play();
        setPlaybackState("playing");
        startPlaybackProgress();
      } catch (err) {
        setPlaybackState("error");
        setErrorState({
          code: "PLAYBACK_FAILED",
          message:
            err instanceof Error ? err.message : "Playback failed",
          timestamp: Date.now(),
          recoverable: true,
        });
      }
    },
    [playback_volume, startPlaybackProgress, stopPlaybackProgress]
  );

  const pause = useCallback((): void => {
    audio_element_ref.current?.pause();
    stopPlaybackProgress();
    setPlaybackState("paused");
  }, [stopPlaybackProgress]);

  const resume = useCallback(async (): Promise<void> => {
    const audio = audio_element_ref.current;
    if (!audio) return;
    try {
      await audio.play();
      setPlaybackState("playing");
      startPlaybackProgress();
    } catch {
      setPlaybackState("error");
    }
  }, [startPlaybackProgress]);

  const stop = useCallback((): void => {
    const audio = audio_element_ref.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    stopPlaybackProgress();
    setPlaybackState("idle");
    setPlaybackProgressMs(0);
    setCurrentPlaybackUrl(null);
  }, [stopPlaybackProgress]);

  const seek = useCallback((position_ms: number): void => {
    const audio = audio_element_ref.current;
    if (audio) {
      audio.currentTime = position_ms / 1000;
      setPlaybackProgressMs(position_ms);
    }
  }, []);

  const setVolume = useCallback((volume: number): void => {
    const clamped = Math.max(0, Math.min(1, volume));
    setPlaybackVolumeState(clamped);
    if (audio_element_ref.current) {
      audio_element_ref.current.volume = clamped;
    }
  }, []);

  const playSynthesized = useCallback(
    async (
      audio_data: ArrayBuffer,
      format: AudioFormat = "webm"
    ): Promise<void> => {
      const mimeMap: Record<AudioFormat, string> = {
        webm: "audio/webm",
        ogg: "audio/ogg",
        mp4: "audio/mp4",
        wav: "audio/wav",
      };

      const blob = new Blob([audio_data], { type: mimeMap[format] });
      const url = URL.createObjectURL(blob);
      created_urls_ref.current.push(url);

      await play(url, {
        on_end: () => {
          URL.revokeObjectURL(url);
          created_urls_ref.current = created_urls_ref.current.filter(
            (u) => u !== url
          );
        },
      });
    },
    [play]
  );

  // ── VAD ───────────────────────────────────────────────────────────────────

  const setVADEnabled = useCallback((enabled: boolean): void => {
    setVADEnabledState(enabled);
    if (!enabled) setIsVoiceActive(false);
  }, []);

  const setVADThreshold = useCallback((threshold: number): void => {
    setVADThresholdState(Math.max(0, Math.min(1, threshold)));
  }, []);

  const setNoiseCancellationEnabled = useCallback(
    (enabled: boolean): void => {
      setNoiseCancellationEnabledState(enabled);
    },
    []
  );

  // ── Error ─────────────────────────────────────────────────────────────────

  const clearError = useCallback((): void => setErrorState(null), []);

  // ── Init ──────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!is_supported) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void checkPermission();
    void refreshDevices();

    const handleDeviceChange = (): void => {
      void refreshDevices();
    };
    navigator.mediaDevices.addEventListener(
      "devicechange",
      handleDeviceChange
    );
    return () => {
      navigator.mediaDevices.removeEventListener(
        "devicechange",
        handleDeviceChange
      );
    };
  }, [is_supported, checkPermission, refreshDevices]);

  // ── Cleanup ───────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      cancelRecording();
      stop();
      created_urls_ref.current.forEach((url) => URL.revokeObjectURL(url));
      created_urls_ref.current = [];
    };
  }, [cancelRecording, stop]);

  // ── Context value ─────────────────────────────────────────────────────────

  const value = useMemo<VoiceContextValue>(
    () => ({
      permission,
      requestPermission,
      checkPermission,
      input_devices,
      output_devices,
      selected_input_device_id,
      selected_output_device_id,
      selectInputDevice,
      selectOutputDevice,
      refreshDevices,
      recording_state,
      current_session,
      sessions,
      startRecording,
      stopRecording,
      pauseRecording,
      resumeRecording,
      cancelRecording,
      deleteSession,
      clearSessions,
      audio_level,
      audio_level_history,
      is_voice_active,
      playback_state,
      current_playback_url,
      playback_progress_ms,
      playback_duration_ms,
      playback_volume,
      play,
      pause,
      resume,
      stop,
      seek,
      setVolume,
      playSynthesized,
      vad_enabled,
      vad_threshold,
      setVADEnabled,
      setVADThreshold,
      noise_cancellation_enabled,
      setNoiseCancellationEnabled,
      is_supported,
      is_recording:
        recording_state === "recording" ||
        recording_state === "paused",
      is_playing: playback_state === "playing",
      is_loading:
        recording_state === "requesting" ||
        recording_state === "processing" ||
        playback_state === "loading",
      error,
      clearError,
    }),
    [
      permission,
      requestPermission,
      checkPermission,
      input_devices,
      output_devices,
      selected_input_device_id,
      selected_output_device_id,
      selectInputDevice,
      selectOutputDevice,
      refreshDevices,
      recording_state,
      current_session,
      sessions,
      startRecording,
      stopRecording,
      pauseRecording,
      resumeRecording,
      cancelRecording,
      deleteSession,
      clearSessions,
      audio_level,
      audio_level_history,
      is_voice_active,
      playback_state,
      current_playback_url,
      playback_progress_ms,
      playback_duration_ms,
      playback_volume,
      play,
      pause,
      resume,
      stop,
      seek,
      setVolume,
      playSynthesized,
      vad_enabled,
      vad_threshold,
      setVADEnabled,
      setVADThreshold,
      noise_cancellation_enabled,
      setNoiseCancellationEnabled,
      is_supported,
      error,
      clearError,
    ]
  );

  return (
    <VoiceContext.Provider value={value}>{children}</VoiceContext.Provider>
  );
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export const useVoiceContext = (): VoiceContextValue => {
  const context = useContext(VoiceContext);
  if (context === null) {
    throw new Error(
      "[useVoiceContext] must be used within a <VoiceProvider>. " +
        "Ensure VoiceProvider wraps your component tree."
    );
  }
  return context;
};