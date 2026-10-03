import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, X, Loader2 } from 'lucide-react';
import { VoiceService } from '../services/voice.service';
import { eventManager } from '../services/event-manager';

// ── Whisper silence detection constants ────────────────────────────────────────
const SILENCE_THRESHOLD = 0.02;   // RMS below this is "silence"
const SILENCE_DURATION_MS = 1500; // ms of silence before we stop
const MAX_RECORDING_MS = 15000;   // hard cap on recording length

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

type VoiceState = 'idle' | 'listening' | 'thinking' | 'speaking';

const STATE_CONFIG: Record<VoiceState, {
  label: string;
  sub: string;
  ringColor: string;
  orb: string;
  pulse: boolean;
}> = {
  idle: {
    label: 'Hey Aspire',
    sub: 'Say the wake word or tap to activate',
    ringColor: 'rgba(99,102,241,0.2)',
    orb: 'linear-gradient(135deg, #4b4b6b 0%, #2d2d42 100%)',
    pulse: false,
  },
  listening: {
    label: 'Listening',
    sub: 'Speak now…',
    ringColor: 'rgba(34,197,94,0.35)',
    orb: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
    pulse: true,
  },
  thinking: {
    label: 'Thinking',
    sub: 'Processing your request…',
    ringColor: 'rgba(139,92,246,0.35)',
    orb: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
    pulse: true,
  },
  speaking: {
    label: 'Speaking',
    sub: 'Aspire is responding…',
    ringColor: 'rgba(59,130,246,0.35)',
    orb: 'linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)',
    pulse: true,
  },
};

// Precompute stable heights per bar to avoid re-randomizing on render
const BAR_HEIGHTS = Array.from({ length: 32 }, (_, i) => {
  const seed = Math.sin(i * 2.399) * 0.5 + 0.5;
  const seed2 = Math.sin(i * 4.1) * 0.5 + 0.5;
  return {
    h1: Math.floor(seed * 40 + 8),
    h2: Math.floor(seed2 * 50 + 12),
    dur: 0.45 + seed * 0.55,
  };
});

function WaveformBars({ active, color }: { active: boolean; color: string }) {
  return (
    <div className="flex items-center justify-center gap-0.5" style={{ height: 64 }}>
      {BAR_HEIGHTS.map((bar, i) => (
        <motion.div
          key={i}
          className="rounded-full"
          style={{
            width: 3,
            background: color,
            opacity: active ? 0.7 : 0.25,
          }}
          animate={active ? {
            height: [6, bar.h1, bar.h2, 6],
          } : { height: 6 }}
          transition={active ? {
            duration: bar.dur,
            repeat: Infinity,
            repeatType: 'mirror',
            ease: 'easeInOut',
            delay: i * 0.03,
          } : { duration: 0.4 }}
        />
      ))}
    </div>
  );
}

function OrbRings({ color }: { color: string }) {
  return (
    <>
      {[1, 2, 3].map(i => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-full"
          style={{
            border: `2px solid ${color}`,
            margin: -(i * 24),
          }}
          animate={{
            scale: [1, 1.08, 1],
            opacity: [0.6 / i, 0.2 / i, 0.6 / i],
          }}
          transition={{
            duration: 2 + i * 0.5,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: i * 0.3,
          }}
        />
      ))}
    </>
  );
}

const WAKE_WORD = 'hey aspire';

function createSpeechRecognition(): SpeechRecognition | null {
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Ctor) return null;
  const recognition = new Ctor();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  return recognition;
}

export default function VoiceView() {
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wakeWordRecRef = useRef<SpeechRecognition | null>(null);
  const commandRecRef = useRef<SpeechRecognition | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const transitioningRef = useRef(false);

  // Refs for functions called across callback boundaries (avoids stale closures)
  const startWakeWordDetectionRef = useRef<() => void>(() => {});
  const startCommandCaptureRef = useRef<() => void>(() => {});
  const handleLLMRequestRef = useRef<(text: string) => Promise<void>>(async () => {});
  const handleWhisperSTTRef = useRef<(blob: Blob) => Promise<void>>(async () => {});
  const goToIdleRef = useRef<() => void>(() => {});
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const config = STATE_CONFIG[voiceState];

  const goToIdle = useCallback(() => {
    transitioningRef.current = false;
    setVoiceState('idle');
    setTranscript('');
    setResponse('');
    setError(null);
  }, []);
  goToIdleRef.current = goToIdle;

  const startWakeWordDetection = useCallback(() => {
    const rec = createSpeechRecognition();
    if (!rec) return;

    rec.onresult = (event: SpeechRecognitionEvent) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          const text = result[0].transcript.toLowerCase().trim();
          if (text.includes(WAKE_WORD)) {
            transitioningRef.current = true;
            rec.stop();
            setVoiceState('listening');
            setTranscript('');
            setResponse('');
            setError(null);
            setTimeout(() => {
              transitioningRef.current = false;
              startCommandCaptureRef.current();
            }, 300);
            return;
          }
        }
      }
    };

    rec.onerror = () => {
      if (!transitioningRef.current) {
        setTimeout(() => startWakeWordDetectionRef.current(), 1000);
      }
    };

    rec.onend = () => {
      if (!transitioningRef.current) {
        setTimeout(() => startWakeWordDetectionRef.current(), 200);
      }
    };

    wakeWordRecRef.current = rec;
    rec.start();
  }, []);
  startWakeWordDetectionRef.current = startWakeWordDetection;

  const stopMediaRecording = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  }, []);

  const handleWhisperSTT = useCallback(async (blob: Blob) => {
    try {
      const response = await VoiceService.speechToText({ audio: blob });
      const text = response.transcription?.text?.trim();
      if (text) {
        setTranscript(text);
        handleLLMRequestRef.current(text);
      } else {
        goToIdleRef.current();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Whisper transcription failed');
      goToIdleRef.current();
    }
  }, []);
  handleWhisperSTTRef.current = handleWhisperSTT;

  const startCommandCapture = useCallback(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
        streamRef.current = stream;

        // Set up silence detection via AnalyserNode
        const audioCtx = new AudioContext();
        audioContextRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        source.connect(analyser);
        analyserRef.current = analyser;

        // Start MediaRecorder
        recordingChunksRef.current = [];
        const recorder = new MediaRecorder(stream);
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) recordingChunksRef.current.push(e.data);
        };

        recorder.onstop = () => {
          const blob = new Blob(recordingChunksRef.current, { type: recorder.mimeType });
          stopMediaRecording();
          handleWhisperSTTRef.current(blob);
        };

        recorder.start();

        // Silence detection loop
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        let silenceStart = 0;

        const checkSilence = () => {
          if (mediaRecorderRef.current?.state !== 'recording') return;
          analyser.getByteTimeDomainData(dataArray);
          let sum = 0;
          for (let i = 0; i < bufferLength; i++) {
            const val = (dataArray[i] - 128) / 128;
            sum += val * val;
          }
          const rms = Math.sqrt(sum / bufferLength);

          if (rms < SILENCE_THRESHOLD) {
            silenceStart += 50;
            if (silenceStart >= SILENCE_DURATION_MS && recordingChunksRef.current.length > 0) {
              recorder.stop();
              return;
            }
          } else {
            silenceStart = 0;
          }
          setTimeout(checkSilence, 50);
        };
        setTimeout(checkSilence, 500); // brief grace period before silence detection starts

        // Max recording time hard cap
        timerRef.current = setTimeout(() => {
          if (mediaRecorderRef.current?.state === 'recording') {
            recorder.stop();
          }
        }, MAX_RECORDING_MS);
      })
      .catch(() => {
        goToIdleRef.current();
      });
  }, []);
  startCommandCaptureRef.current = startCommandCapture;

  const handleTTS = useCallback(async (text: string) => {
    setVoiceState('speaking');
    try {
      const voices = await VoiceService.listVoices();
      const defaultVoice = voices.voices?.[0];
      const ttsResponse = await VoiceService.textToSpeech({
        text,
        voiceId: defaultVoice?.id ?? 'default',
      });

      const audio = new Audio(ttsResponse.audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        audioRef.current = null;
        goToIdleRef.current();
        setTimeout(() => startWakeWordDetectionRef.current(), 500);
      };

      audio.onerror = () => {
        audioRef.current = null;
        goToIdleRef.current();
      };

      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'TTS failed');
      goToIdleRef.current();
    }
  }, []);

  const handleTTSRef = useRef(handleTTS);
  handleTTSRef.current = handleTTS;

  const handleLLMRequest = useCallback(async (text: string) => {
    setVoiceState('thinking');
    setError(null);

    let llmResponse = '';
    try {
      await eventManager.streamMessage(text, {
        webEnabled: false,
        memoryEnabled: true,
        computerEnabled: false,
      }, {
        onStateChange: () => {},
        onChunk: (chunk) => {
          llmResponse += chunk.delta;
        },
        onError: (err) => {
          setError(err.message);
          goToIdleRef.current();
        },
      });

      if (!llmResponse) {
        setError('No response generated');
        goToIdleRef.current();
        return;
      }

      setResponse(llmResponse);
      await handleTTSRef.current(llmResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'LLM processing failed');
      goToIdleRef.current();
    }
  }, []);
  handleLLMRequestRef.current = handleLLMRequest;

  const cleanup = useCallback(() => {
    if (wakeWordRecRef.current) {
      wakeWordRecRef.current.stop();
      wakeWordRecRef.current = null;
    }
    if (commandRecRef.current) {
      commandRecRef.current.stop();
      commandRecRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    stopMediaRecording();
  }, [stopMediaRecording]);

  useEffect(() => {
    if (voiceState === 'idle') {
      startWakeWordDetectionRef.current();
    }
    return () => cleanup();
    // Only run when voiceState changes to 'idle'; cleanup on unmount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceState]);

  const activate = () => {
    if (muted) return;
    if (voiceState !== 'idle') {
      cleanup();
      goToIdleRef.current();
      return;
    }

    cleanup();
    setVoiceState('listening');
    setTranscript('');
    setResponse('');
    setError(null);
    setTimeout(() => startCommandCaptureRef.current(), 200);
  };

  return (
    <div
      className="flex flex-col items-center justify-center h-full relative overflow-hidden"
      style={{ background: 'radial-gradient(ellipse at 50% 40%, rgba(99,102,241,0.06) 0%, transparent 65%)' }}
    >
      {/* Background ambient */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(ellipse at 50% 50%, ${config.ringColor} 0%, transparent 60%)`,
          transition: 'background 1s ease',
        }}
      />

      {/* Top label */}
      <div className="absolute top-8 left-0 right-0 text-center">
        <AnimatePresence mode="wait">
          <motion.div
            key={voiceState + '-label'}
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.3 }}
          >
            <div
              className="text-xs uppercase tracking-widest font-medium mb-1"
              style={{ color: 'rgba(255,255,255,0.25)' }}
            >
              Voice Mode
            </div>
            <div className="text-2xl font-semibold" style={{ color: '#f0f0f4', letterSpacing: '-0.02em' }}>
              {config.label}
            </div>
            <div className="text-sm mt-1" style={{ color: 'rgba(255,255,255,0.35)' }}>
              {config.sub}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Orb */}
      <div className="relative flex items-center justify-center" style={{ marginTop: -40 }}>
        {/* Rings */}
        <AnimatePresence>
          {config.pulse && <OrbRings color={config.ringColor} />}
        </AnimatePresence>

        {/* Main orb */}
        <motion.button
          whileTap={{ scale: 0.93 }}
          onClick={activate}
          className="relative flex items-center justify-center rounded-full z-10"
          style={{
            width: 120,
            height: 120,
            background: config.orb,
            boxShadow: `0 0 60px ${config.ringColor}, 0 0 120px ${config.ringColor}40`,
            transition: 'background 0.8s ease, box-shadow 0.8s ease',
          }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={voiceState + '-icon'}
              initial={{ scale: 0.7, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.7, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {voiceState === 'idle' ? (
                <Mic size={40} style={{ color: 'rgba(255,255,255,0.5)' }} />
              ) : voiceState === 'listening' ? (
                <Mic size={40} style={{ color: 'white' }} />
              ) : voiceState === 'thinking' ? (
                <Loader2 size={36} className="animate-spin" style={{ color: 'white' }} />
              ) : (
                <div style={{ fontSize: 36 }}>⚡</div>
              )}
            </motion.div>
          </AnimatePresence>
        </motion.button>
      </div>

      {/* Waveform */}
      <div className="mt-12 w-full max-w-md px-8">
        <WaveformBars
          active={voiceState === 'listening' || voiceState === 'speaking'}
          color={voiceState === 'speaking' ? '#3b82f6' : '#22c55e'}
        />
      </div>

      {/* Transcript + Response */}
      <div className="mt-6 w-full max-w-lg px-8 text-center" style={{ minHeight: 80 }}>
        <AnimatePresence>
          {transcript && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-3"
            >
              <div className="text-xs uppercase tracking-wider mb-1.5" style={{ color: 'rgba(255,255,255,0.25)' }}>
                You said
              </div>
              <p className="text-sm" style={{ color: 'rgba(255,255,255,0.65)' }}>
                "{transcript}"
              </p>
            </motion.div>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {response && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="text-xs uppercase tracking-wider mb-1.5" style={{ color: 'rgba(255,255,255,0.25)' }}>
                Aspire
              </div>
              <p className="text-sm leading-relaxed" style={{ color: 'rgba(255,255,255,0.75)' }}>
                {response}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Error display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-28 left-0 right-0 text-center px-8"
          >
            <div
              className="inline-block rounded-lg px-4 py-2 text-xs"
              style={{
                background: 'rgba(239,68,68,0.12)',
                border: '1px solid rgba(239,68,68,0.25)',
                color: '#f87171',
              }}
            >
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom controls */}
      <div className="absolute bottom-10 flex items-center gap-4">
        <button
          onClick={() => setMuted(!muted)}
          className="flex items-center justify-center rounded-full transition-all"
          style={{
            width: 48,
            height: 48,
            background: muted ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.05)',
            border: `1px solid ${muted ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.08)'}`,
            color: muted ? '#ef4444' : 'rgba(255,255,255,0.4)',
          }}
        >
          {muted ? <MicOff size={18} /> : <Mic size={18} />}
        </button>

        <button
          onClick={() => {
            cleanup();
            goToIdle();
          }}
          className="flex items-center justify-center rounded-full transition-all hover:bg-white/[0.07]"
          style={{
            width: 48,
            height: 48,
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.08)',
            color: 'rgba(255,255,255,0.4)',
          }}
        >
          <X size={18} />
        </button>
      </div>

      {/* Wake word hint */}
      <div
        className="absolute bottom-4 text-center text-xs"
        style={{ color: 'rgba(255,255,255,0.18)' }}
      >
        Wake word: <span style={{ color: 'rgba(255,255,255,0.35)' }}>Hey Aspire</span>
      </div>
    </div>
  );
}
