import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Mic, MicOff, Square } from "lucide-react";
import { cn } from "@/lib/utils";

interface VoiceModeProps {
  onTranscript: (text: string) => void;
  onStop: () => void;
  isProcessing: boolean;
  className?: string;
}

export function VoiceMode({ onTranscript, onStop, isProcessing, className }: VoiceModeProps) {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState("");
  const animationRef = useRef<number>(0);

  const startListening = useCallback(() => {
    setIsListening(true);
    setTranscript("");
    const animate = () => {
      setAudioLevel(Math.random());
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
  }, []);

  const stopListening = useCallback(() => {
    setIsListening(false);
    cancelAnimationFrame(animationRef.current);
    setAudioLevel(0);
    if (transcript.trim()) onTranscript(transcript.trim());
  }, [transcript, onTranscript]);

  return (
    <div className={cn("flex flex-col items-center justify-center gap-6 py-8", className)}>
      <motion.div
        animate={{
          scale: isListening ? [1, 1.05, 1] : 1,
        }}
        transition={{ duration: 1.5, repeat: isListening ? Infinity : 0 }}
        className={cn(
          "relative flex h-24 w-24 items-center justify-center rounded-full transition-all",
          isListening
            ? "bg-violet-500/20 ring-4 ring-violet-500/30"
            : "bg-white/[0.04] ring-1 ring-white/[0.08]"
        )}
      >
        {isListening ? (
          <Mic className="h-10 w-10 text-violet-400" />
        ) : (
          <MicOff className="h-10 w-10 text-gray-600" />
        )}

        {isListening && (
          <svg className="absolute -inset-2 h-28 w-28" viewBox="0 0 120 120">
            <motion.circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="rgb(139, 92, 246)"
              strokeWidth="2"
              strokeOpacity={0.3 + audioLevel * 0.3}
              animate={{ scale: [1, 1 + audioLevel * 0.1, 1] }}
              transition={{ duration: 0.3 }}
            />
          </svg>
        )}
      </motion.div>

      <div className="text-center">
        <p className="text-sm text-gray-400">
          {isListening ? "Listening..." : isProcessing ? "Processing..." : "Tap to speak"}
        </p>
        {transcript && (
          <p className="mt-2 text-xs text-gray-500 max-w-xs">{transcript}</p>
        )}
      </div>

      <div className="flex items-center gap-2">
        {isProcessing ? (
          <button
            onClick={onStop}
            className="flex items-center gap-2 rounded-lg bg-red-500/20 px-4 py-2 text-sm text-red-400 hover:bg-red-500/30"
          >
            <Square className="h-4 w-4" /> Stop
          </button>
        ) : isListening ? (
          <button
            onClick={stopListening}
            className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-500"
          >
            <MicOff className="h-4 w-4" /> Stop Recording
          </button>
        ) : (
          <button
            onClick={startListening}
            className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-500"
          >
            <Mic className="h-4 w-4" /> Start Speaking
          </button>
        )}
      </div>
    </div>
  );
}
