import { motion } from 'framer-motion';
import { Wifi, Brain, Mic } from 'lucide-react';
import type { AIState } from '../types';
import { useChatStore } from '../store/chatStore';
import { MODEL_DISPLAY_NAMES } from '../constants/models';

const STATE_LABEL: Partial<Record<AIState, string>> = {
  thinking: 'Thinking…',
  searching: 'Searching…',
  coding: 'Coding…',
  reading: 'Reading…',
  browsing: 'Browsing…',
  remembering: 'Memory…',
  speaking: 'Speaking…',
  finished: 'Done',
  listening: 'Listening…',
  idle: '',
};

interface Props {
  aiState: AIState;
  memoryEnabled: boolean;
  webEnabled?: boolean;
}

export default function StatusBar({ aiState, memoryEnabled }: Props) {
  const stateLabel = STATE_LABEL[aiState];
  const selectedModel = useChatStore((s) => s.selectedModel);
  const displayName = MODEL_DISPLAY_NAMES[selectedModel] ?? selectedModel;

  return (
    <div
      className="status-bar flex items-center justify-between px-4 flex-shrink-0"
      style={{ height: 32 }}
    >
      {/* Left: Online + Provider + Model */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="status-dot status-dot-green" />
          <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.35)' }}>Online</span>
        </div>

        <div style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.08)' }} />

        <div className="flex items-center gap-1">
          <Wifi size={10} style={{ color: 'rgba(255,255,255,0.25)' }} />
          <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.35)' }}>NVIDIA NIM</span>
        </div>

        <div style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.08)' }} />

        <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.35)', fontFamily: 'monospace' }}>
          {displayName}
        </span>

        <div style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.08)' }} />

        <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}>
          ~84ms
        </span>
      </div>

      {/* Center: AI state */}
      <div style={{ minWidth: 80, textAlign: 'center' }}>
        {stateLabel && (
          <motion.span
            key={stateLabel}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ fontSize: '0.7rem', color: '#8b5cf6', fontWeight: 500 }}
          >
            {stateLabel}
          </motion.span>
        )}
      </div>

      {/* Right: Voice + Memory */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <Mic
            size={10}
            style={{ color: 'rgba(255,255,255,0.25)' }}
          />
          <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.25)' }}>
            Voice off
          </span>
        </div>

        <div style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.08)' }} />

        <div className="flex items-center gap-1">
          <Brain
            size={10}
            style={{ color: memoryEnabled ? '#8b5cf6' : 'rgba(255,255,255,0.25)' }}
          />
          <span style={{ fontSize: '0.7rem', color: memoryEnabled ? '#8b5cf6' : 'rgba(255,255,255,0.25)' }}>
            {memoryEnabled ? 'Memory on' : 'Memory off'}
          </span>
        </div>
      </div>
    </div>
  );
}
