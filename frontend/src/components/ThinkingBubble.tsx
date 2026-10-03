import { motion } from 'framer-motion';
import type { AIState } from '../types';

const STATE_LABELS: Partial<Record<AIState, string>> = {
  thinking: 'Thinking',
  searching: 'Searching the web',
  coding: 'Writing code',
  reading: 'Reading document',
  browsing: 'Browsing',
  remembering: 'Accessing memory',
};

interface Props {
  state: AIState;
}

export default function ThinkingBubble({ state }: Props) {
  const label = STATE_LABELS[state] || 'Working';

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.2 }}
      className="flex gap-3"
    >
      {/* Avatar */}
      <div
        className="flex-shrink-0 flex items-center justify-center rounded-full"
        style={{
          width: 30,
          height: 30,
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
          boxShadow: '0 0 10px rgba(99,102,241,0.3)',
          marginTop: 2,
        }}
      >
        <span style={{ fontSize: 12 }}>⚡</span>
      </div>

      <div
        className="message-ai px-4 py-3 flex items-center gap-3"
        style={{ minHeight: 48 }}
      >
        {/* Dots */}
        <div className="flex gap-1.5 items-center">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="rounded-full animate-thinking"
              style={{
                width: 6,
                height: 6,
                background: '#8b5cf6',
                display: 'inline-block',
                animationDelay: `${i * 0.18}s`,
              }}
            />
          ))}
        </div>
        <span
          className="text-xs"
          style={{ color: 'rgba(255,255,255,0.3)' }}
        >
          {label}
        </span>
      </div>
    </motion.div>
  );
}
