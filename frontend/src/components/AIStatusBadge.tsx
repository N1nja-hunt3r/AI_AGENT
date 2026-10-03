import { motion, AnimatePresence } from 'framer-motion';
import type { AIState } from '../types';

const STATE_CONFIG: Record<AIState, { label: string; color: string; dotClass: string }> = {
  idle: { label: 'Ready', color: 'rgba(255,255,255,0.3)', dotClass: '' },
  listening: { label: 'Listening', color: '#22c55e', dotClass: 'status-dot-green' },
  thinking: { label: 'Thinking', color: '#8b5cf6', dotClass: 'status-dot-purple' },
  searching: { label: 'Searching', color: '#3b82f6', dotClass: 'status-dot-blue' },
  coding: { label: 'Coding', color: '#f59e0b', dotClass: 'status-dot-amber' },
  reading: { label: 'Reading', color: '#06b6d4', dotClass: 'status-dot-blue' },
  browsing: { label: 'Browsing', color: '#3b82f6', dotClass: 'status-dot-blue' },
  remembering: { label: 'Using Memory', color: '#8b5cf6', dotClass: 'status-dot-purple' },
  speaking: { label: 'Speaking', color: '#22c55e', dotClass: 'status-dot-green' },
  finished: { label: 'Finished', color: '#22c55e', dotClass: 'status-dot-green' },
};

interface Props {
  state: AIState;
}

export default function AIStatusBadge({ state }: Props) {
  const config = STATE_CONFIG[state];
  if (state === 'idle') return null;

  return (
    <AnimatePresence>
      <motion.div
        key={state}
        initial={{ opacity: 0, y: -6, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -6, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        className="flex items-center gap-2 rounded-full px-3 py-1"
        style={{
          background: 'rgba(0,0,0,0.4)',
          border: `1px solid ${config.color}30`,
          backdropFilter: 'blur(12px)',
        }}
      >
        {/* Dot */}
        <span
          className={`status-dot ${config.dotClass} animate-pulse-soft`}
        />

        {/* Thinking dots or label */}
        {state === 'thinking' ? (
          <div className="flex items-center gap-1" style={{ color: config.color }}>
            <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.7)' }}>Thinking</span>
            <div className="flex gap-0.5 ml-1">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="animate-thinking rounded-full"
                  style={{
                    width: 4,
                    height: 4,
                    background: config.color,
                    display: 'inline-block',
                    animationDelay: `${i * 0.16}s`,
                  }}
                />
              ))}
            </div>
          </div>
        ) : (
          <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.7)' }}>
            {config.label}
          </span>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
