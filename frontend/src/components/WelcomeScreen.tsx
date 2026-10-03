import { motion } from 'framer-motion';
import type { CapabilityCard } from '../types';

const CAPABILITIES: CapabilityCard[] = [
  {
    id: 'research',
    icon: '🔍',
    label: 'Research',
    prompt: 'Research the latest advancements in AI language models and summarize the key breakthroughs.',
    color: '#3b82f6',
  },
  {
    id: 'code',
    icon: '⚡',
    label: 'Code',
    prompt: 'Write a TypeScript function that fetches data from an API with automatic retry and exponential backoff.',
    color: '#f59e0b',
  },
  {
    id: 'remember',
    icon: '🧠',
    label: 'Remember',
    prompt: 'Remember that I prefer concise answers, I work in TypeScript, and my current project is an AI operating system.',
    color: '#8b5cf6',
  },
  {
    id: 'analyze',
    icon: '📊',
    label: 'Analyze',
    prompt: 'Analyze this document and provide a structured summary with key insights and recommended actions.',
    color: '#06b6d4',
  },
  {
    id: 'control',
    icon: '🖥️',
    label: 'Control',
    prompt: 'Open Chrome, go to GitHub, and find the latest trending TypeScript repositories.',
    color: '#ec4899',
  },
  {
    id: 'create',
    icon: '✨',
    label: 'Create',
    prompt: 'Create a detailed product roadmap for an AI-powered task management application.',
    color: '#22c55e',
  },
  {
    id: 'voice',
    icon: '🎙️',
    label: 'Voice',
    prompt: 'Switch to voice mode so I can talk to you hands-free.',
    color: '#6366f1',
  },
  {
    id: 'automate',
    icon: '🔄',
    label: 'Automate',
    prompt: 'Set up an automation that checks my email every morning and summarizes important messages.',
    color: '#f97316',
  },
];

interface Props {
  onPrompt: (prompt: string) => void;
}

export default function WelcomeScreen({ onPrompt }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 pb-4">
      {/* Wordmark */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-10"
      >
        {/* Orb */}
        <div className="flex justify-center mb-6">
          <div className="relative">
            <div
              className="rounded-full flex items-center justify-center"
              style={{
                width: 72,
                height: 72,
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #3b82f6 100%)',
                boxShadow: '0 0 40px rgba(99,102,241,0.35), 0 0 80px rgba(99,102,241,0.15)',
              }}
            >
              <span style={{ fontSize: 28 }}>⚡</span>
            </div>
            {/* Ring */}
            <div
              className="absolute inset-0 rounded-full animate-ripple"
              style={{
                border: '2px solid rgba(99,102,241,0.4)',
                margin: -2,
              }}
            />
          </div>
        </div>

        <h1
          className="font-semibold tracking-tight"
          style={{
            fontSize: '1.875rem',
            color: '#f0f0f4',
            letterSpacing: '-0.02em',
          }}
        >
          Welcome to Aspire
        </h1>
        <p
          className="mt-2"
          style={{ fontSize: '0.9375rem', color: 'rgba(255,255,255,0.4)' }}
        >
          Your Personal AI Operating System
        </p>
      </motion.div>

      {/* Capability Cards */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-2xl"
      >
        <p
          className="text-center mb-4 text-xs font-medium uppercase tracking-widest"
          style={{ color: 'rgba(255,255,255,0.2)' }}
        >
          What would you like to do?
        </p>
        <div className="grid grid-cols-4 gap-2">
          {CAPABILITIES.map((card, i) => (
            <motion.button
              key={card.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.04, duration: 0.3 }}
              onClick={() => onPrompt(card.prompt)}
              className="capability-card flex flex-col items-start gap-2 text-left"
            >
              <div
                className="flex items-center justify-center rounded-lg"
                style={{
                  width: 32,
                  height: 32,
                  background: `${card.color}15`,
                  border: `1px solid ${card.color}25`,
                }}
              >
                <span style={{ fontSize: 14 }}>{card.icon}</span>
              </div>
              <div>
                <div
                  className="text-sm font-medium"
                  style={{ color: '#d4d4e8' }}
                >
                  {card.label}
                </div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
