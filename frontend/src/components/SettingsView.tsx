import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cpu,
  Key,
  Mic,
  Palette,
  Shield,
  Brain,
  Code,
  ChevronRight,
  Check,
  Eye,
  EyeOff,
} from 'lucide-react';

type SettingsSection =
  | 'models'
  | 'providers'
  | 'apikeys'
  | 'voice'
  | 'theme'
  | 'permissions'
  | 'memory'
  | 'developer';

const SECTIONS: { id: SettingsSection; icon: React.ElementType; label: string; sub: string }[] = [
  { id: 'models', icon: Cpu, label: 'Models', sub: 'AI model selection & parameters' },
  { id: 'providers', icon: ChevronRight, label: 'Providers', sub: 'NVIDIA NIM, OpenAI (fallback)' },
  { id: 'apikeys', icon: Key, label: 'API Keys', sub: 'Manage provider credentials' },
  { id: 'voice', icon: Mic, label: 'Voice & Wake Word', sub: 'Voice settings and activation' },
  { id: 'theme', icon: Palette, label: 'Themes', sub: 'Appearance and display' },
  { id: 'permissions', icon: Shield, label: 'Permissions', sub: 'System access controls' },
  { id: 'memory', icon: Brain, label: 'Memory', sub: 'Persistent context settings' },
  { id: 'developer', icon: Code, label: 'Developer Mode', sub: 'Advanced tools and debugging' },
];

const MODEL_OPTIONS = [
  { id: 'deepseek-v4-pro', label: 'DeepSeek V4 Pro', provider: 'NVIDIA NIM', badge: 'Chief' },
  { id: 'qwen-3.5-122b', label: 'Qwen 3.5 122B', provider: 'NVIDIA NIM', badge: 'Chief (Fallback)' },
  { id: 'llama-3.3-70b', label: 'Llama 3.3 70B', provider: 'NVIDIA NIM', badge: 'Coder' },
  { id: 'llama-3.1-70b', label: 'Llama 3.1 70B', provider: 'NVIDIA NIM', badge: 'Coder (Fallback)' },
  { id: 'qwen-3.5-397b', label: 'Qwen 3.5 397B', provider: 'NVIDIA NIM', badge: 'Research' },
  { id: 'qwen-3-next-80b', label: 'Qwen 3 Next 80B', provider: 'NVIDIA NIM', badge: 'Research (Fallback)' },
  { id: 'llama-3.2-90b-vision', label: 'Llama 3.2 90B Vision', provider: 'NVIDIA NIM', badge: 'Vision' },
  { id: 'llama-3.2-11b-vision', label: 'Llama 3.2 11B Vision', provider: 'NVIDIA NIM', badge: 'Vision (Fallback)' },
  { id: 'gpt-4o', label: 'GPT-4o', provider: 'OpenAI (Emergency Fallback)', badge: null },
];

const VOICE_OPTIONS = ['Whisper Large v3', 'Parakeet 1.1B', 'Magpie TTS', 'Chatterbox TTS'];

function ModelsPanel() {
  const [selected, setSelected] = useState('deepseek-v4-pro');
  const [temp, setTemp] = useState(0.7);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold mb-1" style={{ color: '#d4d4e8' }}>Primary Model</h3>
        <p className="text-xs mb-3" style={{ color: 'rgba(255,255,255,0.3)' }}>Used for all general requests unless overridden by the planner</p>
        <div className="grid grid-cols-2 gap-2">
          {MODEL_OPTIONS.map(m => (
            <button
              key={m.id}
              onClick={() => setSelected(m.id)}
              className="flex items-center justify-between rounded-xl p-3 text-left transition-all"
              style={{
                background: selected === m.id ? 'rgba(99,102,241,0.12)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${selected === m.id ? 'rgba(99,102,241,0.35)' : 'rgba(255,255,255,0.07)'}`,
              }}
            >
              <div>
                <div className="text-sm font-medium" style={{ color: selected === m.id ? '#a5b4fc' : 'rgba(255,255,255,0.7)' }}>
                  {m.label}
                </div>
                <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>{m.provider}</div>
              </div>
              <div className="flex items-center gap-2">
                {m.badge && (
                  <span
                    className="text-xs px-1.5 py-0.5 rounded-md font-medium"
                    style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}
                  >
                    {m.badge}
                  </span>
                )}
                {selected === m.id && <Check size={14} style={{ color: '#6366f1' }} />}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold" style={{ color: '#d4d4e8' }}>Temperature</h3>
          <span className="text-sm font-mono" style={{ color: '#6366f1' }}>{temp.toFixed(1)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          value={temp}
          onChange={e => setTemp(parseFloat(e.target.value))}
          className="w-full"
          style={{ accentColor: '#6366f1' }}
        />
        <div className="flex justify-between mt-1">
          <span className="text-xs" style={{ color: 'rgba(255,255,255,0.25)' }}>Precise</span>
          <span className="text-xs" style={{ color: 'rgba(255,255,255,0.25)' }}>Creative</span>
        </div>
      </div>
    </div>
  );
}

function APIKeysPanel() {
  const [show, setShow] = useState<Record<string, boolean>>({});
  const [keys, setKeys] = useState({
    nvidia_nim: '',
    openai: '',
  });

  const PROVIDERS = [
    { id: 'nvidia_nim', label: 'NVIDIA NIM', placeholder: 'nvapi-...' },
    { id: 'openai', label: 'OpenAI (Fallback)', placeholder: 'sk-...' },
  ] as const;

  return (
    <div className="space-y-4">
      <p className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
        API keys are stored locally and never sent to our servers.
      </p>
      {PROVIDERS.map(p => (
        <div key={p.id}>
          <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(255,255,255,0.5)' }}>
            {p.label}
          </label>
          <div
            className="flex items-center rounded-xl overflow-hidden"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            <input
              type={show[p.id] ? 'text' : 'password'}
              value={keys[p.id]}
              onChange={e => setKeys(prev => ({ ...prev, [p.id]: e.target.value }))}
              placeholder={p.placeholder}
              className="flex-1 bg-transparent px-3 py-2.5 text-sm outline-none"
              style={{ color: 'rgba(255,255,255,0.7)' }}
            />
            <button
              onClick={() => setShow(prev => ({ ...prev, [p.id]: !prev[p.id] }))}
              className="px-3 py-2.5"
              style={{ color: 'rgba(255,255,255,0.3)' }}
            >
              {show[p.id] ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function VoicePanel() {
  const [selectedVoice, setSelectedVoice] = useState('Nova');
  const [wakeWord, setWakeWord] = useState('Hey Aspire');
  const [continuousMode, setContinuousMode] = useState(true);

  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-sm font-semibold mb-2" style={{ color: '#d4d4e8' }}>AI Voice</h3>
        <div className="grid grid-cols-3 gap-2">
          {VOICE_OPTIONS.map(v => (
            <button
              key={v}
              onClick={() => setSelectedVoice(v)}
              className="rounded-xl py-2 text-sm font-medium transition-all"
              style={{
                background: selectedVoice === v ? 'rgba(99,102,241,0.15)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${selectedVoice === v ? 'rgba(99,102,241,0.4)' : 'rgba(255,255,255,0.07)'}`,
                color: selectedVoice === v ? '#a5b4fc' : 'rgba(255,255,255,0.5)',
              }}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-1.5" style={{ color: '#d4d4e8' }}>Wake Word</h3>
        <input
          value={wakeWord}
          onChange={e => setWakeWord(e.target.value)}
          className="w-full rounded-xl px-3 py-2.5 text-sm outline-none"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            color: 'rgba(255,255,255,0.7)',
          }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold" style={{ color: '#d4d4e8' }}>Continuous Listening</div>
          <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
            Stays active after each response
          </div>
        </div>
        <button
          onClick={() => setContinuousMode(!continuousMode)}
          className="relative rounded-full transition-all"
          style={{
            width: 44,
            height: 24,
            background: continuousMode ? '#6366f1' : 'rgba(255,255,255,0.1)',
          }}
        >
          <div
            className="absolute top-1 rounded-full transition-all"
            style={{
              width: 16,
              height: 16,
              background: 'white',
              left: continuousMode ? 24 : 4,
            }}
          />
        </button>
      </div>
    </div>
  );
}

function ThemePanel() {
  const [theme, setTheme] = useState('graphite');
  const THEMES = [
    { id: 'graphite', label: 'Graphite', colors: ['#0d0d0f', '#6366f1', '#8b5cf6'] },
    { id: 'midnight', label: 'Midnight Blue', colors: ['#050a1a', '#3b82f6', '#60a5fa'] },
    { id: 'carbon', label: 'Carbon', colors: ['#0a0a0a', '#22c55e', '#4ade80'] },
    { id: 'rose', label: 'Rose', colors: ['#0f0a10', '#ec4899', '#f472b6'] },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {THEMES.map(t => (
          <button
            key={t.id}
            onClick={() => setTheme(t.id)}
            className="rounded-xl p-4 text-left transition-all"
            style={{
              background: t.colors[0],
              border: `2px solid ${theme === t.id ? t.colors[1] : 'rgba(255,255,255,0.07)'}`,
            }}
          >
            <div className="flex gap-1 mb-3">
              {t.colors.map((c, i) => (
                <div key={i} className="rounded-full" style={{ width: 12, height: 12, background: c }} />
              ))}
            </div>
            <div className="text-sm font-medium" style={{ color: theme === t.id ? t.colors[1] : 'rgba(255,255,255,0.6)' }}>
              {t.label}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function GenericPanel({ section }: { section: SettingsSection }) {
  const CONTENT: Partial<Record<SettingsSection, { items: { label: string; sub: string; value: boolean }[] }>> = {
    permissions: {
      items: [
        { label: 'File System Access', sub: 'Read and write local files', value: true },
        { label: 'Browser Control', sub: 'Open and control Chrome', value: false },
        { label: 'Screen Capture', sub: 'Take screenshots for Vision', value: true },
        { label: 'Clipboard Access', sub: 'Read clipboard content', value: true },
        { label: 'Microphone', sub: 'Voice mode and transcription', value: true },
      ],
    },
    memory: {
      items: [
        { label: 'Auto-save memories', sub: 'Automatically extract and store context', value: true },
        { label: 'Cross-session memory', sub: 'Remember context between sessions', value: true },
        { label: 'Project memory', sub: 'Track per-project context separately', value: false },
        { label: 'Memory search', sub: 'Use memory in every response', value: true },
      ],
    },
    developer: {
      items: [
        { label: 'Show reasoning', sub: 'Display AI thought process', value: false },
        { label: 'Show token usage', sub: 'Display token counts per message', value: true },
        { label: 'Show latency', sub: 'Display response time per request', value: true },
        { label: 'Enable request logging', sub: 'Log all API requests for debugging', value: false },
        { label: 'Show raw API responses', sub: 'Display unprocessed API response data', value: false },
      ],
    },
  };

  const [values, setValues] = useState<Record<string, boolean>>(
    (CONTENT[section]?.items || []).reduce((acc, item) => ({ ...acc, [item.label]: item.value }), {})
  );

  const items = CONTENT[section]?.items || [];

  return (
    <div className="space-y-2">
      {items.map(item => (
        <div
          key={item.label}
          className="flex items-center justify-between rounded-xl p-3"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div>
            <div className="text-sm font-medium" style={{ color: 'rgba(255,255,255,0.75)' }}>{item.label}</div>
            <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>{item.sub}</div>
          </div>
          <button
            onClick={() => setValues(prev => ({ ...prev, [item.label]: !prev[item.label] }))}
            className="relative rounded-full transition-all flex-shrink-0"
            style={{
              width: 40,
              height: 22,
              background: values[item.label] ? '#6366f1' : 'rgba(255,255,255,0.1)',
            }}
          >
            <div
              className="absolute top-1 rounded-full transition-all"
              style={{
                width: 14,
                height: 14,
                background: 'white',
                left: values[item.label] ? 22 : 4,
              }}
            />
          </button>
        </div>
      ))}
    </div>
  );
}

function ProvidersPanel() {
  const PROVIDERS = [
    { name: 'NVIDIA NIM', status: 'connected', models: 'DeepSeek V4 Pro, Llama 3.3 70B, Qwen 3.5 397B, Vision, Embeddings, Whisper, Magpie TTS + fallbacks' },
    { name: 'OpenAI (Emergency Fallback)', status: 'disconnected', models: 'GPT-4o (used only when NVIDIA NIM is unavailable)' },
  ];
  return (
    <div className="space-y-2">
      {PROVIDERS.map(p => (
        <div
          key={p.name}
          className="flex items-center justify-between rounded-xl p-3"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div>
            <div className="text-sm font-medium" style={{ color: 'rgba(255,255,255,0.75)' }}>{p.name}</div>
            <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>{p.models}</div>
          </div>
          <div
            className="text-xs px-2.5 py-1 rounded-full font-medium"
            style={{
              background: p.status === 'connected' ? 'rgba(34,197,94,0.12)' :
                p.status === 'available' ? 'rgba(59,130,246,0.12)' : 'rgba(255,255,255,0.06)',
              color: p.status === 'connected' ? '#22c55e' :
                p.status === 'available' ? '#60a5fa' : 'rgba(255,255,255,0.3)',
            }}
          >
            {p.status}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function SettingsView() {
  const [active, setActive] = useState<SettingsSection>('models');

  const renderPanel = () => {
    switch (active) {
      case 'models': return <ModelsPanel />;
      case 'apikeys': return <APIKeysPanel />;
      case 'voice': return <VoicePanel />;
      case 'theme': return <ThemePanel />;
      case 'providers': return <ProvidersPanel />;
      default: return <GenericPanel section={active} />;
    }
  };

  const activeSection = SECTIONS.find(s => s.id === active)!;

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <div
        className="flex-shrink-0 flex flex-col overflow-y-auto"
        style={{
          width: 220,
          borderRight: '1px solid rgba(255,255,255,0.06)',
          background: 'rgba(0,0,0,0.15)',
          padding: '16px 8px',
        }}
      >
        <div className="px-3 mb-4">
          <h2 className="text-sm font-semibold" style={{ color: '#c4c4d4' }}>Settings</h2>
          <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.25)' }}>Configure Aspire</p>
        </div>
        <div className="flex flex-col gap-0.5">
          {SECTIONS.map(s => {
            const Icon = s.icon;
            const isActive = active === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setActive(s.id)}
                className="nav-item flex items-center gap-2.5 px-3 py-2.5 text-left"
                style={{
                  background: isActive ? 'rgba(99,102,241,0.12)' : undefined,
                  color: isActive ? '#818cf8' : 'rgba(255,255,255,0.45)',
                }}
              >
                <Icon size={15} style={{ flexShrink: 0 }} />
                <span className="text-sm font-medium">{s.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <div className="max-w-lg">
            <div className="mb-6">
              <h2 className="text-lg font-semibold" style={{ color: '#f0f0f4', letterSpacing: '-0.01em' }}>
                {activeSection.label}
              </h2>
              <p className="text-sm mt-1" style={{ color: 'rgba(255,255,255,0.35)' }}>
                {activeSection.sub}
              </p>
            </div>
            {renderPanel()}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
