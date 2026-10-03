import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  MoreHorizontal,
  Database,
  FolderOpen,
  Cpu,
  Trash2,
  ChevronRight,
} from 'lucide-react';
import AIStatusBadge from './AIStatusBadge';
import MessageBubble from './MessageBubble';
import WelcomeScreen from './WelcomeScreen';
import ChatInput from './ChatInput';
import type { Message, AIState, DrawerType } from '../types';

interface Props {
  messages: Message[];
  aiState: AIState;
  isWelcome: boolean;
  drawer: DrawerType;
  toggleDrawer: (t: DrawerType) => void;
  inputValue: string;
  setInputValue: (v: string) => void;
  onSend: (v: string) => void;
  onClear: () => void;
  webEnabled: boolean;
  setWebEnabled: (v: boolean) => void;
  memoryEnabled: boolean;
  setMemoryEnabled: (v: boolean) => void;
  computerEnabled: boolean;
  setComputerEnabled: (v: boolean) => void;
  dragOver: boolean;
  setDragOver: (v: boolean) => void;
}

const DRAWER_ITEMS: { id: DrawerType; icon: React.ElementType; label: string }[] = [
  { id: 'memory', icon: Database, label: 'Memory' },
  { id: 'files', icon: FolderOpen, label: 'Files' },
  { id: 'tasks', icon: Cpu, label: 'Tasks' },
];

const isThinking = (s: AIState) =>
  ['thinking', 'searching', 'coding', 'reading', 'browsing', 'remembering'].includes(s);

export default function ChatView({
  messages, aiState, isWelcome,
  drawer, toggleDrawer,
  inputValue, setInputValue,
  onSend, onClear,
  webEnabled, setWebEnabled,
  memoryEnabled, setMemoryEnabled,
  computerEnabled, setComputerEnabled,
  dragOver, setDragOver,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, aiState]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3 flex-shrink-0"
        style={{
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          background: 'rgba(0,0,0,0.2)',
        }}
      >
        <div className="flex items-center gap-3">
          <div>
            <h2 className="font-semibold text-sm" style={{ color: '#d4d4e8' }}>Aspire</h2>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.28)' }}>Personal AI Operating System</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* AI Status */}
          <AIStatusBadge state={aiState} />

          {/* Drawer toggles */}
          <div className="flex items-center gap-1 ml-2">
            {DRAWER_ITEMS.map(item => {
              const Icon = item.icon;
              const active = drawer === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => toggleDrawer(item.id)}
                  className="flex items-center justify-center rounded-lg transition-all"
                  style={{
                    width: 30,
                    height: 30,
                    background: active ? 'rgba(99,102,241,0.15)' : 'transparent',
                    color: active ? '#818cf8' : 'rgba(255,255,255,0.3)',
                  }}
                  title={item.label}
                >
                  <Icon size={14} />
                </button>
              );
            })}
          </div>

          {!isWelcome && (
            <button
              onClick={onClear}
              className="flex items-center justify-center rounded-lg transition-all hover:bg-white/[0.05]"
              style={{ width: 30, height: 30, color: 'rgba(255,255,255,0.25)' }}
              title="Clear conversation"
            >
              <Trash2 size={13} />
            </button>
          )}

          <button
            className="flex items-center justify-center rounded-lg transition-all hover:bg-white/[0.05]"
            style={{ width: 30, height: 30, color: 'rgba(255,255,255,0.25)' }}
          >
            <MoreHorizontal size={14} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Message area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto px-5 py-4" style={{ scrollbarWidth: 'thin' }}>
            {isWelcome ? (
              <WelcomeScreen onPrompt={onSend} />
            ) : (
              <div className="flex flex-col gap-4 max-w-3xl mx-auto pb-4">
                {messages.map((msg, i) => (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    showThinking={msg.role === 'assistant' && msg.content === '' && i === messages.length - 1 && isThinking(aiState)}
                  />
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div className="flex-shrink-0 max-w-3xl mx-auto w-full">
            <ChatInput
              value={inputValue}
              onChange={setInputValue}
              onSend={onSend}
              disabled={isThinking(aiState)}
              webEnabled={webEnabled}
              setWebEnabled={setWebEnabled}
              memoryEnabled={memoryEnabled}
              setMemoryEnabled={setMemoryEnabled}
              computerEnabled={computerEnabled}
              setComputerEnabled={setComputerEnabled}
              dragOver={dragOver}
              setDragOver={setDragOver}
            />
          </div>
        </div>

        {/* Contextual Drawer */}
        <AnimatePresence>
          {drawer && (
            <DrawerPanel drawer={drawer} onClose={() => toggleDrawer(null)} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function DrawerPanel({ drawer, onClose }: { drawer: DrawerType; onClose: () => void }) {

  const DRAWER_CONTENT: Record<NonNullable<DrawerType>, { title: string; items: { icon: string; label: string; sub: string }[] }> = {
    memory: {
      title: 'Memory',
      items: [
        { icon: '🧠', label: 'Fetching memories…', sub: 'From backend' },
      ],
    },
    files: {
      title: 'Files',
      items: [
        { icon: '📄', label: 'No files attached', sub: 'Drop files in chat to attach' },
      ],
    },
    tasks: {
      title: 'Running Tasks',
      items: [
        { icon: '⚡', label: 'No active tasks', sub: 'Tasks appear here during agent execution' },
      ],
    },
    reasoning: {
      title: 'Reasoning',
      items: [
        { icon: '💭', label: 'No reasoning data', sub: 'Appears during AI processing' },
      ],
    },
    tokens: {
      title: 'Token Usage',
      items: [
        { icon: '📊', label: 'Context: awaiting session', sub: 'Updated during conversation' },
      ],
    },
  };

  const content = DRAWER_CONTENT[drawer!];

  return (
    <motion.div
      key="drawer"
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 260, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className="drawer flex-shrink-0 overflow-hidden"
    >
      <div className="w-[260px] h-full flex flex-col">
        {/* Drawer header */}
        <div
          className="flex items-center justify-between px-4 py-3 flex-shrink-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        >
          <span className="text-sm font-semibold" style={{ color: '#c4c4d4' }}>
            {content.title}
          </span>
          <button
            onClick={onClose}
            className="flex items-center justify-center rounded-lg w-6 h-6 transition-all hover:bg-white/[0.07]"
            style={{ color: 'rgba(255,255,255,0.3)' }}
          >
            <ChevronRight size={13} />
          </button>
        </div>

        {/* Drawer content */}
        <div className="flex-1 overflow-y-auto p-3">
          <div className="flex flex-col gap-1">
            {content.items.map((item, i) => (
              <div
                key={i}
                className="flex items-start gap-2.5 rounded-lg p-2.5 transition-all hover:bg-white/[0.04] cursor-pointer"
              >
                <span style={{ fontSize: 14, flexShrink: 0, marginTop: 1 }}>{item.icon}</span>
                <div className="min-w-0">
                  <div className="text-xs font-medium truncate" style={{ color: 'rgba(240,240,244,0.75)' }}>
                    {item.label}
                  </div>
                  <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
                    {item.sub}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
