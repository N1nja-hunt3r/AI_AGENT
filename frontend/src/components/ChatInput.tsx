import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Mic,
  Paperclip,
  Image,
  Globe,
  Brain,
  Monitor,
  X,
} from 'lucide-react';
import { cn } from '../utils/cn';

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: (v: string) => void;
  onSendVoice?: () => void;
  disabled?: boolean;
  webEnabled: boolean;
  setWebEnabled: (v: boolean) => void;
  memoryEnabled: boolean;
  setMemoryEnabled: (v: boolean) => void;
  computerEnabled: boolean;
  setComputerEnabled: (v: boolean) => void;
  dragOver: boolean;
  setDragOver: (v: boolean) => void;
}

interface PendingFile {
  id: string;
  name: string;
  type: string;
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  disabled,
  webEnabled, setWebEnabled,
  memoryEnabled, setMemoryEnabled,
  computerEnabled, setComputerEnabled,
  dragOver, setDragOver,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [files, setFiles] = useState<PendingFile[]>([]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value);
    setFiles([]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files);
    const newFiles: PendingFile[] = dropped.map(f => ({
      id: Math.random().toString(36).slice(2),
      name: f.name,
      type: f.type,
    }));
    setFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (id: string) => setFiles(prev => prev.filter(f => f.id !== id));

  const TOGGLES = [
    {
      id: 'web',
      icon: Globe,
      label: 'Web',
      active: webEnabled,
      toggle: () => setWebEnabled(!webEnabled),
      color: '#3b82f6',
    },
    {
      id: 'memory',
      icon: Brain,
      label: 'Memory',
      active: memoryEnabled,
      toggle: () => setMemoryEnabled(!memoryEnabled),
      color: '#8b5cf6',
    },
    {
      id: 'computer',
      icon: Monitor,
      label: 'Computer',
      active: computerEnabled,
      toggle: () => setComputerEnabled(!computerEnabled),
      color: '#ec4899',
    },
  ];

  return (
    <div className="px-4 pb-2 pt-1">
      <div
        className={cn('input-area', dragOver && 'accent-glow')}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {/* Attached files */}
        <AnimatePresence>
          {files.length > 0 && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="px-3 pt-3 flex flex-wrap gap-2"
            >
              {files.map(f => (
                <div
                  key={f.id}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1"
                  style={{
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid rgba(255,255,255,0.09)',
                    fontSize: '0.78rem',
                    color: 'rgba(255,255,255,0.65)',
                  }}
                >
                  <Paperclip size={10} style={{ color: 'rgba(255,255,255,0.4)' }} />
                  <span>{f.name}</span>
                  <button onClick={() => removeFile(f.id)} className="ml-1" style={{ color: 'rgba(255,255,255,0.35)' }}>
                    <X size={10} />
                  </button>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Drop overlay */}
        <AnimatePresence>
          {dragOver && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 rounded-2xl flex items-center justify-center z-10"
              style={{ background: 'rgba(99,102,241,0.08)', border: '2px dashed rgba(99,102,241,0.4)' }}
            >
              <div className="text-center">
                <div style={{ fontSize: 28 }}>📎</div>
                <p className="text-sm mt-1" style={{ color: '#818cf8' }}>Drop files or images</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Textarea */}
        <div className="px-4 pt-3 pb-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={e => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Aspire anything…"
            rows={1}
            disabled={disabled}
            className="w-full"
            style={{
              minHeight: 28,
              maxHeight: 180,
              color: 'rgba(240,240,244,0.9)',
            }}
          />
        </div>

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between px-3 pb-3 gap-2">
          {/* Left: file/image attach */}
          <div className="flex items-center gap-1">
            <label
              className="flex items-center justify-center rounded-lg cursor-pointer transition-all hover:bg-white/[0.06]"
              style={{ width: 32, height: 32 }}
              title="Attach file"
            >
              <input type="file" multiple className="hidden" onChange={e => {
                const newFiles: PendingFile[] = Array.from(e.target.files || []).map(f => ({
                  id: Math.random().toString(36).slice(2),
                  name: f.name,
                  type: f.type,
                }));
                setFiles(prev => [...prev, ...newFiles]);
              }} />
              <Paperclip size={15} style={{ color: 'rgba(255,255,255,0.35)' }} />
            </label>

            <label
              className="flex items-center justify-center rounded-lg cursor-pointer transition-all hover:bg-white/[0.06]"
              style={{ width: 32, height: 32 }}
              title="Attach image"
            >
              <input type="file" accept="image/*" multiple className="hidden" onChange={e => {
                const newFiles: PendingFile[] = Array.from(e.target.files || []).map(f => ({
                  id: Math.random().toString(36).slice(2),
                  name: f.name,
                  type: f.type,
                }));
                setFiles(prev => [...prev, ...newFiles]);
              }} />
              <Image size={15} style={{ color: 'rgba(255,255,255,0.35)' }} />
            </label>

            {/* Divider */}
            <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />

            {/* Toggles */}
            {TOGGLES.map(t => {
              const Icon = t.icon;
              return (
                <button
                  key={t.id}
                  onClick={t.toggle}
                  className="flex items-center gap-1 rounded-lg px-2 py-1 transition-all"
                  style={{
                    height: 28,
                    background: t.active ? `${t.color}18` : 'transparent',
                    border: `1px solid ${t.active ? `${t.color}35` : 'transparent'}`,
                  }}
                  title={t.label}
                >
                  <Icon
                    size={13}
                    style={{ color: t.active ? t.color : 'rgba(255,255,255,0.3)' }}
                  />
                  <span
                    className="text-xs font-medium hidden sm:block"
                    style={{ color: t.active ? t.color : 'rgba(255,255,255,0.3)' }}
                  >
                    {t.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Right: voice + send */}
          <div className="flex items-center gap-2">
            <button
              className="flex items-center justify-center rounded-lg transition-all hover:bg-white/[0.06]"
              style={{ width: 32, height: 32 }}
              title="Voice input"
            >
              <Mic size={15} style={{ color: 'rgba(255,255,255,0.35)' }} />
            </button>

            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={handleSend}
              disabled={!value.trim() || disabled}
              className="flex items-center justify-center rounded-lg transition-all"
              style={{
                width: 32,
                height: 32,
                background: value.trim() && !disabled
                  ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
                  : 'rgba(255,255,255,0.06)',
                opacity: !value.trim() || disabled ? 0.4 : 1,
                boxShadow: value.trim() && !disabled ? '0 0 12px rgba(99,102,241,0.4)' : 'none',
              }}
            >
              <Send size={13} style={{ color: value.trim() && !disabled ? 'white' : 'rgba(255,255,255,0.4)' }} />
            </motion.button>
          </div>
        </div>
      </div>

      {/* Hint */}
      <div className="text-center mt-1.5">
        <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.18)' }}>
          Aspire automatically routes your request to the right capability
        </span>
      </div>
    </div>
  );
}
