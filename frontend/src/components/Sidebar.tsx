import { motion } from 'framer-motion';
import {
  MessageSquare,
  Mic,
  Settings,
  Zap,
  ChevronRight,
} from 'lucide-react';
import { cn } from '../utils/cn';
import type { AppView } from '../types';

interface Props {
  view: AppView;
  setView: (v: AppView) => void;
}

const NAV = [
  { id: 'chat' as AppView, icon: MessageSquare, label: 'Chat' },
  { id: 'voice' as AppView, icon: Mic, label: 'Voice' },
  { id: 'settings' as AppView, icon: Settings, label: 'Settings' },
];

export default function Sidebar({ view, setView }: Props) {
  return (
    <aside
      className="flex flex-col h-full"
      style={{
        width: 60,
        background: 'rgba(0,0,0,0.35)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center justify-center py-4 px-2">
        <div
          className="flex items-center justify-center rounded-xl"
          style={{
            width: 36,
            height: 36,
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            boxShadow: '0 0 16px rgba(99,102,241,0.4)',
          }}
        >
          <Zap size={16} fill="white" color="white" />
        </div>
      </div>

      <div className="flex-1 flex flex-col gap-1 px-2 pt-4">
        {NAV.map(item => {
          const Icon = item.icon;
          const active = view === item.id;
          return (
            <div key={item.id} className="relative group">
              <motion.button
                whileTap={{ scale: 0.92 }}
                onClick={() => setView(item.id)}
                className={cn(
                  'nav-item w-full flex items-center justify-center',
                  active && 'active'
                )}
                style={{ height: 40 }}
                title={item.label}
              >
                <Icon
                  size={18}
                  style={{
                    color: active ? '#818cf8' : 'rgba(255,255,255,0.45)',
                    transition: 'color 0.15s',
                  }}
                />
              </motion.button>

              {/* Tooltip */}
              <div
                className="absolute left-full ml-2 top-1/2 -translate-y-1/2 pointer-events-none z-50 opacity-0 group-hover:opacity-100 transition-opacity duration-150"
              >
                <div
                  className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium whitespace-nowrap"
                  style={{
                    background: '#1e1e28',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#c4c4d4',
                  }}
                >
                  <ChevronRight size={10} style={{ color: '#6366f1' }} />
                  {item.label}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom ornament */}
      <div className="flex flex-col items-center pb-4 gap-2">
        <div
          className="w-6"
          style={{
            height: 1,
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)',
          }}
        />
        <div
          className="text-center"
          style={{
            fontSize: 9,
            color: 'rgba(255,255,255,0.2)',
            letterSpacing: '0.05em',
            writingMode: 'vertical-rl',
          }}
        >
          ASPIRE
        </div>
      </div>
    </aside>
  );
}
