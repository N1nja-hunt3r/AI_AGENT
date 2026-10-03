import { motion } from 'framer-motion';
import type { Message } from '../types';

interface Props {
  message: Message;
  showThinking?: boolean;
}

const THINKING_LABELS: Record<string, string> = {
  thinking: 'Thinking',
  searching: 'Searching the web',
  coding: 'Writing code',
  reading: 'Reading document',
  browsing: 'Browsing',
  remembering: 'Accessing memory',
};

function ThinkingDots() {
  return (
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
  );
}

function renderContent(content: string) {
  // Simple markdown-like rendering
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let inCode = false;
  let codeLines: string[] = [];
  let codeLang = '';

  lines.forEach((line, idx) => {
    if (line.startsWith('```')) {
      if (!inCode) {
        inCode = true;
        codeLang = line.slice(3).trim();
        codeLines = [];
      } else {
        inCode = false;
        elements.push(
          <div key={`code-${idx}`} className="my-3 rounded-lg overflow-hidden" style={{ background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.08)' }}>
            {codeLang && (
              <div className="flex items-center justify-between px-3 py-1.5" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                <span className="text-xs font-mono" style={{ color: 'rgba(255,255,255,0.3)' }}>{codeLang}</span>
                <span className="text-xs" style={{ color: 'rgba(255,255,255,0.2)' }}>code</span>
              </div>
            )}
            <pre className="p-4 overflow-x-auto" style={{ margin: 0 }}>
              <code className="text-sm font-mono" style={{ color: '#c9d1d9', fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace" }}>
                {codeLines.join('\n')}
              </code>
            </pre>
          </div>
        );
      }
      return;
    }

    if (inCode) {
      codeLines.push(line);
      return;
    }

    if (!line.trim()) {
      elements.push(<div key={idx} className="h-2" />);
      return;
    }

    // Bold
    const parts = line.split(/(\*\*[^*]+\*\*)/g).map((part, pi) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={pi} style={{ color: '#e8e8f4', fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
      }
      // Inline code
      const codeParts = part.split(/(`[^`]+`)/g).map((cp, ci) => {
        if (cp.startsWith('`') && cp.endsWith('`')) {
          return (
            <code key={ci} style={{
              background: 'rgba(255,255,255,0.07)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 4,
              padding: '1px 5px',
              fontSize: '0.825em',
              fontFamily: 'monospace',
              color: '#a5b4fc',
            }}>
              {cp.slice(1, -1)}
            </code>
          );
        }
        return cp;
      });
      return <span key={pi}>{codeParts}</span>;
    });

    if (line.startsWith('- ') || line.startsWith('• ')) {
      elements.push(
        <div key={idx} className="flex gap-2 items-start" style={{ marginBottom: 2 }}>
          <span style={{ color: '#6366f1', marginTop: 2, flexShrink: 0 }}>•</span>
          <span>{parts}</span>
        </div>
      );
    } else if (/^\d+\.\s/.test(line)) {
      const num = line.match(/^(\d+)\./)?.[1];
      const rest = line.replace(/^\d+\.\s/, '');
      const restParts = rest.split(/(\*\*[^*]+\*\*)/g).map((part, pi) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={pi} style={{ color: '#e8e8f4', fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });
      elements.push(
        <div key={idx} className="flex gap-2 items-start" style={{ marginBottom: 2 }}>
          <span style={{ color: '#6366f1', fontSize: '0.8em', marginTop: 3, flexShrink: 0, minWidth: 16 }}>{num}.</span>
          <span>{restParts}</span>
        </div>
      );
    } else {
      elements.push(
        <p key={idx} style={{ margin: '2px 0' }}>{parts}</p>
      );
    }
  });

  return elements;
}

export default function MessageBubble({ message, showThinking }: Props) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      {!isUser && (
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
      )}

      <div
        className={`max-w-[80%] ${isUser ? 'message-user' : 'message-ai'} px-4 py-3`}
        style={{ minWidth: 0 }}
      >
        {/* AI label */}
        {!isUser && (
          <div
            className="text-xs font-medium mb-2"
            style={{ color: '#6366f1', letterSpacing: '0.02em' }}
          >
            Aspire
          </div>
        )}

        {/* Thinking state shown inline inside the assistant bubble */}
        {showThinking ? (
          <div className="flex items-center gap-3" style={{ minHeight: 28 }}>
            <ThinkingDots />
            <span className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
              {THINKING_LABELS.thinking}
            </span>
          </div>
        ) : (
          <>
            <div
              className="prose-chat"
              style={{
                fontSize: '0.9rem',
                lineHeight: 1.7,
                color: isUser ? 'rgba(255,255,255,0.9)' : 'rgba(240,240,244,0.88)',
              }}
            >
              {renderContent(message.content ?? '')}
            </div>

            {/* Timestamp */}
            {message.content && (
              <div
                className="mt-2 text-right"
                style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.2)' }}
              >
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}
