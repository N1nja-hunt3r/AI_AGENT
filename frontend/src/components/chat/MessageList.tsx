import { memo, useEffect, useRef } from "react";
import { MessageSquare, Code2, FileText, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Message } from "./Message";
import type { ChatMessage } from "./types";

interface MessageListProps {
  messages: ChatMessage[];
  isTyping?: boolean;
  onSendMessage?: (content: string) => void;
  onEdit?: (messageId: string, newContent: string) => void;
  onBranch?: (messageId: string) => void;
  className?: string;
}

const SUGGESTIONS = [
  { icon: MessageSquare, text: "Explain quantum computing in simple terms" },
  { icon: Code2, text: "Write a React hook for debouncing" },
  { icon: FileText, text: "Summarize this article for me" },
  { icon: Sparkles, text: "Help me plan a coding project" },
];

export const MessageList = memo<MessageListProps>(
  ({ messages, isTyping = false, onSendMessage, onEdit, onBranch, className }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isTyping]);

    if (messages.length === 0 && !isTyping) {
      return (
        <div className={cn("flex h-full flex-col items-center justify-center gap-4 px-6 text-center", className)}>
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-500/10 ring-1 ring-violet-500/20">
            <Sparkles className="h-6 w-6 text-violet-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-100">How can I help you today?</h2>
            <p className="mt-1 text-sm text-gray-500">Ask anything, or choose a suggestion below.</p>
          </div>
          {onSendMessage && (
            <div className="grid w-full max-w-md grid-cols-1 gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => {
                const Icon = s.icon;
                return (
                  <button
                    key={s.text}
                    onClick={() => onSendMessage(s.text)}
                    className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2.5 text-left text-xs text-gray-400 transition-all hover:border-violet-500/30 hover:bg-violet-500/10 hover:text-gray-200"
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0 text-violet-400" />
                    <span className="leading-snug">{s.text}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      );
    }

    return (
      <div className={cn("flex flex-col h-full", className)}>
        <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
          <div className="mx-auto max-w-3xl px-4 py-4 space-y-1">
            {messages.map((msg, i) => (
              <Message
                key={msg.id}
                message={msg}
                isLast={i === messages.length - 1}
                onEdit={onEdit}
                onBranch={onBranch}
              />
            ))}
            {isTyping && (
              <div className="flex items-center gap-2 px-2 py-3 text-gray-500">
                <div className="flex gap-1">
                  <span className="h-2 w-2 rounded-full bg-gray-600 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="h-2 w-2 rounded-full bg-gray-600 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="h-2 w-2 rounded-full bg-gray-600 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                <span className="text-xs">AI is thinking...</span>
              </div>
            )}
          </div>
          <div ref={bottomRef} />
        </div>
      </div>
    );
  }
);

MessageList.displayName = "MessageList";
