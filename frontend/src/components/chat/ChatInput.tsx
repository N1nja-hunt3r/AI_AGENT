import { memo, useRef, useCallback } from "react";
import type { KeyboardEvent, ClipboardEvent } from "react";
import { Send, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { FileUploader } from "./FileUploader";
import type { ChatAttachment } from "./types";

interface ChatInputProps {
  onSend: (content: string, attachments: ChatAttachment[]) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
  attachments?: ChatAttachment[];
  onAttach?: (files: File[]) => void;
  onRemoveAttachment?: (id: string) => void;
  className?: string;
}

export const ChatInput = memo<ChatInputProps>(
  ({
    onSend,
    onStop,
    isStreaming = false,
    disabled = false,
    placeholder = "Ask Aspire anything…",
    attachments = [],
    onAttach,
    onRemoveAttachment,
    className,
  }) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSend = useCallback(() => {
      const el = textareaRef.current;
      if (!el) return;
      const text = el.value.trim();
      if (!text && attachments.length === 0) return;
      onSend(text, attachments);
      el.value = "";
      el.style.height = "auto";
    }, [onSend, attachments]);

    const handleKeyDown = useCallback(
      (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      },
      [handleSend]
    );

    const handlePaste = useCallback(
      (e: ClipboardEvent<HTMLTextAreaElement>) => {
        const files = Array.from(e.clipboardData.files);
        if (files.length > 0 && onAttach) {
          e.preventDefault();
          onAttach(files);
        }
      },
      [onAttach]
    );

    const resize = useCallback(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }, []);

    return (
      <div className={cn("relative w-full", className)}>
        <div className="flex flex-col rounded-2xl border border-white/10 bg-gray-900/60 backdrop-blur-xl focus-within:border-violet-500/40 transition-all">
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-1.5 px-3 pt-3">
              {attachments.map((att) => (
                <div
                  key={att.id}
                  className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-gray-800/60 px-2.5 py-1.5 text-xs"
                >
                  <span className="max-w-30 truncate font-medium text-gray-300">{att.name}</span>
                  <button
                    type="button"
                    onClick={() => onRemoveAttachment?.(att.id)}
                    className="ml-0.5 text-gray-600 hover:text-gray-300 transition-colors"
                    aria-label={`Remove ${att.name}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            ref={textareaRef}
            onChange={resize}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={placeholder}
            disabled={disabled || isStreaming}
            rows={1}
            className="flex-1 resize-none bg-transparent px-4 py-3.5 text-sm text-gray-200 placeholder-gray-600 outline-none scrollbar-thin leading-relaxed max-h-40"
            aria-label="Message input"
          />

          <div className="flex items-center justify-between px-3 pb-3">
            <div className="flex items-center gap-1">
              {onAttach && (
                <FileUploader
                  attachments={attachments}
                  onAttach={onAttach}
                  onRemove={onRemoveAttachment ?? (() => {})}
                  disabled={disabled}
                />
              )}
            </div>

            <div className="flex items-center gap-2">
              {isStreaming ? (
                <button
                  type="button"
                  onClick={onStop}
                  className="flex h-8 w-8 items-center justify-center rounded-xl bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                  aria-label="Stop generating"
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={disabled}
                  className="flex h-8 w-8 items-center justify-center rounded-xl bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                  aria-label="Send message"
                >
                  <Send className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }
);

ChatInput.displayName = "ChatInput";
