import { memo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Copy, Check, RotateCcw, ThumbsUp, ThumbsDown, Pencil, X, Save, GitBranch,
  User, AlertCircle, Brain, ChevronDown, ChevronRight, BookOpen, FileText, Sparkles
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MarkdownRenderer } from "./MarkdownRenderer";
import type { ChatMessage } from "./types";

interface MessageProps {
  message: ChatMessage;
  isLast?: boolean;
  onRegenerate?: (messageId: string) => void;
  onFeedback?: (messageId: string, feedback: "up" | "down") => void;
  onEdit?: (messageId: string, newContent: string) => void;
  onBranch?: (messageId: string) => void;
}

export const Message = memo<MessageProps>(
  ({ message, isLast = false, onRegenerate, onFeedback, onEdit, onBranch }) => {
    const [copied, setCopied] = useState(false);
    const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
    const [showReasoning, setShowReasoning] = useState(false);
    const [showCitations, setShowCitations] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editContent, setEditContent] = useState(message.content);

    const isUser = message.role === "user";
    const isStreaming = message.status === "streaming";
    const isError = message.status === "error";

    const handleCopy = useCallback(async () => {
      try {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch { /* ignore */ }
    }, [message.content]);

    const handleFeedback = useCallback(
      (type: "up" | "down") => {
        setFeedback(type);
        onFeedback?.(message.id, type);
      },
      [message.id, onFeedback]
    );

    const handleEditSave = useCallback(() => {
      if (editContent.trim() && editContent !== message.content) {
        onEdit?.(message.id, editContent.trim());
      }
      setIsEditing(false);
    }, [editContent, message.content, message.id, onEdit]);

    const handleEditCancel = useCallback(() => {
      setEditContent(message.content);
      setIsEditing(false);
    }, [message.content]);

    const handleEditKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleEditSave();
        }
        if (e.key === "Escape") {
          handleEditCancel();
        }
      },
      [handleEditSave, handleEditCancel]
    );

    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        className={cn("group relative flex w-full gap-3 px-2 py-3", isUser ? "flex-row-reverse" : "flex-row")}
        aria-label={`${message.role} message`}
      >
        <div
          className={cn(
            "flex h-7 w-7 shrink-0 items-center justify-center rounded-full ring-1 select-none",
            isUser
              ? "bg-primary text-primary-foreground ring-primary/30"
              : "bg-primary/15 text-primary ring-primary/20"
          )}
        >
          {isUser ? <User className="h-3.5 w-3.5" /> : <Sparkles className="h-3.5 w-3.5" />}
        </div>

        <div className={cn("flex max-w-[80%] flex-col gap-1.5", isUser ? "items-end" : "items-start")}>
          <div
            className={cn(
              "relative rounded-2xl px-4 py-3 text-sm shadow-sm transition-all duration-200",
              isUser
                ? "rounded-tr-sm bg-primary text-primary-foreground shadow-primary/10"
                : "rounded-tl-sm bg-muted/60 backdrop-blur-sm border border-border/40 text-foreground",
              isError && "border-destructive/50 bg-destructive/10"
            )}
          >
            {isError ? (
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span className="text-sm">{message.content}</span>
              </div>
            ) : isEditing ? (
              <div className="w-full">
                <textarea
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  onKeyDown={handleEditKeyDown}
                  className="w-full bg-transparent border border-primary/30 rounded-lg p-2 text-sm resize-none outline-none focus:border-primary/60"
                  rows={3}
                  autoFocus
                />
                <div className="flex items-center gap-2 mt-2">
                  <button
                    onClick={handleEditSave}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium bg-primary text-primary-foreground hover:bg-primary/90"
                  >
                    <Save className="h-3 w-3" /> Save
                  </button>
                  <button
                    onClick={handleEditCancel}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium bg-muted hover:bg-muted/80"
                  >
                    <X className="h-3 w-3" /> Cancel
                  </button>
                  <span className="text-[10px] text-muted-foreground ml-2">Enter to save, Esc to cancel</span>
                </div>
              </div>
            ) : isUser ? (
              <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
            ) : (
              <MarkdownRenderer content={message.content} isStreaming={isStreaming} />
            )}

            {isStreaming && (
              <motion.span
                className="ml-0.5 inline-block h-4 w-0.5 rounded-sm bg-current align-middle"
                animate={{ opacity: [1, 0] }}
                transition={{ duration: 0.7, repeat: Infinity, repeatType: "reverse" }}
              />
            )}
          </div>

          {!isUser && !!message.metadata?.reasoning && (
            <div className="w-full max-w-150">
              <button
                onClick={() => setShowReasoning((prev) => !prev)}
                className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] text-amber-400/70 hover:text-amber-300 hover:bg-amber-500/5 transition-colors"
              >
                {showReasoning ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <Brain className="h-3 w-3" />
                Reasoning steps
              </button>
              <AnimatePresence>
                {showReasoning && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-1 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300/80 leading-relaxed">
                      {String(message.metadata?.reasoning ?? "")}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {!isUser && Array.isArray(message.metadata?.citations) && (message.metadata.citations as unknown[]).length > 0 && (
            <div className="w-full max-w-150">
              <button
                onClick={() => setShowCitations((prev) => !prev)}
                className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] text-blue-400/70 hover:text-blue-300 hover:bg-blue-500/5 transition-colors"
              >
                {showCitations ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <BookOpen className="h-3 w-3" />
                {(message.metadata.citations as unknown[]).length} source{(message.metadata.citations as unknown[]).length !== 1 ? "s" : ""}
              </button>
              <AnimatePresence>
                {showCitations && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-1 space-y-1.5">
                      {(message.metadata.citations as Array<{ title?: string; source?: string; text?: string }>).map((citation, i) => (
                        <div key={i} className="flex items-start gap-2 rounded-lg border border-blue-500/20 bg-blue-500/5 p-2.5 text-xs">
                          <FileText className="mt-0.5 h-3 w-3 shrink-0 text-blue-400" />
                          <div>
                            <p className="font-medium text-blue-300/90">{citation.title ?? `Source ${i + 1}`}</p>
                            {citation.source && <p className="text-[10px] text-blue-400/60">{citation.source}</p>}
                            {citation.text && <p className="mt-0.5 text-blue-300/70 line-clamp-2">{citation.text}</p>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          <div className={cn("flex items-center gap-2 px-1", isUser ? "flex-row-reverse" : "flex-row")}>
            <span className="text-[10px] text-muted-foreground/60 select-none">
              {new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
            {message.tokens !== undefined && (
              <span className="text-[10px] text-muted-foreground/40 select-none">{message.tokens} tokens</span>
            )}
          </div>

          {!isUser && !isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={cn(
                "flex items-center gap-0.5 opacity-0 transition-opacity duration-150 group-hover:opacity-100",
                isLast && "opacity-100"
              )}
            >
              <button
                onClick={handleCopy}
                className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                aria-label="Copy message"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
              {onRegenerate && (
                <button
                  onClick={() => onRegenerate(message.id)}
                  className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                  aria-label="Regenerate response"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              )}
              {onBranch && (
                <button
                  onClick={() => onBranch(message.id)}
                  className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                  aria-label="Create branch"
                  title="Create conversation branch from here"
                >
                  <GitBranch className="h-3.5 w-3.5" />
                </button>
              )}
              <button
                onClick={() => handleFeedback("up")}
                className={cn("flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground", feedback === "up" && "text-emerald-500")}
                aria-label="Thumbs up"
                disabled={feedback !== null}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => handleFeedback("down")}
                className={cn("flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground", feedback === "down" && "text-rose-500")}
                aria-label="Thumbs down"
                disabled={feedback !== null}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </button>
            </motion.div>
          )}

          {isUser && !isEditing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-0.5 opacity-0 transition-opacity duration-150 group-hover:opacity-100"
            >
              <button
                onClick={() => {
                  setEditContent(message.content);
                  setIsEditing(true);
                }}
                className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                aria-label="Edit message"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
              {onBranch && (
                <button
                  onClick={() => onBranch(message.id)}
                  className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                  aria-label="Create branch"
                  title="Create conversation branch from here"
                >
                  <GitBranch className="h-3.5 w-3.5" />
                </button>
              )}
            </motion.div>
          )}
        </div>
      </motion.div>
    );
  }
);

Message.displayName = "Message";
