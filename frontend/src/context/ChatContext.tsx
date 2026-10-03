// frontend/src/context/ChatContext.tsx
/* eslint-disable react-refresh/only-export-components */

"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type {
  Conversation,
  Message,
  TokenUsage,
} from "../types/chat";
import { MessageRole, MessageStatus } from "../types/chat";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ChatContextError = {
  code: string;
  message: string;
  conversation_id?: string;
  message_id?: string;
  timestamp: number;
  recoverable: boolean;
};

export type TypingIndicatorState = {
  is_typing: boolean;
  started_at: number | null;
};

export type ConversationContextMode =
  | "chat"
  | "agent"
  | "document"
  | "voice"
  | "playground";

export type ScrollBehavior = "auto" | "smooth" | "instant";

export interface MessageDraft {
  content: string;
  conversation_id: string;
  updated_at: number;
}

export interface ConversationSession {
  conversation: Conversation | null;
  is_loading_messages: boolean;
  is_at_context_limit: boolean;
  context_usage_percent: number;
}

export interface ChatContextValue {
  // Session
  mode: ConversationContextMode;
  setMode: (mode: ConversationContextMode) => void;

  // Active conversation
  session: ConversationSession;
  setConversation: (conversation: Conversation | null) => void;
  clearSession: () => void;

  // Messages
  messages: Message[];
  appendMessage: (message: Message) => void;
  updateMessage: (id: string, patch: Partial<Message>) => void;
  removeMessage: (id: string) => void;
  clearMessages: () => void;
  appendStreamToken: (message_id: string, token: string) => void;
  finalizeMessage: (
    message_id: string,
    patch: Partial<Message>
  ) => void;

  // Draft
  draft: string;
  setDraft: (content: string) => void;
  clearDraft: () => void;
  saveDraftForConversation: (conversation_id: string) => void;
  loadDraftForConversation: (conversation_id: string) => void;

  // Streaming
  is_streaming: boolean;
  streaming_message_id: string | null;
  startStreaming: (message_id: string) => AbortController;
  stopStreaming: () => void;
  abort_controller_ref: React.MutableRefObject<AbortController | null>;

  // Typing indicator
  typing: TypingIndicatorState;
  setTyping: (is_typing: boolean) => void;

  // Scroll
  scroll_anchor_ref: React.RefObject<HTMLDivElement | null>;
  should_auto_scroll: boolean;
  setShouldAutoScroll: (value: boolean) => void;
  scrollToBottom: (behavior?: ScrollBehavior) => void;

  // Error
  error: ChatContextError | null;
  setError: (error: ChatContextError | null) => void;
  clearError: () => void;

  // Token tracking
  current_token_usage: TokenUsage | null;
  setTokenUsage: (usage: TokenUsage | null) => void;
  accumulated_cost_usd: number;
  addCost: (cost_usd: number) => void;

  // Message helpers
  getMessageById: (id: string) => Message | undefined;
  getLastMessage: () => Message | undefined;
  getLastAssistantMessage: () => Message | undefined;
  getMessageCount: () => number;
  isMessageStreaming: (id: string) => boolean;

  // Edit & Branch
  editMessage: (id: string, newContent: string) => void;
  createBranch: (fromMessageId: string) => string;
  activeBranchId: string | null;
  branches: Record<string, { id: string; name: string; parentMessageId: string }>;
  switchBranch: (branchId: string | null) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const computeContextUsage = (
  conversation: Conversation | null
): { is_at_limit: boolean; percent: number } => {
  if (!conversation) return { is_at_limit: false, percent: 0 };
  const used = conversation.context_window_used ?? 0;
  const total = conversation.context_window_total ?? 0;
  if (total === 0) return { is_at_limit: false, percent: 0 };
  const percent = Math.min(100, Math.round((used / total) * 100));
  return { is_at_limit: percent >= 95, percent };
};

// ─── Default value ────────────────────────────────────────────────────────────

const DEFAULT_SESSION: ConversationSession = {
  conversation: null,
  is_loading_messages: false,
  is_at_context_limit: false,
  context_usage_percent: 0,
};

// ─── Context ─────────────────────────────────────────────────────────────────

export const ChatContext = createContext<ChatContextValue | null>(null);
ChatContext.displayName = "ChatContext";

// ─── Draft storage key ────────────────────────────────────────────────────────

const DRAFT_STORAGE_PREFIX = "ai_os_draft:";

const getDraftKey = (conversation_id: string): string =>
  `${DRAFT_STORAGE_PREFIX}${conversation_id}`;

// ─── Props ────────────────────────────────────────────────────────────────────

export interface ChatProviderProps {
  children: ReactNode;
  initial_mode?: ConversationContextMode;
  initial_conversation?: Conversation | null;
  max_message_history?: number;
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const ChatProvider = ({
  children,
  initial_mode = "chat",
  initial_conversation = null,
  max_message_history = 1000,
}: ChatProviderProps): React.JSX.Element => {
  const [mode, setMode] = useState<ConversationContextMode>(initial_mode);
  const [session, setSession] = useState<ConversationSession>({
    ...DEFAULT_SESSION,
    conversation: initial_conversation,
  });
  const [messages, setMessages] = useState<Message[]>(
    initial_conversation?.messages ?? []
  );
  const [draft, setDraftState] = useState<string>("");
  const [is_streaming, setIsStreaming] = useState(false);
  const [streaming_message_id, setStreamingMessageId] = useState<
    string | null
  >(null);
  const [typing, setTypingState] = useState<TypingIndicatorState>({
    is_typing: false,
    started_at: null,
  });
  const [should_auto_scroll, setShouldAutoScroll] = useState(true);
  const [error, setErrorState] = useState<ChatContextError | null>(null);
  const [current_token_usage, setCurrentTokenUsage] =
    useState<TokenUsage | null>(null);
  const [accumulated_cost_usd, setAccumulatedCost] = useState(0);
  const [activeBranchId, setActiveBranchId] = useState<string | null>(null);
  const [branches, setBranches] = useState<Record<string, { id: string; name: string; parentMessageId: string }>>({});

  const abort_controller_ref = useRef<AbortController | null>(null);
  const scroll_anchor_ref = useRef<HTMLDivElement | null>(null);
  const typing_timer_ref = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Session management ────────────────────────────────────────────────────

  const setConversation = useCallback(
    (conversation: Conversation | null) => {
      const { is_at_limit, percent } = computeContextUsage(conversation);
      setSession({
        conversation,
        is_loading_messages: false,
        is_at_context_limit: is_at_limit,
        context_usage_percent: percent,
      });
      setMessages(conversation?.messages ?? []);
      setErrorState(null);
      setIsStreaming(false);
      setStreamingMessageId(null);
      setCurrentTokenUsage(null);
    },
    []
  );

  const clearSession = useCallback(() => {
    abort_controller_ref.current?.abort();
    abort_controller_ref.current = null;
    setSession(DEFAULT_SESSION);
    setMessages([]);
    setDraftState("");
    setIsStreaming(false);
    setStreamingMessageId(null);
    setErrorState(null);
    setCurrentTokenUsage(null);
    setAccumulatedCost(0);
  }, []);

  // ── Message management ────────────────────────────────────────────────────

  const appendMessage = useCallback(
    (message: Message) => {
      setMessages((prev) => {
        const next = [...prev, message];
        return next.length > max_message_history
          ? next.slice(next.length - max_message_history)
          : next;
      });
    },
    [max_message_history]
  );

  const updateMessage = useCallback(
    (id: string, patch: Partial<Message>) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? { ...m, ...patch, updated_at: Date.now() }
            : m
        )
      );
    },
    []
  );

  const removeMessage = useCallback((id: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== id));
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const appendStreamToken = useCallback(
    (message_id: string, token: string) => {
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== message_id) return m;
          const content =
            typeof m.content === "string" ? m.content + token : token;
          return { ...m, content, updated_at: Date.now() };
        })
      );
    },
    []
  );

  const finalizeMessage = useCallback(
    (message_id: string, patch: Partial<Message>) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === message_id
            ? {
                ...m,
                ...patch,
                status: patch.status ?? MessageStatus.Complete,
                updated_at: Date.now(),
              }
            : m
        )
      );
      setIsStreaming(false);
      setStreamingMessageId(null);
    },
    []
  );

  // ── Draft management ──────────────────────────────────────────────────────

  const setDraft = useCallback((content: string) => {
    setDraftState(content);
  }, []);

  const clearDraft = useCallback(() => {
    setDraftState("");
  }, []);

  const saveDraftForConversation = useCallback(
    (conversation_id: string) => {
      try {
        if (draft.trim()) {
          sessionStorage.setItem(
            getDraftKey(conversation_id),
            JSON.stringify({
              content: draft,
              conversation_id,
              updated_at: Date.now(),
            })
          );
        } else {
          sessionStorage.removeItem(getDraftKey(conversation_id));
        }
      } catch {
        // Storage unavailable
      }
    },
    [draft]
  );

  const loadDraftForConversation = useCallback(
    (conversation_id: string) => {
      try {
        const raw = sessionStorage.getItem(
          getDraftKey(conversation_id)
        );
        if (!raw) {
          setDraftState("");
          return;
        }
        const parsed = JSON.parse(raw) as MessageDraft;
        setDraftState(parsed.content ?? "");
      } catch {
        setDraftState("");
      }
    },
    []
  );

  // ── Streaming ─────────────────────────────────────────────────────────────

  const startStreaming = useCallback(
    (message_id: string): AbortController => {
      abort_controller_ref.current?.abort();
      const controller = new AbortController();
      abort_controller_ref.current = controller;
      setIsStreaming(true);
      setStreamingMessageId(message_id);
      return controller;
    },
    []
  );

  const stopStreaming = useCallback(() => {
    abort_controller_ref.current?.abort();
    abort_controller_ref.current = null;
    setIsStreaming(false);
    setStreamingMessageId(null);
  }, []);

  // ── Typing indicator ──────────────────────────────────────────────────────

  const setTyping = useCallback((is_typing: boolean) => {
    if (typing_timer_ref.current) {
      clearTimeout(typing_timer_ref.current);
    }
    if (is_typing) {
      setTypingState({ is_typing: true, started_at: Date.now() });
      typing_timer_ref.current = setTimeout(() => {
        setTypingState({ is_typing: false, started_at: null });
      }, 3_000);
    } else {
      setTypingState({ is_typing: false, started_at: null });
    }
  }, []);

  // ── Scroll ────────────────────────────────────────────────────────────────

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = "smooth") => {
      scroll_anchor_ref.current?.scrollIntoView({ behavior });
    },
    []
  );

  useEffect(() => {
    if (should_auto_scroll && is_streaming) {
      scrollToBottom("instant");
    }
  }, [messages, should_auto_scroll, is_streaming, scrollToBottom]);

  // ── Error ─────────────────────────────────────────────────────────────────

  const setError = useCallback((err: ChatContextError | null) => {
    setErrorState(err);
  }, []);

  const clearError = useCallback(() => setErrorState(null), []);

  // ── Token usage ───────────────────────────────────────────────────────────

  const setTokenUsage = useCallback((usage: TokenUsage | null) => {
    setCurrentTokenUsage(usage);
  }, []);

  const addCost = useCallback((cost_usd: number) => {
    setAccumulatedCost((prev) => prev + cost_usd);
  }, []);

  // ── Edit & Branch ────────────────────────────────────────────────────────

  const editMessage = useCallback(
    (id: string, newContent: string) => {
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === id);
        if (idx === -1) return prev;
        const msg = prev[idx];
        if (msg.role !== "user") return prev;

        // If not on a branch, create a new one
        if (!activeBranchId) {
          const branchId = `branch-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
          const branchName = `Edit branch`;
          const newBranch = { id: branchId, name: branchName, parentMessageId: id };

          setBranches((b) => ({ ...b, [branchId]: newBranch }));
          setActiveBranchId(branchId);

          // Clone messages from this point into the branch
          const branched = prev.slice(idx).map((m, i) => ({
            ...m,
            id: i === 0 ? `edit-${Date.now().toString(36)}` : m.id,
            content: i === 0 ? newContent : m.content,
            branchId,
            updated_at: Date.now(),
          }));

          return [...prev.slice(0, idx), ...branched];
        }

        // Already on a branch — just update
        return prev.map((m) =>
          m.id === id ? { ...m, content: newContent, updated_at: Date.now() } : m
        );
      });
    },
    [activeBranchId]
  );

  const createBranch = useCallback(
    (fromMessageId: string): string => {
      const branchId = `branch-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
      const branchName = `Branch ${Object.keys(branches).length + 1}`;
      const newBranch = { id: branchId, name: branchName, parentMessageId: fromMessageId };

      setBranches((prev) => ({ ...prev, [branchId]: newBranch }));
      setActiveBranchId(branchId);
      return branchId;
    },
    [branches]
  );

  const switchBranch = useCallback((branchId: string | null) => {
    setActiveBranchId(branchId);
  }, []);

  // ── Derived helpers ───────────────────────────────────────────────────────

  const getMessageById = useCallback(
    (id: string): Message | undefined =>
      messages.find((m) => m.id === id),
    [messages]
  );

  const getLastMessage = useCallback(
    (): Message | undefined => messages[messages.length - 1],
    [messages]
  );

  const getLastAssistantMessage = useCallback(
    (): Message | undefined =>
      [...messages]
        .reverse()
        .find((m) => m.role === MessageRole.Assistant),
    [messages]
  );

  const getMessageCount = useCallback(
    (): number => messages.length,
    [messages]
  );

  const isMessageStreaming = useCallback(
    (id: string): boolean =>
      is_streaming && streaming_message_id === id,
    [is_streaming, streaming_message_id]
  );

  // ── Cleanup ───────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      abort_controller_ref.current?.abort();
      if (typing_timer_ref.current) {
        clearTimeout(typing_timer_ref.current);
      }
    };
  }, []);

  // ── Context value ─────────────────────────────────────────────────────────

  const value = useMemo<ChatContextValue>(
    () => ({
      mode,
      setMode,
      session,
      setConversation,
      clearSession,
      messages,
      appendMessage,
      updateMessage,
      removeMessage,
      clearMessages,
      appendStreamToken,
      finalizeMessage,
      draft,
      setDraft,
      clearDraft,
      saveDraftForConversation,
      loadDraftForConversation,
      is_streaming,
      streaming_message_id,
      startStreaming,
      stopStreaming,
      abort_controller_ref,
      typing,
      setTyping,
      scroll_anchor_ref,
      should_auto_scroll,
      setShouldAutoScroll,
      scrollToBottom,
      error,
      setError,
      clearError,
      current_token_usage,
      setTokenUsage,
      accumulated_cost_usd,
      addCost,
      getMessageById,
      getLastMessage,
      getLastAssistantMessage,
      getMessageCount,
      isMessageStreaming,
      editMessage,
      createBranch,
      activeBranchId,
      branches,
      switchBranch,
    }),
    [
      mode,
      session,
      messages,
      draft,
      is_streaming,
      streaming_message_id,
      typing,
      should_auto_scroll,
      error,
      current_token_usage,
      accumulated_cost_usd,
      setMode,
      setConversation,
      clearSession,
      appendMessage,
      updateMessage,
      removeMessage,
      clearMessages,
      appendStreamToken,
      finalizeMessage,
      setDraft,
      clearDraft,
      saveDraftForConversation,
      loadDraftForConversation,
      startStreaming,
      stopStreaming,
      setTyping,
      setShouldAutoScroll,
      scrollToBottom,
      setError,
      clearError,
      setTokenUsage,
      addCost,
      getMessageById,
      getLastMessage,
      getLastAssistantMessage,
      getMessageCount,
      isMessageStreaming,
      editMessage,
      createBranch,
      activeBranchId,
      branches,
      switchBranch,
    ]
  );

  return (
    <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
  );
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export const useChatContext = (): ChatContextValue => {
  const context = useContext(ChatContext);
  if (context === null) {
    throw new Error(
      "[useChatContext] must be used within a <ChatProvider>. " +
        "Ensure ChatProvider wraps your component tree."
    );
  }
  return context;
};