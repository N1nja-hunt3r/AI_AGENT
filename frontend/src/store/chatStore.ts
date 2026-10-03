import { create } from "zustand";
import { devtools, persist, subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ─── Types ────────────────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant" | "system" | "tool";

export type MessageStatus = "pending" | "streaming" | "complete" | "error";

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: "low" | "high" | "auto" } }
  | { type: "tool_use"; tool_use_id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export interface Message {
  id: string;
  role: MessageRole;
  content: string | ContentPart[];
  status: MessageStatus;
  model?: string;
  tokens?: {
    prompt: number;
    completion: number;
    total: number;
  };
  metadata?: Record<string, unknown>;
  branchId?: string;
  parentMessageId?: string;
  createdAt: number;
  updatedAt: number;
}

export interface ConversationBranch {
  id: string;
  name: string;
  parentMessageId: string;
  conversationId: string;
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  systemPrompt?: string;
  messages: Message[];
  branches: Record<string, ConversationBranch>;
  activeBranchId?: string;
  metadata?: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

export interface StreamingState {
  isStreaming: boolean;
  messageId: string | null;
  abortController: AbortController | null;
  buffer: string;
}

export interface ModelConfig {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsTools: boolean;
  inputCostPer1kTokens: number;
  outputCostPer1kTokens: number;
}

export interface ChatState {
  // Conversations
  conversations: Record<string, Conversation>;
  currentConversationId: string | null;

  // Model selection
  selectedModel: string;
  availableModels: ModelConfig[];

  // Streaming
  streaming: StreamingState;

  // Error
  error: string | null;
}

export interface ChatActions {
  // Conversation management
  createConversation: (options?: Partial<Pick<Conversation, "title" | "model" | "systemPrompt">>) => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  updateConversation: (id: string, patch: Partial<Pick<Conversation, "title" | "systemPrompt" | "model">>) => void;
  clearAllConversations: () => void;

  // Message management
  addMessage: (conversationId: string, message: Omit<Message, "id" | "createdAt" | "updatedAt">) => string;
  updateMessage: (conversationId: string, messageId: string, patch: Partial<Omit<Message, "id" | "createdAt">>) => void;
  deleteMessage: (conversationId: string, messageId: string) => void;
  appendToMessage: (conversationId: string, messageId: string, chunk: string) => void;

  // Model selection
  setSelectedModel: (modelId: string) => void;
  setAvailableModels: (models: ModelConfig[]) => void;

  // Branch management
  createBranch: (conversationId: string, fromMessageId: string, name?: string) => string;
  switchBranch: (conversationId: string, branchId: string) => void;
  deleteBranch: (conversationId: string, branchId: string) => void;
  getBranchMessages: (conversationId: string, branchId: string) => Message[];

  // Edit message (creates branch if not on main)
  editMessage: (conversationId: string, messageId: string, newContent: string) => void;

  // Streaming control
  startStreaming: (messageId: string) => void;
  stopStreaming: () => void;
  appendStreamChunk: (chunk: string) => void;
  setStreamBuffer: (buffer: string) => void;

  // Error handling
  setError: (error: string | null) => void;

  // Derived helpers
  getCurrentConversation: () => Conversation | null;
  getMessagesByConversation: (conversationId: string) => Message[];
}

export type ChatStore = ChatState & ChatActions;

// ─── Helpers ─────────────────────────────────────────────────────────────────

const generateId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

const now = (): number => Date.now();

const DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro";

const DEFAULT_STREAMING_STATE: StreamingState = {
  isStreaming: false,
  messageId: null,
  abortController: null,
  buffer: "",
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          // ── Initial state ──────────────────────────────────────────────────
          conversations: {},
          currentConversationId: null,
          selectedModel: DEFAULT_MODEL,
          availableModels: [],
          streaming: DEFAULT_STREAMING_STATE,
          error: null,

          // ── Conversation management ────────────────────────────────────────
          createConversation: (options = {}) => {
            const id = generateId();
            const timestamp = now();
            const conversation: Conversation = {
              id,
              title: options.title ?? "New Conversation",
              model: options.model ?? get().selectedModel,
              systemPrompt: options.systemPrompt,
              messages: [],
              branches: {},
              createdAt: timestamp,
              updatedAt: timestamp,
            };

            set((state) => {
              state.conversations[id] = conversation;
              state.currentConversationId = id;
            });

            return id;
          },

          selectConversation: (id) => {
            set((state) => {
              if (state.conversations[id]) {
                state.currentConversationId = id;
                state.error = null;
              }
            });
          },

          deleteConversation: (id) => {
            set((state) => {
              delete state.conversations[id];
              if (state.currentConversationId === id) {
                const remaining = Object.keys(state.conversations);
                state.currentConversationId = remaining[remaining.length - 1] ?? null;
              }
            });
          },

          updateConversation: (id, patch) => {
            set((state) => {
              const conversation = state.conversations[id];
              if (conversation) {
                Object.assign(conversation, patch, { updatedAt: now() });
              }
            });
          },

          clearAllConversations: () => {
            set((state) => {
              state.conversations = {};
              state.currentConversationId = null;
              state.streaming = DEFAULT_STREAMING_STATE;
              state.error = null;
            });
          },

          // ── Message management ─────────────────────────────────────────────
          addMessage: (conversationId, message) => {
            const id = generateId();
            const timestamp = now();
            const conversation = get().conversations[conversationId];
            const branchId = conversation?.activeBranchId;
            const newMessage: Message = {
              ...message,
              id,
              branchId,
              createdAt: timestamp,
              updatedAt: timestamp,
            };

            set((state) => {
              const conversation = state.conversations[conversationId];
              if (conversation) {
                conversation.messages.push(newMessage);
                conversation.updatedAt = timestamp;
              }
            });

            return id;
          },

          updateMessage: (conversationId, messageId, patch) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;

              const message = conversation.messages.find((m) => m.id === messageId);
              if (message) {
                Object.assign(message, patch, { updatedAt: now() });
              }
            });
          },

          deleteMessage: (conversationId, messageId) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (conversation) {
                conversation.messages = conversation.messages.filter(
                  (m) => m.id !== messageId
                );
                conversation.updatedAt = now();
              }
            });
          },

          appendToMessage: (conversationId, messageId, chunk) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;

              const message = conversation.messages.find((m) => m.id === messageId);
              if (message) {
                if (typeof message.content === "string") {
                  message.content += chunk;
                } else {
                  const lastPart = message.content[message.content.length - 1];
                  if (lastPart?.type === "text") {
                    lastPart.text += chunk;
                  } else {
                    message.content.push({ type: "text", text: chunk });
                  }
                }
                message.updatedAt = now();
              }
            });
          },

          // ── Model selection ────────────────────────────────────────────────
          setSelectedModel: (modelId) => {
            set((state) => {
              state.selectedModel = modelId;
            });
          },

          setAvailableModels: (models) => {
            set((state) => {
              state.availableModels = models;
            });
          },

          // ── Branch management ───────────────────────────────────────────
          createBranch: (conversationId, fromMessageId, name) => {
            const branchId = `branch-${generateId()}`;
            const timestamp = now();
            const branchName = name ?? `Branch ${Object.keys(get().conversations[conversationId]?.branches ?? {}).length + 1}`;

            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;

              conversation.branches[branchId] = {
                id: branchId,
                name: branchName,
                parentMessageId: fromMessageId,
                conversationId,
                createdAt: timestamp,
              };
              conversation.activeBranchId = branchId;
              conversation.updatedAt = timestamp;
            });

            return branchId;
          },

          switchBranch: (conversationId, branchId) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;
              conversation.activeBranchId = branchId || undefined;
              conversation.updatedAt = now();
            });
          },

          deleteBranch: (conversationId, branchId) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;
              delete conversation.branches[branchId];
              if (conversation.activeBranchId === branchId) {
                conversation.activeBranchId = undefined;
              }
              conversation.updatedAt = now();
            });
          },

          getBranchMessages: (conversationId, branchId) => {
            const conversation = get().conversations[conversationId];
            if (!conversation) return [];
            const branch = conversation.branches[branchId];
            if (!branch) return [];
            return conversation.messages.filter(
              (m) => m.branchId === branchId || m.id === branch.parentMessageId
            );
          },

          editMessage: (conversationId, messageId, newContent) => {
            set((state) => {
              const conversation = state.conversations[conversationId];
              if (!conversation) return;

              const messageIndex = conversation.messages.findIndex((m) => m.id === messageId);
              if (messageIndex === -1) return;

              const message = conversation.messages[messageIndex];
              if (message.role !== "user") return;

              // If on main branch (no activeBranchId), create a new branch
              if (!conversation.activeBranchId) {
                const branchId = `branch-${generateId()}`;
                const timestamp = now();
                conversation.branches[branchId] = {
                  id: branchId,
                  name: `Edit branch`,
                  parentMessageId: messageId,
                  conversationId,
                  createdAt: timestamp,
                };
                conversation.activeBranchId = branchId;

                // Clone messages from this point forward into the branch
                for (let i = messageIndex; i < conversation.messages.length; i++) {
                  const original = conversation.messages[i];
                  conversation.messages.push({
                    ...original,
                    id: i === messageIndex ? `edit-${generateId()}` : original.id,
                    content: i === messageIndex ? newContent : original.content,
                    branchId,
                    createdAt: timestamp,
                    updatedAt: timestamp,
                  });
                }
              } else {
                // Already on a branch — just update the message
                message.content = newContent;
                message.updatedAt = now();
              }

              conversation.updatedAt = now();
            });
          },

          // ── Streaming control ──────────────────────────────────────────────
          startStreaming: (messageId) => {
            const abortController = new AbortController();
            set((state) => {
              state.streaming = {
                isStreaming: true,
                messageId,
                abortController,
                buffer: "",
              };
            });
          },

          stopStreaming: () => {
            const { streaming } = get();
            streaming.abortController?.abort();
            set((state) => {
              state.streaming = DEFAULT_STREAMING_STATE;
            });
          },

          appendStreamChunk: (chunk) => {
            set((state) => {
              state.streaming.buffer += chunk;
            });
          },

          setStreamBuffer: (buffer) => {
            set((state) => {
              state.streaming.buffer = buffer;
            });
          },

          // ── Error handling ─────────────────────────────────────────────────
          setError: (error) => {
            set((state) => {
              state.error = error;
            });
          },

          // ── Derived helpers ────────────────────────────────────────────────
          getCurrentConversation: () => {
            const { conversations, currentConversationId } = get();
            return currentConversationId ? (conversations[currentConversationId] ?? null) : null;
          },

          getMessagesByConversation: (conversationId) => {
            return get().conversations[conversationId]?.messages ?? [];
          },
        }))
      ),
      {
        name: "chat-store",
        partialize: (state) => ({
          conversations: state.conversations,
          currentConversationId: state.currentConversationId,
          selectedModel: state.selectedModel,
        }),
      }
    ),
    { name: "ChatStore" }
  )
);