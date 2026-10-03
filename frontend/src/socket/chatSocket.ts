// frontend/src/socket/chatSocket.ts

import { SocketManager, type SocketManagerConfig } from "./socket";

// ─── Event payload types ──────────────────────────────────────────────────────

export interface ChatTokenChunk {
  conversation_id: string;
  message_id: string;
  token: string;
  index: number;
  is_final: boolean;
}

export interface ChatStreamStart {
  conversation_id: string;
  message_id: string;
  model: string;
  provider: string;
  started_at: number;
}

export interface ChatStreamEnd {
  conversation_id: string;
  message_id: string;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | "error";
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  duration_ms: number;
  cost_usd?: number;
}

export interface ChatStreamError {
  conversation_id: string;
  message_id: string;
  code: string;
  message: string;
  recoverable: boolean;
  timestamp: number;
}

export interface ChatToolCallStart {
  conversation_id: string;
  message_id: string;
  tool_call_id: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
}

export interface ChatToolCallResult {
  conversation_id: string;
  message_id: string;
  tool_call_id: string;
  tool_name: string;
  result: unknown;
  is_error: boolean;
  duration_ms: number;
}

export interface ChatThinkingChunk {
  conversation_id: string;
  message_id: string;
  thinking: string;
  index: number;
}

export interface ChatSendMessage {
  conversation_id: string;
  content: string;
  model: string;
  provider: string;
  parent_message_id?: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  tools?: ChatTool[];
  stream: boolean;
}

export interface ChatTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ChatCancelStream {
  conversation_id: string;
  message_id: string;
}

export interface ChatRegenerateMessage {
  conversation_id: string;
  message_id: string;
  model?: string;
  provider?: string;
}

// ─── Channel names ────────────────────────────────────────────────────────────

export const CHAT_CHANNELS = {
  stream: (conversation_id: string) => `chat:stream:${conversation_id}`,
  conversation: (conversation_id: string) =>
    `chat:conversation:${conversation_id}`,
  tools: (conversation_id: string) => `chat:tools:${conversation_id}`,
} as const;

// ─── Listener types ───────────────────────────────────────────────────────────

export interface ChatSocketListeners {
  onTokenChunk?: (chunk: ChatTokenChunk) => void;
  onStreamStart?: (event: ChatStreamStart) => void;
  onStreamEnd?: (event: ChatStreamEnd) => void;
  onStreamError?: (error: ChatStreamError) => void;
  onToolCallStart?: (event: ChatToolCallStart) => void;
  onToolCallResult?: (event: ChatToolCallResult) => void;
  onThinkingChunk?: (chunk: ChatThinkingChunk) => void;
}

// ─── Chat Socket ──────────────────────────────────────────────────────────────

export class ChatSocket {
  private readonly manager: SocketManager;
  private readonly activeSubscriptions = new Map<string, () => void>();
  private readonly unsubscribeFns: Array<() => void> = [];

  constructor(config: SocketManagerConfig) {
    this.manager = new SocketManager(config);
  }

  getManager(): SocketManager {
    return this.manager;
  }

  // ── Send actions ───────────────────────────────────────────────────────────

  async sendMessage(payload: ChatSendMessage): Promise<void> {
    await this.manager.send("event", payload, {
      channel: CHAT_CHANNELS.stream(payload.conversation_id),
      waitForAck: true,
      timeoutMs: 10_000,
    });
  }

  async cancelStream(payload: ChatCancelStream): Promise<void> {
    await this.manager.send("event", payload, {
      channel: CHAT_CHANNELS.stream(payload.conversation_id),
    });
  }

  async regenerateMessage(payload: ChatRegenerateMessage): Promise<void> {
    await this.manager.send("event", payload, {
      channel: CHAT_CHANNELS.stream(payload.conversation_id),
    });
  }

  // ── Stream subscriptions ───────────────────────────────────────────────────

  subscribeToStream(
    conversation_id: string,
    listeners: ChatSocketListeners
  ): () => void {
    const channel = CHAT_CHANNELS.stream(conversation_id);
    const existingKey = `stream:${conversation_id}`;

    // Unsubscribe previous if exists
    this.activeSubscriptions.get(existingKey)?.();

    const unsubscribe = this.manager.subscribe<
      | ChatTokenChunk
      | ChatStreamStart
      | ChatStreamEnd
      | ChatStreamError
      | ChatToolCallStart
      | ChatToolCallResult
      | ChatThinkingChunk
    >(channel, (payload, frame) => {
      if (!frame.id) return;

      // Route by frame metadata
      switch (frame.id.split(":")[0]) {
        case "token":
          listeners.onTokenChunk?.(payload as ChatTokenChunk);
          break;
        case "stream_start":
          listeners.onStreamStart?.(payload as ChatStreamStart);
          break;
        case "stream_end":
          listeners.onStreamEnd?.(payload as ChatStreamEnd);
          break;
        case "stream_error":
          listeners.onStreamError?.(payload as ChatStreamError);
          break;
        case "tool_start":
          listeners.onToolCallStart?.(payload as ChatToolCallStart);
          break;
        case "tool_result":
          listeners.onToolCallResult?.(payload as ChatToolCallResult);
          break;
        case "thinking":
          listeners.onThinkingChunk?.(payload as ChatThinkingChunk);
          break;
        default: {
          // Route by shape inspection
          const p = payload as unknown as Record<string, unknown>;
          if ("token" in p) {
            listeners.onTokenChunk?.(payload as ChatTokenChunk);
          } else if ("finish_reason" in p) {
            listeners.onStreamEnd?.(payload as ChatStreamEnd);
          } else if ("prompt_tokens" in p) {
            listeners.onStreamStart?.(payload as ChatStreamStart);
          } else if ("tool_call_id" in p && "result" in p) {
            listeners.onToolCallResult?.(payload as ChatToolCallResult);
          } else if ("tool_call_id" in p) {
            listeners.onToolCallStart?.(payload as ChatToolCallStart);
          } else if ("thinking" in p) {
            listeners.onThinkingChunk?.(payload as ChatThinkingChunk);
          }
          break;
        }
      }
    });

    this.activeSubscriptions.set(existingKey, unsubscribe);
    return () => {
      unsubscribe();
      this.activeSubscriptions.delete(existingKey);
    };
  }

  subscribeToTokenChunks(
    conversation_id: string,
    onChunk: (chunk: ChatTokenChunk) => void
  ): () => void {
    return this.manager.subscribe<ChatTokenChunk>(
      CHAT_CHANNELS.stream(conversation_id),
      (payload) => {
        if ("token" in (payload as object)) {
          onChunk(payload);
        }
      }
    );
  }

  subscribeToStreamEnd(
    conversation_id: string,
    onEnd: (event: ChatStreamEnd) => void
  ): () => void {
    return this.manager.subscribe<ChatStreamEnd>(
      CHAT_CHANNELS.stream(conversation_id),
      (payload) => {
        if ("finish_reason" in (payload as object)) {
          onEnd(payload);
        }
      }
    );
  }

  subscribeToStreamErrors(
    conversation_id: string,
    onError: (error: ChatStreamError) => void
  ): () => void {
    return this.manager.subscribe<ChatStreamError>(
      CHAT_CHANNELS.stream(conversation_id),
      (payload) => {
        if ("recoverable" in (payload as object)) {
          onError(payload);
        }
      }
    );
  }

  unsubscribeFromStream(conversation_id: string): void {
    const key = `stream:${conversation_id}`;
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.delete(key);
  }

  unsubscribeAll(): void {
    this.activeSubscriptions.forEach((fn) => fn());
    this.activeSubscriptions.clear();
    this.unsubscribeFns.forEach((fn) => fn());
    this.unsubscribeFns.length = 0;
  }

  destroy(): void {
    this.unsubscribeAll();
    this.manager.destroy();
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export const createChatSocket = (
  config: SocketManagerConfig
): ChatSocket => new ChatSocket(config);