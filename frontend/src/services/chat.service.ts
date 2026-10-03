import api, { getAuthToken } from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const MessageRole = {
  User: "user",
  Assistant: "assistant",
  System: "system",
} as const;

export type MessageRole = (typeof MessageRole)[keyof typeof MessageRole];

export const MessageStatus = {
  Pending: "pending",
  Streaming: "streaming",
  Complete: "complete",
  Error: "error",
} as const;

export type MessageStatus = (typeof MessageStatus)[keyof typeof MessageStatus];

// ── Core types ────────────────────────────────────────────────────────────────
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  createdAt: string;
  tokens?: number;
  metadata?: Record<string, unknown>;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface SendMessagePayload {
  conversationId?: string;
  content: string;
  role?: MessageRole;
  systemPrompt?: string;
  metadata?: Record<string, unknown>;
}

export interface StreamMessagePayload extends SendMessagePayload {
  onChunk: (chunk: StreamChunk) => void;
  onComplete: (message: Message) => void;
  onError: (error: Error) => void;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface SendMessageResponse {
  message: Message;
  conversationId: string;
  usage: TokenUsage;
}

export interface StreamChunk {
  id: string;
  conversationId: string;
  delta: string;
  done: boolean;
  usage?: TokenUsage;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface GetConversationResponse {
  conversation: Conversation;
}

export interface ClearConversationResponse {
  success: boolean;
  conversationId: string;
  clearedAt: string;
}

// ── API endpoint constants ────────────────────────────────────────────────────
const ENDPOINTS = {
  SEND: "/api/v1/chat/message",
  STREAM: "/api/v1/chat/stream",
  CONVERSATION: (id: string) => `/api/v1/chat/conversations/${id}`,
  CLEAR: (id: string) => `/api/v1/chat/conversations/${id}/clear`,
} as const;

// ── Service functions ─────────────────────────────────────────────────────────

/**
 * Sends a single chat message and awaits the complete AI response.
 * Use this for non-streaming, request-response interactions.
 */
export const sendMessage = async (
  payload: SendMessagePayload
): Promise<SendMessageResponse> => {
  const response: AxiosResponse<SendMessageResponse> = await api.post<
    SendMessageResponse
  >(ENDPOINTS.SEND, {
    conversationId: payload.conversationId ?? null,
    content: payload.content,
    role: payload.role ?? MessageRole.User,
    systemPrompt: payload.systemPrompt ?? null,
    metadata: payload.metadata ?? {},
  });

  return response.data;
};

/**
 * Initiates a streaming chat message using the Fetch API with ReadableStream.
 * The Axios instance is intentionally bypassed here to support SSE / NDJSON
 * streaming which Axios does not natively handle with progressive decoding.
 *
 * Callbacks:
 *  - onChunk:    fired for each streamed delta chunk
 *  - onComplete: fired when the stream closes with the final Message
 *  - onError:    fired on network or parse failures
 *
 * Returns an AbortController so callers can cancel the stream.
 */
export const streamMessage = (
  payload: StreamMessagePayload
): AbortController => {
  const controller = new AbortController();
  const { onChunk, onComplete, onError, ...body } = payload;

  const baseURL: string = api.defaults.baseURL as string;
  const token: string | null = getAuthToken();

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  void (async (): Promise<void> => {
    try {
      const response = await fetch(`${baseURL}${ENDPOINTS.STREAM}`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          conversationId: body.conversationId ?? null,
          content: body.content,
          role: body.role ?? MessageRole.User,
          systemPrompt: body.systemPrompt ?? null,
          metadata: body.metadata ?? {},
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(
          `Stream request failed with status ${response.status}: ${response.statusText}`
        );
      }

      if (!response.body) {
        throw new Error("Response body is null; streaming is not supported.");
      }

      const reader: ReadableStreamDefaultReader<Uint8Array> =
        response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let finalMessage: Message | null = null;

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        const raw = decoder.decode(value, { stream: true });

        // Each line is a JSON-encoded StreamChunk (NDJSON / SSE data lines)
        const lines = raw
          .split("\n")
          .map((l) => l.replace(/^data:\s*/, "").trim())
          .filter((l) => l.length > 0 && l !== "[DONE]");

        for (const line of lines) {
          try {
            const chunk = JSON.parse(line) as StreamChunk;
            onChunk(chunk);

            if (chunk.done) {
              finalMessage = {
                id: chunk.id,
                role: MessageRole.Assistant,
                content: "", // Caller accumulates content via onChunk deltas
                status: MessageStatus.Complete,
                createdAt: new Date().toISOString(),
                tokens: chunk.usage?.totalTokens,
              };
            }
          } catch {
            // Non-JSON lines (comments, keep-alives) are silently ignored
          }
        }
      }

      if (finalMessage) {
        onComplete(finalMessage);
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.name === "AbortError") {
        // Caller-initiated cancellation – not an error condition
        return;
      }

      onError(
        error instanceof Error
          ? error
          : new Error("Unknown streaming error occurred.")
      );
    }
  })();

  return controller;
};

/**
 * Fetches the full message history for a given conversation ID.
 */
export const getConversation = async (
  conversationId: string
): Promise<GetConversationResponse> => {
  if (!conversationId.trim()) {
    throw new Error("[chat.service] conversationId must not be empty.");
  }

  const response: AxiosResponse<GetConversationResponse> =
    await api.get<GetConversationResponse>(
      ENDPOINTS.CONVERSATION(conversationId)
    );

  return response.data;
};

/**
 * Clears all messages in a conversation without deleting the conversation
 * itself, preserving its ID and metadata.
 */
export const clearConversation = async (
  conversationId: string
): Promise<ClearConversationResponse> => {
  if (!conversationId.trim()) {
    throw new Error("[chat.service] conversationId must not be empty.");
  }

  const response: AxiosResponse<ClearConversationResponse> =
    await api.delete<ClearConversationResponse>(
      ENDPOINTS.CLEAR(conversationId)
    );

  return response.data;
};

// ── Namespace export for ergonomic usage ──────────────────────────────────────
export const ChatService = {
  sendMessage,
  streamMessage,
  getConversation,
  clearConversation,
} as const;

export default ChatService;