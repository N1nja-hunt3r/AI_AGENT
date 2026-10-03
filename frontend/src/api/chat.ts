import api from "@/api/client";

export const MessageRole = { User: "user", Assistant: "assistant", System: "system" } as const;
export type MessageRole = (typeof MessageRole)[keyof typeof MessageRole];

export const MessageStatus = { Pending: "pending", Streaming: "streaming", Complete: "complete", Error: "error" } as const;
export type MessageStatus = (typeof MessageStatus)[keyof typeof MessageStatus];

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

const ENDPOINTS = {
  SEND: "/api/v1/chat/message",
  STREAM: "/api/v1/chat/stream",
  CONVERSATION: (id: string) => `/api/v1/chat/conversations/${id}`,
  CLEAR: (id: string) => `/api/v1/chat/conversations/${id}/clear`,
} as const;

export const sendMessage = async (payload: SendMessagePayload): Promise<SendMessageResponse> => {
  const { data } = await api.post<SendMessageResponse>(ENDPOINTS.SEND, {
    conversationId: payload.conversationId ?? null,
    content: payload.content,
    role: payload.role ?? MessageRole.User,
    systemPrompt: payload.systemPrompt ?? null,
    metadata: payload.metadata ?? {},
  });
  return data;
};

export const streamMessage = (payload: StreamMessagePayload): AbortController => {
  const controller = new AbortController();
  const { onChunk, onComplete, onError, ...body } = payload;
  const token = localStorage.getItem("auth_token");

  void (async () => {
    try {
      const response = await fetch(`${api.defaults.baseURL}${ENDPOINTS.STREAM}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          conversationId: body.conversationId ?? null,
          content: body.content,
          role: body.role ?? MessageRole.User,
          systemPrompt: body.systemPrompt ?? null,
          metadata: body.metadata ?? {},
        }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`Stream failed: ${response.status}`);
      if (!response.body) throw new Error("Streaming not supported.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let currentEvent = "";
      let finalMessage: Message | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n");
        buffer = parts.pop() ?? "";

        for (const line of parts) {
          const trimmed = line.trim();
          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim();
          } else if (trimmed.startsWith("data:")) {
            const data = trimmed.slice(5).trim();
            if (currentEvent === "done") {
              try {
                const parsed = JSON.parse(data);
                finalMessage = {
                  id: parsed.session_id ?? `asst_${Date.now()}`,
                  role: MessageRole.Assistant,
                  content: "",
                  status: MessageStatus.Complete,
                  createdAt: parsed.created_at ?? new Date().toISOString(),
                  tokens: parsed.usage?.total_tokens ?? parsed.usage?.totalTokens,
                };
              } catch {
                finalMessage = {
                  id: `asst_${Date.now()}`,
                  role: MessageRole.Assistant,
                  content: "",
                  status: MessageStatus.Complete,
                  createdAt: new Date().toISOString(),
                };
              }
            } else {
              try {
                const chunk = JSON.parse(data) as StreamChunk;
                onChunk(chunk);
                if (chunk.done) {
                  finalMessage = {
                    id: chunk.id,
                    role: MessageRole.Assistant,
                    content: "",
                    status: MessageStatus.Complete,
                    createdAt: new Date().toISOString(),
                    tokens: chunk.usage?.totalTokens,
                  };
                }
              } catch {
                // skip non-JSON data lines
              }
            }
          } else if (trimmed === "[DONE]") {
            // Stream complete marker, nothing more to do
          }
        }
      }
      if (finalMessage) onComplete(finalMessage);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") return;
      onError(error instanceof Error ? error : new Error("Unknown streaming error"));
    }
  })();

  return controller;
};

export const getConversation = async (conversationId: string): Promise<GetConversationResponse> => {
  const { data } = await api.get<GetConversationResponse>(ENDPOINTS.CONVERSATION(conversationId));
  return data;
};

export const clearConversation = async (conversationId: string): Promise<ClearConversationResponse> => {
  const { data } = await api.delete<ClearConversationResponse>(ENDPOINTS.CLEAR(conversationId));
  return data;
};
