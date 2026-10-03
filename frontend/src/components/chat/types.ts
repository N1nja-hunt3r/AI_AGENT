export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  status: "pending" | "streaming" | "complete" | "error";
  createdAt: string;
  tokens?: number;
  metadata?: Record<string, unknown>;
}

export interface ChatModel {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
}

export interface ChatAttachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url?: string;
}
