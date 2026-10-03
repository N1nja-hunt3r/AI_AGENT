import api from "@/api/client";

export interface FrontendMemoryItem {
  id: string;
  type: string;
  status: string;
  content: string;
  summary?: string;
  tags: string[];
  score?: number;
  conversationId?: string;
  agentId?: string;
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
  metadata?: Record<string, unknown>;
}

export interface CreateMemoryPayload {
  type: string;
  content: string;
  summary?: string;
  tags?: string[];
  conversationId?: string;
  agentId?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateMemoryPayload {
  type?: string;
  content?: string;
  summary?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface SearchMemoriesPayload {
  query: string;
  type?: string;
  limit?: number;
  threshold?: number;
  agentId?: string;
  conversationId?: string;
}

export interface PurgeRequest {
  type: string;
}

export interface MemoryResponse {
  memory: FrontendMemoryItem;
}

export interface MemoryListResponse {
  memories: FrontendMemoryItem[];
  total: number;
  page: number;
  limit: number;
}

export interface SearchResult {
  memory: FrontendMemoryItem;
  similarity: number;
  relevanceScore: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
}

export interface DeleteMemoryResponse {
  success: boolean;
  id: string;
  deletedAt: string;
}

export interface PurgeResponse {
  success: boolean;
  purgedCount: number;
  purgedAt: string;
}

const ENDPOINTS = {
  BASE: "/api/v1/memories",
  BY_ID: (id: string) => `/api/v1/memories/${id}`,
  SEARCH: "/api/v1/memories/search",
  PURGE: "/api/v1/memories/purge",
} as const;

export const createMemory = async (payload: CreateMemoryPayload): Promise<MemoryResponse> => {
  const { data } = await api.post<MemoryResponse>(ENDPOINTS.BASE, payload);
  return data;
};

export const listMemories = async (params?: { type?: string; page?: number; limit?: number }): Promise<MemoryListResponse> => {
  const { data } = await api.get<MemoryListResponse>(ENDPOINTS.BASE, { params });
  return data;
};

export const getMemory = async (id: string): Promise<MemoryResponse> => {
  const { data } = await api.get<MemoryResponse>(ENDPOINTS.BY_ID(id));
  return data;
};

export const updateMemory = async (id: string, payload: UpdateMemoryPayload): Promise<MemoryResponse> => {
  const { data } = await api.patch<MemoryResponse>(ENDPOINTS.BY_ID(id), payload);
  return data;
};

export const deleteMemory = async (id: string): Promise<DeleteMemoryResponse> => {
  const { data } = await api.delete<DeleteMemoryResponse>(ENDPOINTS.BY_ID(id));
  return data;
};

export const searchMemories = async (payload: SearchMemoriesPayload): Promise<SearchResponse> => {
  const { data } = await api.post<SearchResponse>(ENDPOINTS.SEARCH, payload);
  return data;
};

export const purgeMemories = async (payload: PurgeRequest): Promise<PurgeResponse> => {
  const { data } = await api.delete<PurgeResponse>(ENDPOINTS.PURGE, { data: payload });
  return data;
};
