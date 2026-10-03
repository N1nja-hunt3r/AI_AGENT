import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const MemoryType = {
  ShortTerm: "short_term",
  LongTerm: "long_term",
  Episodic: "episodic",
  Semantic: "semantic",
} as const;

export type MemoryType = (typeof MemoryType)[keyof typeof MemoryType];

export const MemoryStatus = {
  Active: "active",
  Archived: "archived",
  Deleted: "deleted",
} as const;

export type MemoryStatus = (typeof MemoryStatus)[keyof typeof MemoryStatus];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface Memory {
  id: string;
  type: MemoryType;
  status: MemoryStatus;
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

export interface MemorySearchResult {
  memory: Memory;
  similarity: number;
  relevanceScore: number;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface CreateMemoryPayload {
  type: MemoryType;
  content: string;
  summary?: string;
  tags?: string[];
  conversationId?: string;
  agentId?: string;
  expiresAt?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateMemoryPayload {
  content?: string;
  summary?: string;
  tags?: string[];
  status?: MemoryStatus;
  expiresAt?: string;
  metadata?: Record<string, unknown>;
}

export interface SearchMemoryPayload {
  query: string;
  type?: MemoryType;
  tags?: string[];
  limit?: number;
  threshold?: number;
  agentId?: string;
  conversationId?: string;
}

export interface ListMemoriesParams {
  type?: MemoryType;
  status?: MemoryStatus;
  tags?: string[];
  agentId?: string;
  conversationId?: string;
  page?: number;
  limit?: number;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface MemoryResponse {
  memory: Memory;
}

export interface MemoryListResponse {
  memories: Memory[];
  total: number;
  page: number;
  limit: number;
}

export interface MemorySearchResponse {
  results: MemorySearchResult[];
  total: number;
  query: string;
}

export interface DeleteMemoryResponse {
  success: boolean;
  id: string;
  deletedAt: string;
}

export interface PurgeMemoriesResponse {
  success: boolean;
  purgedCount: number;
  purgedAt: string;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  BASE: "/api/v1/memories",
  BY_ID: (id: string) => `/api/v1/memories/${id}`,
  SEARCH: "/api/v1/memories/search",
  PURGE: "/api/v1/memories/purge",
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const createMemory = async (
  payload: CreateMemoryPayload
): Promise<MemoryResponse> => {
  const response: AxiosResponse<MemoryResponse> =
    await api.post<MemoryResponse>(ENDPOINTS.BASE, payload);
  return response.data;
};

const listMemories = async (
  params?: ListMemoriesParams
): Promise<MemoryListResponse> => {
  const response: AxiosResponse<MemoryListResponse> =
    await api.get<MemoryListResponse>(ENDPOINTS.BASE, { params });
  return response.data;
};

const getMemory = async (id: string): Promise<MemoryResponse> => {
  const response: AxiosResponse<MemoryResponse> =
    await api.get<MemoryResponse>(ENDPOINTS.BY_ID(id));
  return response.data;
};

const updateMemory = async (
  id: string,
  payload: UpdateMemoryPayload
): Promise<MemoryResponse> => {
  const response: AxiosResponse<MemoryResponse> =
    await api.patch<MemoryResponse>(ENDPOINTS.BY_ID(id), payload);
  return response.data;
};

const deleteMemory = async (id: string): Promise<DeleteMemoryResponse> => {
  const response: AxiosResponse<DeleteMemoryResponse> =
    await api.delete<DeleteMemoryResponse>(ENDPOINTS.BY_ID(id));
  return response.data;
};

const searchMemories = async (
  payload: SearchMemoryPayload
): Promise<MemorySearchResponse> => {
  const response: AxiosResponse<MemorySearchResponse> =
    await api.post<MemorySearchResponse>(ENDPOINTS.SEARCH, payload);
  return response.data;
};

const purgeMemories = async (
  type?: MemoryType
): Promise<PurgeMemoriesResponse> => {
  const response: AxiosResponse<PurgeMemoriesResponse> =
    await api.delete<PurgeMemoriesResponse>(ENDPOINTS.PURGE, {
      data: { type },
    });
  return response.data;
};

export const MemoryService = {
  createMemory,
  listMemories,
  getMemory,
  updateMemory,
  deleteMemory,
  searchMemories,
  purgeMemories,
} as const;

export default MemoryService;