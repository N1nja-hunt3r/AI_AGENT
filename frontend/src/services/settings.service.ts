import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Settings sub-models (matching backend Pydantic schemas) ──────────────
export interface ModelSettings {
  provider: string;
  model: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
}

export interface MemorySettings {
  enabled: boolean;
  max_items: number;
  ttl_seconds: number | null;
  retrieval_top_k: number;
}

export interface RAGSettings {
  enabled: boolean;
  default_namespace: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
}

export interface FeatureFlagSettings {
  enable_streaming: boolean;
  enable_function_calling: boolean;
  enable_multi_agent: boolean;
  enable_vector_search: boolean;
  enable_experimental: boolean;
}

// ── API Key types ────────────────────────────────────────────────────────
export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  scopes: string[];
  expiresAt: string | null;
}

export interface CreateApiKeyRequest {
  name: string;
  scopes: string[];
}

export interface ListApiKeysResponse {
  keys: ApiKey[];
}

export interface CreateApiKeyResponse {
  key: ApiKey;
  secret: string;
}

// ── Top-level response / request types ───────────────────────────────────
export interface UserSettings {
  user_id: string;
  model: ModelSettings;
  memory: MemorySettings;
  rag: RAGSettings;
  features: FeatureFlagSettings;
  updated_at: string;
}

export interface UpdateSettingsRequest {
  model?: ModelSettings;
  memory?: MemorySettings;
  rag?: RAGSettings;
  features?: FeatureFlagSettings;
}

// ── Endpoints ────────────────────────────────────────────────────────────
const ENDPOINTS = {
  SETTINGS: "/api/v1/settings",
  RESET: "/api/v1/settings/reset",
  FEATURES: "/api/v1/settings/features",
  API_KEYS: "/api/v1/api-keys",
} as const;

// ── Service methods ──────────────────────────────────────────────────────
const getSettings = async (): Promise<UserSettings> => {
  const response: AxiosResponse<UserSettings> =
    await api.get<UserSettings>(ENDPOINTS.SETTINGS);
  return response.data;
};

const updateSettings = async (
  payload: UpdateSettingsRequest
): Promise<UserSettings> => {
  const response: AxiosResponse<UserSettings> =
    await api.put<UserSettings>(ENDPOINTS.SETTINGS, payload);
  return response.data;
};

const resetSettings = async (): Promise<UserSettings> => {
  const response: AxiosResponse<UserSettings> =
    await api.post<UserSettings>(ENDPOINTS.RESET);
  return response.data;
};

const getFeatureFlags = async (): Promise<FeatureFlagSettings> => {
  const response: AxiosResponse<FeatureFlagSettings> =
    await api.get<FeatureFlagSettings>(ENDPOINTS.FEATURES);
  return response.data;
};

const listApiKeys = async (): Promise<ListApiKeysResponse> => {
  const response: AxiosResponse<ListApiKeysResponse> =
    await api.get<ListApiKeysResponse>(ENDPOINTS.API_KEYS);
  return response.data;
};

const createApiKey = async (
  payload: CreateApiKeyRequest
): Promise<CreateApiKeyResponse> => {
  const response: AxiosResponse<CreateApiKeyResponse> =
    await api.post<CreateApiKeyResponse>(ENDPOINTS.API_KEYS, payload);
  return response.data;
};

const deleteApiKey = async (id: string): Promise<void> => {
  await api.delete(`${ENDPOINTS.API_KEYS}/${id}`);
};

export const SettingsService = {
  getSettings,
  updateSettings,
  resetSettings,
  getFeatureFlags,
  listApiKeys,
  createApiKey,
  deleteApiKey,
} as const;

export default SettingsService;
