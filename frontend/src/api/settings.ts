import api from "@/api/client";

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

export interface UserSettings {
  user_id: string;
  model: ModelSettings;
  memory: MemorySettings;
  rag: RAGSettings;
  features: FeatureFlagSettings;
  updated_at: string;
}

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

export interface UpdateSettingsRequest {
  model?: ModelSettings;
  memory?: MemorySettings;
  rag?: RAGSettings;
  features?: FeatureFlagSettings;
}

const ENDPOINTS = {
  SETTINGS: "/api/v1/settings",
  RESET: "/api/v1/settings/reset",
  FEATURES: "/api/v1/settings/features",
  API_KEYS: "/api/v1/api-keys",
} as const;

export const getSettings = async (): Promise<UserSettings> => {
  const { data } = await api.get<UserSettings>(ENDPOINTS.SETTINGS);
  return data;
};

export const updateSettings = async (payload: UpdateSettingsRequest): Promise<UserSettings> => {
  const { data } = await api.put<UserSettings>(ENDPOINTS.SETTINGS, payload);
  return data;
};

export const resetSettings = async (): Promise<UserSettings> => {
  const { data } = await api.post<UserSettings>(ENDPOINTS.RESET);
  return data;
};

export const getFeatureFlags = async (): Promise<FeatureFlagSettings> => {
  const { data } = await api.get<FeatureFlagSettings>(ENDPOINTS.FEATURES);
  return data;
};

export const listApiKeys = async (): Promise<ListApiKeysResponse> => {
  const { data } = await api.get<ListApiKeysResponse>(ENDPOINTS.API_KEYS);
  return data;
};

export const createApiKey = async (payload: CreateApiKeyRequest): Promise<CreateApiKeyResponse> => {
  const { data } = await api.post<CreateApiKeyResponse>(ENDPOINTS.API_KEYS, payload);
  return data;
};

export const deleteApiKey = async (id: string): Promise<void> => {
  await api.delete(`${ENDPOINTS.API_KEYS}/${id}`);
};
