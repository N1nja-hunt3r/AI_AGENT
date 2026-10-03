// frontend/src/constants/providers.ts
// Aspire AI — Provider registry (NVIDIA NIM primary, OpenAI fallback only)

import { ProviderType, ProviderAuthType } from "../types/settings";

// ─── Provider metadata ────────────────────────────────────────────────────────

export interface ProviderMeta {
  id: string;
  name: string;
  type: ProviderType;
  auth_type: ProviderAuthType;
  base_url: string;
  docs_url: string;
  icon: string;
  color: string;
  requires_api_key: boolean;
  supports_organization_id: boolean;
  supports_project_id: boolean;
  supports_region: boolean;
  regions?: string[];
  default_max_tokens: number;
  default_context_window: number;
  rate_limit_rpm?: number;
  rate_limit_tpm?: number;
}

// ─── Provider registry ────────────────────────────────────────────────────────

export const PROVIDERS: Record<string, ProviderMeta> = {
  nvidia_nim: {
    id: "nvidia_nim",
    name: "NVIDIA NIM",
    type: ProviderType.Custom,
    auth_type: ProviderAuthType.ApiKey,
    base_url: "https://integrate.api.nvidia.com/v1",
    docs_url: "https://docs.nvidia.com/nim/",
    icon: "nvidia",
    color: "#76B900",
    requires_api_key: true,
    supports_organization_id: false,
    supports_project_id: false,
    supports_region: false,
    default_max_tokens: 4096,
    default_context_window: 131_072,
    rate_limit_rpm: 10_000,
    rate_limit_tpm: 500_000,
  },
  openai: {
    id: "openai",
    name: "OpenAI (Fallback)",
    type: ProviderType.OpenAI,
    auth_type: ProviderAuthType.ApiKey,
    base_url: "https://api.openai.com/v1",
    docs_url: "https://platform.openai.com/docs",
    icon: "openai",
    color: "#10A37F",
    requires_api_key: true,
    supports_organization_id: true,
    supports_project_id: false,
    supports_region: false,
    default_max_tokens: 4096,
    default_context_window: 128_000,
    rate_limit_rpm: 10_000,
    rate_limit_tpm: 150_000,
  },
} as const;

// ─── Provider ID list ────────────────────────────────────────────────────────

export const PROVIDER_IDS = Object.keys(PROVIDERS) as Array<
  keyof typeof PROVIDERS
>;

// ─── Provider labels for UI ──────────────────────────────────────────────────

export const PROVIDER_LABELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDERS).map(([id, meta]) => [id, meta.name])
);

// ─── Provider order for display ──────────────────────────────────────────────

export const PROVIDER_ORDER: string[] = [
  "nvidia_nim",
  "openai",
];

// ─── Helper functions ────────────────────────────────────────────────────────

/**
 * Returns provider meta by ID.
 */
export const getProvider = (id: string): ProviderMeta | undefined => {
  return PROVIDERS[id];
};

/**
 * Returns all providers that require an API key.
 */
export const getApiKeyProviders = (): ProviderMeta[] => {
  return Object.values(PROVIDERS).filter((p) => p.requires_api_key);
};

/**
 * Returns providers that do not require auth (local models).
 */
export const getLocalProviders = (): ProviderMeta[] => {
  return Object.values(PROVIDERS).filter(
    (p) => p.auth_type === ProviderAuthType.None
  );
};

/**
 * Returns providers supporting organization ID.
 */
export const getOrgSupportingProviders = (): ProviderMeta[] => {
  return Object.values(PROVIDERS).filter(
    (p) => p.supports_organization_id
  );
};

/**
 * Returns providers supporting region selection.
 */
export const getRegionSupportingProviders = (): ProviderMeta[] => {
  return Object.values(PROVIDERS).filter((p) => p.supports_region);
};

/**
 * Validates a provider base URL.
 */
export const isValidProviderUrl = (url: string): boolean => {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol);
  } catch {
    return false;
  }
};
