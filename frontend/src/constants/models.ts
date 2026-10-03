// Aspire AI — Backend model registry (NVIDIA NIM primary, OpenAI fallback)

import { ModelCapability } from "../types/settings";

// ─── Model IDs (NVIDIA NIM backend) ──────────────────────────────────────────

export const ASPIRE_MODELS = {
  // Chief (primary)
  DEEPSEEK_V4_PRO: "deepseek-ai/deepseek-v4-pro",
  // Chief (fallback)
  QWEN_3_5_122B: "qwen/qwen3.5-122b-a10b",
  // Coder (primary)
  LLAMA_3_3_70B: "meta/llama-3.3-70b-instruct",
  // Coder (fallback)
  LLAMA_3_1_70B: "meta/llama-3.1-70b-instruct",
  // Research (primary)
  QWEN_3_5_397B: "qwen/qwen3.5-397b-a17b",
  // Research (fallback)
  QWEN_3_NEXT_80B: "qwen/qwen3-next-80b-a3b-instruct",
  // Vision (primary)
  LLAMA_3_2_90B_VISION: "meta/llama-3.2-90b-vision-instruct",
  // Vision (fallback)
  LLAMA_3_2_11B_VISION: "meta/llama-3.2-11b-vision-instruct",
  // Embeddings
  NV_EMBED_V1: "nvidia/nv-embed-v1",
  // Speech-to-Text (primary)
  WHISPER_LARGE_V3: "openai/whisper-large-v3",
  // Speech-to-Text (fallback)
  PARAKEET_1_1B: "parakeet-1.1b-rnnt-multilingual-asr",
  // Text-to-Speech (primary)
  MAGPIE_TTS: "magpie-tts-multilingual",
  // Text-to-Speech (fallback)
  CHATTERBOX_TTS: "chatterbox-multilingual-tts",
  // Emergency fallback
  GPT_4O: "gpt-4o",
} as const;

// ─── Model display names ─────────────────────────────────────────────────────

export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  [ASPIRE_MODELS.DEEPSEEK_V4_PRO]: "DeepSeek V4 Pro",
  [ASPIRE_MODELS.QWEN_3_5_122B]: "Qwen 3.5 122B",
  [ASPIRE_MODELS.LLAMA_3_3_70B]: "Llama 3.3 70B",
  [ASPIRE_MODELS.LLAMA_3_1_70B]: "Llama 3.1 70B",
  [ASPIRE_MODELS.QWEN_3_5_397B]: "Qwen 3.5 397B",
  [ASPIRE_MODELS.QWEN_3_NEXT_80B]: "Qwen 3 Next 80B",
  [ASPIRE_MODELS.LLAMA_3_2_90B_VISION]: "Llama 3.2 90B Vision",
  [ASPIRE_MODELS.LLAMA_3_2_11B_VISION]: "Llama 3.2 11B Vision",
  [ASPIRE_MODELS.NV_EMBED_V1]: "NV-Embed-v1",
  [ASPIRE_MODELS.WHISPER_LARGE_V3]: "Whisper Large v3",
  [ASPIRE_MODELS.PARAKEET_1_1B]: "Parakeet 1.1B",
  [ASPIRE_MODELS.MAGPIE_TTS]: "Magpie TTS",
  [ASPIRE_MODELS.CHATTERBOX_TTS]: "Chatterbox TTS",
  [ASPIRE_MODELS.GPT_4O]: "GPT-4o (Emergency Fallback)",
};

// ─── Model roles (maps to backend agent routing) ─────────────────────────────

export const MODEL_ROLES = {
  CHIEF: "chief",
  CHIEF_FALLBACK: "chief_fallback",
  CODER: "coder",
  CODER_FALLBACK: "coder_fallback",
  RESEARCHER: "researcher",
  RESEARCHER_FALLBACK: "researcher_fallback",
  VISION: "vision",
  VISION_FALLBACK: "vision_fallback",
  EMBEDDING: "embedding",
  STT: "stt",
  STT_FALLBACK: "stt_fallback",
  TTS: "tts",
  TTS_FALLBACK: "tts_fallback",
  EMERGENCY: "emergency",
} as const;

// ─── Model capabilities lookup ────────────────────────────────────────────────

export const MODEL_CAPABILITIES: Record<string, ModelCapability[]> = {
  [ASPIRE_MODELS.DEEPSEEK_V4_PRO]: [
    ModelCapability.Chat, ModelCapability.Reasoning, ModelCapability.Streaming,
    ModelCapability.CodeGeneration, ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.QWEN_3_5_122B]: [
    ModelCapability.Chat, ModelCapability.Reasoning, ModelCapability.Streaming,
    ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.LLAMA_3_3_70B]: [
    ModelCapability.Chat, ModelCapability.Streaming,
    ModelCapability.CodeGeneration, ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.LLAMA_3_1_70B]: [
    ModelCapability.Chat, ModelCapability.Streaming,
    ModelCapability.CodeGeneration, ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.QWEN_3_5_397B]: [
    ModelCapability.Chat, ModelCapability.Reasoning, ModelCapability.Streaming,
    ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.QWEN_3_NEXT_80B]: [
    ModelCapability.Chat, ModelCapability.Reasoning, ModelCapability.Streaming,
    ModelCapability.FunctionCalling,
  ],
  [ASPIRE_MODELS.LLAMA_3_2_90B_VISION]: [
    ModelCapability.Chat, ModelCapability.Vision, ModelCapability.Streaming,
  ],
  [ASPIRE_MODELS.LLAMA_3_2_11B_VISION]: [
    ModelCapability.Chat, ModelCapability.Vision, ModelCapability.Streaming,
  ],
  [ASPIRE_MODELS.NV_EMBED_V1]: [ModelCapability.Embedding],
  [ASPIRE_MODELS.WHISPER_LARGE_V3]: [ModelCapability.Audio],
  [ASPIRE_MODELS.PARAKEET_1_1B]: [ModelCapability.Audio],
  [ASPIRE_MODELS.MAGPIE_TTS]: [ModelCapability.Audio],
  [ASPIRE_MODELS.CHATTERBOX_TTS]: [ModelCapability.Audio],
  [ASPIRE_MODELS.GPT_4O]: [
    ModelCapability.Chat, ModelCapability.Vision, ModelCapability.FunctionCalling,
    ModelCapability.Streaming, ModelCapability.CodeGeneration, ModelCapability.Reasoning,
  ],
} as const;

// ─── Model context windows ───────────────────────────────────────────────────

export const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
  [ASPIRE_MODELS.DEEPSEEK_V4_PRO]: 131_072,
  [ASPIRE_MODELS.QWEN_3_5_122B]: 131_072,
  [ASPIRE_MODELS.LLAMA_3_3_70B]: 128_000,
  [ASPIRE_MODELS.LLAMA_3_1_70B]: 128_000,
  [ASPIRE_MODELS.QWEN_3_5_397B]: 131_072,
  [ASPIRE_MODELS.QWEN_3_NEXT_80B]: 131_072,
  [ASPIRE_MODELS.LLAMA_3_2_90B_VISION]: 128_000,
  [ASPIRE_MODELS.LLAMA_3_2_11B_VISION]: 128_000,
  [ASPIRE_MODELS.NV_EMBED_V1]: 2_048,
  [ASPIRE_MODELS.GPT_4O]: 128_000,
};

// ─── Model lists by category ─────────────────────────────────────────────────

export const VISION_MODELS = [
  ASPIRE_MODELS.LLAMA_3_2_90B_VISION,
  ASPIRE_MODELS.LLAMA_3_2_11B_VISION,
  ASPIRE_MODELS.GPT_4O,
] as const;

export const EMBEDDING_MODELS = [
  ASPIRE_MODELS.NV_EMBED_V1,
] as const;

export const AUDIO_MODELS = [
  ASPIRE_MODELS.WHISPER_LARGE_V3,
  ASPIRE_MODELS.PARAKEET_1_1B,
  ASPIRE_MODELS.MAGPIE_TTS,
  ASPIRE_MODELS.CHATTERBOX_TTS,
] as const;

export const REASONING_MODELS = [
  ASPIRE_MODELS.DEEPSEEK_V4_PRO,
  ASPIRE_MODELS.QWEN_3_5_122B,
  ASPIRE_MODELS.QWEN_3_5_397B,
  ASPIRE_MODELS.QWEN_3_NEXT_80B,
] as const;

export const CHAT_MODELS = [
  ASPIRE_MODELS.DEEPSEEK_V4_PRO,
  ASPIRE_MODELS.DEEPSEEK_V4_FLASH,
  ASPIRE_MODELS.QWEN_3_5_122B,
  ASPIRE_MODELS.LLAMA_3_3_70B,
  ASPIRE_MODELS.LLAMA_3_1_70B,
  ASPIRE_MODELS.QWEN_3_5_397B,
  ASPIRE_MODELS.QWEN_3_NEXT_80B,
  ASPIRE_MODELS.GPT_4O,
] as const;

// ─── Default model constants ─────────────────────────────────────────────────

export const DEFAULT_MODEL = ASPIRE_MODELS.DEEPSEEK_V4_PRO;
export const DEFAULT_PROVIDER = "nvidia_nim";
export const DEFAULT_REASONING_MODEL = ASPIRE_MODELS.DEEPSEEK_V4_PRO;
export const DEFAULT_CODE_MODEL = ASPIRE_MODELS.LLAMA_3_3_70B;
export const DEFAULT_VISION_MODEL = ASPIRE_MODELS.LLAMA_3_2_90B_VISION;
export const DEFAULT_AUDIO_MODEL = ASPIRE_MODELS.WHISPER_LARGE_V3;
export const DEFAULT_EMBEDDING_MODEL = ASPIRE_MODELS.NV_EMBED_V1;
