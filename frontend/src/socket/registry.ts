// frontend/src/socket/registry.ts

// ─── Core socket manager ──────────────────────────────────────────────────────

export {
  SocketManager,
  type SocketManagerConfig,
  type SocketReadyState,
  type SocketCloseCode,
  type SocketFrameType,
  type SocketFrame,
  type AuthFrame,
  type AuthAckFrame,
  type ErrorFrame,
  type SubscribeFrame,
  type SocketEventListener,
  type SocketStateListener,
  type SocketErrorListener,
  type SocketManagerError,
} from "./socket";

// ─── Chat socket ──────────────────────────────────────────────────────────────

export {
  ChatSocket,
  createChatSocket,
  CHAT_CHANNELS,
  type ChatTokenChunk,
  type ChatStreamStart,
  type ChatStreamEnd,
  type ChatStreamError,
  type ChatToolCallStart,
  type ChatToolCallResult,
  type ChatThinkingChunk,
  type ChatSendMessage,
  type ChatTool,
  type ChatCancelStream,
  type ChatRegenerateMessage,
  type ChatSocketListeners,
} from "./chatSocket";

// ─── Voice socket ─────────────────────────────────────────────────────────────

export {
  VoiceSocket,
  createVoiceSocket,
  VOICE_CHANNELS,
  type AudioFormat,
  type AudioSampleRate,
  type VoiceActivityState,
  type VoiceSessionConfig,
  type VoiceSessionStarted,
  type VoiceSessionEnded,
  type VoiceAudioChunkMeta,
  type VoiceTranscription,
  type VoiceWordTimestamp,
  type VoiceActivityEvent,
  type VoiceSynthesisRequest,
  type VoiceSynthesisChunkMeta,
  type VoiceSynthesisComplete,
  type VoiceError,
  type VoiceSocketListeners,
} from "./voiceSocket";

// ─── Monitoring socket ────────────────────────────────────────────────────────

export {
  MonitoringSocket,
  createMonitoringSocket,
  MONITORING_CHANNELS,
  type MetricUnit,
  type MetricDataPoint,
  type Metric,
  type AgentMonitorStatus,
  type AgentMetricSnapshot,
  type AgentHealthCheck,
  type SystemMetricSnapshot,
  type ProviderMetricSnapshot,
  type CostEvent,
  type CostSummary,
  type AlertSeverity,
  type AlertCategory,
  type MonitoringAlert,
  type LogLevel,
  type LogEntry,
  type MonitoringSocketListeners,
} from "./monitoringSocket";

// ─── Singleton registry ───────────────────────────────────────────────────────

import { SocketManager, type SocketManagerConfig } from "./socket";
import { ChatSocket } from "./chatSocket";
import { VoiceSocket } from "./voiceSocket";
import { MonitoringSocket } from "./monitoringSocket";

export interface SocketRegistryConfig {
  chat: SocketManagerConfig;
  voice: SocketManagerConfig;
  monitoring: SocketManagerConfig;
}

export interface SocketRegistryInstance {
  chat: ChatSocket;
  voice: VoiceSocket;
  monitoring: MonitoringSocket;
  destroyAll: () => void;
  connectAll: () => Promise<void>;
  disconnectAll: () => void;
}

let registryInstance: SocketRegistryInstance | null = null;

export const createSocketRegistry = (
  config: SocketRegistryConfig
): SocketRegistryInstance => {
  if (registryInstance) {
    registryInstance.destroyAll();
  }

  const chat = new ChatSocket(config.chat);
  const voice = new VoiceSocket(config.voice);
  const monitoring = new MonitoringSocket(config.monitoring);

  registryInstance = {
    chat,
    voice,
    monitoring,

    connectAll: async () => {
      await Promise.all([
        chat.getManager().connect(),
        voice.getManager().connect(),
        monitoring.getManager().connect(),
      ]);
    },

    disconnectAll: () => {
      chat.getManager().disconnect();
      voice.getManager().disconnect();
      monitoring.getManager().disconnect();
    },

    destroyAll: () => {
      chat.destroy();
      voice.destroy();
      monitoring.destroy();
      registryInstance = null;
    },
  };

  return registryInstance;
};

export const getSocketRegistry = (): SocketRegistryInstance | null =>
  registryInstance;

// ─── Standalone factory helpers ───────────────────────────────────────────────

export const createSocketManager = (
  config: SocketManagerConfig
): SocketManager => new SocketManager(config);