// frontend/src/socket/socket.ts

import type { TokenPair } from "../auth/AuthContext";

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_RECONNECT_ATTEMPTS = 10;
const DEFAULT_RECONNECT_BASE_DELAY_MS = 500;
const DEFAULT_RECONNECT_MAX_DELAY_MS = 30_000;
const DEFAULT_RECONNECT_JITTER_MS = 300;
const DEFAULT_HEARTBEAT_INTERVAL_MS = 25_000;
const DEFAULT_HEARTBEAT_TIMEOUT_MS = 10_000;
const DEFAULT_CONNECT_TIMEOUT_MS = 15_000;
const PROTOCOL_VERSION = "1";

// ─── Enums ────────────────────────────────────────────────────────────────────

export type SocketReadyState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "closing"
  | "closed"
  | "error";

export type SocketCloseCode =
  | 1000 // Normal closure
  | 1001 // Going away
  | 1006 // Abnormal closure
  | 1008 // Policy violation
  | 1011 // Internal error
  | 4000 // Auth required
  | 4001 // Auth failed
  | 4002 // Auth expired
  | 4003 // Rate limited
  | 4004 // Duplicate session
  | 4005; // Server shutdown

// ─── Frame types ─────────────────────────────────────────────────────────────

export type SocketFrameType =
  | "ping"
  | "pong"
  | "auth"
  | "auth_ack"
  | "subscribe"
  | "unsubscribe"
  | "event"
  | "error"
  | "ack";

export interface SocketFrame<TPayload = unknown> {
  type: SocketFrameType;
  id?: string;
  channel?: string;
  payload?: TPayload;
  timestamp: number;
  version: string;
}

export interface AuthFrame {
  access_token: string;
  session_id?: string;
  device_id?: string;
}

export interface AuthAckFrame {
  user_id: string;
  session_id: string;
  expires_at: number;
}

export interface ErrorFrame {
  code: string;
  message: string;
  recoverable: boolean;
  request_id?: string;
}

export interface SubscribeFrame {
  channel: string;
  params?: Record<string, unknown>;
}

// ─── Listener types ───────────────────────────────────────────────────────────

export type SocketEventListener<TPayload = unknown> = (
  payload: TPayload,
  frame: SocketFrame<TPayload>
) => void;

export type SocketStateListener = (state: SocketReadyState) => void;

export type SocketErrorListener = (error: SocketManagerError) => void;

// ─── Error type ───────────────────────────────────────────────────────────────

export interface SocketManagerError {
  code: string;
  message: string;
  timestamp: number;
  recoverable: boolean;
  attempt?: number;
}

// ─── Configuration ────────────────────────────────────────────────────────────

export interface SocketManagerConfig {
  url: string;
  protocols?: string[];
  reconnectAttempts?: number;
  reconnectBaseDelayMs?: number;
  reconnectMaxDelayMs?: number;
  reconnectJitterMs?: number;
  heartbeatIntervalMs?: number;
  heartbeatTimeoutMs?: number;
  connectTimeoutMs?: number;
  getTokens?: () => Promise<TokenPair | null> | TokenPair | null;
  onTokenExpired?: () => Promise<TokenPair | null>;
  deviceId?: string;
  binaryType?: BinaryType;
  autoConnect?: boolean;
}

// ─── Subscription record ──────────────────────────────────────────────────────

interface ChannelSubscription {
  channel: string;
  params?: Record<string, unknown>;
  listeners: Set<SocketEventListener<unknown>>;
}

// ─── Socket Manager ───────────────────────────────────────────────────────────

export class SocketManager {
  private readonly config: Required<
    Omit<SocketManagerConfig, "getTokens" | "onTokenExpired" | "deviceId" | "protocols">
  > &
    Pick<SocketManagerConfig, "getTokens" | "onTokenExpired" | "deviceId" | "protocols">;

  private ws: WebSocket | null = null;
  private state: SocketReadyState = "closed";
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private connectTimeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private isAuthenticated = false;
  private pendingFrameQueue: SocketFrame[] = [];
  private messageIdCounter = 0;

  private readonly stateListeners = new Set<SocketStateListener>();
  private readonly errorListeners = new Set<SocketErrorListener>();
  private readonly eventListeners = new Map<
    string,
    Set<SocketEventListener<unknown>>
  >();
  private readonly channelSubscriptions = new Map<
    string,
    ChannelSubscription
  >();
  private readonly ackCallbacks = new Map<
    string,
    (frame: SocketFrame) => void
  >();

  constructor(config: SocketManagerConfig) {
    this.config = {
      protocols: config.protocols,
      reconnectAttempts:
        config.reconnectAttempts ?? DEFAULT_RECONNECT_ATTEMPTS,
      reconnectBaseDelayMs:
        config.reconnectBaseDelayMs ?? DEFAULT_RECONNECT_BASE_DELAY_MS,
      reconnectMaxDelayMs:
        config.reconnectMaxDelayMs ?? DEFAULT_RECONNECT_MAX_DELAY_MS,
      reconnectJitterMs:
        config.reconnectJitterMs ?? DEFAULT_RECONNECT_JITTER_MS,
      heartbeatIntervalMs:
        config.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      heartbeatTimeoutMs:
        config.heartbeatTimeoutMs ?? DEFAULT_HEARTBEAT_TIMEOUT_MS,
      connectTimeoutMs:
        config.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS,
      getTokens: config.getTokens,
      onTokenExpired: config.onTokenExpired,
      deviceId: config.deviceId,
      binaryType: config.binaryType ?? "arraybuffer",
      url: config.url,
      autoConnect: config.autoConnect ?? false,
    };

    if (this.config.autoConnect) {
      void this.connect();
    }
  }

  // ── State management ───────────────────────────────────────────────────────

  getState(): SocketReadyState {
    return this.state;
  }

  isConnected(): boolean {
    return this.state === "connected" && this.isAuthenticated;
  }

  private setState(next: SocketReadyState): void {
    if (this.state === next) return;
    this.state = next;
    this.stateListeners.forEach((fn) => fn(next));
  }

  // ── Connection lifecycle ───────────────────────────────────────────────────

  async connect(): Promise<void> {
    if (
      this.state === "connecting" ||
      this.state === "connected"
    ) {
      return;
    }

    this.setState("connecting");
    this.clearReconnectTimer();

    this.connectTimeoutTimer = setTimeout(() => {
      this.emitError({
        code: "CONNECT_TIMEOUT",
        message: "WebSocket connection timed out",
        timestamp: Date.now(),
        recoverable: true,
      });
      this.ws?.close();
      this.scheduleReconnect();
    }, this.config.connectTimeoutMs);

    try {
      this.ws = new WebSocket(this.config.url, this.config.protocols);
      this.ws.binaryType = this.config.binaryType;

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
    } catch (err) {
      this.clearConnectTimeout();
      this.emitError({
        code: "CONNECT_FAILED",
        message:
          err instanceof Error ? err.message : "Failed to create WebSocket",
        timestamp: Date.now(),
        recoverable: true,
      });
      this.scheduleReconnect();
    }
  }

  disconnect(code: SocketCloseCode = 1000, reason = "Client disconnect"): void {
    this.reconnectAttempt = this.config.reconnectAttempts; // Prevent auto-reconnect
    this.clearAllTimers();
    this.setState("closing");
    this.ws?.close(code, reason);
    this.ws = null;
    this.isAuthenticated = false;
    this.pendingFrameQueue = [];
    this.setState("closed");
  }

  // ── WebSocket event handlers ───────────────────────────────────────────────

  private async handleOpen(): Promise<void> {
    this.clearConnectTimeout();
    await this.authenticate();
  }

  private handleMessage(event: MessageEvent): void {
    try {
      let frame: SocketFrame;

      if (event.data instanceof ArrayBuffer) {
        const text = new TextDecoder().decode(event.data);
        frame = JSON.parse(text) as SocketFrame;
      } else if (typeof event.data === "string") {
        frame = JSON.parse(event.data) as SocketFrame;
      } else {
        return;
      }

      this.routeFrame(frame);
    } catch {
      this.emitError({
        code: "PARSE_ERROR",
        message: "Failed to parse incoming WebSocket frame",
        timestamp: Date.now(),
        recoverable: true,
      });
    }
  }

  private handleClose(event: CloseEvent): void {
    this.clearHeartbeat();
    this.clearConnectTimeout();
    this.isAuthenticated = false;

    const normalClosure =
      event.code === 1000 || event.code === 1001;
    const authFailure =
      event.code === 4001 || event.code === 4002;

    if (normalClosure || this.state === "closing") {
      this.setState("closed");
      return;
    }

    if (authFailure) {
      this.emitError({
        code: "AUTH_FAILED",
        message: `Authentication failed (code: ${event.code})`,
        timestamp: Date.now(),
        recoverable: event.code === 4002,
      });
      if (event.code === 4002 && this.config.onTokenExpired) {
        void this.handleTokenExpiry();
        return;
      }
      this.setState("error");
      return;
    }

    this.setState("reconnecting");
    this.scheduleReconnect();
  }

  private handleError(): void {
    this.emitError({
      code: "WEBSOCKET_ERROR",
      message: "WebSocket encountered an error",
      timestamp: Date.now(),
      recoverable: true,
      attempt: this.reconnectAttempt,
    });
  }

  // ── Frame routing ──────────────────────────────────────────────────────────

  private routeFrame(frame: SocketFrame): void {
    switch (frame.type) {
      case "pong":
        this.clearHeartbeatTimeout();
        break;

      case "auth_ack":
        this.handleAuthAck(frame as SocketFrame<AuthAckFrame>);
        break;

      case "event":
        this.dispatchEvent(frame);
        break;

      case "error":
        this.handleServerError(frame as SocketFrame<ErrorFrame>);
        break;

      case "ack":
        if (frame.id) {
          const cb = this.ackCallbacks.get(frame.id);
          if (cb) {
            cb(frame);
            this.ackCallbacks.delete(frame.id);
          }
        }
        break;

      default:
        break;
    }
  }

  private dispatchEvent(frame: SocketFrame): void {
    // Dispatch to ack callbacks
    if (frame.id) {
      const cb = this.ackCallbacks.get(frame.id);
      if (cb) {
        cb(frame);
        this.ackCallbacks.delete(frame.id);
      }
    }

    // Dispatch to channel listeners
    if (frame.channel) {
      const sub = this.channelSubscriptions.get(frame.channel);
      if (sub) {
        sub.listeners.forEach((fn) =>
          fn(frame.payload, frame)
        );
      }
    }

    // Dispatch to wildcard event listeners keyed by channel or type
    const key = frame.channel ?? frame.type;
    const listeners = this.eventListeners.get(key);
    if (listeners) {
      listeners.forEach((fn) => fn(frame.payload, frame));
    }

    // Global wildcard
    const wildcardListeners = this.eventListeners.get("*");
    if (wildcardListeners) {
      wildcardListeners.forEach((fn) => fn(frame.payload, frame));
    }
  }

  private handleServerError(frame: SocketFrame<ErrorFrame>): void {
    const payload = frame.payload;
    if (!payload) return;
    this.emitError({
      code: payload.code,
      message: payload.message,
      timestamp: Date.now(),
      recoverable: payload.recoverable,
    });
  }

  // ── Authentication ─────────────────────────────────────────────────────────

  private async authenticate(): Promise<void> {
    if (!this.config.getTokens) {
      this.finalizeConnection();
      return;
    }

    try {
      const tokens = await this.config.getTokens();
      if (!tokens) {
        this.emitError({
          code: "AUTH_NO_TOKENS",
          message: "No authentication tokens available",
          timestamp: Date.now(),
          recoverable: false,
        });
        this.ws?.close(4000, "Auth required");
        return;
      }

      const authFrame: SocketFrame<AuthFrame> = {
        type: "auth",
        payload: {
          access_token: tokens.access_token,
          device_id: this.config.deviceId,
        },
        timestamp: Date.now(),
        version: PROTOCOL_VERSION,
      };

      this.sendRaw(authFrame);
    } catch {
      this.emitError({
        code: "AUTH_ERROR",
        message: "Failed to retrieve auth tokens",
        timestamp: Date.now(),
        recoverable: true,
      });
      this.scheduleReconnect();
    }
  }

  private handleAuthAck(frame: SocketFrame<AuthAckFrame>): void {
    void frame;
    this.isAuthenticated = true;
    this.reconnectAttempt = 0;
    this.setState("connected");
    this.startHeartbeat();
    this.flushPendingQueue();
    this.resubscribeChannels();
  }

  private async handleTokenExpiry(): Promise<void> {
    if (!this.config.onTokenExpired) {
      this.setState("error");
      return;
    }

    try {
      const tokens = await this.config.onTokenExpired();
      if (tokens) {
        void this.connect();
      } else {
        this.setState("error");
      }
    } catch {
      this.setState("error");
    }
  }

  private finalizeConnection(): void {
    this.isAuthenticated = true;
    this.reconnectAttempt = 0;
    this.setState("connected");
    this.startHeartbeat();
    this.flushPendingQueue();
    this.resubscribeChannels();
  }

  // ── Heartbeat ──────────────────────────────────────────────────────────────

  private startHeartbeat(): void {
    this.clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (!this.isConnected()) {
        this.clearHeartbeat();
        return;
      }

      this.sendRaw<void>({
        type: "ping",
        timestamp: Date.now(),
        version: PROTOCOL_VERSION,
      });

      this.heartbeatTimeoutTimer = setTimeout(() => {
        this.emitError({
          code: "HEARTBEAT_TIMEOUT",
          message: "Heartbeat response timed out — connection lost",
          timestamp: Date.now(),
          recoverable: true,
        });
        this.ws?.close(1006, "Heartbeat timeout");
        this.scheduleReconnect();
      }, this.config.heartbeatTimeoutMs);
    }, this.config.heartbeatIntervalMs);
  }

  private clearHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    this.clearHeartbeatTimeout();
  }

  private clearHeartbeatTimeout(): void {
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
  }

  // ── Reconnection ───────────────────────────────────────────────────────────

  private scheduleReconnect(): void {
    if (this.reconnectAttempt >= this.config.reconnectAttempts) {
      this.setState("error");
      this.emitError({
        code: "MAX_RECONNECT_ATTEMPTS",
        message: `Max reconnect attempts (${this.config.reconnectAttempts}) reached`,
        timestamp: Date.now(),
        recoverable: false,
        attempt: this.reconnectAttempt,
      });
      return;
    }

    const exponential =
      this.config.reconnectBaseDelayMs *
      2 ** this.reconnectAttempt;
    const jitter =
      Math.random() * this.config.reconnectJitterMs;
    const delay = Math.min(
      exponential + jitter,
      this.config.reconnectMaxDelayMs
    );

    this.reconnectAttempt += 1;
    this.setState("reconnecting");

    this.reconnectTimer = setTimeout(() => {
      void this.connect();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private clearConnectTimeout(): void {
    if (this.connectTimeoutTimer) {
      clearTimeout(this.connectTimeoutTimer);
      this.connectTimeoutTimer = null;
    }
  }

  private clearAllTimers(): void {
    this.clearReconnectTimer();
    this.clearHeartbeat();
    this.clearConnectTimeout();
  }

  // ── Send utilities ─────────────────────────────────────────────────────────

  private sendRaw<TPayload>(frame: SocketFrame<TPayload>): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify(frame));
      return true;
    } catch {
      return false;
    }
  }

  send<TPayload>(
    type: SocketFrameType,
    payload: TPayload,
    options: {
      channel?: string;
      waitForAck?: boolean;
      timeoutMs?: number;
    } = {}
  ): Promise<SocketFrame> {
    const id = this.generateId();
    const frame: SocketFrame<TPayload> = {
      id,
      type,
      channel: options.channel,
      payload,
      timestamp: Date.now(),
      version: PROTOCOL_VERSION,
    };

    if (!this.isConnected()) {
      this.pendingFrameQueue.push(frame as SocketFrame);
      return Promise.resolve(frame as SocketFrame);
    }

    if (!options.waitForAck) {
      this.sendRaw(frame);
      return Promise.resolve(frame as SocketFrame);
    }

    return new Promise<SocketFrame>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.ackCallbacks.delete(id);
        reject(
          new Error(`Ack timeout for frame ${id} (type: ${type})`)
        );
      }, options.timeoutMs ?? 10_000);

      this.ackCallbacks.set(id, (ackFrame) => {
        clearTimeout(timeout);
        resolve(ackFrame);
      });

      if (!this.sendRaw(frame)) {
        clearTimeout(timeout);
        this.ackCallbacks.delete(id);
        this.pendingFrameQueue.push(frame as SocketFrame);
        reject(new Error("Failed to send frame"));
      }
    });
  }

  sendBinary(data: ArrayBuffer | ArrayBufferView<ArrayBuffer>): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(data);
      return true;
    } catch {
      return false;
    }
  }

  private flushPendingQueue(): void {
    while (this.pendingFrameQueue.length > 0) {
      const frame = this.pendingFrameQueue.shift();
      if (frame) this.sendRaw(frame);
    }
  }

  // ── Channel subscriptions ──────────────────────────────────────────────────

  subscribe<TPayload = unknown>(
    channel: string,
    listener: SocketEventListener<TPayload>,
    params?: Record<string, unknown>
  ): () => void {
    let sub = this.channelSubscriptions.get(channel);

    if (!sub) {
      sub = { channel, params, listeners: new Set() };
      this.channelSubscriptions.set(channel, sub);

      if (this.isConnected()) {
        void this.send<SubscribeFrame>(
          "subscribe",
          { channel, params },
          {}
        );
      }
    }

    sub.listeners.add(listener as SocketEventListener<unknown>);

    return () => {
      const current = this.channelSubscriptions.get(channel);
      if (!current) return;
      current.listeners.delete(listener as SocketEventListener<unknown>);
      if (current.listeners.size === 0) {
        this.channelSubscriptions.delete(channel);
        if (this.isConnected()) {
          void this.send<SubscribeFrame>(
            "unsubscribe",
            { channel },
            {}
          );
        }
      }
    };
  }

  private resubscribeChannels(): void {
    this.channelSubscriptions.forEach((sub, channel) => {
      if (sub.listeners.size > 0) {
        void this.send<SubscribeFrame>(
          "subscribe",
          { channel, params: sub.params },
          {}
        );
      }
    });
  }

  // ── Event listeners ────────────────────────────────────────────────────────

  on<TPayload = unknown>(
    event: string,
    listener: SocketEventListener<TPayload>
  ): () => void {
    let listeners = this.eventListeners.get(event);
    if (!listeners) {
      listeners = new Set();
      this.eventListeners.set(event, listeners);
    }
    listeners.add(listener as SocketEventListener<unknown>);

    return () => this.off(event, listener);
  }

  off<TPayload = unknown>(
    event: string,
    listener: SocketEventListener<TPayload>
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(listener as SocketEventListener<unknown>);
    }
  }

  onStateChange(listener: SocketStateListener): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  onError(listener: SocketErrorListener): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  // ── Error emission ─────────────────────────────────────────────────────────

  private emitError(error: SocketManagerError): void {
    this.errorListeners.forEach((fn) => fn(error));
  }

  // ── Utility ───────────────────────────────────────────────────────────────

  private generateId(): string {
    return `${Date.now().toString(36)}-${(++this.messageIdCounter)
      .toString(36)
      .padStart(4, "0")}`;
  }

  destroy(): void {
    this.disconnect(1000, "Manager destroyed");
    this.stateListeners.clear();
    this.errorListeners.clear();
    this.eventListeners.clear();
    this.channelSubscriptions.clear();
    this.ackCallbacks.clear();
    this.pendingFrameQueue = [];
  }
}