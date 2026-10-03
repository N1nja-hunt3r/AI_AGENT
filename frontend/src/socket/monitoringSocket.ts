// frontend/src/socket/monitoringSocket.ts

import { SocketManager, type SocketManagerConfig } from "./socket";

// ─── Metric types ─────────────────────────────────────────────────────────────

export type MetricUnit =
  | "count"
  | "percent"
  | "ms"
  | "bytes"
  | "bytes_per_sec"
  | "tokens"
  | "tokens_per_sec"
  | "usd"
  | "requests_per_sec";

export interface MetricDataPoint {
  timestamp: number;
  value: number;
  labels?: Record<string, string>;
}

export interface Metric {
  id: string;
  name: string;
  unit: MetricUnit;
  value: number;
  delta?: number;
  trend?: "up" | "down" | "stable";
  timestamp: number;
  labels?: Record<string, string>;
  history?: MetricDataPoint[];
}

// ─── Agent monitoring ─────────────────────────────────────────────────────────

export type AgentMonitorStatus =
  | "idle"
  | "running"
  | "error"
  | "unavailable"
  | "degraded";

export interface AgentMetricSnapshot {
  agent_id: string;
  agent_name: string;
  status: AgentMonitorStatus;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  active_tasks: number;
  queued_tasks: number;
  completed_tasks_1m: number;
  failed_tasks_1m: number;
  success_rate_1m: number;
  tokens_per_second: number;
  memory_mb: number;
  cpu_percent: number;
  timestamp: number;
}

export interface AgentHealthCheck {
  agent_id: string;
  status: "healthy" | "degraded" | "unhealthy";
  checks: Array<{
    name: string;
    status: "pass" | "fail" | "warn";
    message?: string;
    duration_ms: number;
  }>;
  checked_at: number;
}

// ─── System metrics ───────────────────────────────────────────────────────────

export interface SystemMetricSnapshot {
  cpu_percent: number;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_percent: number;
  disk_total_gb: number;
  disk_used_gb: number;
  disk_percent: number;
  network_rx_bytes_per_sec: number;
  network_tx_bytes_per_sec: number;
  active_connections: number;
  uptime_ms: number;
  timestamp: number;
}

// ─── LLM/Provider metrics ─────────────────────────────────────────────────────

export interface ProviderMetricSnapshot {
  provider_id: string;
  provider_name: string;
  is_available: boolean;
  latency_ms: number;
  requests_per_minute: number;
  tokens_per_minute: number;
  error_rate: number;
  rate_limit_remaining: number;
  rate_limit_reset_at: number;
  cost_usd_today: number;
  cost_usd_month: number;
  timestamp: number;
}

// ─── Cost tracking ────────────────────────────────────────────────────────────

export interface CostEvent {
  request_id: string;
  conversation_id?: string;
  agent_id?: string;
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
  timestamp: number;
}

export interface CostSummary {
  period: "hour" | "day" | "week" | "month";
  total_cost_usd: number;
  total_tokens: number;
  breakdown_by_provider: Record<string, number>;
  breakdown_by_model: Record<string, number>;
  breakdown_by_agent: Record<string, number>;
  computed_at: number;
}

// ─── Alerts ───────────────────────────────────────────────────────────────────

export type AlertSeverity = "info" | "warning" | "critical";

export type AlertCategory =
  | "performance"
  | "cost"
  | "error_rate"
  | "availability"
  | "security"
  | "capacity";

export interface MonitoringAlert {
  id: string;
  severity: AlertSeverity;
  category: AlertCategory;
  title: string;
  message: string;
  agent_id?: string;
  provider_id?: string;
  metric_name?: string;
  metric_value?: number;
  threshold?: number;
  is_resolved: boolean;
  fired_at: number;
  resolved_at?: number;
}

// ─── Real-time log streaming ──────────────────────────────────────────────────

export type LogLevel = "debug" | "info" | "warn" | "error" | "fatal";

export interface LogEntry {
  id: string;
  level: LogLevel;
  message: string;
  source: string;
  agent_id?: string;
  conversation_id?: string;
  task_id?: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

// ─── Channel names ────────────────────────────────────────────────────────────

export const MONITORING_CHANNELS = {
  agentMetrics: (agent_id: string) => `monitor:agent:${agent_id}`,
  agentHealth: (agent_id: string) => `monitor:health:${agent_id}`,
  allAgents: () => `monitor:agents:all`,
  system: () => `monitor:system`,
  provider: (provider_id: string) => `monitor:provider:${provider_id}`,
  allProviders: () => `monitor:providers:all`,
  costs: () => `monitor:costs`,
  alerts: () => `monitor:alerts`,
  logs: (filter?: string) => (filter ? `monitor:logs:${filter}` : `monitor:logs`),
  metrics: (metric_id: string) => `monitor:metric:${metric_id}`,
} as const;

// ─── Listener types ───────────────────────────────────────────────────────────

export interface MonitoringSocketListeners {
  onAgentMetrics?: (snapshot: AgentMetricSnapshot) => void;
  onAgentHealth?: (health: AgentHealthCheck) => void;
  onSystemMetrics?: (snapshot: SystemMetricSnapshot) => void;
  onProviderMetrics?: (snapshot: ProviderMetricSnapshot) => void;
  onCostEvent?: (event: CostEvent) => void;
  onCostSummary?: (summary: CostSummary) => void;
  onAlert?: (alert: MonitoringAlert) => void;
  onAlertResolved?: (alert: MonitoringAlert) => void;
  onLogEntry?: (entry: LogEntry) => void;
  onMetricUpdate?: (metric: Metric) => void;
}

// ─── Monitoring Socket ────────────────────────────────────────────────────────

export class MonitoringSocket {
  private readonly manager: SocketManager;
  private readonly activeSubscriptions = new Map<string, () => void>();

  constructor(config: SocketManagerConfig) {
    this.manager = new SocketManager(config);
  }

  getManager(): SocketManager {
    return this.manager;
  }

  // ── Agent subscriptions ────────────────────────────────────────────────────

  subscribeToAgent(
    agent_id: string,
    listeners: Pick<
      MonitoringSocketListeners,
      "onAgentMetrics" | "onAgentHealth"
    >
  ): () => void {
    const unsubscribers: Array<() => void> = [];

    if (listeners.onAgentMetrics) {
      unsubscribers.push(
        this.manager.subscribe<AgentMetricSnapshot>(
          MONITORING_CHANNELS.agentMetrics(agent_id),
          listeners.onAgentMetrics
        )
      );
    }

    if (listeners.onAgentHealth) {
      unsubscribers.push(
        this.manager.subscribe<AgentHealthCheck>(
          MONITORING_CHANNELS.agentHealth(agent_id),
          listeners.onAgentHealth
        )
      );
    }

    const key = `agent:${agent_id}`;
    const cleanup = () => unsubscribers.forEach((fn) => fn());
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, cleanup);

    return () => {
      cleanup();
      this.activeSubscriptions.delete(key);
    };
  }

  subscribeToAllAgents(
    onMetrics: (snapshot: AgentMetricSnapshot) => void
  ): () => void {
    const key = "agents:all";
    const unsub = this.manager.subscribe<AgentMetricSnapshot>(
      MONITORING_CHANNELS.allAgents(),
      onMetrics
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── System subscriptions ───────────────────────────────────────────────────

  subscribeToSystemMetrics(
    onMetrics: (snapshot: SystemMetricSnapshot) => void
  ): () => void {
    const key = "system";
    const unsub = this.manager.subscribe<SystemMetricSnapshot>(
      MONITORING_CHANNELS.system(),
      onMetrics
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Provider subscriptions ─────────────────────────────────────────────────

  subscribeToProvider(
    provider_id: string,
    onMetrics: (snapshot: ProviderMetricSnapshot) => void
  ): () => void {
    const key = `provider:${provider_id}`;
    const unsub = this.manager.subscribe<ProviderMetricSnapshot>(
      MONITORING_CHANNELS.provider(provider_id),
      onMetrics
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  subscribeToAllProviders(
    onMetrics: (snapshot: ProviderMetricSnapshot) => void
  ): () => void {
    const key = "providers:all";
    const unsub = this.manager.subscribe<ProviderMetricSnapshot>(
      MONITORING_CHANNELS.allProviders(),
      onMetrics
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Cost subscriptions ─────────────────────────────────────────────────────

  subscribeToCosts(
    listeners: Pick<
      MonitoringSocketListeners,
      "onCostEvent" | "onCostSummary"
    >
  ): () => void {
    const key = "costs";
    const unsub = this.manager.subscribe<CostEvent | CostSummary>(
      MONITORING_CHANNELS.costs(),
      (payload) => {
        const p = payload as unknown as Record<string, unknown>;
        if ("period" in p) {
          listeners.onCostSummary?.(payload as CostSummary);
        } else {
          listeners.onCostEvent?.(payload as CostEvent);
        }
      }
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Alert subscriptions ────────────────────────────────────────────────────

  subscribeToAlerts(
    listeners: Pick<
      MonitoringSocketListeners,
      "onAlert" | "onAlertResolved"
    >
  ): () => void {
    const key = "alerts";
    const unsub = this.manager.subscribe<MonitoringAlert>(
      MONITORING_CHANNELS.alerts(),
      (alert) => {
        if (alert.is_resolved) {
          listeners.onAlertResolved?.(alert);
        } else {
          listeners.onAlert?.(alert);
        }
      }
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Log streaming ──────────────────────────────────────────────────────────

  subscribeToLogs(
    onEntry: (entry: LogEntry) => void,
    filter?: string
  ): () => void {
    const key = `logs:${filter ?? "all"}`;
    const unsub = this.manager.subscribe<LogEntry>(
      MONITORING_CHANNELS.logs(filter),
      onEntry
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Custom metric subscriptions ────────────────────────────────────────────

  subscribeToMetric(
    metric_id: string,
    onMetric: (metric: Metric) => void
  ): () => void {
    const key = `metric:${metric_id}`;
    const unsub = this.manager.subscribe<Metric>(
      MONITORING_CHANNELS.metrics(metric_id),
      onMetric
    );
    this.activeSubscriptions.get(key)?.();
    this.activeSubscriptions.set(key, unsub);
    return () => {
      unsub();
      this.activeSubscriptions.delete(key);
    };
  }

  // ── Composite subscription ─────────────────────────────────────────────────

  subscribeAll(listeners: MonitoringSocketListeners): () => void {
    const unsubscribers: Array<() => void> = [];

    if (listeners.onSystemMetrics) {
      unsubscribers.push(
        this.subscribeToSystemMetrics(listeners.onSystemMetrics)
      );
    }

    if (listeners.onAgentMetrics) {
      unsubscribers.push(
        this.subscribeToAllAgents(listeners.onAgentMetrics)
      );
    }

    if (listeners.onProviderMetrics) {
      unsubscribers.push(
        this.subscribeToAllProviders(listeners.onProviderMetrics)
      );
    }

    if (listeners.onCostEvent || listeners.onCostSummary) {
      unsubscribers.push(
        this.subscribeToCosts({
          onCostEvent: listeners.onCostEvent,
          onCostSummary: listeners.onCostSummary,
        })
      );
    }

    if (listeners.onAlert || listeners.onAlertResolved) {
      unsubscribers.push(
        this.subscribeToAlerts({
          onAlert: listeners.onAlert,
          onAlertResolved: listeners.onAlertResolved,
        })
      );
    }

    if (listeners.onLogEntry) {
      unsubscribers.push(this.subscribeToLogs(listeners.onLogEntry));
    }

    return () => unsubscribers.forEach((fn) => fn());
  }

  // ── Request historical data ────────────────────────────────────────────────

  async requestCostSummary(
    period: CostSummary["period"]
  ): Promise<void> {
    await this.manager.send(
      "event",
      { action: "request_summary", period },
      { channel: MONITORING_CHANNELS.costs() }
    );
  }

  async requestAgentHealthCheck(agent_id: string): Promise<void> {
    await this.manager.send(
      "event",
      { action: "health_check", agent_id },
      { channel: MONITORING_CHANNELS.agentHealth(agent_id) }
    );
  }

  unsubscribeAll(): void {
    this.activeSubscriptions.forEach((fn) => fn());
    this.activeSubscriptions.clear();
  }

  destroy(): void {
    this.unsubscribeAll();
    this.manager.destroy();
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export const createMonitoringSocket = (
  config: SocketManagerConfig
): MonitoringSocket => new MonitoringSocket(config);