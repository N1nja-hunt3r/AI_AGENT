// frontend/src/types/monitoring.ts

import type { AgentStatus } from "./agent";

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum MetricUnit {
  Count = "count",
  Percent = "percent",
  Milliseconds = "ms",
  Seconds = "s",
  Bytes = "bytes",
  Kilobytes = "kb",
  Megabytes = "mb",
  Gigabytes = "gb",
  BytesPerSec = "bytes_per_sec",
  Tokens = "tokens",
  TokensPerSec = "tokens_per_sec",
  USD = "usd",
  RequestsPerSec = "requests_per_sec",
  RequestsPerMin = "requests_per_min",
  Ratio = "ratio",
}

export enum MetricTrend {
  Up = "up",
  Down = "down",
  Stable = "stable",
}

export enum AlertSeverity {
  Info = "info",
  Warning = "warning",
  Critical = "critical",
}

export enum AlertCategory {
  Performance = "performance",
  Cost = "cost",
  ErrorRate = "error_rate",
  Availability = "availability",
  Security = "security",
  Capacity = "capacity",
  Latency = "latency",
  RateLimit = "rate_limit",
}

export enum AlertStatus {
  Active = "active",
  Resolved = "resolved",
  Acknowledged = "acknowledged",
  Silenced = "silenced",
}

export enum LogLevel {
  Debug = "debug",
  Info = "info",
  Warn = "warn",
  Error = "error",
  Fatal = "fatal",
}

export enum TimeWindow {
  OneMinute = "1m",
  FiveMinutes = "5m",
  FifteenMinutes = "15m",
  OneHour = "1h",
  SixHours = "6h",
  TwentyFourHours = "24h",
  SevenDays = "7d",
  ThirtyDays = "30d",
}

export enum CostPeriod {
  Hourly = "hour",
  Daily = "day",
  Weekly = "week",
  Monthly = "month",
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface MetricDataPoint {
  timestamp: number;
  value: number;
  labels?: Record<string, string>;
}

export interface MetricSeries {
  metric_id: string;
  name: string;
  unit: MetricUnit;
  data_points: MetricDataPoint[];
  window: TimeWindow;
}

export interface Metric {
  id: string;
  name: string;
  description?: string;
  unit: MetricUnit;
  value: number;
  previous_value?: number;
  delta?: number;
  delta_percent?: number;
  trend?: MetricTrend;
  labels?: Record<string, string>;
  timestamp: number;
}

// ─── System metrics ───────────────────────────────────────────────────────────

export interface SystemMetricSnapshot {
  cpu_percent: number;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_percent: number;
  swap_total_mb: number;
  swap_used_mb: number;
  disk_total_gb: number;
  disk_used_gb: number;
  disk_percent: number;
  disk_read_bytes_per_sec: number;
  disk_write_bytes_per_sec: number;
  network_rx_bytes_per_sec: number;
  network_tx_bytes_per_sec: number;
  active_connections: number;
  open_file_descriptors: number;
  process_count: number;
  uptime_ms: number;
  timestamp: number;
}

// ─── Agent monitoring ─────────────────────────────────────────────────────────

export interface AgentMetricSnapshot {
  agent_id: string;
  agent_name: string;
  status: AgentStatus;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  active_tasks: number;
  queued_tasks: number;
  completed_tasks_1m: number;
  failed_tasks_1m: number;
  success_rate_1m: number;
  tokens_per_second: number;
  total_tokens_used: number;
  total_cost_usd: number;
  memory_mb: number;
  cpu_percent: number;
  timestamp: number;
}

// ─── Provider monitoring ──────────────────────────────────────────────────────

export interface ProviderMetricSnapshot {
  provider_id: string;
  provider_name: string;
  is_available: boolean;
  latency_ms: number;
  latency_p95_ms: number;
  requests_per_minute: number;
  tokens_per_minute: number;
  error_rate: number;
  error_count_1h: number;
  rate_limit_remaining: number;
  rate_limit_total: number;
  rate_limit_reset_at: number;
  cost_usd_today: number;
  cost_usd_month: number;
  models_available: number;
  timestamp: number;
}

// ─── Cost ─────────────────────────────────────────────────────────────────────

export interface CostEvent {
  request_id: string;
  conversation_id?: string;
  agent_id?: string;
  task_id?: string;
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  input_cost_usd: number;
  output_cost_usd: number;
  total_cost_usd: number;
  timestamp: number;
}

export interface CostBreakdown {
  by_provider: Record<string, number>;
  by_model: Record<string, number>;
  by_agent: Record<string, number>;
  by_conversation: Record<string, number>;
}

export interface CostSummary {
  period: CostPeriod;
  total_cost_usd: number;
  total_tokens: number;
  total_requests: number;
  avg_cost_per_request_usd: number;
  breakdown: CostBreakdown;
  budget_limit_usd?: number;
  budget_used_percent?: number;
  computed_at: number;
}

// ─── Alerts ───────────────────────────────────────────────────────────────────

export interface AlertThreshold {
  metric_name: string;
  operator: "gt" | "lt" | "gte" | "lte" | "eq" | "neq";
  value: number;
  unit: MetricUnit;
  window: TimeWindow;
}

export interface MonitoringAlert {
  id: string;
  severity: AlertSeverity;
  category: AlertCategory;
  status: AlertStatus;
  title: string;
  message: string;
  agent_id?: string;
  provider_id?: string;
  threshold?: AlertThreshold;
  current_value?: number;
  labels?: Record<string, string>;
  acknowledged_by?: string;
  silenced_until?: number;
  fired_at: number;
  resolved_at?: number;
  acknowledged_at?: number;
  metadata?: Record<string, unknown>;
}

// ─── Logs ────────────────────────────────────────────────────────────────────

export interface LogEntry {
  id: string;
  level: LogLevel;
  message: string;
  source: string;
  agent_id?: string;
  task_id?: string;
  conversation_id?: string;
  request_id?: string;
  stack_trace?: string;
  duration_ms?: number;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface LogFilter {
  levels?: LogLevel[];
  sources?: string[];
  agent_id?: string;
  date_from?: number;
  date_to?: number;
  search?: string;
  limit?: number;
  offset?: number;
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export interface MonitoringDashboard {
  system: SystemMetricSnapshot;
  agents: AgentMetricSnapshot[];
  providers: ProviderMetricSnapshot[];
  cost_summary: CostSummary;
  active_alerts: MonitoringAlert[];
  recent_logs: LogEntry[];
  computed_at: number;
}