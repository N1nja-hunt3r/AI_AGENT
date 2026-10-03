import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const AlertSeverity = {
  Info: "info",
  Warning: "warning",
  Error: "error",
  Critical: "critical",
} as const;

export type AlertSeverity = (typeof AlertSeverity)[keyof typeof AlertSeverity];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface DashboardSummary {
  activeAgents: number;
  activeSessions: number;
  requestsPerMinute: number;
  errorRate: number;
  avgLatencyMs: number;
  costTodayUsd: number;
  generatedAt: string;
}

export interface ProfilerSnapshot {
  component: string;
  cpuPercent: number;
  memoryMb: number;
  openConnections: number;
  capturedAt: string;
}

export interface Alert {
  alertId: string;
  severity: string;
  title: string;
  description: string;
  component: string;
  triggeredAt: string;
  resolved: boolean;
  resolvedAt?: string;
}

export interface AlertListResponse {
  alerts: Alert[];
  total: number;
}

export interface AlertResolveResponse {
  alertId: string;
  resolved: boolean;
  resolvedAt: string;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  requestId?: string;
  extra: Record<string, unknown>;
}

export interface LogListResponse {
  entries: LogEntry[];
  total: number;
  page: number;
  pageSize: number;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface GetLogsParams {
  level?: string;
  requestId?: string;
  search?: string;
  page?: number;
  pageSize?: number;
}

export interface ListAlertsParams {
  severity?: string;
  resolved?: boolean;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  METRICS: "/api/v1/monitoring/metrics",
  DASHBOARD: "/api/v1/monitoring/dashboard",
  LOGS: "/api/v1/monitoring/logs",
  PROFILER: "/api/v1/monitoring/profiler",
  ALERTS: "/api/v1/monitoring/alerts",
  RESOLVE: (id: string) => `/api/v1/monitoring/alerts/${id}/resolve`,
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const getMetrics = async (): Promise<string> => {
  const response: AxiosResponse<string> =
    await api.get<string>(ENDPOINTS.METRICS);
  return response.data;
};

const getDashboard = async (): Promise<DashboardSummary> => {
  const response: AxiosResponse<DashboardSummary> =
    await api.get<DashboardSummary>(ENDPOINTS.DASHBOARD);
  return response.data;
};

const getProfilerSnapshot = async (
  component?: string
): Promise<ProfilerSnapshot[]> => {
  const response: AxiosResponse<ProfilerSnapshot[]> =
    await api.get<ProfilerSnapshot[]>(ENDPOINTS.PROFILER, {
      params: { component },
    });
  return response.data;
};

const listAlerts = async (
  params?: ListAlertsParams
): Promise<AlertListResponse> => {
  const response: AxiosResponse<AlertListResponse> =
    await api.get<AlertListResponse>(ENDPOINTS.ALERTS, { params });
  return response.data;
};

const resolveAlert = async (id: string): Promise<AlertResolveResponse> => {
  const response: AxiosResponse<AlertResolveResponse> =
    await api.post<AlertResolveResponse>(ENDPOINTS.RESOLVE(id));
  return response.data;
};

const getLogs = async (params?: GetLogsParams): Promise<LogListResponse> => {
  const response: AxiosResponse<LogListResponse> =
    await api.get<LogListResponse>(ENDPOINTS.LOGS, { params });
  return response.data;
};

export const MonitoringService = {
  getMetrics,
  getDashboard,
  getProfilerSnapshot,
  listAlerts,
  resolveAlert,
  getLogs,
} as const;

export default MonitoringService;
