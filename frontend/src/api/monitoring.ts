import api from "@/api/client";

export const AlertSeverity = { Info: "info", Warning: "warning", Error: "error", Critical: "critical" } as const;
export type AlertSeverity = (typeof AlertSeverity)[keyof typeof AlertSeverity];

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

export interface LogQueryResponse {
  entries: LogEntry[];
  total: number;
  page: number;
  pageSize: number;
}

export interface QueryLogsParams {
  level?: string;
  logger?: string;
  requestId?: string;
  search?: string;
  from?: string;
  to?: string;
  page?: number;
  pageSize?: number;
}

const ENDPOINTS = {
  METRICS: "/api/v1/monitoring/metrics",
  DASHBOARD: "/api/v1/monitoring/dashboard",
  LOGS: "/api/v1/monitoring/logs",
  PROFILER: "/api/v1/monitoring/profiler",
  ALERTS: "/api/v1/monitoring/alerts",
  RESOLVE: (id: string) => `/api/v1/monitoring/alerts/${id}/resolve`,
} as const;

export const getMetrics = async (): Promise<string> => {
  const { data } = await api.get<string>(ENDPOINTS.METRICS);
  return data;
};

export const getDashboard = async (): Promise<DashboardSummary> => {
  const { data } = await api.get<DashboardSummary>(ENDPOINTS.DASHBOARD);
  return data;
};

export const getProfilerSnapshot = async (component?: string): Promise<ProfilerSnapshot[]> => {
  const { data } = await api.get<ProfilerSnapshot[]>(ENDPOINTS.PROFILER, { params: { component } });
  return data;
};

export const listAlerts = async (params?: { severity?: string; resolved?: boolean }): Promise<AlertListResponse> => {
  const { data } = await api.get<AlertListResponse>(ENDPOINTS.ALERTS, { params });
  return data;
};

export const resolveAlert = async (alertId: string): Promise<AlertResolveResponse> => {
  const { data } = await api.post<AlertResolveResponse>(ENDPOINTS.RESOLVE(alertId));
  return data;
};

export const queryLogs = async (params?: QueryLogsParams): Promise<LogQueryResponse> => {
  const { data } = await api.get<LogQueryResponse>(ENDPOINTS.LOGS, { params });
  return data;
};
