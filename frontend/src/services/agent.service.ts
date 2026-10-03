import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const AgentStatus = {
  Idle: "idle",
  Running: "running",
  Paused: "paused",
  Error: "error",
  Completed: "completed",
} as const;

export type AgentStatus = (typeof AgentStatus)[keyof typeof AgentStatus];

export const AgentType = {
  Conversational: "conversational",
  TaskBased: "task_based",
  Autonomous: "autonomous",
  Supervisor: "supervisor",
} as const;

export type AgentType = (typeof AgentType)[keyof typeof AgentType];

// ── Types (matching backend AgentInfo) ───────────────────────────────────────
export interface AgentInfo {
  agent_id: string;
  name: string;
  status: string;
  capabilities: string[];
  parent_agent_id?: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface SpawnAgentPayload {
  name: string;
  agent_type?: string;
  capabilities?: string[];
  parent_agent_id?: string;
  config?: Record<string, unknown>;
  session_id?: string;
}

export interface DelegateTaskPayload {
  from_agent_id: string;
  to_agent_id: string;
  task_description: string;
  context?: Record<string, unknown>;
  priority?: number;
}

export interface ListAgentsParams {
  status_filter?: string;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface SpawnAgentResponse {
  agent: AgentInfo;
}

export interface AgentListResponse {
  agents: AgentInfo[];
  total: number;
}

export interface ShutdownResponse {
  agent_id: string;
  status: string;
  shutdown_at: string;
}

export interface DelegateTaskResponse {
  delegation_id: string;
  from_agent_id: string;
  to_agent_id: string;
  status: string;
  created_at: string;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  BASE: "/api/v1/agents",
  BY_ID: (id: string) => `/api/v1/agents/${id}`,
  SPAWN: "/api/v1/agents/spawn",
  SHUTDOWN: (id: string) => `/api/v1/agents/${id}/shutdown`,
  DELEGATE: "/api/v1/agents/delegate",
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const createAgent = async (
  payload: SpawnAgentPayload
): Promise<SpawnAgentResponse> => {
  const response: AxiosResponse<SpawnAgentResponse> =
    await api.post<SpawnAgentResponse>(ENDPOINTS.SPAWN, payload);
  return response.data;
};

const listAgents = async (
  params?: ListAgentsParams
): Promise<AgentListResponse> => {
  const response: AxiosResponse<AgentListResponse> =
    await api.get<AgentListResponse>(ENDPOINTS.BASE, { params });
  return response.data;
};

const getAgent = async (id: string): Promise<AgentInfo> => {
  const response: AxiosResponse<AgentInfo> =
    await api.get<AgentInfo>(ENDPOINTS.BY_ID(id));
  return response.data;
};

const shutdownAgent = async (
  id: string,
  force?: boolean
): Promise<ShutdownResponse> => {
  const response: AxiosResponse<ShutdownResponse> =
    await api.post<ShutdownResponse>(ENDPOINTS.SHUTDOWN(id), null, {
      params: { ...(force !== undefined && { force }) },
    });
  return response.data;
};

const delegateTask = async (
  payload: DelegateTaskPayload
): Promise<DelegateTaskResponse> => {
  const response: AxiosResponse<DelegateTaskResponse> =
    await api.post<DelegateTaskResponse>(ENDPOINTS.DELEGATE, payload);
  return response.data;
};

export const AgentService = {
  createAgent,
  listAgents,
  getAgent,
  shutdownAgent,
  delegateTask,
} as const;

export default AgentService;
