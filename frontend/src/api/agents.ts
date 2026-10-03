import api from "@/api/client";

export interface AgentInfo {
  agent_id: string;
  name: string;
  status: string;
  capabilities: string[];
  parent_agent_id?: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface AgentListResponse {
  agents: AgentInfo[];
  total: number;
}

export interface SpawnAgentPayload {
  name: string;
  agent_type?: string;
  capabilities?: string[];
  parent_agent_id?: string;
  config?: Record<string, unknown>;
  session_id?: string;
}

export interface SpawnAgentResponse {
  agent: AgentInfo;
}

export interface ShutdownResponse {
  agent_id: string;
  status: string;
  shutdown_at: string;
}

export interface DelegateTaskPayload {
  from_agent_id: string;
  to_agent_id: string;
  task_description: string;
  context?: Record<string, unknown>;
  priority?: number;
}

export interface DelegateTaskResponse {
  delegation_id: string;
  from_agent_id: string;
  to_agent_id: string;
  status: string;
  created_at: string;
}

const ENDPOINTS = {
  BASE: "/api/v1/agents",
  BY_ID: (id: string) => `/api/v1/agents/${id}`,
  SPAWN: "/api/v1/agents/spawn",
  SHUTDOWN: (id: string) => `/api/v1/agents/${id}/shutdown`,
  DELEGATE: "/api/v1/agents/delegate",
} as const;

export const listAgents = async (status_filter?: string): Promise<AgentListResponse> => {
  const { data } = await api.get<AgentListResponse>(ENDPOINTS.BASE, { params: { status_filter } });
  return data;
};

export const getAgent = async (id: string): Promise<AgentInfo> => {
  const { data } = await api.get<AgentInfo>(ENDPOINTS.BY_ID(id));
  return data;
};

export const spawnAgent = async (payload: SpawnAgentPayload): Promise<SpawnAgentResponse> => {
  const { data } = await api.post<SpawnAgentResponse>(ENDPOINTS.SPAWN, payload);
  return data;
};

export const shutdownAgent = async (id: string, force?: boolean): Promise<ShutdownResponse> => {
  const { data } = await api.post<ShutdownResponse>(ENDPOINTS.SHUTDOWN(id), null, { params: { force } });
  return data;
};

export const delegateTask = async (payload: DelegateTaskPayload): Promise<DelegateTaskResponse> => {
  const { data } = await api.post<DelegateTaskResponse>(ENDPOINTS.DELEGATE, payload);
  return data;
};
