import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const TaskStatus = {
  Draft: "draft",
  Active: "active",
  Paused: "paused",
  Archived: "archived",
} as const;

export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

export const TriggerType = {
  Manual: "manual",
  Schedule: "schedule",
  Webhook: "webhook",
  Event: "event",
} as const;

export type TriggerType = (typeof TriggerType)[keyof typeof TriggerType];

export const StepType = {
  LLM: "llm",
  Tool: "tool",
  Condition: "condition",
  Transform: "transform",
  HttpRequest: "http_request",
  Delay: "delay",
  SubWorkflow: "sub_workflow",
} as const;

export type StepType = (typeof StepType)[keyof typeof StepType];

export const RunStatus = {
  Queued: "queued",
  Running: "running",
  Success: "success",
  Failed: "failed",
  Cancelled: "cancelled",
  TimedOut: "timed_out",
} as const;

export type RunStatus = (typeof RunStatus)[keyof typeof RunStatus];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface TaskStep {
  id: string;
  name: string;
  type: StepType;
  order: number;
  config: Record<string, unknown>;
  nextStepId?: string;
  onErrorStepId?: string;
}

export interface TaskTrigger {
  type: TriggerType;
  schedule?: string;
  webhookUrl?: string;
  eventName?: string;
  config?: Record<string, unknown>;
}

export interface Task {
  id: string;
  name: string;
  description: string;
  status: TaskStatus;
  trigger: TaskTrigger;
  steps: TaskStep[];
  version: number;
  createdAt: string;
  updatedAt: string;
  lastRunAt?: string;
  metadata?: Record<string, unknown>;
}

export interface TaskRun {
  id: string;
  taskId: string;
  status: RunStatus;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  stepResults: StepResult[];
  startedAt: string;
  completedAt?: string;
  errorMessage?: string;
  durationMs?: number;
}

export interface StepResult {
  stepId: string;
  status: RunStatus;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  startedAt: string;
  completedAt?: string;
  errorMessage?: string;
  durationMs?: number;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface CreateTaskPayload {
  name: string;
  description?: string;
  schedule?: string;
  payload?: Record<string, unknown>;
  enabled?: boolean;
}

export interface ListTasksParams {
  status?: TaskStatus;
  page?: number;
  limit?: number;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface TaskResponse {
  task: Task;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  page: number;
  limit: number;
}

export interface DeleteTaskResponse {
  task_id: string;
  deleted: boolean;
}

export interface TaskActionResponse {
  task_id: string;
  status: string;
  updated_at: string;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  BASE: "/api/v1/automation/tasks",
  BY_ID: (id: string) => `/api/v1/automation/tasks/${id}`,
  PAUSE: (id: string) => `/api/v1/automation/tasks/${id}/pause`,
  RESUME: (id: string) => `/api/v1/automation/tasks/${id}/resume`,
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const createTask = async (
  payload: CreateTaskPayload
): Promise<TaskResponse> => {
  const response: AxiosResponse<TaskResponse> =
    await api.post<TaskResponse>(ENDPOINTS.BASE, payload);
  return response.data;
};

const listTasks = async (
  params?: ListTasksParams
): Promise<TaskListResponse> => {
  const response: AxiosResponse<TaskListResponse> =
    await api.get<TaskListResponse>(ENDPOINTS.BASE, { params });
  return response.data;
};

const getTask = async (id: string): Promise<TaskResponse> => {
  const response: AxiosResponse<TaskResponse> =
    await api.get<TaskResponse>(ENDPOINTS.BY_ID(id));
  return response.data;
};

const deleteTask = async (
  id: string
): Promise<DeleteTaskResponse> => {
  const response: AxiosResponse<DeleteTaskResponse> =
    await api.delete<DeleteTaskResponse>(ENDPOINTS.BY_ID(id));
  return response.data;
};

const pauseTask = async (id: string): Promise<TaskActionResponse> => {
  const response: AxiosResponse<TaskActionResponse> =
    await api.post<TaskActionResponse>(ENDPOINTS.PAUSE(id));
  return response.data;
};

const resumeTask = async (id: string): Promise<TaskActionResponse> => {
  const response: AxiosResponse<TaskActionResponse> =
    await api.post<TaskActionResponse>(ENDPOINTS.RESUME(id));
  return response.data;
};

export const AutomationService = {
  createTask,
  listTasks,
  getTask,
  deleteTask,
  pauseTask,
  resumeTask,
} as const;

export default AutomationService;
