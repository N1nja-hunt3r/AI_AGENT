import api from "@/api/client";

export const TaskStatus = { Draft: "draft", Active: "active", Paused: "paused", Archived: "archived" } as const;
export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

export interface AutomationTaskInfo {
  task_id: string;
  name: string;
  status: string;
  schedule?: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface CreateTaskPayload {
  name: string;
  description?: string;
  schedule?: string;
  payload?: Record<string, unknown>;
  enabled?: boolean;
}

export interface CreateTaskResponse {
  task: AutomationTaskInfo;
}

export interface TaskListResponse {
  tasks: AutomationTaskInfo[];
  total: number;
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

const ENDPOINTS = {
  TASKS: "/api/v1/automation/tasks",
  TASK_BY_ID: (id: string) => `/api/v1/automation/tasks/${id}`,
  PAUSE: (id: string) => `/api/v1/automation/tasks/${id}/pause`,
  RESUME: (id: string) => `/api/v1/automation/tasks/${id}/resume`,
} as const;

export const createTask = async (payload: CreateTaskPayload): Promise<CreateTaskResponse> => {
  const { data } = await api.post<CreateTaskResponse>(ENDPOINTS.TASKS, payload);
  return data;
};

export const listTasks = async (status_filter?: string): Promise<TaskListResponse> => {
  const { data } = await api.get<TaskListResponse>(ENDPOINTS.TASKS, { params: { status_filter } });
  return data;
};

export const getTask = async (taskId: string): Promise<AutomationTaskInfo> => {
  const { data } = await api.get<AutomationTaskInfo>(ENDPOINTS.TASK_BY_ID(taskId));
  return data;
};

export const deleteTask = async (taskId: string): Promise<DeleteTaskResponse> => {
  const { data } = await api.delete<DeleteTaskResponse>(ENDPOINTS.TASK_BY_ID(taskId));
  return data;
};

export const pauseTask = async (taskId: string): Promise<TaskActionResponse> => {
  const { data } = await api.post<TaskActionResponse>(ENDPOINTS.PAUSE(taskId));
  return data;
};

export const resumeTask = async (taskId: string): Promise<TaskActionResponse> => {
  const { data } = await api.post<TaskActionResponse>(ENDPOINTS.RESUME(taskId));
  return data;
};
