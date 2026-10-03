import { create } from "zustand";
import { devtools, subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ─── Types ────────────────────────────────────────────────────────────────────

export type AgentStatus =
  | "idle"
  | "initializing"
  | "running"
  | "paused"
  | "stopping"
  | "stopped"
  | "error"
  | "degraded";

export type TaskStatus = "queued" | "running" | "completed" | "failed" | "error" | "cancelled" | "retrying";

export type TaskPriority = "low" | "normal" | "high" | "critical";

export type ToolCallStatus = "pending" | "executing" | "success" | "error";

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
  status: ToolCallStatus;
  error?: string;
  durationMs?: number;
  startedAt: number;
  completedAt?: number;
}

export interface AgentTask {
  id: string;
  agentId: string;
  type: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  progress: number; // 0–100
  result?: unknown;
  error?: string;
  retryCount: number;
  maxRetries: number;
  toolCalls: ToolCall[];
  parentTaskId?: string;
  childTaskIds: string[];
  metadata?: Record<string, unknown>;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  estimatedDurationMs?: number;
}

export interface HealthCheckResult {
  status: "healthy" | "degraded" | "unhealthy";
  message?: string;
  checkedAt: number;
  details?: Record<string, unknown>;
}

export interface LatencyMetrics {
  p50: number;
  p95: number;
  p99: number;
  avg: number;
  samples: number;
  windowMs: number;
  updatedAt: number;
}

export interface ResourceUsage {
  cpuPercent: number;
  memoryMb: number;
  activeConnections: number;
  queueDepth: number;
  updatedAt: number;
}

export interface Agent {
  id: string;
  name: string;
  description?: string;
  version: string;
  status: AgentStatus;
  capabilities: string[];
  health: HealthCheckResult | null;
  latency: LatencyMetrics | null;
  resources: ResourceUsage | null;
  currentTaskId: string | null;
  errorMessage?: string;
  metadata?: Record<string, unknown>;
  registeredAt: number;
  lastSeenAt: number;
}

export interface AgentState {
  // Agents
  agents: Record<string, Agent>;
  selectedAgentId: string | null;

  // Tasks
  tasks: Record<string, AgentTask>;
  taskQueue: string[]; // ordered task IDs

  // Global metrics
  globalHealth: "healthy" | "degraded" | "unhealthy" | "unknown";
  isPolling: boolean;
  pollingIntervalMs: number;

  // Error
  error: string | null;
}

export interface AgentActions {
  // Agent management
  registerAgent: (agent: Omit<Agent, "health" | "latency" | "resources" | "registeredAt" | "lastSeenAt">) => void;
  unregisterAgent: (agentId: string) => void;
  updateAgentStatus: (agentId: string, status: AgentStatus, errorMessage?: string) => void;
  updateAgentHealth: (agentId: string, health: HealthCheckResult) => void;
  updateAgentLatency: (agentId: string, latency: LatencyMetrics) => void;
  updateAgentResources: (agentId: string, resources: ResourceUsage) => void;
  heartbeat: (agentId: string) => void;
  setAgents: (agents: Agent[]) => void;
  selectAgent: (agentId: string | null) => void;

  // Task management
  enqueueTask: (task: Omit<AgentTask, "id" | "status" | "progress" | "retryCount" | "toolCalls" | "childTaskIds" | "createdAt">) => string;
  updateTask: (taskId: string, patch: Partial<Omit<AgentTask, "id" | "createdAt">>) => void;
  startTask: (taskId: string) => void;
  completeTask: (taskId: string, result?: unknown) => void;
  failTask: (taskId: string, error: string) => void;
  cancelTask: (taskId: string) => void;
  retryTask: (taskId: string) => void;
  setTaskProgress: (taskId: string, progress: number) => void;
  dequeueTask: (taskId: string) => void;
  clearCompletedTasks: () => void;

  // Tool call management
  addToolCall: (taskId: string, toolCall: Omit<ToolCall, "id" | "startedAt">) => string;
  updateToolCall: (taskId: string, toolCallId: string, patch: Partial<Omit<ToolCall, "id" | "startedAt">>) => void;
  completeToolCall: (taskId: string, toolCallId: string, output: unknown) => void;
  failToolCall: (taskId: string, toolCallId: string, error: string) => void;

  // Polling
  setPolling: (isPolling: boolean) => void;
  setPollingInterval: (intervalMs: number) => void;

  // Global health
  computeGlobalHealth: () => void;

  // Error
  setError: (error: string | null) => void;

  // Derived
  getAgentById: (agentId: string) => Agent | null;
  getTasksByAgent: (agentId: string) => AgentTask[];
  getActiveTasksByAgent: (agentId: string) => AgentTask[];
  getPendingTasks: () => AgentTask[];
  getRunningTasks: () => AgentTask[];
}

export type AgentStore = AgentState & AgentActions;

// ─── Helpers ─────────────────────────────────────────────────────────────────

const generateId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

const ACTIVE_TASK_STATUSES: TaskStatus[] = ["queued", "running", "retrying"];

// ─── Store ────────────────────────────────────────────────────────────────────

export const useAgentStore = create<AgentStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // ── Initial state ────────────────────────────────────────────────────
        agents: {},
        selectedAgentId: null,
        tasks: {},
        taskQueue: [],
        globalHealth: "unknown",
        isPolling: false,
        pollingIntervalMs: 5_000,
        error: null,

        // ── Agent management ─────────────────────────────────────────────────
        registerAgent: (agent) => {
          const timestamp = Date.now();
          set((state) => {
            state.agents[agent.id] = {
              ...agent,
              health: null,
              latency: null,
              resources: null,
              registeredAt: timestamp,
              lastSeenAt: timestamp,
            };
          });
        },

        unregisterAgent: (agentId) => {
          set((state) => {
            delete state.agents[agentId];
            if (state.selectedAgentId === agentId) {
              state.selectedAgentId = null;
            }
          });
        },

        updateAgentStatus: (agentId, status, errorMessage) => {
          set((state) => {
            const agent = state.agents[agentId];
            if (agent) {
              agent.status = status;
              agent.errorMessage = errorMessage;
              agent.lastSeenAt = Date.now();
            }
          });
        },

        updateAgentHealth: (agentId, health) => {
          set((state) => {
            const agent = state.agents[agentId];
            if (agent) {
              agent.health = health;
              agent.lastSeenAt = Date.now();
            }
          });
          get().computeGlobalHealth();
        },

        updateAgentLatency: (agentId, latency) => {
          set((state) => {
            const agent = state.agents[agentId];
            if (agent) {
              agent.latency = latency;
            }
          });
        },

        updateAgentResources: (agentId, resources) => {
          set((state) => {
            const agent = state.agents[agentId];
            if (agent) {
              agent.resources = resources;
            }
          });
        },

        heartbeat: (agentId) => {
          set((state) => {
            const agent = state.agents[agentId];
            if (agent) {
              agent.lastSeenAt = Date.now();
            }
          });
        },

        setAgents: (agents) => {
          set((state) => {
            state.agents = Object.fromEntries(agents.map((a) => [a.id, a]));
          });
          get().computeGlobalHealth();
        },

        selectAgent: (agentId) => {
          set((state) => {
            state.selectedAgentId = agentId;
          });
        },

        // ── Task management ──────────────────────────────────────────────────
        enqueueTask: (task) => {
          const id = generateId();
          const newTask: AgentTask = {
            ...task,
            id,
            status: "queued",
            progress: 0,
            retryCount: 0,
            toolCalls: [],
            childTaskIds: [],
            createdAt: Date.now(),
          };

          set((state) => {
            state.tasks[id] = newTask;

            // Insert by priority
            const priorityOrder: Record<TaskPriority, number> = {
              critical: 0,
              high: 1,
              normal: 2,
              low: 3,
            };

            const insertIndex = state.taskQueue.findIndex((qid) => {
              const qtask = state.tasks[qid];
              return qtask ? priorityOrder[qtask.priority] > priorityOrder[task.priority] : false;
            });

            if (insertIndex === -1) {
              state.taskQueue.push(id);
            } else {
              state.taskQueue.splice(insertIndex, 0, id);
            }

            // Link to parent
            if (task.parentTaskId) {
              const parent = state.tasks[task.parentTaskId];
              if (parent) {
                parent.childTaskIds.push(id);
              }
            }

            // Assign to agent
            const agent = state.agents[task.agentId];
            if (agent && !agent.currentTaskId) {
              agent.currentTaskId = id;
            }
          });

          return id;
        },

        updateTask: (taskId, patch) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              Object.assign(task, patch);
            }
          });
        },

        startTask: (taskId) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              task.status = "running";
              task.startedAt = Date.now();
            }
          });
        },

        completeTask: (taskId, result) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              task.status = "completed";
              task.progress = 100;
              task.result = result;
              task.completedAt = Date.now();

              const agent = state.agents[task.agentId];
              if (agent?.currentTaskId === taskId) {
                agent.currentTaskId = null;
                agent.status = "idle";
              }
            }
          });
        },

        failTask: (taskId, error) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              task.status = "failed";
              task.error = error;
              task.completedAt = Date.now();

              const agent = state.agents[task.agentId];
              if (agent?.currentTaskId === taskId) {
                agent.currentTaskId = null;
                agent.status = "error";
                agent.errorMessage = error;
              }
            }
          });
        },

        cancelTask: (taskId) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task && ACTIVE_TASK_STATUSES.includes(task.status)) {
              task.status = "cancelled";
              task.completedAt = Date.now();
              state.taskQueue = state.taskQueue.filter((id) => id !== taskId);
            }
          });
        },

        retryTask: (taskId) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task && (task.status === "failed" || task.status === "error")) {
              task.status = "retrying";
              task.retryCount += 1;
              task.error = undefined;
              task.completedAt = undefined;
              if (!state.taskQueue.includes(taskId)) {
                state.taskQueue.unshift(taskId);
              }
            }
          });
        },

        setTaskProgress: (taskId, progress) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              task.progress = Math.max(0, Math.min(100, progress));
            }
          });
        },

        dequeueTask: (taskId) => {
          set((state) => {
            state.taskQueue = state.taskQueue.filter((id) => id !== taskId);
          });
        },

        clearCompletedTasks: () => {
          set((state) => {
            const completedStatuses: TaskStatus[] = ["completed", "cancelled", "failed"];
            const completedIds = Object.keys(state.tasks).filter(
              (id) => completedStatuses.includes(state.tasks[id]!.status)
            );
            completedIds.forEach((id) => {
              delete state.tasks[id];
            });
          });
        },

        // ── Tool call management ─────────────────────────────────────────────
        addToolCall: (taskId, toolCall) => {
          const id = generateId();
          set((state) => {
            const task = state.tasks[taskId];
            if (task) {
              task.toolCalls.push({
                ...toolCall,
                id,
                startedAt: Date.now(),
              });
            }
          });
          return id;
        },

        updateToolCall: (taskId, toolCallId, patch) => {
          set((state) => {
            const task = state.tasks[taskId];
            if (!task) return;
            const toolCall = task.toolCalls.find((tc) => tc.id === toolCallId);
            if (toolCall) {
              Object.assign(toolCall, patch);
            }
          });
        },

        completeToolCall: (taskId, toolCallId, output) => {
          const completedAt = Date.now();
          set((state) => {
            const task = state.tasks[taskId];
            if (!task) return;
            const toolCall = task.toolCalls.find((tc) => tc.id === toolCallId);
            if (toolCall) {
              toolCall.status = "success";
              toolCall.output = output;
              toolCall.completedAt = completedAt;
              toolCall.durationMs = completedAt - toolCall.startedAt;
            }
          });
        },

        failToolCall: (taskId, toolCallId, error) => {
          const completedAt = Date.now();
          set((state) => {
            const task = state.tasks[taskId];
            if (!task) return;
            const toolCall = task.toolCalls.find((tc) => tc.id === toolCallId);
            if (toolCall) {
              toolCall.status = "error";
              toolCall.error = error;
              toolCall.completedAt = completedAt;
              toolCall.durationMs = completedAt - toolCall.startedAt;
            }
          });
        },

        // ── Polling ───────────────────────────────────────────────────────────
        setPolling: (isPolling) => {
          set((state) => {
            state.isPolling = isPolling;
          });
        },

        setPollingInterval: (intervalMs) => {
          set((state) => {
            state.pollingIntervalMs = Math.max(1_000, intervalMs);
          });
        },

        // ── Global health ─────────────────────────────────────────────────────
        computeGlobalHealth: () => {
          set((state) => {
            const agents = Object.values(state.agents);
            if (agents.length === 0) {
              state.globalHealth = "unknown";
              return;
            }

            const healthStatuses = agents
              .map((a) => a.health?.status)
              .filter(Boolean) as Array<"healthy" | "degraded" | "unhealthy">;

            if (healthStatuses.length === 0) {
              state.globalHealth = "unknown";
            } else if (healthStatuses.some((s) => s === "unhealthy")) {
              state.globalHealth = "unhealthy";
            } else if (healthStatuses.some((s) => s === "degraded")) {
              state.globalHealth = "degraded";
            } else {
              state.globalHealth = "healthy";
            }
          });
        },

        // ── Error ─────────────────────────────────────────────────────────────
        setError: (error) => {
          set((state) => {
            state.error = error;
          });
        },

        // ── Derived ───────────────────────────────────────────────────────────
        getAgentById: (agentId) => get().agents[agentId] ?? null,

        getTasksByAgent: (agentId) =>
          Object.values(get().tasks).filter((t) => t.agentId === agentId),

        getActiveTasksByAgent: (agentId) =>
          Object.values(get().tasks).filter(
            (t) => t.agentId === agentId && ACTIVE_TASK_STATUSES.includes(t.status)
          ),

        getPendingTasks: () =>
          get()
            .taskQueue.map((id) => get().tasks[id])
            .filter((t): t is AgentTask => t !== undefined && t.status === "queued"),

        getRunningTasks: () =>
          Object.values(get().tasks).filter((t) => t.status === "running"),
      }))
    ),
    { name: "AgentStore" }
  )
);