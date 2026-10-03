// frontend/src/types/agent.ts

import type { TokenUsage } from "./chat";

// ─── Enums ────────────────────────────────────────────────────────────────────

export enum AgentId {
  Orchestrator = "orchestrator",
  Planner = "planner",
  Researcher = "researcher",
  Coder = "coder",
  Reflector = "reflector",
  MemoryManager = "memory_manager",
  ToolExecutor = "tool_executor",
  Critic = "critic",
  Summarizer = "summarizer",
}

export enum AgentStatus {
  Idle = "idle",
  Initializing = "initializing",
  Running = "running",
  Paused = "paused",
  Stopping = "stopping",
  Stopped = "stopped",
  Error = "error",
  Degraded = "degraded",
  Unavailable = "unavailable",
  RateLimited = "rate_limited",
}

export enum TaskStatus {
  Queued = "queued",
  Planning = "planning",
  Running = "running",
  AwaitingTool = "awaiting_tool",
  AwaitingHuman = "awaiting_human",
  Reflecting = "reflecting",
  Completed = "completed",
  Failed = "failed",
  Cancelled = "cancelled",
  TimedOut = "timed_out",
  Retrying = "retrying",
}

export enum TaskPriority {
  Low = "low",
  Normal = "normal",
  High = "high",
  Critical = "critical",
}

export enum AgentStepType {
  Thought = "thought",
  Action = "action",
  Observation = "observation",
  Reflection = "reflection",
  Plan = "plan",
  Message = "message",
  ToolCall = "tool_call",
  Memory = "memory",
  Critique = "critique",
}

export enum ToolExecutionStatus {
  Pending = "pending",
  Executing = "executing",
  Success = "success",
  Error = "error",
  Timeout = "timeout",
  Cancelled = "cancelled",
}

export enum HealthStatus {
  Healthy = "healthy",
  Degraded = "degraded",
  Unhealthy = "unhealthy",
  Unknown = "unknown",
}

// ─── Tool types ───────────────────────────────────────────────────────────────

export interface AgentTool {
  name: string;
  description: string;
  parameters_schema: Record<string, unknown>;
  category?: string;
  requires_confirmation?: boolean;
  timeout_ms?: number;
  is_enabled: boolean;
}

export interface ToolExecution {
  id: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_output?: unknown;
  status: ToolExecutionStatus;
  error?: string;
  started_at: number;
  completed_at?: number;
  duration_ms?: number;
  retry_count: number;
  cost_usd?: number;
}

// ─── Agent step ───────────────────────────────────────────────────────────────

export interface AgentStep {
  id: string;
  type: AgentStepType;
  content: string;
  tool_execution?: ToolExecution;
  token_usage?: TokenUsage;
  model?: string;
  created_at: number;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
}

// ─── Task ─────────────────────────────────────────────────────────────────────

export interface AgentTask {
  id: string;
  agent_id: string;
  type: string;
  description: string;
  goal?: string;
  status: TaskStatus;
  priority: TaskPriority;
  progress_percent: number;
  steps: AgentStep[];
  result?: unknown;
  error?: string;
  parent_task_id?: string;
  child_task_ids: string[];
  conversation_id?: string;
  session_id?: string;
  retry_count: number;
  max_retries: number;
  timeout_ms?: number;
  token_usage?: TokenUsage;
  cost_usd?: number;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  updated_at: number;
  metadata?: Record<string, unknown>;
}

export type AgentTaskCreateInput = Pick<
  AgentTask,
  | "agent_id"
  | "type"
  | "description"
  | "goal"
  | "priority"
  | "parent_task_id"
  | "conversation_id"
  | "timeout_ms"
  | "max_retries"
  | "metadata"
>;

// ─── Health ───────────────────────────────────────────────────────────────────

export interface HealthCheck {
  name: string;
  status: "pass" | "fail" | "warn";
  message?: string;
  duration_ms: number;
  checked_at: number;
  metadata?: Record<string, unknown>;
}

export interface AgentHealth {
  status: HealthStatus;
  checks: HealthCheck[];
  last_checked_at: number;
  message?: string;
}

// ─── Latency ─────────────────────────────────────────────────────────────────

export interface LatencyPercentiles {
  p50_ms: number;
  p75_ms: number;
  p90_ms: number;
  p95_ms: number;
  p99_ms: number;
  p999_ms: number;
}

export interface LatencyMetrics extends LatencyPercentiles {
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  stddev_ms: number;
  sample_count: number;
  window_start: number;
  window_end: number;
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface AgentMetrics {
  tasks_queued: number;
  tasks_running: number;
  tasks_completed: number;
  tasks_failed: number;
  tasks_cancelled: number;
  success_rate_1m: number;
  success_rate_5m: number;
  success_rate_1h: number;
  avg_task_duration_ms: number;
  total_tokens_used: number;
  total_tool_calls: number;
  total_cost_usd: number;
  uptime_ms: number;
  last_active_at: number | null;
  updated_at: number;
}

// ─── Resource usage ───────────────────────────────────────────────────────────

export interface AgentResourceUsage {
  cpu_percent: number;
  memory_mb: number;
  active_connections: number;
  queue_depth: number;
  updated_at: number;
}

// ─── Config ───────────────────────────────────────────────────────────────────

export interface AgentConfig {
  model: string;
  provider: string;
  temperature: number;
  max_tokens: number;
  max_concurrent_tasks: number;
  max_retries: number;
  timeout_ms: number;
  memory_enabled: boolean;
  reflection_enabled: boolean;
  planning_enabled: boolean;
  allowed_tools: string[];
  max_steps: number;
  custom_instructions?: string;
  metadata?: Record<string, unknown>;
}

// ─── Agent ────────────────────────────────────────────────────────────────────

export interface Agent {
  id: string;
  name: string;
  description: string;
  version: string;
  status: AgentStatus;
  health: AgentHealth;
  latency: LatencyMetrics | null;
  metrics: AgentMetrics;
  resources: AgentResourceUsage | null;
  config: AgentConfig;
  capabilities: string[];
  available_tools: AgentTool[];
  current_task_id: string | null;
  error_message?: string;
  registered_at: number;
  last_heartbeat_at: number | null;
  metadata?: Record<string, unknown>;
}

export type AgentUpdateInput = Partial<
  Pick<Agent, "config" | "metadata" | "status">
>;

// ─── Agent events ─────────────────────────────────────────────────────────────

export enum AgentEventType {
  StatusChanged = "status_changed",
  TaskCreated = "task_created",
  TaskStarted = "task_started",
  TaskCompleted = "task_completed",
  TaskFailed = "task_failed",
  StepAdded = "step_added",
  ToolCalled = "tool_called",
  HealthChecked = "health_checked",
  MetricsUpdated = "metrics_updated",
}

export interface AgentEvent<TPayload = unknown> {
  type: AgentEventType;
  agent_id: string;
  task_id?: string;
  payload: TPayload;
  timestamp: number;
}

// ─── Orchestration ────────────────────────────────────────────────────────────

export interface OrchestrationSession {
  id: string;
  conversation_id?: string;
  goal: string;
  agent_ids: string[];
  status: TaskStatus;
  master_task_id: string;
  started_at: number;
  completed_at?: number;
  result?: unknown;
  metadata?: Record<string, unknown>;
}

export interface AgentPlan {
  id: string;
  session_id: string;
  steps: PlanStep[];
  estimated_duration_ms: number;
  confidence: number;
  created_at: number;
}

export interface PlanStep {
  id: string;
  order: number;
  agent_id: string;
  action: string;
  description: string;
  depends_on: string[];
  estimated_duration_ms: number;
  is_optional: boolean;
}