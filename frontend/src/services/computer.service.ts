import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ---------------------------------------------------------------------------
// Request types (mirror backend Pydantic models)
// ---------------------------------------------------------------------------
export interface MouseClickRequest {
  session_id: string;
  x: number;
  y: number;
  button?: "left" | "right" | "middle";
  double_click?: boolean;
}

export interface MouseMoveRequest {
  session_id: string;
  x: number;
  y: number;
}

export interface KeyboardTypeRequest {
  session_id: string;
  text: string;
}

export interface KeyboardKeyRequest {
  session_id: string;
  keys: string[];
}

export interface BrowserNavigateRequest {
  session_id: string;
  url: string;
}

export interface BrowserActionRequest {
  session_id: string;
  action: "back" | "forward" | "reload" | "close";
}

export interface TerminalExecRequest {
  session_id: string;
  command: string;
  working_dir?: string | null;
  timeout_seconds?: number;
}

// ---------------------------------------------------------------------------
// Response types (mirror backend Pydantic models)
// ---------------------------------------------------------------------------
export interface ActionResponse {
  session_id: string;
  action: string;
  status: string;
  result?: unknown;
  error?: string | null;
  executed_at: string;
}

export interface ScreenshotResponse {
  session_id: string;
  image_base64: string;
  width: number;
  height: number;
  captured_at: string;
}

export interface TerminalExecResponse {
  session_id: string;
  command: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
  executed_at: string;
}

export interface ApprovalRequiredResponse {
  requires_approval: boolean;
  action: string;
  reason: string;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
const ENDPOINTS = {
  MOUSE_CLICK: "/api/v1/computer/mouse/click",
  MOUSE_MOVE: "/api/v1/computer/mouse/move",
  KEYBOARD_TYPE: "/api/v1/computer/keyboard/type",
  KEYBOARD_KEY: "/api/v1/computer/keyboard/key",
  BROWSER_NAVIGATE: "/api/v1/computer/browser/navigate",
  BROWSER_ACTION: "/api/v1/computer/browser/action",
  SCREENSHOT: (session_id: string) =>
    `/api/v1/computer/screenshot/${session_id}`,
  TERMINAL_EXEC: "/api/v1/computer/terminal/exec",
  APPROVALS_CHECK: "/api/v1/computer/approvals/check",
} as const;

// ---------------------------------------------------------------------------
// Action functions
// ---------------------------------------------------------------------------
const mouseClick = async (
  payload: MouseClickRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.MOUSE_CLICK, payload);
  return response.data;
};

const mouseMove = async (
  payload: MouseMoveRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.MOUSE_MOVE, payload);
  return response.data;
};

const keyboardType = async (
  payload: KeyboardTypeRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.KEYBOARD_TYPE, payload);
  return response.data;
};

const keyboardKey = async (
  payload: KeyboardKeyRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.KEYBOARD_KEY, payload);
  return response.data;
};

const browserNavigate = async (
  payload: BrowserNavigateRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.BROWSER_NAVIGATE, payload);
  return response.data;
};

const browserAction = async (
  payload: BrowserActionRequest
): Promise<ActionResponse> => {
  const response: AxiosResponse<ActionResponse> =
    await api.post<ActionResponse>(ENDPOINTS.BROWSER_ACTION, payload);
  return response.data;
};

const terminalExec = async (
  payload: TerminalExecRequest
): Promise<TerminalExecResponse> => {
  const response: AxiosResponse<TerminalExecResponse> =
    await api.post<TerminalExecResponse>(ENDPOINTS.TERMINAL_EXEC, payload);
  return response.data;
};

const takeScreenshot = async (
  session_id: string
): Promise<ScreenshotResponse> => {
  const response: AxiosResponse<ScreenshotResponse> =
    await api.get<ScreenshotResponse>(ENDPOINTS.SCREENSHOT(session_id));
  return response.data;
};

const checkApproval = async (
  action: string,
  session_id: string
): Promise<ApprovalRequiredResponse> => {
  const response: AxiosResponse<ApprovalRequiredResponse> =
    await api.get<ApprovalRequiredResponse>(ENDPOINTS.APPROVALS_CHECK, {
      params: { action, session_id },
    });
  return response.data;
};

export const ComputerService = {
  mouseClick,
  mouseMove,
  keyboardType,
  keyboardKey,
  browserNavigate,
  browserAction,
  terminalExec,
  takeScreenshot,
  checkApproval,
} as const;

export default ComputerService;
