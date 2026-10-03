import api from "@/api/client";

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

export interface ActionResult {
  success: boolean;
  screenshot?: Screenshot;
  output?: string;
  error?: string;
  duration_ms: number;
}

export interface Screenshot {
  id: string;
  data: string;
  resolution: ScreenResolution;
  capturedAt: string;
}

export interface ScreenResolution {
  width: number;
  height: number;
}

export interface ActionResultResponse {
  result: ActionResult;
}

export interface ApprovalCheckResponse {
  requires_approval: boolean;
  message?: string;
}

const ENDPOINTS = {
  MOUSE_CLICK: "/api/v1/computer/mouse/click",
  MOUSE_MOVE: "/api/v1/computer/mouse/move",
  KEYBOARD_TYPE: "/api/v1/computer/keyboard/type",
  KEYBOARD_KEY: "/api/v1/computer/keyboard/key",
  BROWSER_NAVIGATE: "/api/v1/computer/browser/navigate",
  BROWSER_ACTION: "/api/v1/computer/browser/action",
  SCREENSHOT: (session_id: string) => `/api/v1/computer/screenshot/${session_id}`,
  TERMINAL_EXEC: "/api/v1/computer/terminal/exec",
  APPROVALS_CHECK: "/api/v1/computer/approvals/check",
} as const;

export const mouseClick = async (payload: MouseClickRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.MOUSE_CLICK, payload);
  return data;
};

export const mouseMove = async (payload: MouseMoveRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.MOUSE_MOVE, payload);
  return data;
};

export const keyboardType = async (payload: KeyboardTypeRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.KEYBOARD_TYPE, payload);
  return data;
};

export const keyboardKey = async (payload: KeyboardKeyRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.KEYBOARD_KEY, payload);
  return data;
};

export const browserNavigate = async (payload: BrowserNavigateRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.BROWSER_NAVIGATE, payload);
  return data;
};

export const browserAction = async (payload: BrowserActionRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.BROWSER_ACTION, payload);
  return data;
};

export const takeScreenshot = async (session_id: string): Promise<Screenshot> => {
  const { data } = await api.get<Screenshot>(ENDPOINTS.SCREENSHOT(session_id));
  return data;
};

export const terminalExec = async (payload: TerminalExecRequest): Promise<ActionResultResponse> => {
  const { data } = await api.post<ActionResultResponse>(ENDPOINTS.TERMINAL_EXEC, payload);
  return data;
};

export const checkApproval = async (): Promise<ApprovalCheckResponse> => {
  const { data } = await api.get<ApprovalCheckResponse>(ENDPOINTS.APPROVALS_CHECK);
  return data;
};
