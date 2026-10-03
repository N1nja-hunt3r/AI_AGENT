import axios, { AxiosError, type AxiosInstance, type AxiosResponse, type InternalAxiosRequestConfig } from "axios";

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const TIMEOUT_MS: number = Number(import.meta.env.VITE_API_TIMEOUT) || 30_000;

export interface ApiErrorPayload {
  message: string;
  statusCode: number;
  errors?: Record<string, string[]>;
}

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly errors?: Record<string, string[]>;

  constructor(payload: ApiErrorPayload) {
    super(payload.message);
    this.name = "ApiError";
    this.statusCode = payload.statusCode;
    this.errors = payload.errors;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

const getAuthToken = (): string | null => localStorage.getItem("auth_token");
const clearAuthToken = (): void => { localStorage.removeItem("auth_token"); };

const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT_MS,
  headers: { "Content-Type": "application/json", Accept: "application/json" },
  withCredentials: true,
});

api.interceptors.request.use((config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
  const token = getAuthToken();
  if (token && config.headers) config.headers.Authorization = `Bearer ${token}`;
  config.headers["X-Request-Time"] = new Date().toISOString();
  return config;
});

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError<ApiErrorPayload>) => {
    const status = error.response?.status;
    const msg = error.response?.data?.message ?? error.message ?? "An error occurred";
    const errs = error.response?.data?.errors;

    if (status === 401) {
      clearAuthToken();
      if (window.location.pathname !== "/login") window.location.replace("/login");
      return Promise.reject(new ApiError({ statusCode: 401, message: "Session expired.", errors: errs }));
    }
    if (status === 403) return Promise.reject(new ApiError({ statusCode: 403, message: "Permission denied.", errors: errs }));
    if (status === 404) return Promise.reject(new ApiError({ statusCode: 404, message: msg, errors: errs }));
    if (status === 422) return Promise.reject(new ApiError({ statusCode: 422, message: msg || "Validation failed.", errors: errs }));
    if (status === 429) return Promise.reject(new ApiError({ statusCode: 429, message: "Too many requests.", errors: errs }));
    if (status === 500) return Promise.reject(new ApiError({ statusCode: 500, message: "Internal server error.", errors: errs }));
    if (status && status >= 502) return Promise.reject(new ApiError({ statusCode: status, message: "Service unavailable.", errors: errs }));
    if (error.code === "ECONNABORTED") return Promise.reject(new ApiError({ statusCode: 408, message: `Request timed out.` }));
    if (!error.response) return Promise.reject(new ApiError({ statusCode: 0, message: "Network error." }));
    return Promise.reject(new ApiError({ statusCode: status ?? 0, message: msg, errors: errs }));
  },
);

export { BASE_URL, getAuthToken, clearAuthToken };
export default api;
