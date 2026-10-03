import api, {
  BASE_URL,
  getAuthToken,
  clearAuthToken,
} from "@/api/client";
import type { ApiError, ApiErrorPayload } from "@/api/client";

export { BASE_URL, getAuthToken, clearAuthToken };
export type { ApiError, ApiErrorPayload };
export default api;