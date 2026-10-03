// frontend/src/utils/clipboard.ts

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ClipboardReadResult {
  text: string;
  success: boolean;
  error?: string;
}

export interface ClipboardWriteResult {
  success: boolean;
  error?: string;
}

export type ClipboardFormat = "text/plain" | "text/html" | "image/png";

// ─── Feature detection ────────────────────────────────────────────────────────

export const isClipboardAPIAvailable = (): boolean =>
  typeof navigator !== "undefined" &&
  typeof navigator.clipboard !== "undefined" &&
  typeof navigator.clipboard.writeText === "function";

export const isClipboardReadAvailable = (): boolean =>
  typeof navigator !== "undefined" &&
  typeof navigator.clipboard !== "undefined" &&
  typeof navigator.clipboard.readText === "function";

// ─── Write ────────────────────────────────────────────────────────────────────

/**
 * Copies text to the clipboard.
 * Falls back to the legacy execCommand method if the Clipboard API
 * is not available or permission is denied.
 */
export const copyToClipboard = async (
  text: string
): Promise<ClipboardWriteResult> => {
  if (!text) {
    return { success: false, error: "Nothing to copy" };
  }

  if (isClipboardAPIAvailable()) {
    try {
      await navigator.clipboard.writeText(text);
      return { success: true };
    } catch {
      // Fall through to legacy method
    }
  }

  return copyToClipboardLegacy(text);
};

/**
 * Copies HTML to the clipboard (where supported).
 */
export const copyHtmlToClipboard = async (
  html: string,
  plainText?: string
): Promise<ClipboardWriteResult> => {
  if (
    typeof navigator === "undefined" ||
    !navigator.clipboard ||
    typeof ClipboardItem === "undefined"
  ) {
    return copyToClipboard(plainText ?? stripHtmlTags(html));
  }

  try {
    const items: Record<string, Blob> = {
      "text/html": new Blob([html], { type: "text/html" }),
    };

    if (plainText) {
      items["text/plain"] = new Blob([plainText], { type: "text/plain" });
    }

    await navigator.clipboard.write([new ClipboardItem(items)]);
    return { success: true };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Copy failed",
    };
  }
};

/**
 * Legacy clipboard copy using document.execCommand.
 * Only works in browser environments during user-initiated events.
 */
const copyToClipboardLegacy = (text: string): ClipboardWriteResult => {
  if (typeof document === "undefined") {
    return { success: false, error: "DOM not available" };
  }

  const el = document.createElement("textarea");
  el.value = text;
  el.setAttribute("readonly", "");
  el.style.cssText =
    "position:fixed;top:-9999px;left:-9999px;opacity:0;pointer-events:none;";

  document.body.appendChild(el);

  try {
    el.focus();
    el.select();
    el.setSelectionRange(0, el.value.length);
    const success = document.execCommand("copy");
    return success
      ? { success: true }
      : { success: false, error: "execCommand returned false" };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Legacy copy failed",
    };
  } finally {
    document.body.removeChild(el);
  }
};

// ─── Read ─────────────────────────────────────────────────────────────────────

/**
 * Reads text from the clipboard.
 */
export const readFromClipboard = async (): Promise<ClipboardReadResult> => {
  if (!isClipboardReadAvailable()) {
    return {
      text: "",
      success: false,
      error: "Clipboard read API not available",
    };
  }

  try {
    const text = await navigator.clipboard.readText();
    return { text, success: true };
  } catch (err) {
    return {
      text: "",
      success: false,
      error: err instanceof Error ? err.message : "Read failed",
    };
  }
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Strips HTML tags from a string.
 */
const stripHtmlTags = (html: string): string =>
  html.replace(/<[^>]*>/g, "").replace(/\s{2,}/g, " ").trim();

/**
 * Checks if the clipboard permission is granted.
 * Returns null if the Permissions API is not available.
 */
export const checkClipboardPermission = async (
  mode: "read" | "write" = "write"
): Promise<PermissionState | null> => {
  if (
    typeof navigator === "undefined" ||
    !navigator.permissions
  ) {
    return null;
  }

  try {
    const permName =
      mode === "read" ? "clipboard-read" : "clipboard-write";
    const status = await navigator.permissions.query({
      name: permName as PermissionName,
    });
    return status.state;
  } catch {
    return null;
  }
};

/**
 * Copies a value and returns a boolean indicating success.
 * Convenience wrapper for simple use-cases.
 */
export const copy = async (text: string): Promise<boolean> => {
  const result = await copyToClipboard(text);
  return result.success;
};

/**
 * Copies the current page URL to the clipboard.
 */
export const copyCurrentUrl = async (): Promise<ClipboardWriteResult> => {
  if (typeof window === "undefined") {
    return { success: false, error: "Window not available" };
  }
  return copyToClipboard(window.location.href);
};

/**
 * Copies a code block with optional language annotation.
 */
export const copyCode = async (
  code: string
): Promise<ClipboardWriteResult> => copyToClipboard(code);