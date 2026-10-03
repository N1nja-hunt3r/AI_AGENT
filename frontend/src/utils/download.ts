// frontend/src/utils/download.ts

// ─── Types ────────────────────────────────────────────────────────────────────

export type DownloadMimeType =
  | "text/plain"
  | "text/html"
  | "text/csv"
  | "text/markdown"
  | "application/json"
  | "application/pdf"
  | "application/octet-stream"
  | "image/png"
  | "image/jpeg"
  | "image/svg+xml"
  | "audio/mpeg"
  | "audio/wav"
  | "audio/ogg"
  | (string & NonNullable<unknown>);

export interface DownloadOptions {
  filename: string;
  mime_type?: DownloadMimeType;
  charset?: string;
}

export interface DownloadResult {
  success: boolean;
  error?: string;
}

export interface CSVOptions {
  delimiter?: string;
  quote_char?: string;
  include_bom?: boolean;
  line_ending?: "lf" | "crlf";
}

// ─── Core download ────────────────────────────────────────────────────────────

/**
 * Triggers a file download from a Blob object.
 */
export const downloadBlob = (
  blob: Blob,
  filename: string
): DownloadResult => {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return { success: false, error: "Not in a browser environment" };
  }

  const url = URL.createObjectURL(blob);

  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = sanitiseFilename(filename);
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    return { success: true };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Download failed",
    };
  } finally {
    // Defer revocation to allow the browser to initiate download
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
};

/**
 * Triggers a file download from a URL.
 */
export const downloadFromUrl = (
  url: string,
  filename: string
): DownloadResult => {
  if (typeof document === "undefined") {
    return { success: false, error: "Not in a browser environment" };
  }

  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = sanitiseFilename(filename);
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    return { success: true };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Download failed",
    };
  }
};

// ─── Text download ────────────────────────────────────────────────────────────

/**
 * Downloads a string as a plain text file.
 */
export const downloadText = (
  content: string,
  options: DownloadOptions
): DownloadResult => {
  const {
    filename,
    mime_type = "text/plain",
    charset = "utf-8",
  } = options;

  const blob = new Blob([content], {
    type: `${mime_type};charset=${charset}`,
  });

  return downloadBlob(blob, filename);
};

/**
 * Downloads an object as a formatted JSON file.
 */
export const downloadJSON = (
  data: unknown,
  filename: string,
  indent = 2
): DownloadResult => {
  try {
    const json = JSON.stringify(data, null, indent);
    return downloadText(json, {
      filename: ensureExtension(filename, ".json"),
      mime_type: "application/json",
    });
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "JSON serialisation failed",
    };
  }
};

// ─── CSV download ─────────────────────────────────────────────────────────────

/**
 * Serialises a 2D array of values to a CSV string.
 */
export const arrayToCSV = (
  rows: Array<Array<string | number | boolean | null | undefined>>,
  options: CSVOptions = {}
): string => {
  const {
    delimiter = ",",
    quote_char = '"',
    include_bom = true,
    line_ending = "crlf",
  } = options;

  const lineEnd = line_ending === "crlf" ? "\r\n" : "\n";

  const escape = (value: string | number | boolean | null | undefined): string => {
    if (value === null || value === undefined) return "";
    const str = String(value);
    if (
      str.includes(delimiter) ||
      str.includes(quote_char) ||
      str.includes("\n") ||
      str.includes("\r")
    ) {
      return `${quote_char}${str.replace(
        new RegExp(quote_char, "g"),
        `${quote_char}${quote_char}`
      )}${quote_char}`;
    }
    return str;
  };

  const csv = rows
    .map((row) => row.map(escape).join(delimiter))
    .join(lineEnd);

  return include_bom ? `\uFEFF${csv}` : csv;
};

/**
 * Downloads an array of objects as a CSV file.
 */
export const downloadCSV = <TRow extends Record<string, unknown>>(
  data: TRow[],
  filename: string,
  options: CSVOptions & { headers?: string[] } = {}
): DownloadResult => {
  if (data.length === 0) {
    return { success: false, error: "No data to export" };
  }

  const { headers, ...csvOptions } = options;
  const keys = headers ?? (Object.keys(data[0]!) as string[]);
  const rows: Array<Array<string | number | boolean | null | undefined>> = [
    keys,
    ...data.map((row) =>
      keys.map((key) => row[key] as string | number | boolean | null | undefined)
    ),
  ];

  const csv = arrayToCSV(rows, csvOptions);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  return downloadBlob(blob, ensureExtension(filename, ".csv"));
};

// ─── Binary download ──────────────────────────────────────────────────────────

/**
 * Downloads an ArrayBuffer as a binary file.
 */
export const downloadArrayBuffer = (
  buffer: ArrayBuffer,
  options: DownloadOptions
): DownloadResult => {
  const { filename, mime_type = "application/octet-stream" } = options;
  const blob = new Blob([buffer], { type: mime_type });
  return downloadBlob(blob, filename);
};

/**
 * Downloads a base64-encoded string as a binary file.
 */
export const downloadBase64 = (
  base64: string,
  options: DownloadOptions
): DownloadResult => {
  const { filename, mime_type = "application/octet-stream" } = options;

  try {
    const byteString = atob(base64.split(",").pop() ?? base64);
    const buffer = new Uint8Array(byteString.length);
    for (let i = 0; i < byteString.length; i++) {
      buffer[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([buffer], { type: mime_type });
    return downloadBlob(blob, filename);
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Base64 decode failed",
    };
  }
};

/**
 * Downloads a canvas element as a PNG image.
 */
export const downloadCanvas = (
  canvas: HTMLCanvasElement,
  filename: string,
  quality = 1.0
): DownloadResult => {
  try {
    const url = canvas.toDataURL("image/png", quality);
    return downloadFromUrl(url, ensureExtension(filename, ".png"));
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Canvas export failed",
    };
  }
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Sanitises a filename by removing illegal characters.
 */
export const sanitiseFilename = (filename: string): string => {
  return filename
    .replace(/[/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, "_")
    .replace(/-{2,}/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "")
    .slice(0, 255);
};

/**
 * Ensures a filename has the specified extension.
 */
export const ensureExtension = (filename: string, ext: string): string => {
  const normalised = ext.startsWith(".") ? ext : `.${ext}`;
  return filename.toLowerCase().endsWith(normalised.toLowerCase())
    ? filename
    : `${filename}${normalised}`;
};

/**
 * Extracts the file extension from a filename or URL.
 */
export const getFileExtension = (filename: string): string => {
  const base = filename.split("/").pop() ?? filename;
  const dotIndex = base.lastIndexOf(".");
  return dotIndex > 0 ? base.slice(dotIndex).toLowerCase() : "";
};

/**
 * Formats a file size in bytes to a human-readable string.
 */
export const formatFileSize = (bytes: number, precision = 1): string => {
  if (bytes < 0) return "0 B";
  if (bytes === 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(precision)} KB`;
  if (bytes < 1024 ** 3)
    return `${(bytes / 1024 ** 2).toFixed(precision)} MB`;
  if (bytes < 1024 ** 4)
    return `${(bytes / 1024 ** 3).toFixed(precision)} GB`;
  return `${(bytes / 1024 ** 4).toFixed(precision)} TB`;
};

/**
 * Converts bytes to a specific unit.
 */
export const convertBytes = (
  bytes: number,
  to: "KB" | "MB" | "GB" | "TB"
): number => {
  const divisors = { KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 };
  return bytes / divisors[to];
};