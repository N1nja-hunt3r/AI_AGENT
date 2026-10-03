// frontend/src/utils/date.ts

// ─── Constants ────────────────────────────────────────────────────────────────

const MS_PER_SECOND = 1_000;
const MS_PER_MINUTE = 60 * MS_PER_SECOND;
const MS_PER_HOUR = 60 * MS_PER_MINUTE;
const MS_PER_DAY = 24 * MS_PER_HOUR;
const MS_PER_WEEK = 7 * MS_PER_DAY;
const MS_PER_MONTH = 30 * MS_PER_DAY;
const MS_PER_YEAR = 365 * MS_PER_DAY;

// ─── Types ────────────────────────────────────────────────────────────────────

export type DateInput = Date | number | string;

export type RelativeTimeStyle = "narrow" | "short" | "long";

export interface FormatDateOptions {
  locale?: string;
  timeZone?: string;
  includeTime?: boolean;
  includeYear?: boolean;
  includeSeconds?: boolean;
  hour12?: boolean;
}

export interface DurationComponents {
  years: number;
  months: number;
  weeks: number;
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  milliseconds: number;
}

export interface CountdownResult {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  total_ms: number;
  is_past: boolean;
  is_expired: boolean;
}

// ─── Normalisation ────────────────────────────────────────────────────────────

/**
 * Converts any DateInput to a Date object.
 * Returns null if the input is invalid.
 */
export const toDate = (input: DateInput): Date | null => {
  if (input instanceof Date) {
    return isNaN(input.getTime()) ? null : input;
  }
  if (typeof input === "number") {
    const d = new Date(input);
    return isNaN(d.getTime()) ? null : d;
  }
  if (typeof input === "string") {
    if (!input.trim()) return null;
    const d = new Date(input);
    return isNaN(d.getTime()) ? null : d;
  }
  return null;
};

/**
 * Converts any DateInput to a Unix timestamp in milliseconds.
 */
export const toTimestampMs = (input: DateInput): number | null => {
  const d = toDate(input);
  return d ? d.getTime() : null;
};

/**
 * Converts any DateInput to a Unix timestamp in seconds.
 */
export const toTimestampSec = (input: DateInput): number | null => {
  const ms = toTimestampMs(input);
  return ms !== null ? Math.floor(ms / MS_PER_SECOND) : null;
};

// ─── Validation ───────────────────────────────────────────────────────────────

export const isValidDate = (input: unknown): input is DateInput => {
  if (input === null || input === undefined) return false;
  const d = toDate(input as DateInput);
  return d !== null;
};

export const isDateInFuture = (input: DateInput): boolean => {
  const ms = toTimestampMs(input);
  return ms !== null && ms > Date.now();
};

export const isDateInPast = (input: DateInput): boolean => {
  const ms = toTimestampMs(input);
  return ms !== null && ms < Date.now();
};

export const isSameDay = (a: DateInput, b: DateInput): boolean => {
  const da = toDate(a);
  const db = toDate(b);
  if (!da || !db) return false;
  return (
    da.getFullYear() === db.getFullYear() &&
    da.getMonth() === db.getMonth() &&
    da.getDate() === db.getDate()
  );
};

export const isSameMonth = (a: DateInput, b: DateInput): boolean => {
  const da = toDate(a);
  const db = toDate(b);
  if (!da || !db) return false;
  return (
    da.getFullYear() === db.getFullYear() &&
    da.getMonth() === db.getMonth()
  );
};

export const isSameYear = (a: DateInput, b: DateInput): boolean => {
  const da = toDate(a);
  const db = toDate(b);
  if (!da || !db) return false;
  return da.getFullYear() === db.getFullYear();
};

// ─── Formatting ───────────────────────────────────────────────────────────────

/**
 * Formats a date using Intl.DateTimeFormat.
 */
export const formatDate = (
  input: DateInput,
  options: FormatDateOptions = {}
): string => {
  const d = toDate(input);
  if (!d) return "Invalid date";

  const {
    locale = "en-US",
    timeZone,
    includeTime = false,
    includeSeconds = false,
    hour12 = true,
  } = options;

  const dtOptions: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone,
  };

  if (includeTime) {
    dtOptions.hour = "2-digit";
    dtOptions.minute = "2-digit";
    dtOptions.hour12 = hour12;
    if (includeSeconds) {
      dtOptions.second = "2-digit";
    }
  }

  try {
    return new Intl.DateTimeFormat(locale, dtOptions).format(d);
  } catch {
    return d.toLocaleDateString(locale);
  }
};

/**
 * Formats only the time portion of a date.
 */
export const formatTime = (
  input: DateInput,
  options: Pick<FormatDateOptions, "locale" | "timeZone" | "hour12" | "includeSeconds"> = {}
): string => {
  const d = toDate(input);
  if (!d) return "Invalid time";

  const {
    locale = "en-US",
    timeZone,
    hour12 = true,
    includeSeconds = false,
  } = options;

  const dtOptions: Intl.DateTimeFormatOptions = {
    hour: "2-digit",
    minute: "2-digit",
    hour12,
    timeZone,
  };

  if (includeSeconds) {
    dtOptions.second = "2-digit";
  }

  try {
    return new Intl.DateTimeFormat(locale, dtOptions).format(d);
  } catch {
    return d.toLocaleTimeString(locale);
  }
};

/**
 * Returns a human-readable relative time string (e.g. "3 minutes ago").
 */
export const formatRelativeTime = (
  input: DateInput,
  options: {
    locale?: string;
    style?: RelativeTimeStyle;
    nowLabel?: string;
    baseDate?: DateInput;
  } = {}
): string => {
  const d = toDate(input);
  if (!d) return "Invalid date";

  const {
    locale = "en-US",
    style = "short",
    nowLabel = "just now",
    baseDate,
  } = options;

  const base = baseDate ? toDate(baseDate) ?? new Date() : new Date();
  const diffMs = d.getTime() - base.getTime();
  const absDiff = Math.abs(diffMs);

  if (absDiff < 30 * MS_PER_SECOND) return nowLabel;

  const rtf = new Intl.RelativeTimeFormat(locale, {
    numeric: "auto",
    style,
  });

  const sign = diffMs < 0 ? -1 : 1;

  if (absDiff < MS_PER_MINUTE) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_SECOND), "second");
  }
  if (absDiff < MS_PER_HOUR) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_MINUTE), "minute");
  }
  if (absDiff < MS_PER_DAY) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_HOUR), "hour");
  }
  if (absDiff < MS_PER_WEEK) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_DAY), "day");
  }
  if (absDiff < MS_PER_MONTH) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_WEEK), "week");
  }
  if (absDiff < MS_PER_YEAR) {
    return rtf.format(sign * Math.round(absDiff / MS_PER_MONTH), "month");
  }
  return rtf.format(sign * Math.round(absDiff / MS_PER_YEAR), "year");
};

/**
 * Smart date label: "Today", "Yesterday", relative time, or full date.
 */
export const formatSmartDate = (
  input: DateInput,
  locale = "en-US"
): string => {
  const d = toDate(input);
  if (!d) return "Invalid date";

  const now = new Date();
  const diffMs = now.getTime() - d.getTime();

  if (diffMs < MS_PER_MINUTE) return "Just now";
  if (diffMs < MS_PER_HOUR) return formatRelativeTime(d, { locale });
  if (isSameDay(d, now)) return `Today at ${formatTime(d, { locale })}`;
  if (isSameDay(d, new Date(now.getTime() - MS_PER_DAY))) {
    return `Yesterday at ${formatTime(d, { locale })}`;
  }
  if (diffMs < MS_PER_WEEK) return formatRelativeTime(d, { locale });
  if (isSameYear(d, now)) {
    return formatDate(d, { locale, includeYear: false });
  }
  return formatDate(d, { locale });
};

/**
 * Formats a timestamp as ISO 8601 string.
 */
export const toISOString = (input: DateInput): string | null => {
  const d = toDate(input);
  return d ? d.toISOString() : null;
};

/**
 * Formats duration in milliseconds to a human-readable string (e.g. "2h 34m 12s").
 */
export const formatDurationMs = (
  ms: number,
  options: {
    verbose?: boolean;
    maxUnits?: number;
    includeMs?: boolean;
  } = {}
): string => {
  const { verbose = false, maxUnits = 3, includeMs = false } = options;

  if (ms < 0) ms = 0;

  const components: Array<{ value: number; short: string; long: string }> = [
    { value: Math.floor(ms / MS_PER_YEAR), short: "y", long: " year" },
    { value: Math.floor((ms % MS_PER_YEAR) / MS_PER_MONTH), short: "mo", long: " month" },
    { value: Math.floor((ms % MS_PER_MONTH) / MS_PER_WEEK), short: "w", long: " week" },
    { value: Math.floor((ms % MS_PER_WEEK) / MS_PER_DAY), short: "d", long: " day" },
    { value: Math.floor((ms % MS_PER_DAY) / MS_PER_HOUR), short: "h", long: " hour" },
    { value: Math.floor((ms % MS_PER_HOUR) / MS_PER_MINUTE), short: "m", long: " minute" },
    { value: Math.floor((ms % MS_PER_MINUTE) / MS_PER_SECOND), short: "s", long: " second" },
  ];

  if (includeMs) {
    components.push({
      value: ms % MS_PER_SECOND,
      short: "ms",
      long: " millisecond",
    });
  }

  const parts = components
    .filter((c) => c.value > 0)
    .slice(0, maxUnits)
    .map((c) => {
      const label = verbose
        ? `${c.long}${c.value !== 1 ? "s" : ""}`
        : c.short;
      return `${c.value}${label}`;
    });

  return parts.length > 0 ? parts.join(" ") : verbose ? "0 seconds" : "0s";
};

/**
 * Formats a duration as mm:ss or hh:mm:ss (e.g. audio/video players).
 */
export const formatTimecode = (ms: number): string => {
  const totalSeconds = Math.floor(ms / MS_PER_SECOND);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");

  if (hours > 0) {
    const hh = String(hours).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }
  return `${mm}:${ss}`;
};

// ─── Calculation ──────────────────────────────────────────────────────────────

export const diffMs = (a: DateInput, b: DateInput): number | null => {
  const ta = toTimestampMs(a);
  const tb = toTimestampMs(b);
  return ta !== null && tb !== null ? ta - tb : null;
};

export const diffDays = (a: DateInput, b: DateInput): number | null => {
  const diff = diffMs(a, b);
  return diff !== null ? Math.round(diff / MS_PER_DAY) : null;
};

export const addMs = (input: DateInput, ms: number): Date | null => {
  const ts = toTimestampMs(input);
  return ts !== null ? new Date(ts + ms) : null;
};

export const addDays = (input: DateInput, days: number): Date | null =>
  addMs(input, days * MS_PER_DAY);

export const addHours = (input: DateInput, hours: number): Date | null =>
  addMs(input, hours * MS_PER_HOUR);

export const addMinutes = (input: DateInput, minutes: number): Date | null =>
  addMs(input, minutes * MS_PER_MINUTE);

export const startOfDay = (input: DateInput): Date | null => {
  const d = toDate(input);
  if (!d) return null;
  const copy = new Date(d);
  copy.setHours(0, 0, 0, 0);
  return copy;
};

export const endOfDay = (input: DateInput): Date | null => {
  const d = toDate(input);
  if (!d) return null;
  const copy = new Date(d);
  copy.setHours(23, 59, 59, 999);
  return copy;
};

export const startOfMonth = (input: DateInput): Date | null => {
  const d = toDate(input);
  if (!d) return null;
  return new Date(d.getFullYear(), d.getMonth(), 1, 0, 0, 0, 0);
};

export const endOfMonth = (input: DateInput): Date | null => {
  const d = toDate(input);
  if (!d) return null;
  return new Date(d.getFullYear(), d.getMonth() + 1, 0, 23, 59, 59, 999);
};

export const getCountdown = (target: DateInput): CountdownResult => {
  const ts = toTimestampMs(target);
  const now = Date.now();

  if (ts === null) {
    return { days: 0, hours: 0, minutes: 0, seconds: 0, total_ms: 0, is_past: false, is_expired: true };
  }

  const totalMs = ts - now;
  const is_past = totalMs < 0;
  const absDiff = Math.abs(totalMs);

  return {
    days: Math.floor(absDiff / MS_PER_DAY),
    hours: Math.floor((absDiff % MS_PER_DAY) / MS_PER_HOUR),
    minutes: Math.floor((absDiff % MS_PER_HOUR) / MS_PER_MINUTE),
    seconds: Math.floor((absDiff % MS_PER_MINUTE) / MS_PER_SECOND),
    total_ms: totalMs,
    is_past,
    is_expired: is_past,
  };
};

export const decomposeDuration = (ms: number): DurationComponents => ({
  years: Math.floor(ms / MS_PER_YEAR),
  months: Math.floor((ms % MS_PER_YEAR) / MS_PER_MONTH),
  weeks: Math.floor((ms % MS_PER_MONTH) / MS_PER_WEEK),
  days: Math.floor((ms % MS_PER_WEEK) / MS_PER_DAY),
  hours: Math.floor((ms % MS_PER_DAY) / MS_PER_HOUR),
  minutes: Math.floor((ms % MS_PER_HOUR) / MS_PER_MINUTE),
  seconds: Math.floor((ms % MS_PER_MINUTE) / MS_PER_SECOND),
  milliseconds: ms % MS_PER_SECOND,
});