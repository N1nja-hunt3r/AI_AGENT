// frontend/src/utils/registry.ts

// ─── Date utilities ───────────────────────────────────────────────────────────

export {
  toDate,
  toTimestampMs,
  toTimestampSec,
  isValidDate,
  isDateInFuture,
  isDateInPast,
  isSameDay,
  isSameMonth,
  isSameYear,
  formatDate,
  formatTime,
  formatRelativeTime,
  formatSmartDate,
  toISOString,
  formatDurationMs,
  formatTimecode,
  diffMs,
  diffDays,
  addMs,
  addDays,
  addHours,
  addMinutes,
  startOfDay,
  endOfDay,
  startOfMonth,
  endOfMonth,
  getCountdown,
  decomposeDuration,
} from "./date";

export type {
  DateInput,
  RelativeTimeStyle,
  FormatDateOptions,
  DurationComponents,
  CountdownResult,
} from "./date";

// ─── Markdown utilities ───────────────────────────────────────────────────────

export {
  escapeHtml,
  unescapeHtml,
  escapeMarkdown,
  extractCodeBlocks,
  extractLinks,
  extractHeadings,
  generateTableOfContents,
  stripMarkdown,
  slugifyHeading,
  stripTitle,
  extractTitle,
  generateExcerpt,
  countWords,
  estimateReadingTime,
  wrapInCodeBlock,
  wrapInCode,
  normaliseLanguage,
  textToMarkdown,
  normaliseMarkdown,
} from "./markdown";

export type {
  MarkdownParseOptions,
  ExtractedCodeBlock,
  ExtractedLink,
  ExtractedHeading,
  TableOfContentsItem,
} from "./markdown";

// ─── Stream utilities ─────────────────────────────────────────────────────────

export {
  parseSSEBlock,
  readStreamAsText,
  readSSEStream,
  readNDJSONStream,
  createTokenAccumulator,
  collectAsync,
  streamToString,
  createAbortController,
  mergeAbortSignals,
  throttleStream,
  batchStream,
} from "./stream";

export type {
  SSEMessage,
  SSEParser,
  StreamReaderOptions,
  StreamTokenAccumulator,
  JsonStreamOptions,
} from "./stream";

// ─── Clipboard utilities ──────────────────────────────────────────────────────

export {
  isClipboardAPIAvailable,
  isClipboardReadAvailable,
  copyToClipboard,
  copyHtmlToClipboard,
  readFromClipboard,
  checkClipboardPermission,
  copy,
  copyCurrentUrl,
  copyCode,
} from "./clipboard";

export type {
  ClipboardReadResult,
  ClipboardWriteResult,
  ClipboardFormat,
} from "./clipboard";

// ─── Download utilities ───────────────────────────────────────────────────────

export {
  downloadBlob,
  downloadFromUrl,
  downloadText,
  downloadJSON,
  arrayToCSV,
  downloadCSV,
  downloadArrayBuffer,
  downloadBase64,
  downloadCanvas,
  sanitiseFilename,
  ensureExtension,
  getFileExtension,
  formatFileSize,
  convertBytes,
} from "./download";

export type {
  DownloadMimeType,
  DownloadOptions,
  DownloadResult,
  CSVOptions,
} from "./download";

// ─── Validation utilities ─────────────────────────────────────────────────────

export {
  ok,
  fail,
  validate,
  validateFirst,
  required,
  minLength,
  maxLength,
  exactLength,
  matches,
  notMatches,
  oneOf,
  isEmail,
  isUrl,
  isHttpUrl,
  isUUID,
  isSlug,
  isHexColor,
  isJSON,
  isAlphanumeric,
  noWhitespace,
  isNumber,
  minValue,
  maxValue,
  inRange,
  isInteger,
  isPositive,
  isNonNegative,
  checkPasswordStrength,
  isStrongPassword,
  isAllowedFileType,
  maxFileSize,
  minItems,
  maxItems,
  safeParseJSON,
  safeParseNumber,
  safeParseInt,
  safeParseFloat,
  safeParseBoolean,
  isObject,
  isNonEmptyString,
  isFiniteNumber,
} from "./validation";

export type {
  ValidationResult,
  FieldValidationResult,
  ValidatorFn,
  PasswordStrength,
} from "./validation";

// ─── Inline utilities (debounce / throttle / string helpers) ──────────────────

/**
 * Debounces a function — delays invocation until after `wait` ms
 * have elapsed since the last call.
 */
export const debounce = <TArgs extends unknown[]>(
  fn: (...args: TArgs) => void,
  wait: number,
  options: { leading?: boolean; trailing?: boolean } = {}
): ((...args: TArgs) => void) & { cancel: () => void; flush: (...args: TArgs) => void } => {
  const { leading = false, trailing = true } = options;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: TArgs | null = null;

  const invoke = (args: TArgs): void => {
    fn(...args);
    lastArgs = null;
  };

  const debounced = (...args: TArgs): void => {
    lastArgs = args;

    if (leading && timer === null) {
      invoke(args);
    }

    if (timer) clearTimeout(timer);

    timer = setTimeout(() => {
      timer = null;
      if (trailing && lastArgs) invoke(lastArgs);
    }, wait);
  };

  debounced.cancel = (): void => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    lastArgs = null;
  };

  debounced.flush = (...args: TArgs): void => {
    debounced.cancel();
    invoke(args.length > 0 ? args : (lastArgs ?? args));
  };

  return debounced;
};

/**
 * Throttles a function — invokes at most once per `wait` ms.
 */
export const throttle = <TArgs extends unknown[]>(
  fn: (...args: TArgs) => void,
  wait: number,
  options: { leading?: boolean; trailing?: boolean } = {}
): ((...args: TArgs) => void) & { cancel: () => void } => {
  const { leading = true, trailing = true } = options;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastInvokeTime = 0;
  let lastArgs: TArgs | null = null;

  const invoke = (args: TArgs, time: number): void => {
    lastInvokeTime = time;
    fn(...args);
    lastArgs = null;
  };

  const throttled = (...args: TArgs): void => {
    const now = Date.now();
    const elapsed = now - lastInvokeTime;

    if (leading && elapsed >= wait) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      invoke(args, now);
      return;
    }

    lastArgs = args;

    if (!timer && trailing) {
      timer = setTimeout(() => {
        timer = null;
        if (lastArgs) invoke(lastArgs, Date.now());
      }, wait - elapsed);
    }
  };

  throttled.cancel = (): void => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    lastInvokeTime = 0;
    lastArgs = null;
  };

  return throttled;
};

/**
 * Memoizes a function — caches results based on the first argument.
 */
export const memoize = <TArg, TReturn>(
  fn: (arg: TArg) => TReturn,
  keyFn: (arg: TArg) => string = (a) => String(a)
): ((arg: TArg) => TReturn) & { cache: Map<string, TReturn>; clear: () => void } => {
  const cache = new Map<string, TReturn>();

  const memoized = (arg: TArg): TReturn => {
    const key = keyFn(arg);
    if (cache.has(key)) return cache.get(key)!;
    const result = fn(arg);
    cache.set(key, result);
    return result;
  };

  memoized.cache = cache;
  memoized.clear = () => cache.clear();

  return memoized;
};

// ─── String helpers ───────────────────────────────────────────────────────────

/**
 * Capitalises the first letter of a string.
 */
export const capitalise = (str: string): string =>
  str.length === 0 ? str : str[0]!.toUpperCase() + str.slice(1);

/**
 * Converts a string to title case.
 */
export const toTitleCase = (str: string): string =>
  str.replace(/\b\w/g, (c) => c.toUpperCase());

/**
 * Converts camelCase or PascalCase to kebab-case.
 */
export const toKebabCase = (str: string): string =>
  str
    .replace(/([a-z])([A-Z])/g, "$1-$2")
    .replace(/[\s_]+/g, "-")
    .toLowerCase();

/**
 * Converts kebab-case or snake_case to camelCase.
 */
export const toCamelCase = (str: string): string =>
  str
    .toLowerCase()
    .replace(/[-_\s](.)/g, (_, char: string) => char.toUpperCase());

/**
 * Converts a string to snake_case.
 */
export const toSnakeCase = (str: string): string =>
  str
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/[\s-]+/g, "_")
    .toLowerCase();

/**
 * Truncates a string to the given length, appending a suffix.
 */
export const truncate = (
  str: string,
  maxLength: number,
  suffix = "…"
): string =>
  str.length <= maxLength ? str : str.slice(0, maxLength - suffix.length) + suffix;

/**
 * Truncates a string in the middle, preserving the start and end.
 *
 * @example truncateMiddle("abcdefghij", 7) → "abc…hij"
 */
export const truncateMiddle = (
  str: string,
  maxLength: number,
  separator = "…"
): string => {
  if (str.length <= maxLength) return str;
  const half = Math.floor((maxLength - separator.length) / 2);
  return `${str.slice(0, half)}${separator}${str.slice(str.length - half)}`;
};

/**
 * Pads a string to the left.
 */
export const padLeft = (
  str: string,
  length: number,
  char = " "
): string => str.padStart(length, char);

/**
 * Pads a string to the right.
 */
export const padRight = (
  str: string,
  length: number,
  char = " "
): string => str.padEnd(length, char);

/**
 * Removes duplicate whitespace and trims a string.
 */
export const normaliseWhitespace = (str: string): string =>
  str.replace(/\s+/g, " ").trim();

/**
 * Counts occurrences of a substring within a string.
 */
export const countOccurrences = (
  str: string,
  search: string
): number => {
  if (!search) return 0;
  let count = 0;
  let index = 0;
  while ((index = str.indexOf(search, index)) !== -1) {
    count++;
    index += search.length;
  }
  return count;
};

/**
 * Generates a random alphanumeric string of the given length.
 */
export const randomString = (length: number): string => {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  const randomValues = new Uint32Array(length);
  crypto.getRandomValues(randomValues);
  for (let i = 0; i < length; i++) {
    result += chars[randomValues[i]! % chars.length];
  }
  return result;
};

/**
 * Generates a nanoid-style unique ID.
 */
export const nanoid = (size = 21): string => randomString(size);

/**
 * Interpolates template variables into a string.
 *
 * @example interpolate("Hello, {{name}}!", { name: "World" }) → "Hello, World!"
 */
export const interpolate = (
  template: string,
  variables: Record<string, string | number | boolean>
): string =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key: string) =>
    key in variables ? String(variables[key]) : `{{${key}}}`
  );

/**
 * Returns a pluralised word based on the count.
 *
 * @example pluralise(1, "item") → "1 item", pluralise(2, "item") → "2 items"
 */
export const pluralise = (
  count: number,
  singular: string,
  plural?: string
): string =>
  `${count} ${count === 1 ? singular : (plural ?? `${singular}s`)}`;

/**
 * Compares two strings case-insensitively.
 */
export const equalsIgnoreCase = (a: string, b: string): boolean =>
  a.toLowerCase() === b.toLowerCase();

/**
 * Checks if a string contains a substring case-insensitively.
 */
export const includesIgnoreCase = (str: string, search: string): boolean =>
  str.toLowerCase().includes(search.toLowerCase());