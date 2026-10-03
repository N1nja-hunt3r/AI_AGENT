// frontend/src/utils/validation.ts

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface FieldValidationResult {
  valid: boolean;
  error: string | null;
}

export type ValidatorFn<T = string> = (value: T) => FieldValidationResult;

export interface PasswordStrength {
  score: 0 | 1 | 2 | 3 | 4;
  label: "very_weak" | "weak" | "fair" | "strong" | "very_strong";
  feedback: string[];
}

// ─── Core validator builder ───────────────────────────────────────────────────

export const ok = (): FieldValidationResult => ({ valid: true, error: null });
export const fail = (error: string): FieldValidationResult => ({
  valid: false,
  error,
});

/**
 * Runs multiple validators and returns aggregated result.
 */
export const validate = <T>(
  value: T,
  validators: ValidatorFn<T>[]
): ValidationResult => {
  const errors: string[] = [];
  for (const validator of validators) {
    const result = validator(value);
    if (!result.valid && result.error) {
      errors.push(result.error);
    }
  }
  return { valid: errors.length === 0, errors };
};

/**
 * Runs validators and stops at the first failure.
 */
export const validateFirst = <T>(
  value: T,
  validators: ValidatorFn<T>[]
): FieldValidationResult => {
  for (const validator of validators) {
    const result = validator(value);
    if (!result.valid) return result;
  }
  return ok();
};

// ─── String validators ────────────────────────────────────────────────────────

export const required =
  (message = "This field is required"): ValidatorFn<string> =>
  (value) =>
    value.trim().length > 0 ? ok() : fail(message);

export const minLength =
  (min: number, message?: string): ValidatorFn<string> =>
  (value) =>
    value.length >= min
      ? ok()
      : fail(message ?? `Must be at least ${min} characters`);

export const maxLength =
  (max: number, message?: string): ValidatorFn<string> =>
  (value) =>
    value.length <= max
      ? ok()
      : fail(message ?? `Must be at most ${max} characters`);

export const exactLength =
  (length: number, message?: string): ValidatorFn<string> =>
  (value) =>
    value.length === length
      ? ok()
      : fail(message ?? `Must be exactly ${length} characters`);

export const matches =
  (pattern: RegExp, message = "Invalid format"): ValidatorFn<string> =>
  (value) =>
    pattern.test(value) ? ok() : fail(message);

export const notMatches =
  (pattern: RegExp, message = "Invalid format"): ValidatorFn<string> =>
  (value) =>
    !pattern.test(value) ? ok() : fail(message);

export const oneOf =
  <T extends string>(
    values: readonly T[],
    message?: string
  ): ValidatorFn<string> =>
  (value) =>
    (values as readonly string[]).includes(value)
      ? ok()
      : fail(message ?? `Must be one of: ${values.join(", ")}`);

// ─── Specific format validators ───────────────────────────────────────────────

const EMAIL_RE =
  /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$/;

export const isEmail = (
  message = "Invalid email address"
): ValidatorFn<string> =>
  matches(EMAIL_RE, message);

export const isUrl = (
  message = "Invalid URL"
): ValidatorFn<string> =>
  (value) => {
    try {
      new URL(value);
      return ok();
    } catch {
      return fail(message);
    }
  };

export const isHttpUrl = (
  message = "Must be a valid HTTP or HTTPS URL"
): ValidatorFn<string> =>
  (value) => {
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol)
        ? ok()
        : fail(message);
    } catch {
      return fail(message);
    }
  };

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const isUUID = (
  message = "Invalid UUID"
): ValidatorFn<string> =>
  matches(UUID_RE, message);

const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const isSlug = (
  message = "Must be a valid slug (lowercase letters, numbers, hyphens)"
): ValidatorFn<string> =>
  matches(SLUG_RE, message);

const HEX_COLOR_RE = /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;

export const isHexColor = (
  message = "Invalid hex color"
): ValidatorFn<string> =>
  matches(HEX_COLOR_RE, message);

export const isJSON = (
  message = "Invalid JSON"
): ValidatorFn<string> =>
  (value) => {
    try {
      JSON.parse(value);
      return ok();
    } catch {
      return fail(message);
    }
  };

export const isAlphanumeric = (
  message = "Only letters and numbers are allowed"
): ValidatorFn<string> =>
  matches(/^[a-zA-Z0-9]+$/, message);

export const noWhitespace = (
  message = "No spaces allowed"
): ValidatorFn<string> =>
  notMatches(/\s/, message);

// ─── Numeric validators ───────────────────────────────────────────────────────

export const isNumber = (
  message = "Must be a valid number"
): ValidatorFn<unknown> =>
  (value) =>
    !isNaN(Number(value)) ? ok() : fail(message);

export const minValue =
  (min: number, message?: string): ValidatorFn<number> =>
  (value) =>
    value >= min ? ok() : fail(message ?? `Must be at least ${min}`);

export const maxValue =
  (max: number, message?: string): ValidatorFn<number> =>
  (value) =>
    value <= max ? ok() : fail(message ?? `Must be at most ${max}`);

export const inRange =
  (
    min: number,
    max: number,
    message?: string
  ): ValidatorFn<number> =>
  (value) =>
    value >= min && value <= max
      ? ok()
      : fail(message ?? `Must be between ${min} and ${max}`);

export const isInteger = (
  message = "Must be a whole number"
): ValidatorFn<number> =>
  (value) =>
    Number.isInteger(value) ? ok() : fail(message);

export const isPositive = (
  message = "Must be a positive number"
): ValidatorFn<number> =>
  (value) =>
    value > 0 ? ok() : fail(message);

export const isNonNegative = (
  message = "Must be zero or a positive number"
): ValidatorFn<number> =>
  (value) =>
    value >= 0 ? ok() : fail(message);

// ─── Password strength ────────────────────────────────────────────────────────

/**
 * Calculates password strength and returns a structured result.
 */
export const checkPasswordStrength = (
  password: string
): PasswordStrength => {
  const feedback: string[] = [];
  let score = 0;

  if (password.length >= 8) score++;
  else feedback.push("Use at least 8 characters");

  if (password.length >= 12) score++;
  else if (password.length >= 8)
    feedback.push("Consider using 12+ characters for stronger security");

  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
  else feedback.push("Mix uppercase and lowercase letters");

  if (/\d/.test(password)) score++;
  else feedback.push("Include at least one number");

  if (/[!@#$%^&*()_+=[\]{};':"\\|,.<>/?`~-]/.test(password)) score++;
  else feedback.push("Include at least one special character");

  const clampedScore = Math.min(score, 4) as PasswordStrength["score"];
  const labels: PasswordStrength["label"][] = [
    "very_weak",
    "weak",
    "fair",
    "strong",
    "very_strong",
  ];

  return {
    score: clampedScore,
    label: labels[clampedScore] ?? "very_weak",
    feedback,
  };
};

export const isStrongPassword =
  (
    minScore: PasswordStrength["score"] = 3,
    message = "Password is not strong enough"
  ): ValidatorFn<string> =>
  (value) => {
    const strength = checkPasswordStrength(value);
    return strength.score >= minScore ? ok() : fail(message);
  };

// ─── File validators ──────────────────────────────────────────────────────────

export const isAllowedFileType =
  (
    allowedTypes: string[],
    message?: string
  ): ValidatorFn<File> =>
  (file) =>
    allowedTypes.includes(file.type)
      ? ok()
      : fail(
          message ??
            `File type "${file.type}" is not allowed. Accepted: ${allowedTypes.join(", ")}`
        );

export const maxFileSize =
  (maxBytes: number, message?: string): ValidatorFn<File> =>
  (file) =>
    file.size <= maxBytes
      ? ok()
      : fail(
          message ??
            `File size must not exceed ${(maxBytes / 1024 / 1024).toFixed(1)} MB`
        );

// ─── Array validators ─────────────────────────────────────────────────────────

export const minItems =
  <T>(min: number, message?: string): ValidatorFn<T[]> =>
  (value) =>
    value.length >= min
      ? ok()
      : fail(message ?? `Select at least ${min} item${min !== 1 ? "s" : ""}`);

export const maxItems =
  <T>(max: number, message?: string): ValidatorFn<T[]> =>
  (value) =>
    value.length <= max
      ? ok()
      : fail(message ?? `Select at most ${max} item${max !== 1 ? "s" : ""}`);

// ─── Safe parse helpers ───────────────────────────────────────────────────────

/**
 * Safely parses a JSON string. Returns null on failure.
 */
export const safeParseJSON = <T = unknown>(raw: string): T | null => {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
};

/**
 * Safely parses a number. Returns null on failure.
 */
export const safeParseNumber = (value: unknown): number | null => {
  const n = Number(value);
  return isNaN(n) ? null : n;
};

/**
 * Safely parses an integer. Returns null on failure.
 */
export const safeParseInt = (
  value: unknown,
  radix = 10
): number | null => {
  const n = parseInt(String(value), radix);
  return isNaN(n) ? null : n;
};

/**
 * Safely parses a float. Returns null on failure.
 */
export const safeParseFloat = (value: unknown): number | null => {
  const n = parseFloat(String(value));
  return isNaN(n) ? null : n;
};

/**
 * Safely parses a boolean from a string.
 */
export const safeParseBoolean = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value;
  if (value === "true" || value === "1" || value === "yes") return true;
  if (value === "false" || value === "0" || value === "no") return false;
  return null;
};

/**
 * Type guard: checks if a value is a non-null object.
 */
export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Type guard: checks if a value is a non-empty string.
 */
export const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

/**
 * Type guard: checks if a value is a finite number.
 */
export const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && isFinite(value);