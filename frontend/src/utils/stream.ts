// frontend/src/utils/stream.ts

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SSEMessage<TData = unknown> {
  id?: string;
  event?: string;
  data: TData;
  retry?: number;
}

export type SSEParser<TData = unknown> = (raw: string) => TData | null;

export interface StreamReaderOptions {
  signal?: AbortSignal;
  on_chunk?: (chunk: string) => void;
  on_message?: (message: SSEMessage) => void;
  on_error?: (error: Error) => void;
  on_close?: () => void;
  decode?: "text" | "json";
  reconnect?: boolean;
  reconnect_delay_ms?: number;
}

export interface StreamTokenAccumulator {
  push: (token: string) => void;
  flush: () => string;
  peek: () => string;
  reset: () => void;
  length: () => number;
}

export interface JsonStreamOptions {
  signal?: AbortSignal;
}

// ─── SSE line parser ─────────────────────────────────────────────────────────

/**
 * Parses a single SSE event block into a structured SSEMessage.
 * Returns null if the block is empty or a comment-only block.
 */
export const parseSSEBlock = <TData = unknown>(
  block: string,
  parse: SSEParser<TData> = (raw) => {
    try {
      return JSON.parse(raw) as TData;
    } catch {
      return raw as unknown as TData;
    }
  }
): SSEMessage<TData> | null => {
  const lines = block.split("\n");
  const message: Partial<SSEMessage<TData>> = {};
  const dataLines: string[] = [];
  let hasData = false;

  for (const line of lines) {
    if (!line.trim() || line.startsWith(":")) continue;

    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) continue;

    const field = line.slice(0, colonIndex).trim();
    const value = line.slice(colonIndex + 1).trimStart();

    switch (field) {
      case "id":
        message.id = value;
        break;
      case "event":
        message.event = value;
        break;
      case "retry":
        message.retry = parseInt(value, 10);
        break;
      case "data":
        dataLines.push(value);
        hasData = true;
        break;
    }
  }

  if (!hasData) return null;

  const rawData = dataLines.join("\n");

  if (rawData === "[DONE]") return null;

  const parsedData = parse(rawData);
  if (parsedData === null) return null;

  return { ...message, data: parsedData } as SSEMessage<TData>;
};

// ─── Text decoder stream ──────────────────────────────────────────────────────

/**
 * Reads a ReadableStream<Uint8Array> and yields text chunks.
 * Handles the TextDecoder internally.
 */
export async function* readStreamAsText(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal
): AsyncGenerator<string, void, unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");

  try {
    while (true) {
      if (signal?.aborted) {
        reader.cancel();
        break;
      }

      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      if (chunk) yield chunk;
    }

    const final = decoder.decode();
    if (final) yield final;
  } finally {
    reader.releaseLock();
  }
}

// ─── SSE stream reader ────────────────────────────────────────────────────────

/**
 * Reads a Server-Sent Events stream from a Response and yields
 * typed SSEMessage objects.
 */
export async function* readSSEStream<TData = unknown>(
  response: Response,
  options: {
    signal?: AbortSignal;
    parse?: SSEParser<TData>;
  } = {}
): AsyncGenerator<SSEMessage<TData>, void, unknown> {
  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
  }

  if (!response.body) {
    throw new Error("Response body is null — cannot read stream");
  }

  const { signal, parse } = options;
  let buffer = "";

  for await (const chunk of readStreamAsText(response.body, signal)) {
    buffer += chunk;
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const message = parseSSEBlock<TData>(block, parse);
      if (message !== null) {
        yield message;
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    const message = parseSSEBlock<TData>(buffer, parse);
    if (message !== null) {
      yield message;
    }
  }
}

// ─── Token accumulator ────────────────────────────────────────────────────────

/**
 * Creates a mutable token accumulator for streaming text assembly.
 */
export const createTokenAccumulator = (): StreamTokenAccumulator => {
  let buffer = "";

  return {
    push: (token: string) => {
      buffer += token;
    },
    flush: () => {
      const result = buffer;
      buffer = "";
      return result;
    },
    peek: () => buffer,
    reset: () => {
      buffer = "";
    },
    length: () => buffer.length,
  };
};

// ─── Async iterable → array ───────────────────────────────────────────────────

/**
 * Collects all values from an async iterable into an array.
 */
export const collectAsync = async <T>(
  iter: AsyncIterable<T>
): Promise<T[]> => {
  const results: T[] = [];
  for await (const item of iter) {
    results.push(item);
  }
  return results;
};

// ─── Stream to string ─────────────────────────────────────────────────────────

/**
 * Reads a ReadableStream fully and returns the concatenated string.
 */
export const streamToString = async (
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal
): Promise<string> => {
  let result = "";
  for await (const chunk of readStreamAsText(stream, signal)) {
    result += chunk;
  }
  return result;
};

// ─── JSON stream reader ───────────────────────────────────────────────────────

/**
 * Reads a newline-delimited JSON stream and yields parsed objects.
 */
export async function* readNDJSONStream<TData = unknown>(
  response: Response,
  options: JsonStreamOptions = {}
): AsyncGenerator<TData, void, unknown> {
  if (!response.body) {
    throw new Error("Response body is null");
  }

  let buffer = "";

  for await (const chunk of readStreamAsText(response.body, options.signal)) {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        yield JSON.parse(trimmed) as TData;
      } catch {
        // Skip malformed lines
      }
    }
  }

  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer.trim()) as TData;
    } catch {
      // Skip
    }
  }
}

// ─── Abort helpers ────────────────────────────────────────────────────────────

/**
 * Creates an AbortController with an optional timeout.
 */
export const createAbortController = (
  timeoutMs?: number
): AbortController => {
  const controller = new AbortController();

  if (timeoutMs && timeoutMs > 0) {
    const timer = setTimeout(() => {
      controller.abort(new Error(`Stream timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    controller.signal.addEventListener("abort", () => clearTimeout(timer), {
      once: true,
    });
  }

  return controller;
};

/**
 * Merges multiple AbortSignals into one.
 * Aborts if any of the provided signals abort.
 */
export const mergeAbortSignals = (
  ...signals: Array<AbortSignal | undefined>
): AbortController => {
  const controller = new AbortController();

  for (const signal of signals) {
    if (!signal) continue;
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener(
      "abort",
      () => controller.abort(signal.reason),
      { once: true }
    );
  }

  return controller;
};

// ─── Throttled stream consumer ────────────────────────────────────────────────

/**
 * Yields values from an async iterable at a minimum interval.
 * Useful for rate-limiting UI updates during high-frequency streams.
 */
export async function* throttleStream<T>(
  source: AsyncIterable<T>,
  intervalMs: number
): AsyncGenerator<T, void, unknown> {
  let lastYield = 0;
  let pending: T | undefined;
  let hasPending = false;

  for await (const value of source) {
    const now = Date.now();
    if (now - lastYield >= intervalMs) {
      if (hasPending) {
        yield pending as T;
        hasPending = false;
      }
      yield value;
      lastYield = now;
    } else {
      pending = value;
      hasPending = true;
    }
  }

  if (hasPending) {
    yield pending as T;
  }
}

// ─── Batch stream ─────────────────────────────────────────────────────────────

/**
 * Batches items from an async iterable into arrays of a given size.
 */
export async function* batchStream<T>(
  source: AsyncIterable<T>,
  batchSize: number
): AsyncGenerator<T[], void, unknown> {
  let batch: T[] = [];

  for await (const item of source) {
    batch.push(item);
    if (batch.length >= batchSize) {
      yield batch;
      batch = [];
    }
  }

  if (batch.length > 0) {
    yield batch;
  }
}