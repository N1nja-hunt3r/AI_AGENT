// frontend/src/utils/markdown.ts

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MarkdownParseOptions {
  allow_html?: boolean;
  allow_images?: boolean;
  allow_links?: boolean;
  sanitize?: boolean;
  base_url?: string;
}

export interface ExtractedCodeBlock {
  language: string;
  code: string;
  index: number;
  raw: string;
}

export interface ExtractedLink {
  text: string;
  href: string;
  title?: string;
}

export interface ExtractedHeading {
  level: 1 | 2 | 3 | 4 | 5 | 6;
  text: string;
  id: string;
  raw: string;
}

export interface TableOfContentsItem {
  id: string;
  text: string;
  level: number;
  children: TableOfContentsItem[];
}

// ─── Basic escape ─────────────────────────────────────────────────────────────

const HTML_ESCAPE_MAP: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#x27;",
  "/": "&#x2F;",
};

/**
 * Escapes HTML special characters to prevent XSS.
 */
export const escapeHtml = (str: string): string =>
  str.replace(/[&<>"'/]/g, (char) => HTML_ESCAPE_MAP[char] ?? char);

/**
 * Unescapes HTML entities to plain text.
 */
export const unescapeHtml = (str: string): string =>
  str
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&#x2F;/g, "/");

/**
 * Escapes Markdown special characters.
 */
export const escapeMarkdown = (str: string): string =>
  str.replace(/([\\`*_{}[\]()#+.!|-])/g, "\\$1");

// ─── Extraction ───────────────────────────────────────────────────────────────

/**
 * Extracts all fenced code blocks from a Markdown string.
 */
export const extractCodeBlocks = (markdown: string): ExtractedCodeBlock[] => {
  const FENCE_RE = /^```([^\n]*)\n([\s\S]*?)^```$/gm;
  const blocks: ExtractedCodeBlock[] = [];
  let match: RegExpExecArray | null;
  let index = 0;

  while ((match = FENCE_RE.exec(markdown)) !== null) {
    blocks.push({
      language: match[1]?.trim() ?? "",
      code: match[2] ?? "",
      index,
      raw: match[0],
    });
    index++;
  }

  return blocks;
};

/**
 * Extracts all inline and reference links from Markdown.
 */
export const extractLinks = (markdown: string): ExtractedLink[] => {
  const INLINE_LINK_RE = /\[([^\]]*)\]\(([^)]+?)(?:\s+"([^"]*)")?\)/g;
  const links: ExtractedLink[] = [];
  let match: RegExpExecArray | null;

  while ((match = INLINE_LINK_RE.exec(markdown)) !== null) {
    links.push({
      text: match[1] ?? "",
      href: match[2] ?? "",
      title: match[3],
    });
  }

  return links;
};

/**
 * Extracts all headings from a Markdown string.
 */
export const extractHeadings = (markdown: string): ExtractedHeading[] => {
  const HEADING_RE = /^(#{1,6})\s+(.+)$/gm;
  const headings: ExtractedHeading[] = [];
  let match: RegExpExecArray | null;

  while ((match = HEADING_RE.exec(markdown)) !== null) {
    const level = (match[1]?.length ?? 1) as 1 | 2 | 3 | 4 | 5 | 6;
    const text = match[2]?.trim() ?? "";
    headings.push({
      level,
      text,
      id: slugifyHeading(text),
      raw: match[0],
    });
  }

  return headings;
};

/**
 * Generates a table of contents structure from Markdown headings.
 */
export const generateTableOfContents = (
  markdown: string
): TableOfContentsItem[] => {
  const headings = extractHeadings(markdown);
  const root: TableOfContentsItem[] = [];
  const stack: TableOfContentsItem[] = [];

  for (const heading of headings) {
    const item: TableOfContentsItem = {
      id: heading.id,
      text: heading.text,
      level: heading.level,
      children: [],
    };

    while (stack.length > 0 && stack[stack.length - 1]!.level >= heading.level) {
      stack.pop();
    }

    if (stack.length === 0) {
      root.push(item);
    } else {
      stack[stack.length - 1]!.children.push(item);
    }

    stack.push(item);
  }

  return root;
};

// ─── String manipulation ──────────────────────────────────────────────────────

/**
 * Strips all Markdown syntax and returns plain text.
 */
export const stripMarkdown = (markdown: string): string => {
  return markdown
    // Remove HTML tags
    .replace(/<[^>]*>/g, "")
    // Remove fenced code blocks
    .replace(/```[\s\S]*?```/gm, "")
    // Remove inline code
    .replace(/`([^`]+)`/g, "$1")
    // Remove headings
    .replace(/^#{1,6}\s+/gm, "")
    // Remove blockquotes
    .replace(/^>\s*/gm, "")
    // Remove horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, "")
    // Remove bold/italic (***), (**), (*)
    .replace(/(\*{1,3}|_{1,3})(.+?)\1/g, "$2")
    // Remove links but keep text
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    // Remove images
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    // Remove reference-style links
    .replace(/^\[[^\]]*\]:.*$/gm, "")
    // Remove list markers
    .replace(/^[\s]*[-*+]\s+/gm, "")
    .replace(/^[\s]*\d+\.\s+/gm, "")
    // Remove table pipes
    .replace(/\|/g, " ")
    // Remove excessive whitespace
    .replace(/\n{3,}/g, "\n\n")
    .trim();
};

/**
 * Converts a heading text to a URL-safe anchor ID.
 */
export const slugifyHeading = (text: string): string =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");

/**
 * Strips the first H1 heading from markdown (commonly used to extract titles).
 */
export const stripTitle = (markdown: string): string =>
  markdown.replace(/^#\s+.+\n?/m, "").trimStart();

/**
 * Extracts the first H1 heading from markdown.
 */
export const extractTitle = (markdown: string): string | null => {
  const match = /^#\s+(.+)$/m.exec(markdown);
  return match?.[1]?.trim() ?? null;
};

// ─── Preview ──────────────────────────────────────────────────────────────────

/**
 * Generates a plain-text excerpt from markdown.
 * Strips all syntax and truncates to the specified length.
 */
export const generateExcerpt = (
  markdown: string,
  maxLength = 160,
  suffix = "…"
): string => {
  const plain = stripMarkdown(markdown);
  if (plain.length <= maxLength) return plain;
  const trimmed = plain.slice(0, maxLength).trimEnd();
  return trimmed + suffix;
};

/**
 * Counts approximate word count in markdown content.
 */
export const countWords = (markdown: string): number => {
  const plain = stripMarkdown(markdown);
  const words = plain.match(/\b\w+\b/g);
  return words ? words.length : 0;
};

/**
 * Estimates reading time in minutes.
 */
export const estimateReadingTime = (
  markdown: string,
  wordsPerMinute = 200
): number => {
  const words = countWords(markdown);
  return Math.max(1, Math.ceil(words / wordsPerMinute));
};

// ─── Code block utilities ─────────────────────────────────────────────────────

/**
 * Wraps a string in a markdown fenced code block.
 */
export const wrapInCodeBlock = (
  code: string,
  language = ""
): string => `\`\`\`${language}\n${code}\n\`\`\``;

/**
 * Wraps a string in an inline code span.
 */
export const wrapInCode = (code: string): string => `\`${code}\``;

/**
 * Normalises language identifiers to canonical names.
 */
export const normaliseLanguage = (lang: string): string => {
  const MAP: Record<string, string> = {
    js: "javascript",
    ts: "typescript",
    jsx: "jsx",
    tsx: "tsx",
    py: "python",
    rb: "ruby",
    sh: "bash",
    shell: "bash",
    zsh: "bash",
    yml: "yaml",
    md: "markdown",
    rs: "rust",
    go: "go",
    cs: "csharp",
    "c#": "csharp",
    cpp: "cpp",
    "c++": "cpp",
    kt: "kotlin",
    swift: "swift",
    java: "java",
    php: "php",
    sql: "sql",
    html: "html",
    css: "css",
    scss: "scss",
    less: "less",
    json: "json",
    xml: "xml",
    dockerfile: "dockerfile",
    tf: "terraform",
    makefile: "makefile",
  };
  return MAP[lang.toLowerCase()] ?? lang.toLowerCase();
};

// ─── Transformation ───────────────────────────────────────────────────────────

/**
 * Converts a plain text string to basic markdown by preserving
 * paragraph breaks and escaping special characters.
 */
export const textToMarkdown = (text: string): string =>
  text
    .split(/\n{2,}/)
    .map((para) => escapeMarkdown(para.trim()))
    .filter(Boolean)
    .join("\n\n");

/**
 * Removes trailing whitespace from each line in a markdown string.
 */
export const normaliseMarkdown = (markdown: string): string =>
  markdown
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();