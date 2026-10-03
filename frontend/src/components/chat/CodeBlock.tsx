import { memo, useState, useCallback } from "react";
import { Highlight, themes, type Language } from "prism-react-renderer";
import { Check, Copy, ChevronDown, ChevronUp, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface CodeBlockProps {
  code: string;
  language: string;
  className?: string;
  collapsible?: boolean;
  maxLines?: number;
}

const LANGUAGE_ALIASES: Record<string, string> = {
  js: "javascript",
  ts: "typescript",
  jsx: "jsx",
  tsx: "tsx",
  py: "python",
  rb: "ruby",
  sh: "bash",
  yml: "yaml",
  md: "markdown",
};

const normalizeLanguage = (lang: string): Language => {
  const normalized = LANGUAGE_ALIASES[lang.toLowerCase()] ?? lang.toLowerCase();
  return normalized as Language;
};

export const CodeBlock = memo<CodeBlockProps>(
  ({ code, language, className, collapsible = false, maxLines = 30 }) => {
    const [copied, setCopied] = useState(false);
    const [collapsed, setCollapsed] = useState(false);
    const lineCount = code.split("\n").length;
    const shouldCollapse = collapsible && lineCount > maxLines;

    const handleCopy = useCallback(async () => {
      try {
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {
        // Clipboard API not available
      }
    }, [code]);

    const normalizedLang = normalizeLanguage(language);

    return (
      <div
        className={cn(
          "group relative my-3 overflow-hidden rounded-lg border border-border/60",
          "bg-zinc-950 dark:bg-zinc-900",
          "shadow-md",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/40 bg-zinc-900/80 px-4 py-2 dark:bg-zinc-800/80">
          <div className="flex items-center gap-2">
            <Terminal className="h-3.5 w-3.5 text-zinc-400" />
            <span className="font-mono text-xs text-zinc-400 select-none">
              {language || "text"}
            </span>
          </div>

          <div className="flex items-center gap-1">
            {shouldCollapse && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCollapsed((p) => !p)}
                className="h-7 w-7 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50"
                aria-label={collapsed ? "Expand code" : "Collapse code"}
              >
                {collapsed ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronUp className="h-3.5 w-3.5" />
                )}
              </Button>
            )}

            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              className="h-7 w-7 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-all"
              aria-label="Copy code"
            >
              <AnimatePresence mode="wait" initial={false}>
                {copied ? (
                  <motion.div
                    key="check"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <Check className="h-3.5 w-3.5 text-emerald-400" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="copy"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </motion.div>
                )}
              </AnimatePresence>
            </Button>
          </div>
        </div>

        {/* Code content */}
        <AnimatePresence initial={false}>
          {!collapsed && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
            >
              <div
                className={cn(
                  "overflow-x-auto",
                  shouldCollapse && !collapsed && `max-h-[${maxLines * 1.5}rem]`
                )}
              >
                <Highlight
                  theme={themes.oneDark}
                  code={code}
                  language={normalizedLang}
                >
                  {({
                    className: hlClassName,
                    style,
                    tokens,
                    getLineProps,
                    getTokenProps,
                  }) => (
                    <pre
                      className={cn(
                        hlClassName,
                        "overflow-x-auto p-4 text-sm leading-relaxed font-mono"
                      )}
                      style={{ ...style, background: "transparent", margin: 0 }}
                    >
                      {tokens.map((line, lineIdx) => (
                        <div
                          key={lineIdx}
                          {...getLineProps({ line })}
                          className="table-row"
                        >
                          <span className="table-cell select-none pr-4 text-right text-xs text-zinc-600 w-8">
                            {lineIdx + 1}
                          </span>
                          <span className="table-cell">
                            {line.map((token, tokenIdx) => (
                              <span
                                key={tokenIdx}
                                {...getTokenProps({ token })}
                              />
                            ))}
                          </span>
                        </div>
                      ))}
                    </pre>
                  )}
                </Highlight>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {collapsed && (
          <button
            onClick={() => setCollapsed(false)}
            className="w-full py-2 text-xs text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50 transition-colors"
          >
            Show {lineCount} lines
          </button>
        )}
      </div>
    );
  }
);

CodeBlock.displayName = "CodeBlock";