import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Highlight, themes } from "prism-react-renderer";
import { cn } from "@/lib/utils";

interface MarkdownRendererProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

export const MarkdownRenderer = memo<MarkdownRendererProps>(
  ({ content, isStreaming = false, className }) => {
    return (
      <div className={cn("prose prose-sm dark:prose-invert max-w-none leading-relaxed", isStreaming && "streaming", className)}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ className, children, ...props }) {
              const match = /language-(\w+)/.exec(className ?? "");
              const codeString = String(children).replace(/\n$/, "");
              if (match) {
                return (
                  <Highlight theme={themes.nightOwl} code={codeString} language={match[1]}>
                    {({ tokens, getLineProps, getTokenProps }) => (
                      <pre className="overflow-x-auto rounded-lg border border-white/10 bg-gray-900 p-4 text-sm">
                        <code>
                          {tokens.map((line, i) => (
                            <span key={i} {...getLineProps({ line })}>
                              {line.map((token, key) => (
                                <span key={key} {...getTokenProps({ token })} />
                              ))}
                              {i < tokens.length - 1 && "\n"}
                            </span>
                          ))}
                        </code>
                      </pre>
                    )}
                  </Highlight>
                );
              }
              return (
                <code
                  className={cn(
                    "rounded bg-gray-800 px-1.5 py-0.5 text-sm font-mono text-violet-300",
                    className
                  )}
                  {...props}
                >
                  {children}
                </code>
              );
            },
            pre({ children }) {
              return <>{children}</>;
            },
            a({ href, children }) {
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-violet-400 hover:text-violet-300 underline underline-offset-2"
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }
);

MarkdownRenderer.displayName = "MarkdownRenderer";
