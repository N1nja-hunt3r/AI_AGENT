import { memo, useRef, useCallback } from "react";
import { Paperclip } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatAttachment } from "./types";

interface FileUploaderProps {
  attachments: ChatAttachment[];
  onAttach: (files: File[]) => void;
  onRemove: (id: string) => void;
  disabled?: boolean;
  className?: string;
}

export const FileUploader = memo<FileUploaderProps>(
  ({ onAttach, disabled = false, className }) => {
    const inputRef = useRef<HTMLInputElement>(null);

    const handleClick = useCallback(() => {
      inputRef.current?.click();
    }, []);

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files ?? []);
        if (files.length > 0) {
          onAttach(files);
        }
        if (inputRef.current) inputRef.current.value = "";
      },
      [onAttach]
    );

    return (
      <>
        <input
          ref={inputRef}
          type="file"
          multiple
          onChange={handleChange}
          className="hidden"
          aria-hidden="true"
        />
        <button
          type="button"
          onClick={handleClick}
          disabled={disabled}
          className={cn(
            "flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-gray-500 transition-all hover:text-gray-300 hover:bg-white/[0.06]",
            disabled && "opacity-50 cursor-not-allowed",
            className
          )}
          aria-label="Attach files"
        >
          <Paperclip className="h-3.5 w-3.5" />
          <span>Attach</span>
        </button>
      </>
    );
  }
);

FileUploader.displayName = "FileUploader";
