import { memo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatModel } from "./types";

interface ModelSelectorProps {
  models: ChatModel[];
  selectedModel: ChatModel | null;
  onModelChange: (model: ChatModel) => void;
  disabled?: boolean;
  className?: string;
}

export const ModelSelector = memo<ModelSelectorProps>(
  ({ models, selectedModel, onModelChange, disabled = false, className }) => {
    const [open, setOpen] = useState(false);

    const handleSelect = useCallback(
      (model: ChatModel) => {
        onModelChange(model);
        setOpen(false);
      },
      [onModelChange]
    );

    return (
      <div className={cn("relative", className)}>
        <button
          onClick={() => !disabled && setOpen((p) => !p)}
          disabled={disabled}
          className={cn(
            "flex items-center gap-2 rounded-xl px-3 py-1.5 border border-white/10 bg-white/[0.04] text-sm font-medium text-gray-200 transition-all hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500",
            disabled && "opacity-50 cursor-not-allowed"
          )}
          aria-haspopup="listbox"
          aria-expanded={open}
        >
          <Cpu className="h-4 w-4 text-violet-400" />
          <span className="text-[13px]">{selectedModel?.name ?? "Select model"}</span>
          <ChevronDown className={cn("h-3 w-3 text-gray-500 transition-transform", open && "rotate-180")} />
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.96 }}
              transition={{ duration: 0.15 }}
              className="absolute left-1/2 top-11 z-50 w-64 -translate-x-1/2 rounded-2xl border border-white/10 bg-gray-900/95 backdrop-blur-xl p-1.5 shadow-2xl shadow-black/40"
              role="listbox"
            >
              <p className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-gray-600">
                Models
              </p>
              {models.map((model) => (
                <button
                  key={model.id}
                  role="option"
                  aria-selected={selectedModel?.id === model.id}
                  onClick={() => handleSelect(model)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-xl px-3 py-2.5 transition-colors hover:bg-white/[0.06] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500",
                    selectedModel?.id === model.id && "bg-violet-500/10"
                  )}
                >
                  <div className="flex flex-col items-start">
                    <span className="text-sm font-medium text-gray-200">{model.name}</span>
                    <span className="text-[11px] text-gray-500">{model.provider}</span>
                  </div>
                  {selectedModel?.id === model.id && (
                    <div className="h-2 w-2 rounded-full bg-violet-400" />
                  )}
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }
);

ModelSelector.displayName = "ModelSelector";
