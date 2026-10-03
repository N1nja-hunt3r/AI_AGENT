import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Brain,
  Route,
  Code2,
  Search,
  Database,
  Eye,
  Zap,
  FileText,
  Globe,
  Monitor,
  Mic,
  Terminal,
  Clipboard,
} from "lucide-react";

export type PipelineStage = "planner" | "classifier" | "agent" | "capability" | "response";

interface PipelineProps {
  activeStage: PipelineStage;
  activeAgent?: string;
  activeCapability?: string;
  className?: string;
}

interface StageConfig {
  key: PipelineStage;
  label: string;
  icon: React.ElementType;
  color: string;
}

const STAGES: StageConfig[] = [
  { key: "planner", label: "DeepSeek Planner", icon: Brain, color: "violet" },
  { key: "classifier", label: "Intent Classifier", icon: Route, color: "blue" },
  { key: "agent", label: "Agent", icon: Code2, color: "emerald" },
  { key: "capability", label: "Capability", icon: Zap, color: "amber" },
  { key: "response", label: "Response", icon: Mic, color: "rose" },
];

const AGENTS = [
  { id: "coding", label: "Coding", icon: Code2, provider: "Llama", color: "blue" },
  { id: "research", label: "Research", icon: Search, provider: "Qwen", color: "emerald" },
  { id: "memory", label: "Memory", icon: Database, provider: "Qwen", color: "purple" },
  { id: "vision", label: "Vision", icon: Eye, provider: "Vision", color: "amber" },
  { id: "automation", label: "Automation", icon: Zap, provider: "Tools", color: "rose" },
];

const CAPABILITIES = [
  { id: "files", label: "Files", icon: FileText },
  { id: "browser", label: "Browser", icon: Globe },
  { id: "computer", label: "Computer", icon: Monitor },
  { id: "voice", label: "Voice", icon: Mic },
  { id: "terminal", label: "Terminal", icon: Terminal },
  { id: "clipboard", label: "Clipboard", icon: Clipboard },
];

function StageIcon({ stage, active }: { stage: StageConfig; active: boolean }) {
  const Icon = stage.icon;
  return (
    <motion.div
      animate={active ? { scale: [1, 1.15, 1] } : {}}
      transition={{ duration: 0.5, repeat: active ? Infinity : 0, repeatDelay: 1.5 }}
      className={cn(
        "flex h-8 w-8 items-center justify-center rounded-lg transition-all duration-300",
        active
          ? `bg-${stage.color}-500/20 text-${stage.color}-400 shadow-lg shadow-${stage.color}-500/10`
          : "bg-white/4 text-gray-600"
      )}
    >
      <Icon className="h-4 w-4" />
    </motion.div>
  );
}

export function ChatPipeline({ activeStage, activeAgent, activeCapability, className }: PipelineProps) {
  return (
    <div className={cn("w-full", className)}>
      {/* Main pipeline stages */}
      <div className="flex items-center justify-center gap-1">
        {STAGES.map((stage, i) => {
          const isActive = activeStage === stage.key;
          return (
            <div key={stage.key} className="flex items-center">
              <motion.div
                animate={isActive ? { y: -2 } : { y: 0 }}
                className={cn(
                  "flex flex-col items-center gap-1 px-2 py-1.5 rounded-lg transition-all duration-300",
                  isActive ? "bg-white/6" : "opacity-50"
                )}
              >
                <StageIcon stage={stage} active={isActive} />
                <span className={cn(
                  "text-[10px] font-medium whitespace-nowrap",
                  isActive ? "text-gray-300" : "text-gray-600"
                )}>
                  {stage.label}
                </span>
              </motion.div>
              {i < STAGES.length - 1 && (
                <div className={cn(
                  "h-px w-6 transition-colors duration-300",
                  i < STAGES.findIndex((s) => s.key === activeStage) ? "bg-violet-500/40" : "bg-white/6"
                )} />
              )}
            </div>
          );
        })}
      </div>

      {/* Current stage detail */}
      <AnimatePresence mode="wait">
        {(activeStage === "agent" || activeStage === "capability") && (
          <motion.div
            key={activeStage}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-2 overflow-hidden"
          >
            {activeStage === "agent" && (
              <div className="flex items-center justify-center gap-2">
                {AGENTS.map((agent) => {
                  const isSelected = activeAgent === agent.id;
                  const Icon = agent.icon;
                  return (
                    <motion.div
                      key={agent.id}
                      animate={isSelected ? { scale: 1.05 } : { scale: 0.95 }}
                      className={cn(
                        "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 transition-all duration-200",
                        isSelected
                          ? `bg-${agent.color}-500/15 text-${agent.color}-400 ring-1 ring-${agent.color}-500/30`
                          : "bg-white/3 text-gray-600"
                      )}
                    >
                      <Icon className="h-3 w-3" />
                      <span className="text-[11px] font-medium">{agent.label}</span>
                      <span className="text-[9px] text-gray-600">{agent.provider}</span>
                    </motion.div>
                  );
                })}
              </div>
            )}

            {activeStage === "capability" && (
              <div className="flex items-center justify-center gap-2 flex-wrap">
                {CAPABILITIES.map((cap) => {
                  const isSelected = activeCapability === cap.id;
                  const Icon = cap.icon;
                  return (
                    <div
                      key={cap.id}
                      className={cn(
                        "flex items-center gap-1.5 rounded-lg px-2 py-1 transition-all duration-200",
                        isSelected
                          ? "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30"
                          : "bg-white/3 text-gray-600"
                      )}
                    >
                      <Icon className="h-3 w-3" />
                      <span className="text-[10px]">{cap.label}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
