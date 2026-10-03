import React, { memo } from "react";
import {
  Trash2,
  Download,
  Share2,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Settings2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { ModelSelector } from "./ModelSelector";
import type { ChatModel } from "./types";

interface ToolbarProps {
  models: ChatModel[];
  selectedModel: ChatModel | null;
  onModelChange: (model: ChatModel) => void;
  onNewChat?: () => void;
  onClearChat?: () => void;
  onExport?: () => void;
  onShare?: () => void;
  onToggleSidebar?: () => void;
  onSettings?: () => void;
  isSidebarOpen?: boolean;
  isStreaming?: boolean;
  className?: string;
  conversationTitle?: string;
}

interface ToolbarButtonProps {
  icon: React.ElementType;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "ghost" | "destructive";
  className?: string;
}

const ToolbarButton = memo<ToolbarButtonProps>(
  ({ icon: Icon, label, onClick, disabled = false, className }) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClick}
          disabled={disabled}
          aria-label={label}
          className={cn(
            "h-8 w-8 text-muted-foreground transition-all hover:text-foreground hover:bg-muted/60",
            className
          )}
        >
          <Icon className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  )
);

ToolbarButton.displayName = "ToolbarButton";

export const Toolbar = memo<ToolbarProps>(
  ({
    models,
    selectedModel,
    onModelChange,
    onNewChat,
    onClearChat,
    onExport,
    onShare,
    onToggleSidebar,
    onSettings,
    isSidebarOpen = true,
    isStreaming = false,
    className,
    conversationTitle,
  }) => {
    return (
      <header
        className={cn(
          "grid h-14 w-full grid-cols-[1fr_auto_1fr] items-center gap-3 px-6",
          "bg-background/60 backdrop-blur-xl z-10 shrink-0",
          className
        )}
        role="toolbar"
        aria-label="Chat toolbar"
      >
        <div className="flex items-center gap-1.5 justify-self-start">
          <ToolbarButton
            icon={isSidebarOpen ? PanelLeftClose : PanelLeftOpen}
            label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
            onClick={onToggleSidebar}
          />
          {conversationTitle && (
            <>
              <Separator orientation="vertical" className="mx-1 h-4" />
              <span className="hidden truncate text-xs font-medium text-foreground/60 sm:block max-w-[180px]">
                {conversationTitle}
              </span>
            </>
          )}
        </div>

        <div className="flex items-center justify-center">
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onModelChange={onModelChange}
            disabled={isStreaming}
          />
        </div>

        <div className="flex items-center gap-1 justify-self-end pr-1">
          <ToolbarButton icon={Plus} label="New conversation" onClick={onNewChat} disabled={isStreaming} />
          <div className="flex items-center rounded-lg border border-border/30 px-1 py-0.5">
            <ToolbarButton icon={Share2} label="Share conversation" onClick={onShare} disabled={isStreaming} />
            <ToolbarButton icon={Download} label="Export conversation" onClick={onExport} disabled={isStreaming} />
            <span className="mx-0.5 h-4 w-px bg-border/40" />
            <ToolbarButton icon={Trash2} label="Clear conversation" onClick={onClearChat} disabled={isStreaming} variant="destructive" />
          </div>
          <ToolbarButton icon={Settings2} label="Chat settings" onClick={onSettings} />
        </div>
      </header>
    );
  }
);

Toolbar.displayName = "Toolbar";
