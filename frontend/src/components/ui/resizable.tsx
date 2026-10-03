import * as React from "react"

import { cn } from "@/lib/utils"

interface PanelGroupProps extends React.ComponentProps<"div"> {
  direction?: "horizontal" | "vertical"
}

function ResizablePanelGroup({
  className,
  direction = "horizontal",
  ...props
}: PanelGroupProps) {
  return (
    <div
      data-slot="resizable-panel-group"
      data-direction={direction}
      className={cn(
        "flex h-full w-full",
        direction === "vertical" && "flex-col",
        className
      )}
      {...props}
    />
  )
}

interface PanelProps extends React.ComponentProps<"div"> {
  defaultSize?: number
  minSize?: number
  maxSize?: number
}

function ResizablePanel({
  className,
  ...props
}: PanelProps) {
  return (
    <div
      data-slot="resizable-panel"
      className={cn("flex-1 overflow-hidden", className)}
      {...props}
    />
  )
}

function ResizableHandle({
  className,
  withHandle,
  ...props
}: React.ComponentProps<"div"> & { withHandle?: boolean }) {
  return (
    <div
      data-slot="resizable-handle"
      className={cn(
        "relative flex w-px items-center justify-center bg-border after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2",
        "data-[direction=vertical]:h-px data-[direction=vertical]:w-full data-[direction=vertical]:after:left-0 data-[direction=vertical]:after:h-1 data-[direction=vertical]:after:w-full data-[direction=vertical]:after:translate-x-0 data-[direction=vertical]:after:-translate-y-1/2",
        className
      )}
      {...props}
    >
      {withHandle && (
        <div className="z-10 flex h-4 w-3 items-center justify-center rounded-sm border bg-border">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="size-2.5"
          >
            <circle cx="9" cy="12" r="1" />
            <circle cx="15" cy="12" r="1" />
            <circle cx="9" cy="5" r="1" />
            <circle cx="15" cy="5" r="1" />
            <circle cx="9" cy="19" r="1" />
            <circle cx="15" cy="19" r="1" />
          </svg>
        </div>
      )}
    </div>
  )
}

export { ResizableHandle, ResizablePanel, ResizablePanelGroup }
