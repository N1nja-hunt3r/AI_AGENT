import * as React from "react"

import { cn } from "@/lib/utils"

type BadgeVariant = "default" | "secondary" | "outline" | "destructive"

interface BadgeProps extends React.ComponentProps<"span"> {
  variant?: BadgeVariant
}

function Badge({
  className,
  variant = "default",
  ...props
}: BadgeProps) {
  return (
    <span
      data-slot="badge"
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium transition-colors",
        variant === "default" &&
          "border-transparent bg-foreground text-background",
        variant === "secondary" &&
          "border-transparent bg-secondary text-secondary-foreground",
        variant === "outline" &&
          "text-foreground",
        variant === "destructive" &&
          "border-transparent bg-destructive text-destructive-foreground",
        className
      )}
      {...props}
    />
  )
}

export { Badge, type BadgeProps, type BadgeVariant }
