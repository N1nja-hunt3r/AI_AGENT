import { Suspense, type ReactNode } from "react";

export const PageSuspense = ({ children }: { children: ReactNode }) => (
  <Suspense
    fallback={
      <div className="flex h-full w-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    }
  >
    {children}
  </Suspense>
);
