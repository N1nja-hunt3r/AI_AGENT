import { useState, useEffect, type RefObject } from "react";

interface UseIntersectionObserverOptions {
  threshold?: number;
  root?: Element | null;
  rootMargin?: string;
}

export function useIntersectionObserver(
  ref: RefObject<Element | null>,
  options: UseIntersectionObserverOptions = {}
) {
  const [isIntersecting, setIsIntersecting] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
      },
      {
        threshold: options.threshold ?? 0,
        root: options.root ?? null,
        rootMargin: options.rootMargin ?? "0px",
      }
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, options.threshold, options.root, options.rootMargin]);

  return { isIntersecting };
}
