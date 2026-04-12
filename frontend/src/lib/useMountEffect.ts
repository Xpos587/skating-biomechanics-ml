import { useEffect } from "react"

/**
 * Run a callback once on mount. Cleanup function supported.
 * This is the ONLY allowed wrapper around useEffect in this project.
 */
export function useMountEffect(callback: () => undefined | (() => void)) {
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentionally mount-only
  useEffect(callback, [])
}
