import type { TrackType } from "@/types/choreography"

export type TransitionValidity = "ok" | "warning" | "invalid"

/**
 * ISU-based transition validity matrix.
 *
 * - spin → * : skater loses speed in spin, needs time to rebuild
 * - sequence → sequence : needs significant ice coverage, unusual
 * - jump → * : standard — jumps land with momentum
 */
export const transitionRules: Record<TrackType, Record<TrackType, TransitionValidity>> = {
  jumps: { jumps: "ok", spins: "ok", sequences: "ok" },
  spins: { jumps: "warning", spins: "warning", sequences: "ok" },
  sequences: { jumps: "ok", spins: "ok", sequences: "warning" },
}

const COLORS: Record<TransitionValidity, string> = {
  ok: "oklch(0.6 0.02 260)",
  warning: "oklch(0.75 0.15 80)",
  invalid: "oklch(0.65 0.25 25)",
}

export function transitionColor(v: TransitionValidity): string {
  return COLORS[v]
}

export function getTransitionValidity(from: TrackType, to: TrackType): TransitionValidity {
  return transitionRules[from][to]
}
