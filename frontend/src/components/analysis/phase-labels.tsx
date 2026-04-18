"use client"

import type { PhasesData } from "@/types"

interface PhaseLabelsProps {
  phases: PhasesData
  currentFrame: number
  width: number
}

export function PhaseLabels({ phases, currentFrame, width }: PhaseLabelsProps) {
  if (!phases.takeoff && !phases.peak && !phases.landing) return null

  // Calculate label positions (normalized 0-1)
  const takeoffX = phases.takeoff !== undefined ? (phases.takeoff / currentFrame) * width : null
  const peakX = phases.peak !== undefined ? (phases.peak / currentFrame) * width : null
  const landingX = phases.landing !== undefined ? (phases.landing / currentFrame) * width : null

  return (
    <div className="absolute top-2 left-0 right-0 flex justify-between px-4">
      {takeoffX !== null && (
        <div
          className="absolute top-0 rounded-full px-2 py-1 text-xs font-medium"
          style={{
            left: `${takeoffX}px`,
            backgroundColor: "oklch(var(--score-good) / 0.8)",
            color: "oklch(var(--background))",
          }}
        >
          Takeoff
        </div>
      )}
      {peakX !== null && (
        <div
          className="absolute top-0 rounded-full px-2 py-1 text-xs font-medium"
          style={{
            left: `${peakX}px`,
            backgroundColor: "oklch(var(--score-mid) / 0.8)",
            color: "oklch(var(--background))",
          }}
        >
          Peak
        </div>
      )}
      {landingX !== null && (
        <div
          className="absolute top-0 rounded-full px-2 py-1 text-xs font-medium"
          style={{
            left: `${landingX}px`,
            backgroundColor: "oklch(var(--score-bad) / 0.8)",
            color: "oklch(var(--background))",
          }}
        >
          Landing
        </div>
      )}
    </div>
  )
}
