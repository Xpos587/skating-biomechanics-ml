"use client"

import { useAnalysisStore } from "@/stores/analysis"
import type { PhasesData } from "@/types"

interface PhaseTimelineProps {
  totalFrames: number
  phases: PhasesData
}

export function PhaseTimeline({ totalFrames, phases }: PhaseTimelineProps) {
  const { currentFrame, setCurrentFrame } = useAnalysisStore()

  const percentage = (currentFrame / totalFrames) * 100

  const handleSeek = (
    e: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>,
  ) => {
    const rect = e.currentTarget.getBoundingClientRect()
    let x: number

    if ("clientX" in e) {
      x = e.clientX - rect.left
    } else {
      x = rect.width / 2 // Center for keyboard activation
    }

    const seekPercentage = x / rect.width
    const targetFrame = Math.floor(seekPercentage * totalFrames)

    setCurrentFrame(targetFrame)
  }

  const takeoffPercent = phases.takeoff !== undefined ? (phases.takeoff / totalFrames) * 100 : null

  const peakPercent = phases.peak !== undefined ? (phases.peak / totalFrames) * 100 : null

  const landingPercent = phases.landing !== undefined ? (phases.landing / totalFrames) * 100 : null

  return (
    <div
      className="relative w-full h-12 bg-muted rounded-lg overflow-hidden cursor-pointer"
      onClick={handleSeek}
      onKeyDown={e => e.key === "Enter" && handleSeek(e)}
      role="slider"
      aria-valuemin={0}
      aria-valuemax={totalFrames}
      aria-valuenow={currentFrame}
      aria-label="Frame scrubber"
      tabIndex={0}
    >
      {/* Phase zones */}
      {takeoffPercent !== null && peakPercent !== null && (
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: `${takeoffPercent}%`,
            right: `${100 - peakPercent}%`,
            backgroundColor: "oklch(var(--score-good) / 0.2)",
          }}
        />
      )}

      {peakPercent !== null && landingPercent !== null && (
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: `${peakPercent}%`,
            right: `${100 - landingPercent}%`,
            backgroundColor: "oklch(var(--score-mid) / 0.2)",
          }}
        />
      )}

      {landingPercent !== null && (
        <div
          className="absolute top-0 bottom-0"
          style={{ left: `${landingPercent}%`, backgroundColor: "oklch(var(--score-bad) / 0.2)" }}
        />
      )}

      {/* Phase markers */}
      {takeoffPercent !== null && (
        <div
          className="absolute top-0 bottom-0 w-0.5"
          style={{ left: `${takeoffPercent}%`, backgroundColor: "oklch(var(--score-good))" }}
          title="Takeoff"
        />
      )}

      {peakPercent !== null && (
        <div
          className="absolute top-0 bottom-0 w-0.5"
          style={{ left: `${peakPercent}%`, backgroundColor: "oklch(var(--score-mid))" }}
          title="Peak"
        />
      )}

      {landingPercent !== null && (
        <div
          className="absolute top-0 bottom-0 w-0.5"
          style={{ left: `${landingPercent}%`, backgroundColor: "oklch(var(--score-bad))" }}
          title="Landing"
        />
      )}

      {/* Scrubber */}
      <div
        className="absolute top-0 bottom-0 w-1 bg-primary shadow-lg"
        style={{ left: `${percentage}%` }}
      >
        <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-primary rounded-full" />
      </div>

      {/* Frame counter */}
      <div className="absolute bottom-1 right-2 text-xs font-medium text-muted-foreground">
        {currentFrame} / {totalFrames}
      </div>
    </div>
  )
}
