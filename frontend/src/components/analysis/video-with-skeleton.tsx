"use client"

import { useRef } from "react"
import { useAnalysisStore } from "@/stores/analysis"
import type { PhasesData, PoseData } from "@/types"
import { PhaseLabels } from "./phase-labels"
import { SkeletonCanvas } from "./skeleton-canvas"

interface VideoWithSkeletonProps {
  videoUrl: string
  poseData: PoseData | null
  phases: PhasesData | null
  totalFrames: number
  className?: string
}

export function VideoWithSkeleton({
  videoUrl,
  poseData,
  phases,
  totalFrames,
  className = "",
}: VideoWithSkeletonProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { currentFrame, setCurrentFrame } = useAnalysisStore()

  const handleTimeUpdate = () => {
    if (!videoRef.current) return
    const video = videoRef.current
    const frame = Math.floor((video.currentTime / video.duration) * totalFrames)
    setCurrentFrame(frame)
  }

  const handleSeek = (
    e: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>,
  ) => {
    if (!videoRef.current || !containerRef.current) return

    let clientX: number
    if ("clientX" in e) {
      clientX = e.clientX
    } else {
      const rect = (e.target as HTMLDivElement).getBoundingClientRect()
      clientX = rect.left + rect.width / 2
    }

    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percent = x / rect.width
    videoRef.current.currentTime = percent * videoRef.current.duration
  }

  if (!poseData) {
    // Fallback: show video without skeleton
    return (
      <div
        className={`relative ${className}`}
        style={{ backgroundColor: "oklch(var(--background))" }}
      >
        {/* biome-ignore lint/a11y/useMediaCaption: Skating analysis video has no captions */}
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full"
          controls
          onTimeUpdate={handleTimeUpdate}
        />
      </div>
    )
  }

  return (
    // biome-ignore lint/a11y/useSemanticElements: div maintains aspect-video CSS
    <div
      ref={containerRef}
      className={`relative aspect-video ${className}`}
      style={{ backgroundColor: "oklch(var(--background))" }}
      onClick={handleSeek}
      onKeyDown={e => e.key === "Enter" && handleSeek(e)}
      role="button"
      tabIndex={0}
    >
      {/* biome-ignore lint/a11y/useMediaCaption: Skating analysis video has no captions */}
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full h-full object-contain"
        onTimeUpdate={handleTimeUpdate}
      />
      <SkeletonCanvas poseData={poseData} currentFrame={currentFrame} width={1920} height={1080} />
      {phases && <PhaseLabels phases={phases} currentFrame={totalFrames} width={1920} />}
    </div>
  )
}
