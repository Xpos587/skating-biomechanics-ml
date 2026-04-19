"use client"

import WaveSurfer from "wavesurfer.js"
import { useChoreographyEditor } from "./store"

interface WaveformViewProps {
  audioUrl: string | null
}

export function WaveformView({ audioUrl }: WaveformViewProps) {
  const { pixelsPerSecond, setCurrentTime, setIsPlaying } = useChoreographyEditor()

  function handleContainerRef(el: HTMLDivElement | null) {
    if (!el || WaveformViewRef.current) return
    if (!audioUrl) return

    const ws = WaveSurfer.create({
      container: el,
      waveColor: "oklch(var(--muted-foreground) / 0.3)",
      progressColor: "oklch(var(--primary))",
      cursorColor: "oklch(0.6 0.2 25)",
      cursorWidth: 2,
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      hideScrollbar: true,
      minPxPerSec: pixelsPerSecond,
    })

    const unsubscribers = [
      ws.on("timeupdate", time => {
        if (el.isConnected) setCurrentTime(time)
      }),
      ws.on("play", () => {
        if (el.isConnected) setIsPlaying(true)
      }),
      ws.on("pause", () => {
        if (el.isConnected) setIsPlaying(false)
      }),
    ]

    ws.load(audioUrl).catch(() => {})

    WaveformViewRef.current = ws
    WaveformViewRef._cleanup = () => {
      for (const u of unsubscribers) u?.()
      ws.destroy()
      WaveformViewRef.current = null
    }
  }

  // Expose for TransportBar
  void WaveformViewRef

  return (
    <div
      ref={handleContainerRef}
      className="relative w-full overflow-hidden rounded-lg border border-border bg-muted/30"
      style={{ height: 80 }}
    >
      {!audioUrl && (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Загрузите музыку для отображения waveform
        </div>
      )}
    </div>
  )
}

// Module-level ref for TransportBar to access wavesurfer instance
interface WaveformRefObj {
  current: WaveSurfer | null
  _cleanup?: () => void
}
export const WaveformViewRef: WaveformRefObj = { current: null }
