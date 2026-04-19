"use client"

import { Magnet, Pause, Play, SkipBack, SkipForward, ZoomIn } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { SnapMode } from "@/types/choreography"
import { useChoreographyEditor } from "./store"
import { WaveformViewRef } from "./waveform-view"

const SNAP_OPTIONS: { value: SnapMode; labelKey: string }[] = [
  { value: "off", labelKey: "snapOff" },
  { value: "beats", labelKey: "snapBeats" },
  { value: "phrases", labelKey: "snapPhrases" },
]

export function TransportBar() {
  const t = useTranslations("choreography")
  const {
    isPlaying,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    setCurrentTime,
    setPixelsPerSecond,
    setSnapMode,
  } = useChoreographyEditor()

  const ws = WaveformViewRef.current

  function togglePlay() {
    if (!ws) return
    ws.playPause()
  }

  function skipBack() {
    if (!ws) return
    ws.setTime(Math.max(0, ws.getCurrentTime() - 5))
  }

  function skipForward() {
    if (!ws) return
    ws.setTime(Math.min(ws.getDuration(), ws.getCurrentTime() + 5))
  }

  function handleSeek(e: React.ChangeEvent<HTMLInputElement>) {
    const time = Number.parseFloat(e.target.value)
    setCurrentTime(time)
    ws?.setTime(time)
  }

  function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}:${s.toString().padStart(2, "0")}`
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5">
      {/* Playback */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={skipBack}
          className="rounded p-1 text-muted-foreground hover:text-foreground"
          aria-label="Skip back 5s"
        >
          <SkipBack className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={togglePlay}
          className="rounded p-1 text-foreground hover:text-primary"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
        </button>
        <button
          type="button"
          onClick={skipForward}
          className="rounded p-1 text-muted-foreground hover:text-foreground"
          aria-label="Skip forward 5s"
        >
          <SkipForward className="h-4 w-4" />
        </button>
      </div>

      {/* Time + Seek */}
      <div className="flex items-center gap-2 text-xs tabular-nums">
        <span className="text-foreground">{formatTime(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={musicDuration || 180}
          step={0.1}
          value={currentTime}
          onChange={handleSeek}
          className="h-1 w-24 cursor-pointer accent-primary"
          aria-label="Seek"
        />
        <span className="text-muted-foreground">{formatTime(musicDuration)}</span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Zoom */}
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <ZoomIn className="h-3.5 w-3.5" />
        <input
          type="range"
          min={2}
          max={60}
          step={1}
          value={pixelsPerSecond}
          onChange={e => setPixelsPerSecond(Number(e.target.value))}
          className="h-1 w-20 cursor-pointer accent-primary"
          aria-label="Zoom"
        />
      </div>

      {/* Snap */}
      <div className="flex items-center gap-1">
        <Magnet className="h-3.5 w-3.5 text-muted-foreground" />
        <select
          value={snapMode}
          onChange={e => setSnapMode(e.target.value as SnapMode)}
          className="h-7 rounded border border-border bg-background px-1.5 text-xs"
          aria-label="Snap mode"
        >
          {SNAP_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>
              {t(opt.labelKey)}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
