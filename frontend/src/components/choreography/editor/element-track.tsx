"use client"

import { Plus } from "lucide-react"
import { useChoreographyEditor } from "./store"
import { TimelineElement } from "./timeline-element"
import type { TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { useTranslations } from "@/i18n"
import { cn } from "@/lib/utils"

interface ElementTrackProps {
  type: TrackType
}

export function ElementTrack({ type }: ElementTrackProps) {
  const t = useTranslations("choreography")
  const {
    elements,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    beatMarkers,
    phraseMarkers,
    selectedElementId,
    setSelectedElement,
  } = useChoreographyEditor()

  const trackElements = elements.filter((e) => e.trackType === type)
  const config = TRACK_CONFIG[type]
  const trackLabel = t(`${type}Track` as Parameters<typeof t>[0])

  const timelineWidthPx = musicDuration * pixelsPerSecond

  function handleDoubleClick(e: React.MouseEvent) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const timestamp = x / pixelsPerSecond
    // Will be wired to ElementPicker in Phase 2
    setSelectedElement(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Delete" && selectedElementId) {
      useChoreographyEditor.getState().removeElement(selectedElementId)
    }
  }

  return (
    <div
      className="flex border-b border-border last:border-b-0"
      onKeyDown={handleKeyDown}
    >
      {/* Track label */}
      <div className="flex w-28 shrink-0 items-center justify-between border-r border-border px-2 py-1.5">
        <span className={cn("text-xs font-medium", config.color)}>
          {trackLabel}
        </span>
        <span className={cn(
          "text-[10px]",
          trackElements.length > config.maxElements ? "text-red-500" : "text-muted-foreground",
        )}>
          {trackElements.length}/{config.maxElements}
        </span>
      </div>

      {/* Track content */}
      <div
        className="relative flex-1 overflow-hidden bg-muted/20 py-0.5"
        onDoubleClick={handleDoubleClick}
      >
        {/* Beat markers */}
        {beatMarkers.map((time, i) => {
          const x = time * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`beat-${i}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-primary/10"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Phrase markers */}
        {phraseMarkers.map((time, i) => {
          const x = time * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`phrase-${i}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-muted-foreground/20"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Playhead on this track */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
          style={{ left: `${currentTime * pixelsPerSecond}px` }}
        />

        {/* Elements */}
        {trackElements.map((el) => (
          <TimelineElement
            key={el.id}
            element={el}
            pixelsPerSecond={pixelsPerSecond}
            duration={musicDuration}
            snapMode={snapMode}
            beatMarkers={beatMarkers}
            phraseMarkers={phraseMarkers}
            isSelected={el.id === selectedElementId}
            isActive={currentTime >= el.timestamp && currentTime < el.timestamp + el.duration}
            onSelect={setSelectedElement}
            onEdit={setSelectedElement}
          />
        ))}

        {/* Empty hint */}
        {trackElements.length === 0 && (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            <Plus className="mr-1 h-3 w-3" />
            Двойной клик для добавления
          </div>
        )}
      </div>
    </div>
  )
}
