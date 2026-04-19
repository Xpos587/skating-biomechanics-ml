"use client"

import { Plus } from "lucide-react"
import { useState } from "react"
import { useTranslations } from "@/i18n"
import { cn } from "@/lib/utils"
import type { TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { ElementEditor } from "./element-editor"
import { ElementPicker } from "./element-picker"
import { useChoreographyEditor } from "./store"
import { TimelineElement } from "./timeline-element"

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
    addElement,
    setSelectedElement,
  } = useChoreographyEditor()

  const [pickerState, setPickerState] = useState<{
    x: number
    y: number
    timestamp: number
  } | null>(null)
  const [editorState, setEditorState] = useState<{
    x: number
    y: number
    elementId: string
  } | null>(null)

  const trackElements = elements.filter(e => e.trackType === type)
  const config = TRACK_CONFIG[type]
  const trackLabel = t(`${type}Track` as Parameters<typeof t>[0])

  const timelineWidthPx = musicDuration * pixelsPerSecond

  function handleDoubleClick(e: React.MouseEvent) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const timestamp = x / pixelsPerSecond
    setPickerState({ x: e.clientX, y: e.clientY, timestamp })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Delete" && selectedElementId) {
      useChoreographyEditor.getState().removeElement(selectedElementId)
    }
  }

  return (
    <section
      className="flex border-b border-border last:border-b-0"
      onKeyDown={handleKeyDown}
      aria-label={trackLabel}
    >
      {/* Track label */}
      <div className="flex w-28 shrink-0 items-center justify-between border-r border-border px-2 py-1.5">
        <span className={cn("text-xs font-medium", config.color)}>{trackLabel}</span>
        <span
          className={cn(
            "text-[10px]",
            trackElements.length > config.maxElements
              ? "text-destructive"
              : "text-muted-foreground",
          )}
        >
          {trackElements.length}/{config.maxElements}
        </span>
      </div>

      {/* Track content */}
      <div
        role="application"
        aria-label={`${trackLabel} elements`}
        className="relative flex-1 overflow-hidden bg-muted/20 py-0.5"
        onDoubleClick={handleDoubleClick}
      >
        {/* Beat markers */}
        {beatMarkers.map(time => {
          const x = time * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`beat-${time}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-primary/10"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Phrase markers */}
        {phraseMarkers.map(time => {
          const x = time * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`phrase-${time}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-muted-foreground/20"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Playhead on this track */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 w-0.5 z-10"
          style={{
            left: `${currentTime * pixelsPerSecond}px`,
            backgroundColor: "oklch(var(--destructive))",
          }}
        />

        {/* Elements */}
        {trackElements.map(el => (
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
            onEdit={id => {
              const domEl = document.querySelector(`[data-element-id="${id}"]`)
              const rect = domEl?.getBoundingClientRect()
              if (rect) {
                setEditorState({ x: rect.right + 4, y: rect.top, elementId: id })
              }
            }}
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

      {/* Element Picker popover */}
      {pickerState && (
        <div
          role="dialog"
          aria-label="Element picker"
          className="fixed inset-0 z-50"
          onClick={() => setPickerState(null)}
          onKeyDown={e => {
            if (e.key === "Escape") setPickerState(null)
          }}
        >
          <div className="absolute" style={{ left: pickerState.x, top: pickerState.y }}>
            <ElementPicker
              trackType={type}
              onSelect={code => addElement(type, pickerState.timestamp, code)}
              onClose={() => setPickerState(null)}
            />
          </div>
        </div>
      )}

      {/* Element Editor popover */}
      {editorState && (
        <div
          role="dialog"
          aria-label="Element editor"
          className="fixed inset-0 z-50"
          onClick={() => setEditorState(null)}
          onKeyDown={e => {
            if (e.key === "Escape") setEditorState(null)
          }}
        >
          <div className="absolute" style={{ left: editorState.x, top: editorState.y }}>
            <ElementEditor elementId={editorState.elementId} onClose={() => setEditorState(null)} />
          </div>
        </div>
      )}
    </section>
  )
}
