"use client"

import { useCallback, useRef } from "react"
import { cn } from "@/lib/utils"
import type { TimelineElement as TimelineElementType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { useChoreographyEditor } from "./store"

interface TimelineElementProps {
  element: TimelineElementType
  pixelsPerSecond: number
  duration: number
  snapMode: "beats" | "phrases" | "off"
  beatMarkers: number[]
  phraseMarkers: number[]
  isSelected: boolean
  isActive: boolean
  onSelect: (id: string) => void
  onEdit: (id: string) => void
}

const SNAP_THRESHOLD = 0.5

export function TimelineElement({
  element,
  pixelsPerSecond,
  snapMode,
  beatMarkers,
  isSelected,
  isActive,
  onSelect,
  onEdit,
}: TimelineElementProps) {
  const moveElement = useChoreographyEditor(s => s.moveElement)
  const dragRef = useRef({ startX: 0, startTimestamp: 0 })
  const config = TRACK_CONFIG[element.trackType]

  const leftPx = element.timestamp * pixelsPerSecond
  const widthPx = element.duration * pixelsPerSecond

  const snapToMarker = useCallback(
    (timestamp: number): number => {
      if (snapMode === "off") return timestamp
      if (beatMarkers.length === 0) return timestamp
      let closest = beatMarkers[0]
      let minDist = Math.abs(timestamp - closest)
      for (const m of beatMarkers) {
        const dist = Math.abs(timestamp - m)
        if (dist < minDist) {
          minDist = dist
          closest = m
        }
      }
      return minDist < SNAP_THRESHOLD ? closest : timestamp
    },
    [snapMode, beatMarkers],
  )

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      e.stopPropagation()
      onSelect(element.id)
      dragRef.current = { startX: e.clientX, startTimestamp: element.timestamp }
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)

      const ac = new AbortController()

      const handlePointerMove = (ev: PointerEvent) => {
        const dx = ev.clientX - dragRef.current.startX
        const dt = dx / pixelsPerSecond
        let newTimestamp = dragRef.current.startTimestamp + dt
        newTimestamp = snapToMarker(newTimestamp)
        moveElement(element.id, newTimestamp)
      }

      const handlePointerUp = () => {
        ac.abort()
      }

      window.addEventListener("pointermove", handlePointerMove, { signal: ac.signal })
      window.addEventListener("pointerup", handlePointerUp, { signal: ac.signal })
    },
    [element.id, element.timestamp, pixelsPerSecond, onSelect, moveElement, snapToMarker],
  )

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onEdit(element.id)
    },
    [element.id, onEdit],
  )

  const backHalfThreshold = useChoreographyEditor(s => s.musicDuration / 2)
  const isBackHalf = element.timestamp > backHalfThreshold

  return (
    <button
      type="button"
      data-element-id={element.id}
      tabIndex={0}
      className={cn(
        "absolute top-1 bottom-1 cursor-grab rounded-md border px-1.5 py-0.5 text-xs select-none transition-colors text-left",
        "hover:brightness-110 active:cursor-grabbing",
        isSelected && "ring-2 ring-primary ring-offset-1 ring-offset-background",
        isActive && "brightness-125",
        config.colorVar,
        isBackHalf && "bg-gradient-to-r from-transparent to-amber-500/10",
      )}
      style={{
        left: `${leftPx}px`,
        width: `${widthPx}px`,
        minWidth: 40,
      }}
      onPointerDown={handlePointerDown}
      onDoubleClick={handleDoubleClick}
      onKeyDown={e => {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.stopPropagation()
          useChoreographyEditor.getState().removeElement(element.id)
        }
      }}
      title={`${element.code} (${element.timestamp.toFixed(1)}s)`}
    >
      <div className={cn("truncate font-semibold", config.color)}>{element.code}</div>
      <div className="text-muted-foreground text-[10px]">{element.timestamp.toFixed(1)}s</div>
    </button>
  )
}
