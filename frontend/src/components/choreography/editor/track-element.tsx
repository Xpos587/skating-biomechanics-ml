"use client"

import { useCallback, useRef } from "react"
import { useTranslations } from "@/i18n"
import { cn } from "@/lib/utils"
import type { TimelineElement as TEl } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { useChoreographyEditor } from "./store"

interface Props {
  element: TEl
  pixelsPerSecond: number
  musicDuration: number
  snapMode: "beats" | "phrases" | "off"
  beatMarkers: number[]
  isSelected: boolean
  isActive: boolean
  onSelect: (id: string) => void
  onEdit: (id: string) => void
}

const SNAP_T = 0.5

export function TrackElement({
  element,
  pixelsPerSecond,
  snapMode,
  beatMarkers,
  isSelected,
  onSelect,
  onEdit,
}: Props) {
  const moveElement = useChoreographyEditor(s => s.moveElement)
  const t = useTranslations("choreography.timeline")
  const drag = useRef({ x0: 0, t0: 0 })
  const cfg = TRACK_CONFIG[element.trackType]

  const left = element.timestamp * pixelsPerSecond
  const width = element.duration * pixelsPerSecond

  const snap = useCallback(
    (ts: number) => {
      if (snapMode === "off" || beatMarkers.length === 0) return ts
      let best = beatMarkers[0]
      let minD = Math.abs(ts - best)
      for (const m of beatMarkers) {
        const d = Math.abs(ts - m)
        if (d < minD) {
          minD = d
          best = m
        }
      }
      return minD < SNAP_T ? best : ts
    },
    [snapMode, beatMarkers],
  )

  const onDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      e.stopPropagation()
      onSelect(element.id)
      drag.current = { x0: e.clientX, t0: element.timestamp }
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
      const ac = new AbortController()
      window.addEventListener(
        "pointermove",
        ev => {
          const dt = (ev.clientX - drag.current.x0) / pixelsPerSecond
          moveElement(element.id, snap(drag.current.t0 + dt))
        },
        { signal: ac.signal },
      )
      window.addEventListener("pointerup", () => ac.abort(), { signal: ac.signal })
    },
    [element.id, element.timestamp, pixelsPerSecond, onSelect, moveElement, snap],
  )

  const isBackHalf = element.timestamp > useChoreographyEditor.getState().musicDuration / 2

  return (
    <button
      type="button"
      data-el-id={element.id}
      tabIndex={0}
      className={cn(
        "absolute top-1 bottom-1 flex cursor-grab items-center overflow-hidden rounded-md text-[11px] leading-none select-none transition-shadow",
        "active:cursor-grabbing",
        isSelected && "ring-2 ring-white/40",
      )}
      style={{
        left: `${left}px`,
        width: `${Math.max(width, 64)}px`,
        backgroundColor: `${cfg.hex}14`,
        borderLeft: `3px solid ${cfg.hex}`,
      }}
      onPointerDown={onDown}
      onDoubleClick={e => {
        e.stopPropagation()
        onEdit(element.id)
      }}
      onKeyDown={e => {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.stopPropagation()
          useChoreographyEditor.getState().removeElement(element.id)
        }
      }}
      title={`${element.code} @ ${element.timestamp.toFixed(1)}s`}
    >
      <div className="flex min-w-0 flex-1 flex-col justify-center gap-0.5 px-2 py-1">
        <span className="truncate font-semibold" style={{ color: cfg.hex }}>
          {element.code}
        </span>
        <span className="text-[9px] tabular-nums text-muted-foreground">
          {element.timestamp.toFixed(1)}s
        </span>
      </div>
      {isBackHalf && (
        <span
          className="shrink-0 self-start px-1.5 py-0.5 text-[8px] font-bold leading-none"
          style={{ color: "oklch(var(--accent-gold) / 0.7)" }}
        >
          {t("backHalfShort")}
        </span>
      )}
    </button>
  )
}
