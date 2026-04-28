"use client"

import { useCallback, useMemo, useRef } from "react"
import { useTranslations } from "@/i18n"
import type { LayoutElement, TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { useChoreographyEditor } from "./editor/store"
import { JumpTrace, SequenceTrace, SpinMarker } from "./rink-figures"
import { FlowPaths } from "./rink-flow"

const VW = 60
const VH = 30
const PAD = 2

interface RinkElement {
  id: string
  code: string
  x: number
  y: number
  trackType: TrackType
  timestamp: number
}

function trackColor(trackType: TrackType): string {
  return TRACK_CONFIG[trackType].hex
}

function trackLabel(trackType: TrackType, t: (key: string) => string): string {
  return trackType === "jumps"
    ? t("rink.jump")
    : trackType === "spins"
      ? t("rink.spin")
      : t("rink.sequence")
}

function autoLayout(
  elements: { id: string; code: string; trackType: TrackType; timestamp: number }[],
  _duration: number,
): RinkElement[] {
  if (elements.length === 0) return []

  const sorted = [...elements].sort((a, b) => a.timestamp - b.timestamp)
  const usableW = VW - PAD * 2
  const usableH = VH - PAD * 2

  return sorted.map((el, i) => {
    // Serpentine pattern: left→right, right→left
    const row = Math.floor(i / 5)
    const col = i % 5
    const rowDir = row % 2 === 0 ? 1 : -1
    const colNorm = rowDir === 1 ? col / 4 : 1 - col / 4
    const rowNorm = row / Math.max(1, Math.ceil(sorted.length / 5) - 1)
    return {
      ...el,
      x: PAD + colNorm * usableW,
      y: PAD + rowNorm * usableH,
    }
  })
}

export function RinkDiagram({
  className,
  elements: propElements,
}: {
  className?: string
  elements?: LayoutElement[]
}) {
  const t = useTranslations("choreography")
  const storeElements = useChoreographyEditor(s => s.elements)
  const selectedId = useChoreographyEditor(s => s.selectedElementId)
  const select = useChoreographyEditor(s => s.setSelectedElement)
  const updatePos = useChoreographyEditor(s => s.updateElementPosition)
  const musicDuration = useChoreographyEditor(s => s.musicDuration)

  // Readonly mode: use prop elements, no interactivity
  const isReadonly = !!propElements
  const _elements = propElements ?? storeElements
  const svgRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef<{ id: string; sx: number; sy: number; ox: number; oy: number } | null>(
    null,
  )

  const rinkElements = useMemo((): RinkElement[] => {
    if (isReadonly) {
      return (propElements ?? []).map((el, i) => {
        const tt: TrackType = el.code.includes("Sp")
          ? "spins"
          : el.code.startsWith("StSq") || el.code.startsWith("ChSq")
            ? "sequences"
            : "jumps"
        return {
          id: `ro-${i}`,
          code: el.code,
          trackType: tt,
          timestamp: el.timestamp,
          x: el.position?.x ?? 0,
          y: el.position?.y ?? 0,
        }
      })
    }

    const withPos: RinkElement[] = storeElements
      .filter(
        (el): el is typeof el & { position: NonNullable<typeof el.position> } => !!el.position,
      )
      .map(el => ({
        id: el.id,
        code: el.code,
        trackType: el.trackType,
        timestamp: el.timestamp,
        x: el.position.x,
        y: el.position.y,
      }))

    const withoutPos = storeElements.filter(el => !el.position)
    const auto = autoLayout(withoutPos, musicDuration)

    return [...withPos, ...auto].sort((a, b) => a.timestamp - b.timestamp)
  }, [isReadonly, propElements, storeElements, musicDuration])

  const toSvg = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const rect = svg.getBoundingClientRect()
    return {
      x: ((clientX - rect.left) / rect.width) * VW,
      y: ((clientY - rect.top) / rect.height) * VH,
    }
  }, [])

  const onPointerDown = useCallback(
    (e: React.PointerEvent, el: RinkElement) => {
      e.preventDefault()
      e.stopPropagation()
      select(el.id)
      const pt = toSvg(e.clientX, e.clientY)
      dragRef.current = {
        id: el.id,
        sx: e.clientX,
        sy: e.clientY,
        ox: pt.x - el.x,
        oy: pt.y - el.y,
      }
      ;(e.target as Element).setPointerCapture(e.pointerId)
    },
    [select, toSvg],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current) return
      const pt = toSvg(e.clientX, e.clientY)
      const nx = Math.max(PAD, Math.min(VW - PAD, pt.x - dragRef.current.ox))
      const ny = Math.max(PAD, Math.min(VH - PAD, pt.y - dragRef.current.oy))
      updatePos(dragRef.current.id, nx, ny)
    },
    [toSvg, updatePos],
  )

  const onPointerUp = useCallback(() => {
    dragRef.current = null
  }, [])

  const onRinkClick = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as Element).closest("[data-el-marker]")) return
      select(null)
    },
    [select],
  )

  return (
    <div className={`w-full ${className ?? ""}`}>
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: interactive SVG canvas with pointer events */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VW} ${VH}`}
        className="w-full select-none"
        style={{ borderRadius: "var(--radius-sm)", overflow: "hidden" }}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onClick={onRinkClick}
      >
        <title>{t("rink.title")}</title>
        {/* Ice */}
        <rect x={0} y={0} width={VW} height={VH} fill="oklch(0.97 0.01 240)" rx={0.5} />

        {/* Border */}
        <rect
          x={1}
          y={1}
          width={VW - 2}
          height={VH - 2}
          fill="none"
          stroke="oklch(var(--border))"
          strokeWidth={0.15}
          rx={0.3}
        />

        {/* Center line */}
        <line
          x1={VW / 2}
          y1={1}
          x2={VW / 2}
          y2={VH - 1}
          stroke="oklch(0.5 0.15 25)"
          strokeWidth={0.08}
          strokeDasharray="0.5,0.5"
        />

        {/* Center circle */}
        <circle
          cx={VW / 2}
          cy={VH / 2}
          r={4.5}
          fill="none"
          stroke="oklch(0.5 0.15 25)"
          strokeWidth={0.08}
        />
        <circle cx={VW / 2} cy={VH / 2} r={0.15} fill="oklch(0.5 0.15 25)" />

        {/* Zone lines */}
        <line x1={5} y1={1} x2={5} y2={VH - 1} stroke="oklch(var(--border))" strokeWidth={0.06} />
        <line
          x1={VW - 5}
          y1={1}
          x2={VW - 5}
          y2={VH - 1}
          stroke="oklch(var(--border))"
          strokeWidth={0.06}
        />

        {/* Corner circles */}
        {[
          [10, 7.5],
          [10, VH - 7.5],
          [VW - 10, 7.5],
          [VW - 10, VH - 7.5],
        ].map(([cx, cy]) => (
          <g key={`corner-${cx}-${cy}`}>
            <circle
              cx={cx}
              cy={cy}
              r={3}
              fill="none"
              stroke="oklch(var(--border))"
              strokeWidth={0.06}
            />
            <circle cx={cx} cy={cy} r={0.15} fill="oklch(0.5 0.15 25)" />
          </g>
        ))}

        {/* Flow paths between sequential elements */}
        <FlowPaths elements={rinkElements} />

        {/* Elements */}
        {rinkElements.map((el, i) => {
          const color = trackColor(el.trackType)
          const selected = el.id === selectedId
          const num = i + 1
          return (
            <g
              key={el.id}
              data-el-marker
              onPointerDown={isReadonly ? undefined : e => onPointerDown(e, el)}
              style={{ cursor: isReadonly ? "default" : "grab" }}
            >
              {/* Selection ring */}
              {selected && (
                <circle
                  cx={el.x}
                  cy={el.y}
                  r={2}
                  fill="none"
                  stroke={color}
                  strokeWidth={0.2}
                  opacity={0.6}
                />
              )}

              {/* Trace figure by type */}
              {el.trackType === "jumps" && (
                <JumpTrace x={el.x} y={el.y} code={el.code} color={color} elementId={el.id} />
              )}
              {el.trackType === "spins" && (
                <SpinMarker x={el.x} y={el.y} color={color} />
              )}
              {el.trackType === "sequences" && (
                <SequenceTrace x={el.x} y={el.y} code={el.code} color={color} elementId={el.id} />
              )}

              {/* Number badge */}
              <circle
                cx={el.x + 1.2}
                cy={el.y - 1.0}
                r={0.7}
                fill="oklch(var(--background))"
                stroke={color}
                strokeWidth={0.12}
              />
              <text
                x={el.x + 1.2}
                y={el.y - 0.65}
                textAnchor="middle"
                fontSize={0.85}
                fill={color}
                fontWeight="bold"
              >
                {num}
              </text>

              {/* Code label */}
              <text
                x={el.x}
                y={el.y + 2.2}
                textAnchor="middle"
                fontSize={0.9}
                fill="oklch(var(--foreground))"
                fontWeight="600"
              >
                {el.code}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-3 px-1 text-[10px] text-muted-foreground">
        {(["jumps", "spins", "sequences"] as TrackType[]).map(tt => (
          <div key={tt} className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full" style={{ backgroundColor: trackColor(tt) }} />
            {trackLabel(tt, t)}
          </div>
        ))}
      </div>
    </div>
  )
}
