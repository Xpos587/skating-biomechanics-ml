import type { ReactNode } from "react"

// ---------------------------------------------------------------------------
// Deterministic angle from element ID (stable across re-renders)
// ---------------------------------------------------------------------------

export function hashAngle(elementId: string): number {
  let h = 0
  for (let i = 0; i < elementId.length; i++) {
    h = (h * 31 + elementId.charCodeAt(i)) | 0
  }
  return ((h % 360) + 360) % 360
}

// ---------------------------------------------------------------------------
// Jump classification
// ---------------------------------------------------------------------------

type JumpCategory = "toe" | "edge" | "axel"

function jumpCategory(code: string): JumpCategory {
  // Strip rotation count prefix (1-4) to get base jump letter
  const base = code.replace(/^[1-4]/, "")
  if (base === "A") return "axel"
  if (base === "S" || base === "Lo") return "edge"
  return "toe" // T, Lz, F
}

// ---------------------------------------------------------------------------
// JumpTrace — entry arc + hop marker + exit arc
// ---------------------------------------------------------------------------

interface TraceFigureProps {
  x: number
  y: number
  code: string
  color: string
  elementId: string
}

/**
 * Renders a jump trace figure as an SVG <g>.
 *
 * Layout (unrotated, hop at origin):
 *   entry arc → hop circle → exit arc
 *
 * The whole group is rotated by `hashAngle(elementId)` degrees around (x, y).
 */
export function JumpTrace({ x, y, code, color, elementId }: TraceFigureProps): ReactNode {
  const angle = hashAngle(elementId)
  const cat = jumpCategory(code)

  // Arc dimensions per category (in rink units, 1 unit = 1m)
  const entryLen = cat === "axel" ? 3.5 : cat === "edge" ? 3 : 2
  const exitLen = 2
  const hopR = 0.5
  const strokeW = 0.12

  // Build path: entry arc ends at (0,0), exit arc starts at (0,0)
  // Entry: curve from (-entryLen, 0) to (0, 0) with slight bend
  // Exit: curve from (0, 0) to (exitLen, 0) with bend in opposite direction

  const entryR = entryLen * 1.2
  const exitR = exitLen * 1.2

  const d = [
    `M ${-entryLen} 0`,
    `A ${entryR} ${entryR} 0 0 1 0 0`,
    `A ${exitR} ${exitR} 0 0 0 ${exitLen} 0`,
  ].join(" ")

  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`}>
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={strokeW}
        strokeLinecap="round"
        opacity={0.7}
      />
      <circle cx={0} cy={0} r={hopR} fill={color} opacity={0.85} />
    </g>
  )
}

// ---------------------------------------------------------------------------
// SequenceTrace — serpentine S-curve showing ice coverage
// ---------------------------------------------------------------------------

/**
 * Renders a step/choreographic sequence as an S-shaped serpentine path.
 * Three connected arcs create a zigzag pattern through the element position.
 */
export function SequenceTrace({ x, y, color, elementId }: TraceFigureProps): ReactNode {
  const angle = hashAngle(elementId)
  const strokeW = 0.12

  const segLen = 3.5
  const segR = segLen * 1.1

  const d = [
    `M ${-segLen * 1.5} 0`,
    `A ${segR} ${segR} 0 0 1 ${-segLen * 0.5} 0`,
    `A ${segR} ${segR} 0 0 0 ${segLen * 0.5} 0`,
    `A ${segR} ${segR} 0 0 1 ${segLen * 1.5} 0`,
  ].join(" ")

  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`}>
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={strokeW}
        strokeLinecap="round"
        strokeDasharray="0.4,0.2"
        opacity={0.7}
      />
    </g>
  )
}

// ---------------------------------------------------------------------------
// SpinMarker — circle with stroke
// ---------------------------------------------------------------------------

export function SpinMarker({ x, y, color }: Omit<TraceFigureProps, "code" | "elementId">): ReactNode {
  return (
    <g>
      <circle cx={x} cy={y} r={1.3} fill={color} opacity={0.25} stroke={color} strokeWidth={0.15} />
    </g>
  )
}
