import type { ReactNode } from "react"
import type { TrackType } from "@/types/choreography"
import { hashAngle } from "./rink-figures"
import { transitionColor, transitionRules } from "./transition-rules"
import type { TransitionValidity } from "./transition-rules"

// ---------------------------------------------------------------------------
// Trace figure endpoint geometry (mirrors rink-figures.tsx)
// ---------------------------------------------------------------------------

interface ElementEndpoint {
  x: number
  y: number
  angle: number // degrees, 0 = right, clockwise positive
  len: number   // distance from element center to endpoint
}

type JumpCategory = "toe" | "edge" | "axel"

function jumpCategory(code: string): JumpCategory {
  const base = code.replace(/^[1-4]/, "")
  if (base === "A") return "axel"
  if (base === "S" || base === "Lo") return "edge"
  return "toe"
}

/** Exit point of a trace figure (where the skater leaves the element). */
function getExitEndpoint(el: { x: number; y: number; code: string; trackType: TrackType; id: string }): ElementEndpoint {
  const angle = hashAngle(el.id)
  if (el.trackType === "jumps") {
    return { x: el.x, y: el.y, angle, len: 2 }
  }
  if (el.trackType === "sequences") {
    return { x: el.x, y: el.y, angle, len: 5.25 }
  }
  // spins — no directional exit, use hashAngle with radius offset
  return { x: el.x, y: el.y, angle, len: 1.3 }
}

/** Entry point of a trace figure (where the skater arrives at the element). */
function getEntryEndpoint(el: { x: number; y: number; code: string; trackType: TrackType; id: string }): ElementEndpoint {
  const angle = hashAngle(el.id)
  if (el.trackType === "jumps") {
    const cat = jumpCategory(el.code)
    const entryLen = cat === "axel" ? 3.5 : cat === "edge" ? 3 : 2
    // Entry is on the opposite side — angle + 180
    return { x: el.x, y: el.y, angle: angle + 180, len: entryLen }
  }
  if (el.trackType === "sequences") {
    return { x: el.x, y: el.y, angle: angle + 180, len: 5.25 }
  }
  // spins — entry from hashAngle direction
  return { x: el.x, y: el.y, angle: angle + 180, len: 1.3 }
}

/** Convert polar (angle in degrees, distance) to cartesian offset. */
function polarOffset(angleDeg: number, dist: number): { dx: number; dy: number } {
  const rad = (angleDeg * Math.PI) / 180
  return { dx: Math.cos(rad) * dist, dy: Math.sin(rad) * dist }
}

// ---------------------------------------------------------------------------
// FlowPath — single Bezier curve between two elements
// ---------------------------------------------------------------------------

interface FlowPathProps {
  from: { x: number; y: number; code: string; trackType: TrackType; id: string }
  to: { x: number; y: number; code: string; trackType: TrackType; id: string }
  validity: TransitionValidity
}

export function FlowPath({ from, to, validity }: FlowPathProps): ReactNode {
  const exit = getExitEndpoint(from)
  const entry = getEntryEndpoint(to)

  // Compute world-space exit/entry points
  const exitOff = polarOffset(exit.angle, exit.len)
  const exitX = exit.x + exitOff.dx
  const exitY = exit.y + exitOff.dy

  const entryOff = polarOffset(entry.angle, entry.len)
  const entryX = entry.x + entryOff.dx
  const entryY = entry.y + entryOff.dy

  // Control points: tangent direction at exit/entry, length = 30% of distance
  const dx = entryX - exitX
  const dy = entryY - exitY
  const dist = Math.sqrt(dx * dx + dy * dy)
  const tangentLen = dist * 0.3

  const cp1Off = polarOffset(exit.angle, tangentLen)
  const cp2Off = polarOffset(entry.angle, tangentLen)

  const d = [
    `M ${exitX} ${exitY}`,
    `C ${exitX + cp1Off.dx} ${exitY + cp1Off.dy}, ${entryX + cp2Off.dx} ${entryY + cp2Off.dy}, ${entryX} ${entryY}`,
  ].join(" ")

  const color = transitionColor(validity)

  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={0.08}
      strokeDasharray="0.6,0.4"
      opacity={0.5}
      strokeLinecap="round"
    />
  )
}

// ---------------------------------------------------------------------------
// FlowPaths — all flow paths for an element list
// ---------------------------------------------------------------------------

export function FlowPaths({
  elements,
}: {
  elements: { id: string; code: string; x: number; y: number; trackType: TrackType; timestamp: number }[]
}): ReactNode {
  if (elements.length < 2) return null

  return (
    <g>
      {elements.slice(0, -1).map((el, i) => {
        const next = elements[i + 1]!
        const validity = transitionRules[el.trackType][next.trackType]
        return (
          <FlowPath
            key={`flow-${el.id}-${next.id}`}
            from={el}
            to={next}
            validity={validity}
          />
        )
      })}
    </g>
  )
}
