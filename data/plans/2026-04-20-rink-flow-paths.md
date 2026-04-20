# Rink Flow Paths + Transition Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect sequential elements on the rink diagram with Bezier flow paths, colored by transition validity (ok/warning/invalid).

**Architecture:** Two new pure modules — `transition-rules.ts` (static validity matrix) and `rink-flow.tsx` (Bezier SVG generator using endpoint geometry from `rink-figures.tsx`). `rink-diagram.tsx` adds a FlowPaths layer between rink markings and trace figures.

**Tech Stack:** React 19, SVG cubic Bezier paths, TypeScript. No test framework — verify with `bunx tsc --noEmit` and visual check at `:3000`.

**Files:**
- Create: `frontend/src/components/choreography/transition-rules.ts`
- Create: `frontend/src/components/choreography/rink-flow.tsx`
- Modify: `frontend/src/components/choreography/rink-diagram.tsx`

---

### Task 1: Create `transition-rules.ts` — validity matrix + color map

**Files:**
- Create: `frontend/src/components/choreography/transition-rules.ts`

- [ ] **Step 1: Create the file**

```ts
import type { TrackType } from "@/types/choreography"

export type TransitionValidity = "ok" | "warning" | "invalid"

/**
 * ISU-based transition validity matrix.
 *
 * - spin → * : skater loses speed in spin, needs time to rebuild
 * - sequence → sequence : needs significant ice coverage, unusual
 * - jump → * : standard — jumps land with momentum
 */
export const transitionRules: Record<TrackType, Record<TrackType, TransitionValidity>> = {
  jumps: { jumps: "ok", spins: "ok", sequences: "ok" },
  spins: { jumps: "warning", spins: "warning", sequences: "ok" },
  sequences: { jumps: "ok", spins: "ok", sequences: "warning" },
}

const COLORS: Record<TransitionValidity, string> = {
  ok: "oklch(0.6 0.02 260)",
  warning: "oklch(0.75 0.15 80)",
  invalid: "oklch(0.65 0.25 25)",
}

export function transitionColor(v: TransitionValidity): string {
  return COLORS[v]
}

export function getTransitionValidity(from: TrackType, to: TrackType): TransitionValidity {
  return transitionRules[from][to]
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | grep transition-rules || echo "No errors in our file"`
Expected: "No errors in our file"

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/transition-rules.ts
git commit -m "feat(choreography): add transition validity matrix for element flow paths"
```

---

### Task 2: Create `rink-flow.tsx` — Bezier flow paths

**Files:**
- Create: `frontend/src/components/choreography/rink-flow.tsx`

- [ ] **Step 1: Create the file**

This module must mirror the geometry from `rink-figures.tsx` to compute trace figure endpoints. Key values from rink-figures.tsx:

- Jump entryLen: toe=2, edge=3, axel=3.5 (from `jumpCategory()`)
- Jump exitLen: always 2
- Spin radius: 1.3
- Sequence segLen: 3.5, half-span: segLen*1.5 = 5.25
- hashAngle: deterministic rotation per element ID

```tsx
import type { ReactNode } from "react"
import type { TrackType } from "@/types/choreography"
import { hashAngle } from "./rink-figures"
import { transitionColor } from "./transition-rules"
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | grep -E "rink-flow|transition-rules" || echo "No errors in our files"`
Expected: "No errors in our files"

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/rink-flow.tsx
git commit -m "feat(choreography): add Bezier flow paths connecting sequential elements"
```

---

### Task 3: Integrate FlowPaths into RinkDiagram

**Files:**
- Modify: `frontend/src/components/choreography/rink-diagram.tsx`

- [ ] **Step 1: Add import for FlowPaths**

After line 8 (after the rink-figures import), add:

```tsx
import { FlowPaths } from "./rink-flow"
```

- [ ] **Step 2: Add FlowPaths layer between rink markings and trace figures**

Find the line `{/* Elements */}` (line 252) and insert the FlowPaths block immediately before it:

```tsx
        {/* Flow paths between sequential elements */}
        <FlowPaths elements={rinkElements} />

        {/* Elements */}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | grep -E "rink-diagram|rink-flow|transition-rules" || echo "No errors in our files"`
Expected: "No errors in our files"

- [ ] **Step 4: Verify lint passes**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx next lint 2>&1 | head -20`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/choreography/rink-diagram.tsx
git commit -m "feat(choreography): integrate flow paths into rink diagram"
```

---

### Task 4: Visual verification + polish

**Files:**
- Possibly modify: `frontend/src/components/choreography/rink-flow.tsx`

- [ ] **Step 1: Open the app and verify**

1. Go to `http://localhost:3000/choreography`
2. Open a program with multiple elements (jumps, spins, sequences)

Verify:
- Bezier curves connect sequential elements
- Curves start from trace figure exit points, end at entry points
- jump → jump: gray-blue line (ok)
- spin → jump: amber/yellow line (warning)
- spin → spin: amber/yellow line (warning)
- sequence → sequence: amber/yellow line (warning)
- Curves don't overlap with trace figures excessively

- [ ] **Step 2: Fix any visual issues**

Common issues:
- Tangent direction wrong (curve doubles back) → check angle calculation in polarOffset
- Curves too short/long → adjust tangentLen multiplier (currently 0.3)
- Colors too faint → increase opacity (currently 0.5)
- Endpoint doesn't align with trace figure → check exit/entry geometry mirrors rink-figures.tsx

- [ ] **Step 3: Commit polish if needed**

```bash
git add -u
git commit -m "fix(choreography): polish flow path geometry and colors"
```
