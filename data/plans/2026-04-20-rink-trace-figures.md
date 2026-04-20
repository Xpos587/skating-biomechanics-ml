# Rink Trace Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simple geometric element markers on the rink diagram with ISU-style SVG trace figures — curved entry/exit arcs for jumps, serpentine curves for sequences.

**Architecture:** New pure-function module `rink-figures.tsx` exports `JumpTrace`, `SequenceTrace`, `SpinMarker` SVG components. `rink-diagram.tsx` imports these and removes old markers + flow lines. No state, no hooks in the figures module.

**Tech Stack:** React 19, SVG path arcs, TypeScript. No test framework (frontend has none — verify with `bunx tsc --noEmit` and visual check at `:3000`).

**Files:**
- Create: `frontend/src/components/choreography/rink-figures.tsx`
- Modify: `frontend/src/components/choreography/rink-diagram.tsx`

---

### Task 1: Create `rink-figures.tsx` — hashAngle utility + JumpTrace component

**Files:**
- Create: `frontend/src/components/choreography/rink-figures.tsx`

- [ ] **Step 1: Create the file with `hashAngle`, jump classification, and `JumpTrace`**

```tsx
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

  // Entry arc: sweep upward (negative Y in SVG = upward)
  const entryR = entryLen * 1.2 // radius slightly longer than chord
  const exitR = exitLen * 1.2

  const d = [
    // Move to entry start (behind the hop, to the left)
    `M ${-entryLen} 0`,
    // Arc to hop position — curving upward
    `A ${entryR} ${entryR} 0 0 1 0 0`,
    // Arc away from hop — curving downward (opposite sweep)
    `A ${exitR} ${exitR} 0 0 0 ${exitLen} 0`,
  ].join(" ")

  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`}>
      {/* Trace path */}
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={strokeW}
        strokeLinecap="round"
        opacity={0.7}
      />
      {/* Hop marker (filled circle at jump position) */}
      <circle cx={0} cy={0} r={hopR} fill={color} opacity={0.85} />
    </g>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | head -20`
Expected: No new errors (file is new, only exports functions/components)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/rink-figures.tsx
git commit -m "feat(choreography): add rink trace figure generators — hashAngle + JumpTrace"
```

---

### Task 2: Add `SequenceTrace` and `SpinMarker` to `rink-figures.tsx`

**Files:**
- Modify: `frontend/src/components/choreography/rink-figures.tsx`

- [ ] **Step 1: Append SequenceTrace and SpinMarker after JumpTrace**

Add these exports after the `JumpTrace` function:

```tsx
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

  // Serpentine: 3 arcs alternating direction, total length ~10m
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
// SpinMarker — circle with stroke (unchanged from current implementation)
// ---------------------------------------------------------------------------

/**
 * Spin marker: circle with colored stroke and low-opacity fill.
 * Extracted here for consistency with other trace figures.
 */
export function SpinMarker({ x, y, color }: Omit<TraceFigureProps, "code" | "elementId">): ReactNode {
  return (
    <g>
      <circle cx={x} cy={y} r={1.3} fill={color} opacity={0.25} stroke={color} strokeWidth={0.15} />
    </g>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | head -20`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/rink-figures.tsx
git commit -m "feat(choreography): add SequenceTrace serpentine + SpinMarker to rink figures"
```

---

### Task 3: Update `rink-diagram.tsx` — replace markers with trace figures, remove flow lines

**Files:**
- Modify: `frontend/src/components/choreography/rink-diagram.tsx`

- [ ] **Step 1: Add import for trace figures**

At line 7 (after the store import), add:

```tsx
import { JumpTrace, SequenceTrace, SpinMarker } from "./rink-figures"
```

- [ ] **Step 2: Remove the flow lines block**

Delete lines 252–284 (the `{rinkElements.length > 1 && ...}` block that renders dashed lines and arrowheads between consecutive elements).

- [ ] **Step 3: Replace element markers with trace figures**

Replace lines 287–373 (the `{rinkElements.map((el, i) => { ... })}` block) with:

```tsx
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
```

Note: code label Y moved from `+1.8` to `+2.2` to clear the longer trace figures.

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit 2>&1 | head -20`
Expected: No errors

- [ ] **Step 5: Verify lint passes**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx next lint 2>&1 | head -20`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/choreography/rink-diagram.tsx
git commit -m "feat(choreography): replace element markers with trace figures, remove flow lines"
```

---

### Task 4: Visual verification + polish

**Files:**
- Possibly modify: `frontend/src/components/choreography/rink-figures.tsx`

- [ ] **Step 1: Open the app and navigate to a program with elements**

1. Go to `http://localhost:3000/choreography`
2. Open or create a program with jumps, spins, and sequences
3. Check the rink diagram panel on the right

Verify:
- Jumps show curved entry/exit arcs with a filled hop circle
- Sequences show dashed serpentine S-curves
- Spins show the same circle marker as before
- Number badges and code labels are readable
- No flow lines between elements
- Different elements have different rotation angles (visual variety)

- [ ] **Step 2: Fix any visual issues**

Common issues to watch for:
- Trace figures extending outside the rink boundary → reduce arc radii
- Labels overlapping trace paths → increase label Y offset
- Trace figures too small/large → adjust `segLen`, `entryLen`, `hopR`
- Color opacity too faint/strong → adjust `opacity` values

- [ ] **Step 3: Final commit if any polish needed**

```bash
git add -u
git commit -m "fix(choreography): polish trace figure sizes and label positioning"
```
