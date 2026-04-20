# Rink Flow Paths + Transition Rules

**Date:** 2026-04-20
**Status:** Approved

## Problem

Elements on the rink diagram are isolated — no visual connection between them. A choreography program is a sequence of elements connected by skating transitions. The diagram should show these connections and flag physically impossible transitions.

## Scope

**In scope:**
- Bezier curves connecting sequential elements (flow paths)
- Static transition validity matrix (element type → element type)
- Visual distinction between valid/warning/invalid transitions
- Integration into existing RinkDiagram

**Out of scope:**
- Automatic layout optimization (positions come from CSP solver or manual drag)
- Distance-based validation (geometric feasibility — future iteration)
- Physics-based transition simulation (speed, momentum — future iteration)
- Per-element transition rules (specific jump combinations)

## Architecture

### New files

```
frontend/src/components/choreography/
├── rink-figures.tsx      # Existing — trace figures
├── rink-flow.tsx         # NEW — Bezier flow path generator
├── transition-rules.ts   # NEW — transition validity matrix
└── rink-diagram.tsx      # Modified — add FlowPaths layer
```

### transition-rules.ts

Pure data module. No React, no hooks.

```typescript
export type TransitionValidity = "ok" | "warning" | "invalid"
export type TrackType = "jumps" | "spins" | "sequences"

// Matrix: transitionRules[from][to] → validity
export const transitionRules: Record<TrackType, Record<TrackType, TransitionValidity>>
```

**Initial rules (based on ISU + skating physics):**

| from \ to | jumps | spins | sequences |
|-----------|-------|-------|-----------|
| jumps     | ok    | ok    | ok        |
| spins     | warning | warning | ok      |
| sequences | ok    | ok    | warning   |

Rationale:
- **spin → anything**: after a spin the skater loses speed and needs time to accelerate. Two spins in a row = very rare (combo spin counts as one element). Spin → jump needs speed buildup.
- **sequence → sequence**: two step sequences need significant ice coverage. Possible but unusual in competitive programs.
- **jump → *: jumps land with momentum, transition to anything is standard.

**Colors:**
- `ok`: `oklch(0.6 0.02 260)` — neutral gray-blue, dashed
- `warning`: `oklch(0.75 0.15 80)` — amber/yellow, dashed
- `invalid`: `oklch(0.65 0.25 25)` — red, dashed

### rink-flow.tsx

Pure SVG generator. No React hooks, no state.

**Function:**
```typescript
interface FlowPathProps {
  from: { x: number; y: number; angle: number; exitLen: number }
  to: { x: number; y: number; angle: number; entryLen: number }
  validity: TransitionValidity
}

export function FlowPath({ from, to, validity }: FlowPathProps): ReactNode
```

**Algorithm:**
1. Compute exit point: `from` position + exit arc endpoint (offset by exitLen in direction of exit angle)
2. Compute entry point: `to` position - entry arc endpoint (offset by entryLen in opposite direction of entry angle)
3. Cubic Bezier from exit point to entry point
4. Control point 1: exit point + tangent in exit direction (length = 30% of total distance)
5. Control point 2: entry point - tangent in entry direction (length = 30% of total distance)

**Style:**
- Stroke width: 0.08m
- Stroke dasharray: `0.6,0.4`
- Color: based on validity (see above)
- Opacity: 0.5
- No arrowheads (direction implied by element order)

**Exit/entry point computation:**

Each trace figure has known geometry:
- Jump exit: position + exitLen (2m) at exit angle
- Jump entry: position - entryLen (2-3.5m by category) at entry angle
- Spin: no directional exit/entry — use hashAngle direction with offset 1.3m (radius)
- Sequence: exit = position + segLen*1.5 at angle, entry = position - segLen*1.5 at angle

These endpoints are computed by utility functions in `rink-flow.tsx` that mirror the geometry from `rink-figures.tsx`.

### rink-diagram.tsx changes

Add FlowPaths layer between rink markings and trace figures:

```
Render order:
1. Rink markings (ice, lines, circles)
2. Flow paths (new — below elements)
3. Trace figures (existing)
4. Labels (existing)
```

Flow paths are computed from `rinkElements` array (sorted by timestamp). For each consecutive pair (i, i+1):
1. Get transition validity from `transitionRules`
2. Compute exit point of element i and entry point of element i+1
3. Render FlowPath

## API

```typescript
// transition-rules.ts
export function getTransitionValidity(from: TrackType, to: TrackType): TransitionValidity

// rink-flow.tsx
export function FlowPaths({ elements }: { elements: RinkElement[] }): ReactNode

// Utility — compute trace figure endpoints
export function getExitPoint(el: RinkElement): { x: number; y: number; angle: number }
export function getEntryPoint(el: RinkElement): { x: number; y: number; angle: number }
```

## Future enhancements

- Distance-based validation: minimum meters between element types
- Per-element rules: specific jump combination validity (e.g., same-foot landing required)
- Time-based validation: enough time between elements for transition
- Interactive: click flow path to see warning explanation
- Automatic layout: reposition elements to minimize invalid transitions
