# Rink Trace Figure Visualization

**Date:** 2026-04-20
**Status:** Approved

## Problem

Current RinkDiagram renders elements as simple geometric shapes (solid circles, rings, dashed rectangles) with code labels. This doesn't convey skating movement patterns — the most valuable information for choreography planning.

Skreate (github.com/daviddrysdale/skreate) demonstrates ISU-style "trace figure" notation where each element is drawn as SVG arcs showing entry curves, takeoff, landing, and exit paths. We adopt this approach for our rink diagram.

## Scope

**In scope (this iteration):**
- Jumps: trace figures with entry arc + hop marker + exit arc
- Sequences (StSq, ChSq): serpentine/S-curve showing ice coverage
- Spins: unchanged (circle with stroke — separate future work)

**Out of scope:**
- Individual turn notation (three-turns, brackets, rockers, counters)
- Spin trace figures (future iteration)
- Editable entry/exit angles (future: user-draggable direction handles)
- Step sequence internal turn decomposition

## Architecture

### New file: `rink-figures.tsx`

Pure SVG path generators. No React hooks, no state. Each function returns SVG element(s) for a given element type and position.

```
frontend/src/components/choreography/
├── rink-diagram.tsx        # Updated: imports from rink-figures
├── rink-figures.tsx        # NEW: SVG trace figure generators
└── ...
```

### Modified file: `rink-diagram.tsx`

Replace current marker rendering (circles, rectangles) with trace figure components. Remove flow lines.

## Visual Design

### Coordinate system

Rink SVG viewBox: `0 0 60 30` (meters). All trace figure dimensions in rink units.

### Jump trace figures

Each jump renders as:
1. **Entry arc** — curved path approaching the takeoff point
2. **Hop marker** — filled circle at the jump position (takeoff/landing point)
3. **Exit arc** — curved path leaving the landing point

**Sizes:**
- Total trace figure extent: ~4m x ~2m
- Hop marker radius: 0.5m
- Arc stroke width: 0.12m

**Entry/exit angle determination:**
- Deterministic hash of element ID → angle in degrees (0-360)
- Ensures stability across re-renders while providing visual variety
- Hash function: simple string hash mod 360

**Jump subtypes:**

| Category | Jumps | Entry arc | Exit arc |
|----------|-------|-----------|----------|
| Toe jumps | T, Lz, F | Short (2m), moderate curve | Short (2m), opposite curve |
| Edge jumps | S, Lo | Longer (3m), wider curve | Short (2m), moderate curve |
| Axel | A | Longest (3.5m), wide approach | Short (2m), tight exit |

All arcs use SVG `<path>` with arc commands (`a rx ry rotation large-arc-flag sweep-flag x y`).

**Color:** Stroke = track color (orange for jumps), fill = `none`. Hop marker fill = track color, opacity 0.85.

### Sequence trace figures

Step sequences and choreographic sequences render as:
- **Serpentine/S-curve** — 2-3 connected arcs passing through the element position
- Length: ~10m total path length
- Shows that the element covers significant ice area

**Visual:**
- Stroke = track color (emerald for sequences)
- Stroke width: 0.12m
- Stroke dasharray: `0.4,0.2` (dashed, lighter than jumps)
- No fill

### Spin markers

Unchanged from current implementation:
- Circle with track color stroke, 25% opacity fill
- Radius: 1.3m

### Labels

All elements retain:
- **Number badge** — circle with sequence number, offset to upper-right
- **Code label** — element code text below the figure

### Removed elements

- **Flow lines** — dashed straight lines between consecutive elements. Trace figures already show movement direction; flow lines create visual clutter.

## API Design

```typescript
// rink-figures.tsx

interface TraceFigureProps {
  x: number        // Center position X in rink units
  y: number        // Center position Y in rink units
  code: string     // Element code (e.g., "3Lz", "StSq3", "CSp4")
  color: string    // Track color hex
  elementId: string // For deterministic angle hash
}

// Jump trace figure
function JumpTrace({ x, y, code, color, elementId }: TraceFigureProps): JSX.Element

// Sequence trace figure
function SequenceTrace({ x, y, color, elementId }: TraceFigureProps): JSX.Element

// Deterministic angle from element ID
function hashAngle(elementId: string): number
```

## Interaction

- Trace figures remain draggable (pointer events on the hop marker / serpentine group)
- Selection highlighting: subtle glow/shadow on the trace group
- Hover: slight opacity increase on the trace paths

## References

- Skreate: https://github.com/daviddrysdale/skreate — Rust/WASM SVG trace figure renderer
- Skreate live: https://lurklurk.org/skreate
- ISU edge code notation: Foot (L/R) + Direction (F/B) + Edge (O/I)
- Current RinkDiagram: `frontend/src/components/choreography/rink-diagram.tsx`
