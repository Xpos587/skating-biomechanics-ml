# Choreography Timeline Editor — Design Spec

**Date:** 2026-04-19
**Status:** Approved
**Approach:** wavesurfer.js + custom DnD overlay

## Research Summary

### Competitor Landscape

| App | Platform | Key Features | Gap |
|-----|----------|-------------|-----|
| **Skate Plan** | iOS (SwiftUI) | IJS scoring, visual path, music sync, timeline with beat markers, animated preview | iOS only, no web, no multi-track |
| **Choreographic** | iOS + Web beta | Waveform + music sync, formation drag, transition timing (0.01s precision), 3D view, real-time collab | Formation-focused, no IJS scoring |
| **StageKeep** | Web | Music sync with formations, comments, collaboration | Large productions, not skating-specific |
| **Notetracks** | iOS | Notes + markers on audio waveform, regions, export to DAW | For producers, not choreography |
| **8Counts** | iOS | Digital 8-count sheets | No waveform/timeline |

### Open-source

- `cathzvccc/Figure-Skating-Choreography-` (331★) — Python+PyQt5 academic script, not a product.
- Web DAW on wavesurfer.js — CodeSandbox `timeline-react-wavesurfer`, YouTube tutorials exist. Matches our stack.

### Key Insights

1. **Skate Plan** uses a 3-phase workflow: Select Elements → Draw Path → Refine. Simpler than full DAW, better suited for skaters.
2. **Choreographic** proved: waveform + music sync + formation overlay = killer feature.
3. **Notetracks** pattern: notes/markers on audio waveform maps perfectly to elements-as-markers.
4. **Pain point #1** across all apps: syncing elements with music beats.
5. **No web tool** exists combining IJS scoring + music sync + rink visualization.
6. From Olympians: "The process can take months. You have to think about the angles at which the technical panel sees things." — Ben Agosto

### Our Competitive Advantage

IJS scoring + music sync + rink path + ML-powered element analysis + web. No solution combines all five.

---

## Architecture

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ ← Back    "Free Skate Program"     [Auto-saved ✓]    [Export ▼] │
├─────────────────────────────────────────────┬────────────────────┤
│ TRANSPORT BAR                                │                    │
│ [◀◀][▶/❚❚][▶▶]  1:23.4/4:30  🔊━━  [BPM] │                    │
│ Zoom [──●──]  Snap: [beats ▼]  [Auto-fit]   │                    │
├─────────────────────────────────────────────┤   RINK DIAGRAM     │
│ WAVEFORM (wavesurfer.js)                    │                    │
│ ~~~~~~waveform~~~~~ |  beats  |  phrases  ~ │   60×30m Olympic   │
│          ▼ playhead                          │   [Olympic ▼]      │
│ ═════════════════════════════════════════════│   [Custom...]      │
├─────────────────────────────────────────────┤                    │
│ TRACK: JUMPS (7 max)                        │    SVG with        │
│ [3Lz ══][3Lo ══]      [3F+2T]  [3A ══════] │    element         │
│  🟠 5.90  🟠 4.90      🟠 7.50   🟠 8.00   │    markers +       │
├─────────────────────────────────────────────┤    position dots   │
│ TRACK: SPINS (3 max)                        │                    │
│      [CSp4 ═══]  [LSp4 ═══]    [FSp4 ═══]  │                    │
│       🟣 3.20     🟣 3.00       🟣 3.00     │                    │
├─────────────────────────────────────────────┤                    │
│ TRACK: SEQUENCES                            │                    │
│ [StSq4 ═══════════]           [ChSq1 ════]  │                    │
│  🟢 3.90                    🔵 3.00        │                    │
├─────────────────────────────────────────────┤                    │
│ + Click empty area to add element            │                    │
├─────────────────────────────────────────────┼────────────────────┤
│ SCORE BAR (live)                            │                    │
│ TES: 42.30 │ GOE: ±0.00 │ PCS: 35.00       │                    │
│ Total: 77.30 │ ⚠ 0 errors │ 0 warnings     │                    │
└─────────────────────────────────────────────┴────────────────────┘
```

### Responsive

- **Desktop (lg+):** Side-by-side timeline + rink diagram
- **Tablet (md):** Rink diagram collapsible, toggle button
- **Mobile:** Rink diagram hidden behind bottom sheet, timeline full width

---

## Components

### 1. TransportBar

Playback controls and timeline settings.

**Elements:**
- Play/pause, skip back 5s, skip forward 5s buttons
- Current time / total duration display
- Volume slider
- BPM display (from music analysis)
- Zoom slider (controls pixels-per-second density)
- Snap toggle: `beats` / `phrases` / `off`
- Auto-fit button (zoom to fit all elements)

**State:** Uses wavesurfer.js built-in playback. Transport bar is a thin horizontal strip above the timeline.

### 2. WaveformView (wavesurfer.js)

Audio waveform with beat markers as the timeline background.

**Implementation:**
- `wavesurfer.js` v7 with `Waveform` plugin
- Audio source: R2 URL from `music_analysis.audio_url`
- Beat markers: vertical lines from `music_analysis.peaks` array
- Phrase markers: vertical lines from `music_analysis.structure` array (lighter color)
- Playhead: red vertical line, synced with wavesurfer current time
- Waveform height: ~80px, matches timeline width exactly
- Scroll/zoom synchronized with element tracks below

**Key behavior:**
- Click on waveform → seek to position (wavesurfer handles this)
- Playhead position drives element highlighting on tracks
- Zoom slider controls both waveform and element track scroll

### 3. ElementTrack

A single horizontal track for one element type (Jumps / Spins / Sequences).

**Props:**
```typescript
interface ElementTrackProps {
  type: "jumps" | "spins" | "sequences"
  elements: TimelineElement[]
  maxElements: number  // 7 for jumps, 3 for spins
  currentTime: number
  duration: number
  pixelsPerSecond: number
  snapMode: "beats" | "phrases" | "off"
  beatMarkers: number[]  // timestamps from music analysis
  onElementMove: (id: string, newTimestamp: number) => void
  onElementAdd: (trackType: string, timestamp: number) => void
  onElementEdit: (id: string, updates: Partial<TimelineElement>) => void
  onElementDelete: (id: string) => void
}
```

**Visual:**
- Track label on the left: "JUMPS (3/7)", "SPINS (2/3)", "SEQUENCES"
- Track color: orange for jumps, purple for spins, green/blue for sequences
- Empty track shows hint: "+ Double-click to add element"
- Over-capacity: red label when element count > maxElements

**Each element clip renders:**
- Code label (e.g., "3Lz")
- Base value below code (e.g., "5.90")
- Highlight when playhead is within element's time window
- Red border when validation error (Zayak, etc.)
- Back-half indicator: subtle gradient background for elements in second half

### 4. TimelineElement

An element positioned on a track.

**Data model:**
```typescript
interface TimelineElement {
  id: string
  code: string          // "3Lz", "CSp4", "StSq4"
  trackType: "jumps" | "spins" | "sequences"
  timestamp: number     // start time in seconds
  duration: number      // time window width (default: 3s jump, 5s spin, 8s sequence)
  goe: number           // -5 to +5
  jumpPassIndex?: number  // for back-half bonus calculation
  position?: { x: number; y: number }  // rink position
  validationErrors: string[]
  validationWarnings: string[]
}
```

**Interactions:**
- **Drag horizontally** → move element timestamp (snap to beats if enabled)
- **Drag edges** → resize duration window
- **Single click** → select, show inline popover with edit options
- **Double click** → open full edit popover (code picker, GOE slider, position)
- **Right click / long press** → context menu (delete, duplicate, change type)
- **Keyboard:** Delete key removes selected element

### 5. ElementPicker (Popover)

Shown when adding or editing an element.

**Contents:**
- Search input (filter by code or Russian name)
- Element list grouped by type, filtered to current track's type
- Each item: code + base value + Russian name
- Combinations section for jumps: "3Lz+2T", "3F+2T+2Lo", etc.
- Level selector for spins (Lv1-4)
- Selected element highlighted

### 6. ElementEditor (Popover/Sheet)

Shown when clicking an existing element.

**Contents:**
- Element code (clickable to open ElementPicker)
- GOE slider: -5 to +5, step 1
- Duration input (seconds)
- Rink position: click on rink diagram to set, or X/Y inputs
- Notes field (free text, for choreographer notes)
- Validation status: errors/warnings with explanations
- Delete button (with confirmation)
- Duplicate button (creates copy +2s after)

### 7. RinkDiagramPanel

Updated rink diagram with presets and element position editing.

**Changes from current `rink_renderer.py`:**
- Accept `rinkWidth` and `rinkHeight` parameters (meters)
- Presets dropdown: Olympic (60×30), NHL (61×26), Training (56×26), Custom
- Custom option: inline inputs for width × height
- Click on rink → set position for selected element
- Element markers show code + numbered sequence
- Connecting lines between sequential elements (dashed)
- Highlight current element during playback
- Responsive SVG that fills the panel width

**Rink presets:**
```typescript
const RINK_PRESETS = [
  { name: "Olympic", width: 60, height: 30 },
  { name: "NHL", width: 61, height: 26 },
  { name: "Training", width: 56, height: 26 },
]
```

### 8. ScoreBar (Updated)

Live score calculation, already exists. Updates needed:

- Recalculate on every element change (debounced 100ms)
- Back-half bonus: auto-detect from timestamp positions
  - Back half = elements whose timestamp > (duration / 2)
  - Only jump passes qualify for +10% BV
- Inline validation status with error/warning counts
- Click error count → scroll to first invalid element

---

## State Management

### Zustand Store

```typescript
interface ChoreographyEditorState {
  // Program data
  programId: string | null
  title: string
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"

  // Elements
  elements: TimelineElement[]
  selectedElementId: string | null

  // Music
  musicAnalysis: MusicAnalysis | null
  audioUrl: string | null

  // Playback state (derived from wavesurfer)
  currentTime: number
  isPlaying: boolean

  // Timeline settings
  pixelsPerSecond: number  // zoom level
  snapMode: "beats" | "phrases" | "off"

  // Rink
  rinkPreset: "olympic" | "nhl" | "training" | "custom"
  rinkDimensions: { width: number; height: number }

  // Computed (derived, not stored)
  validation: ValidationResult

  // Actions
  addElement: (trackType: string, timestamp: number, code: string) => void
  removeElement: (id: string) => void
  moveElement: (id: string, newTimestamp: number) => void
  updateElement: (id: string, updates: Partial<TimelineElement>) => void
  duplicateElement: (id: string) => void
  setSnapMode: (mode: SnapMode) => void
  setZoom: (pps: number) => void
  setRinkPreset: (preset: RinkPreset) => void
  saveProgram: () => Promise<void>  // debounced auto-save
}
```

### Persistence

- Auto-save: debounced 500ms after any state change
- Uses existing `useSaveProgram` mutation
- Optimistic updates: local state updates immediately, server sync in background
- Conflict resolution: server version wins (last-write-wins, user warned)

---

## Backend Changes

### `rink_renderer.py`

Update `render_rink()` to accept dimensions:

```python
def render_rink(
    elements: list[dict],
    *,
    width: int = 1200,
    height: int = 600,
    rink_width: float = 60.0,
    rink_height: float = 30.0,
) -> str:
```

Scale all coordinates proportionally. Presets defined in Python dict matching frontend constants.

### API Routes

No new routes needed. Existing endpoints cover all operations:
- `POST /choreography/programs/{id}` — save layout with updated elements
- `POST /choreography/validate` — validate layout
- `POST /choreography/render-rink` — render with dimensions
- `GET /choreography/music/{id}/analysis` — get beat markers

### Layout Data Model

Update `ChoreographyProgram.layout` JSON schema:

```json
{
  "elements": [
    {
      "id": "uuid",
      "code": "3Lz",
      "track_type": "jumps",
      "timestamp": 15.3,
      "duration": 3.0,
      "goe": 0,
      "position": { "x": 20.5, "y": 12.0 },
      "jump_pass_index": 0
    }
  ],
  "rink_preset": "olympic",
  "rink_dimensions": { "width": 60, "height": 30 }
}
```

### Back-Half Bonus Detection

Frontend calculates which elements qualify:
- Get `duration` from music analysis
- Group elements by `jump_pass_index` (each unique index = one jump pass)
- A jump pass is "in back half" if its first element's `timestamp > duration/2`
- Up to 3 jump passes can qualify for +10% BV (ISU rule)
- Collect qualifying pass indices into `back_half_indices` set
- Pass to `calculate_tes()` which applies 1.10x multiplier to qualifying elements' base values

---

## Dependencies

### New

- `wavesurfer.js@^7` — audio waveform, playback, zoom, regions
- No DnD library — custom pointer events (simpler, no dependency, works with wavesurfer scroll sync)

### Existing (used)

- `zustand` — editor state (already in package.json)
- `@tanstack/react-query` — server state sync
- `radix-ui` — popovers, dialogs, dropdowns
- `lucide-react` — icons for transport controls

---

## Implementation Phases

### Phase 1: Core Timeline (MVP)

1. `WaveformView` component — wavesurfer.js integration
2. `ElementTrack` component — renders element clips
3. `TimelineElement` component — drag to move
4. `TransportBar` component — playback controls
5. Zustand store with basic CRUD
6. Wire up to existing save API

### Phase 2: Editing & Interaction

7. `ElementPicker` popover — search + filtered list
8. `ElementEditor` popover — GOE slider, duration, notes
9. Snap to beats implementation
10. Inline validation (Zayak, max elements)
11. Keyboard shortcuts (Delete, Ctrl+Z undo)

### Phase 3: Scoring & Rink

12. Live `ScoreBar` updates
13. Back-half bonus auto-detection
14. `RinkDiagramPanel` with presets
15. Click-on-rink position editing
16. Rink element highlighting during playback

### Phase 4: Polish

17. Responsive layout (tablet/mobile)
18. Auto-save indicator
19. Undo/redo stack (zustand middleware)
20. Export improvements (PDF with rink + element list)
21. Accessibility (keyboard navigation, screen reader labels)

---

## i18n Keys Needed

```
choreography.timeline.addelerent
choreography.timeline.deleteElement
choreography.timeline.duplicateElement
choreography.timeline.snapBeats
choreography.timeline.snapPhrases
choreography.timeline.snapOff
choreography.timeline.autoFit
choreography.timeline.rinkPreset
choreography.timeline.rinkCustom
choreography.timeline.validationErrors
choreography.timeline.validationWarnings
choreography.timeline.backHalf
choreography.timeline.jumpsTrack
choreography.timeline.spinsTrack
choreography.timeline.sequencesTrack
```
