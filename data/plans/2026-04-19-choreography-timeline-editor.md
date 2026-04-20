# Choreography Timeline Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current simple element list editor with a multi-track timeline editor (DAW-style) with audio waveform playback, drag-and-drop element placement, snap-to-beats, and live IJS scoring.

**Architecture:** wavesurfer.js for audio waveform/playback + custom pointer-event-based DnD overlay for element tracks. Zustand store for editor state. 3 tracks (Jumps/Spins/Sequences). ScoreBar auto-updates. Rink diagram with presets.

**Tech Stack:** wavesurfer.js v7, zustand, React 19, Radix UI popovers, Tailwind CSS v4, bun

**Spec:** `docs/specs/2026-04-19-choreography-timeline-editor-design.md`

---

## File Structure

### New Frontend Files

| File | Responsibility |
|------|---------------|
| `frontend/src/components/choreography/editor/store.ts` | Zustand store: elements, playback, timeline settings, rink, auto-save |
| `frontend/src/components/choreography/editor/waveform-view.tsx` | wavesurfer.js waveform with beat markers |
| `frontend/src/components/choreography/editor/transport-bar.tsx` | Playback controls, zoom, snap mode |
| `frontend/src/components/choreography/editor/element-track.tsx` | Single track row (jumps/spins/sequences) |
| `frontend/src/components/choreography/editor/timeline-element.tsx` | Draggable element clip on a track |
| `frontend/src/components/choreography/editor/element-picker.tsx` | Search+filter popover for adding elements |
| `frontend/src/components/choreography/editor/element-editor.tsx` | Edit popover: GOE slider, duration, position, delete |
| `frontend/src/components/choreography/editor/rink-panel.tsx` | Updated rink diagram with presets, click-to-position |

### Modified Frontend Files

| File | Change |
|------|--------|
| `frontend/src/app/(app)/choreography/programs/[id]/page.tsx` | Replace simple list with timeline editor layout |
| `frontend/src/components/choreography/score-bar.tsx` | Accept new props for live back-half detection |
| `frontend/src/lib/api/choreography.ts` | Add `useRenderRinkWithDimensions` hook |
| `frontend/src/types/choreography.ts` | Add `TimelineElement`, `SnapMode`, `RinkPreset` types |
| `frontend/messages/ru.json` | Add `choreography.timeline.*` i18n keys |
| `frontend/messages/en.json` | Add `choreography.timeline.*` i18n keys |

### Modified Backend Files

| File | Change |
|------|--------|
| `backend/app/services/choreography/rink_renderer.py` | Add `rink_width`/`rink_height` params |
| `backend/app/routes/choreography.py` | Pass rink dimensions to render_rink |
| `backend/app/schemas.py` | Add `rink_width`/`rink_height` to `RenderRinkRequest` |

---

## Phase 1: Core Timeline (MVP)

### Task 1: Install wavesurfer.js dependency

- [ ] **Step 1: Install wavesurfer.js**

Run: `cd frontend && bun add wavesurfer.js@^7`

Expected: `wavesurfer.js` added to `package.json` dependencies

- [ ] **Step 2: Verify install**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No new type errors

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/bun.lockb
git commit -m "chore(choreography): add wavesurfer.js dependency"
```

---

### Task 2: Add TimelineElement and editor types

**Files:**
- Modify: `frontend/src/types/choreography.ts`

- [ ] **Step 1: Add new types to choreography.ts**

Append after the existing `Inventory` interface:

```typescript
// ---------------------------------------------------------------------------
// Timeline Editor
// ---------------------------------------------------------------------------

export type TrackType = "jumps" | "spins" | "sequences"
export type SnapMode = "beats" | "phrases" | "off"
export type RinkPreset = "olympic" | "nhl" | "training" | "custom"

export const TRACK_CONFIG: Record<TrackType, { maxElements: number; color: string; colorVar: string }> = {
  jumps: { maxElements: 7, color: "text-orange-500", colorVar: "bg-orange-500/20" },
  spins: { maxElements: 3, color: "text-purple-500", colorVar: "bg-purple-500/20" },
  sequences: { maxElements: 10, color: "text-emerald-500", colorVar: "bg-emerald-500/20" },
}

export const RINK_PRESETS: { name: string; width: number; height: number }[] = [
  { name: "Olympic", width: 60, height: 30 },
  { name: "NHL", width: 61, height: 26 },
  { name: "Training", width: 56, height: 26 },
]

export const DEFAULT_DURATIONS: Record<TrackType, number> = {
  jumps: 3,
  spins: 6,
  sequences: 8,
}

export interface TimelineElement {
  id: string
  code: string
  trackType: TrackType
  timestamp: number
  duration: number
  goe: number
  jumpPassIndex?: number
  position?: { x: number; y: number } | null
  notes?: string
}

export function layoutElementToTimeline(el: LayoutElement, index: number): TimelineElement {
  const trackType: TrackType = el.code.includes("Sp")
    ? "spins"
    : el.code.startsWith("StSq") || el.code.startsWith("ChSq")
      ? "sequences"
      : "jumps"
  return {
    id: `el-${index}-${el.code}-${el.timestamp}`,
    code: el.code,
    trackType,
    timestamp: el.timestamp,
    duration: DEFAULT_DURATIONS[trackType],
    goe: el.goe,
    jumpPassIndex: el.jump_pass_index ?? undefined,
    position: el.position ?? undefined,
  }
}

export function timelineToLayoutElement(el: TimelineElement): LayoutElement {
  return {
    code: el.code,
    goe: el.goe,
    timestamp: el.timestamp,
    position: el.position ?? null,
    is_back_half: false,
    is_jump_pass: el.trackType === "jumps",
    jump_pass_index: el.jumpPassIndex ?? null,
  }
}
```

- [ ] **Step 2: Verify types**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/choreography.ts
git commit -m "feat(choreography): add timeline editor types and constants"
```

---

### Task 3: Create Zustand editor store

**Files:**
- Create: `frontend/src/components/choreography/editor/store.ts`

- [ ] **Step 1: Write the store**

```typescript
"use client"

import { create } from "zustand"
import type {
  ChoreographyProgram,
  LayoutElement,
  TimelineElement,
  TrackType,
  SnapMode,
  RinkPreset,
  DEFAULT_DURATIONS,
  layoutElementToTimeline,
  timelineToLayoutElement,
} from "@/types/choreography"
import { calculateClientSideTes } from "./score-utils"

interface ChoreographyEditorState {
  // Program
  programId: string | null
  title: string
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"
  musicAnalysisId: string | null
  audioUrl: string | null
  musicDuration: number
  beatMarkers: number[]
  phraseMarkers: number[]

  // Elements
  elements: TimelineElement[]
  selectedElementId: string | null

  // Playback (set by WaveformView via setCurrentTime/setIsPlaying)
  currentTime: number
  isPlaying: boolean

  // Timeline
  pixelsPerSecond: number
  snapMode: SnapMode

  // Rink
  rinkPreset: RinkPreset
  rinkWidth: number
  rinkHeight: number

  // Loading
  isLoading: boolean

  // Actions — initialization
  initFromProgram: (program: ChoreographyProgram, audioUrl: string | null, musicDuration: number, beatMarkers: number[], phraseMarkers: number[]) => void

  // Actions — elements
  addElement: (trackType: TrackType, timestamp: number, code: string) => void
  removeElement: (id: string) => void
  moveElement: (id: string, newTimestamp: number) => void
  resizeElement: (id: string, newDuration: number) => void
  updateElement: (id: string, updates: Partial<TimelineElement>) => void
  duplicateElement: (id: string) => void
  setSelectedElement: (id: string | null) => void

  // Actions — timeline
  setCurrentTime: (time: number) => void
  setIsPlaying: (playing: boolean) => void
  setPixelsPerSecond: (pps: number) => void
  setSnapMode: (mode: SnapMode) => void

  // Actions — rink
  setRinkPreset: (preset: RinkPreset, width?: number, height?: number) => void
  setRinkDimensions: (width: number, height: number) => void

  // Actions — save
  setTitle: (title: string) => void

  // Computed
  getElementsByTrack: (trackType: TrackType) => TimelineElement[]
  getLayoutForSave: () => { layout: LayoutElement[]; total_tes: number; back_half_indices: number[] }
}

let idCounter = 0
function nextId(): string {
  idCounter += 1
  return `tl-${idCounter}-${Date.now()}`
}

export const useChoreographyEditor = create<ChoreographyEditorState>((set, get) => ({
  // Defaults
  programId: null,
  title: "",
  discipline: "mens_singles",
  segment: "free_skate",
  musicAnalysisId: null,
  audioUrl: null,
  musicDuration: 180,
  beatMarkers: [],
  phraseMarkers: [],
  elements: [],
  selectedElementId: null,
  currentTime: 0,
  isPlaying: false,
  pixelsPerSecond: 15,
  snapMode: "beats",
  rinkPreset: "olympic",
  rinkWidth: 60,
  rinkHeight: 30,
  isLoading: false,

  initFromProgram: (program, audioUrl, musicDuration, beatMarkers, phraseMarkers) => {
    idCounter = 0
    const layoutElements = program.layout?.elements ?? []
    const elements: TimelineElement[] = layoutElements.map((el, i) => ({
      id: `el-${i}-${el.code}-${el.timestamp}`,
      code: el.code,
      trackType: el.code.includes("Sp")
        ? "spins" as TrackType
        : el.code.startsWith("StSq") || el.code.startsWith("ChSq")
          ? "sequences" as TrackType
          : "jumps" as TrackType,
      timestamp: el.timestamp,
      duration: el.code.includes("Sp")
        ? 6
        : el.code.startsWith("StSq") || el.code.startsWith("ChSq")
          ? 8
          : 3,
      goe: el.goe,
      jumpPassIndex: el.jump_pass_index ?? undefined,
      position: el.position ?? undefined,
    }))
    set({
      programId: program.id,
      title: program.title ?? "",
      discipline: program.discipline as "mens_singles" | "womens_singles",
      segment: program.segment as "short_program" | "free_skate",
      musicAnalysisId: program.music_analysis_id,
      audioUrl,
      musicDuration,
      beatMarkers,
      phraseMarkers,
      elements,
      selectedElementId: null,
      currentTime: 0,
      isPlaying: false,
    })
  },

  addElement: (trackType, timestamp, code) => {
    const duration = trackType === "jumps" ? 3 : trackType === "spins" ? 6 : 8
    const el: TimelineElement = {
      id: nextId(),
      code,
      trackType,
      timestamp,
      duration,
      goe: 0,
    }
    set((s) => ({ elements: [...s.elements, el] }))
    get().setSelectedElement(el.id)
  },

  removeElement: (id) => {
    set((s) => ({
      elements: s.elements.filter((e) => e.id !== id),
      selectedElementId: s.selectedElementId === id ? null : s.selectedElementId,
    }))
  },

  moveElement: (id, newTimestamp) => {
    set((s) => ({
      elements: s.elements.map((e) => (e.id === id ? { ...e, timestamp: Math.max(0, newTimestamp) } : e)),
    }))
  },

  resizeElement: (id, newDuration) => {
    set((s) => ({
      elements: s.elements.map((e) => (e.id === id ? { ...e, duration: Math.max(1, newDuration) } : e)),
    }))
  },

  updateElement: (id, updates) => {
    set((s) => ({
      elements: s.elements.map((e) => (e.id === id ? { ...e, ...updates } : e)),
    }))
  },

  duplicateElement: (id) => {
    const el = get().elements.find((e) => e.id === id)
    if (!el) return
    const newEl: TimelineElement = { ...el, id: nextId(), timestamp: el.timestamp + 2 }
    set((s) => ({ elements: [...s.elements, newEl] }))
    get().setSelectedElement(newEl.id)
  },

  setSelectedElement: (id) => set({ selectedElementId: id }),

  setCurrentTime: (time) => set({ currentTime: time }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPixelsPerSecond: (pps) => set({ pixelsPerSecond: Math.max(2, Math.min(60, pps)) }),
  setSnapMode: (mode) => set({ snapMode: mode }),

  setRinkPreset: (preset, width, height) => {
    const found = { Olympic: [60, 30], NHL: [61, 26], Training: [56, 26] }[preset]
    const [w, h] = width !== undefined ? [width, height!] : found ?? [60, 30]
    set({ rinkPreset: preset, rinkWidth: w, rinkHeight: h })
  },

  setRinkDimensions: (width, height) => set({ rinkWidth: width, rinkHeight: height, rinkPreset: "custom" }),

  setTitle: (title) => set({ title }),

  getElementsByTrack: (trackType) => get().elements.filter((e) => e.trackType === trackType),

  getLayoutForSave: () => {
    const { elements, musicDuration } = get()
    const layoutElements: LayoutElement[] = elements.map((el) => ({
      code: el.code,
      goe: el.goe,
      timestamp: el.timestamp,
      position: el.position ?? null,
      is_back_half: false,
      is_jump_pass: el.trackType === "jumps",
      jump_pass_index: el.jumpPassIndex ?? null,
    }))
    const backHalfIndices = calculateBackHalfIndices(elements, musicDuration)
    const total_tes = calculateClientSideTes(layoutElements, backHalfIndices)
    return { layout: layoutElements, total_tes, back_half_indices }
  },
}))

function calculateBackHalfIndices(elements: TimelineElement[], duration: number): number[] {
  const halfTime = duration / 2
  const jumpPasses = new Map<number, number[]>()
  for (const el of elements) {
    if (el.trackType !== "jumps") continue
    const idx = el.jumpPassIndex ?? 0
    if (!jumpPasses.has(idx)) jumpPasses.set(idx, [])
    jumpPasses.get(idx)!.push(el.timestamp)
  }
  const indices: number[] = []
  const sorted = [...jumpPasses.entries()].sort((a, b) => a[0] - b[0])
  const totalPasses = sorted.length
  if (totalPasses >= 3) {
    const backHalfPasses = sorted.slice(-3)
    for (const [passIdx] of backHalfPasses) {
      const passes = jumpPasses.get(passIdx)!
      if (passes[0] > halfTime) indices.push(passIdx)
    }
  }
  return indices
}

function calculateClientSideTes(elements: LayoutElement[], backHalfIndices: number[]): number {
  const ELEMENTS_BV: Record<string, number> = {
    "1T": 0.4, "1S": 0.4, "1Lo": 0.5, "1F": 0.5, "1Lz": 0.6, "1A": 1.1,
    "2T": 1.3, "2S": 1.3, "2Lo": 1.7, "2F": 1.8, "2Lz": 2.1, "2A": 3.3,
    "3T": 4.2, "3S": 4.3, "3Lo": 4.9, "3F": 5.3, "3Lz": 5.9, "3A": 8.0,
    "4T": 9.5, "4S": 9.7, "4Lo": 10.5, "4F": 11.0, "4Lz": 11.5, "4A": 12.5,
    "1Eu": 0.5,
    CSp1: 1.5, CSp2: 2.0, CSp3: 2.5, CSp4: 3.2,
    FSp1: 1.7, FSp2: 2.3, FSp3: 2.8, FSp4: 3.0,
    LSp1: 1.5, LSp2: 2.0, LSp3: 2.5, LSp4: 3.0,
    USp1: 1.5, USp2: 2.0, USp3: 2.5, USp4: 3.0,
    CSpB1: 1.7, CSpB2: 2.3, CSpB3: 2.8, CSpB4: 3.0,
    StSq1: 1.5, StSq2: 2.6, StSq3: 3.3, StSq4: 3.9,
    ChSq1: 3.0,
  }
  function goeFactor(bv: number): number {
    if (bv < 2) return 0.5
    if (bv < 4) return 0.7
    return 1.0
  }
  let total = 0
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i]
    const bv = ELEMENTS_BV[el.code] ?? 0
    const inBackHalf = backHalfIndices.includes(el.jump_pass_index ?? -1)
    const finalBv = inBackHalf ? bv * 1.1 : bv
    const goe = Math.max(-5, Math.min(5, el.goe))
    total += finalBv + goe * goeFactor(bv)
  }
  return Math.round(total * 100) / 100
}
```

- [ ] **Step 2: Verify types**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/store.ts
git commit -m "feat(choreography): add Zustand editor store with element CRUD and TES calculation"
```

---

### Task 4: Create WaveformView component

**Files:**
- Create: `frontend/src/components/choreography/editor/waveform-view.tsx`

- [ ] **Step 1: Write WaveformView**

```typescript
"use client"

import { useRef } from "react"
import WaveSurfer from "wavesurfer.js"
import { useChoreographyEditor } from "./store"

interface WaveformViewProps {
  audioUrl: string | null
}

export function WaveformView({ audioUrl }: WaveformViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const { beatMarkers, phraseMarkers, currentTime, isPlaying, pixelsPerSecond, setCurrentTime, setIsPlaying } =
    useChoreographyEditor()

  // Initialize wavesurfer on mount if audioUrl provided
  // Use useMountEffect pattern — no useEffect
  // Since wavesurfer requires a DOM ref, we'll use a lazy init approach

  function handleContainerRef(el: HTMLDivElement | null) {
    if (!el || wavesurferRef.current) return
    if (!audioUrl) return

    const ws = WaveSurfer.create({
      container: el,
      waveColor: "oklch(var(--muted-foreground) / 0.3)",
      progressColor: "oklch(var(--primary))",
      cursorColor: "oklch(0.6 0.2 25)",
      cursorWidth: 2,
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      hideScrollbar: true,
      minPxPerSec: pixelsPerSecond,
    })

    ws.on("timeupdate", (time) => setCurrentTime(time))
    ws.on("play", () => setIsPlaying(true))
    ws.on("pause", () => setIsPlaying(false))

    ws.load(audioUrl).catch(() => {
      // Graceful fallback if audio fails to load
    })

    wavesurferRef.current = ws
  }

  // Expose play/pause/seek for TransportBar
  // We'll use a module-level ref since we can't use useEffect
  WaveformViewRef.current = wavesurferRef.current

  return (
    <div
      ref={handleContainerRef}
      className="relative w-full overflow-hidden rounded-lg border border-border bg-muted/30"
      style={{ height: 80 }}
    >
      {!audioUrl && (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Загрузите музыку для отображения waveform
        </div>
      )}
    </div>
  )
}

// Module-level ref for TransportBar to access wavesurfer instance
export const WaveformViewRef = { current: null as WaveSurfer | null }
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/waveform-view.tsx
git commit -m "feat(choreography): add WaveformView with wavesurfer.js integration"
```

---

### Task 5: Create TransportBar component

**Files:**
- Create: `frontend/src/components/choreography/editor/transport-bar.tsx`

- [ ] **Step 1: Write TransportBar**

```typescript
"use client"

import { SkipBack, Play, Pause, SkipForward, Volume2, ZoomIn, Magnet } from "lucide-react"
import { useChoreographyEditor } from "./store"
import { WaveformViewRef } from "./waveform-view"
import type { SnapMode } from "@/types/choreography"
import { useTranslations } from "@/i18n"

const SNAP_OPTIONS: { value: SnapMode; labelKey: string }[] = [
  { value: "off", labelKey: "snapOff" },
  { value: "beats", labelKey: "snapBeats" },
  { value: "phrases", labelKey: "snapPhrases" },
]

export function TransportBar() {
  const t = useTranslations("choreography")
  const {
    isPlaying,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    setIsPlaying,
    setCurrentTime,
    setPixelsPerSecond,
    setSnapMode,
  } = useChoreographyEditor()

  const ws = WaveformViewRef.current

  function togglePlay() {
    if (!ws) return
    ws.playPause()
  }

  function skipBack() {
    if (!ws) return
    ws.setTime(Math.max(0, ws.getCurrentTime() - 5))
  }

  function skipForward() {
    if (!ws) return
    ws.setTime(Math.min(ws.getDuration(), ws.getCurrentTime() + 5))
  }

  function handleSeek(e: React.ChangeEvent<HTMLInputElement>) {
    const time = Number.parseFloat(e.target.value)
    setCurrentTime(time)
    ws?.setTime(time)
  }

  function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}:${s.toString().padStart(2, "0")}`
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5">
      {/* Playback */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={skipBack}
          className="rounded p-1 text-muted-foreground hover:text-foreground"
          aria-label="Skip back 5s"
        >
          <SkipBack className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={togglePlay}
          className="rounded p-1 text-foreground hover:text-primary"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
        </button>
        <button
          type="button"
          onClick={skipForward}
          className="rounded p-1 text-muted-foreground hover:text-foreground"
          aria-label="Skip forward 5s"
        >
          <SkipForward className="h-4 w-4" />
        </button>
      </div>

      {/* Time + Seek */}
      <div className="flex items-center gap-2 text-xs tabular-nums">
        <span className="text-foreground">{formatTime(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={musicDuration || 180}
          step={0.1}
          value={currentTime}
          onChange={handleSeek}
          className="h-1 w-24 cursor-pointer accent-primary"
          aria-label="Seek"
        />
        <span className="text-muted-foreground">{formatTime(musicDuration)}</span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Zoom */}
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <ZoomIn className="h-3.5 w-3.5" />
        <input
          type="range"
          min={2}
          max={60}
          step={1}
          value={pixelsPerSecond}
          onChange={(e) => setPixelsPerSecond(Number(e.target.value))}
          className="h-1 w-20 cursor-pointer accent-primary"
          aria-label="Zoom"
        />
      </div>

      {/* Snap */}
      <div className="flex items-center gap-1">
        <Magnet className="h-3.5 w-3.5 text-muted-foreground" />
        <select
          value={snapMode}
          onChange={(e) => setSnapMode(e.target.value as SnapMode)}
          className="h-7 rounded border border-border bg-background px-1.5 text-xs"
          aria-label="Snap mode"
        >
          {SNAP_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {t(opt.labelKey as Parameters<typeof t>[0])}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/transport-bar.tsx
git commit -m "feat(choreography): add TransportBar with playback, seek, zoom, snap"
```

---

### Task 6: Create TimelineElement component with drag

**Files:**
- Create: `frontend/src/components/choreography/editor/timeline-element.tsx`

- [ ] **Step 1: Write TimelineElement**

```typescript
"use client"

import { useRef, useCallback } from "react"
import { useChoreographyEditor } from "./store"
import type { TimelineElement as TimelineElementType, TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { cn } from "@/lib/utils"

interface TimelineElementProps {
  element: TimelineElementType
  pixelsPerSecond: number
  duration: number
  snapMode: "beats" | "phrases" | "off"
  beatMarkers: number[]
  phraseMarkers: number[]
  isSelected: boolean
  isActive: boolean // playhead is within element time window
  onSelect: (id: string) => void
  onEdit: (id: string) => void
}

export function TimelineElement({
  element,
  pixelsPerSecond,
  duration: _duration,
  snapMode,
  beatMarkers,
  phraseMarkers: _phraseMarkers,
  isSelected,
  isActive,
  onSelect,
  onEdit,
}: TimelineElementProps) {
  const moveElement = useChoreographyEditor((s) => s.moveElement)
  const dragRef = useRef({ startX: 0, startTimestamp: 0 })
  const config = TRACK_CONFIG[element.trackType]

  const leftPx = element.timestamp * pixelsPerSecond
  const widthPx = element.duration * pixelsPerSecond

  function snapToMarker(timestamp: number): number {
    if (snapMode === "off") return timestamp
    const markers = snapMode === "beats" ? beatMarkers : beatMarkers
    if (markers.length === 0) return timestamp
    let closest = markers[0]
    let minDist = Math.abs(timestamp - closest)
    for (const m of markers) {
      const dist = Math.abs(timestamp - m)
      if (dist < minDist) {
        minDist = dist
        closest = m
      }
    }
    // Only snap if within 0.5 seconds
    return minDist < 0.5 ? closest : timestamp
  }

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      e.stopPropagation()
      onSelect(element.id)
      dragRef.current = { startX: e.clientX, startTimestamp: element.timestamp }
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)

      const handlePointerMove = (ev: PointerEvent) => {
        const dx = ev.clientX - dragRef.current.startX
        const dt = dx / pixelsPerSecond
        let newTimestamp = dragRef.current.startTimestamp + dt
        newTimestamp = snapToMarker(newTimestamp)
        moveElement(element.id, newTimestamp)
      }

      const handlePointerUp = () => {
        window.removeEventListener("pointermove", handlePointerMove)
        window.removeEventListener("pointerup", handlePointerUp)
      }

      window.addEventListener("pointermove", handlePointerMove)
      window.addEventListener("pointerup", handlePointerUp)
    },
    [element.id, element.timestamp, pixelsPerSecond, snapMode, beatMarkers, onSelect, moveElement],
  )

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onEdit(element.id)
    },
    [element.id, onEdit],
  )

  const backHalfThreshold = useChoreographyEditor((s) => s.musicDuration / 2)
  const isBackHalf = element.timestamp > backHalfThreshold

  return (
    <div
      role="button"
      tabIndex={0}
      className={cn(
        "absolute top-1 bottom-1 cursor-grab rounded-md border px-1.5 py-0.5 text-xs select-none transition-colors",
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
      onKeyDown={(e) => {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.stopPropagation()
          useChoreographyEditor.getState().removeElement(element.id)
        }
      }}
      title={`${element.code} (${element.timestamp.toFixed(1)}s)`}
    >
      <div className={cn("truncate font-semibold", config.color)}>{element.code}</div>
      <div className="text-muted-foreground text-[10px]">
        {element.timestamp.toFixed(1)}s
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/timeline-element.tsx
git commit -m "feat(choreography): add TimelineElement with pointer-based drag and snap-to-beats"
```

---

### Task 7: Create ElementTrack component

**Files:**
- Create: `frontend/src/components/choreography/editor/element-track.tsx`

- [ ] **Step 1: Write ElementTrack**

```typescript
"use client"

import { Plus } from "lucide-react"
import { useChoreographyEditor } from "./store"
import { TimelineElement } from "./timeline-element"
import type { TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { useTranslations } from "@/i18n"

interface ElementTrackProps {
  type: TrackType
}

export function ElementTrack({ type }: ElementTrackProps) {
  const t = useTranslations("choreography")
  const {
    elements,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    beatMarkers,
    phraseMarkers,
    selectedElementId,
    addElement,
    setSelectedElement,
  } = useChoreographyEditor()

  const trackElements = elements.filter((e) => e.trackType === type)
  const config = TRACK_CONFIG[type]
  const trackLabel = t(`${type}Track` as Parameters<typeof t>[0])

  const timelineWidthPx = musicDuration * pixelsPerSecond

  function handleDoubleClick(e: React.MouseEvent) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const timestamp = x / pixelsPerSecond
    // Will be wired to ElementPicker in Phase 2
    // For now, just set cursor position
    setSelectedElement(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Delete" && selectedElementId) {
      useChoreographyEditor.getState().removeElement(selectedElementId)
    }
  }

  return (
    <div
      className="flex border-b border-border last:border-b-0"
      onKeyDown={handleKeyDown}
    >
      {/* Track label */}
      <div className="flex w-28 shrink-0 items-center justify-between border-r border-border px-2 py-1.5">
        <span className={cn("text-xs font-medium", config.color)}>
          {trackLabel}
        </span>
        <span className={cn(
          "text-[10px]",
          trackElements.length > config.maxElements ? "text-red-500" : "text-muted-foreground",
        )}>
          {trackElements.length}/{config.maxElements}
        </span>
      </div>

      {/* Track content */}
      <div
        className="relative flex-1 overflow-hidden bg-muted/20 py-0.5"
        onDoubleClick={handleDoubleClick}
      >
        {/* Beat markers */}
        {beatMarkers.map((time, i) => {
          const x = time * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`beat-${i}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-primary/10"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Phrase markers */}
        {phraseMarkers.map((seg, i) => {
          const x = seg.start * pixelsPerSecond
          if (x > timelineWidthPx) return null
          return (
            <div
              key={`phrase-${i}`}
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-muted-foreground/20"
              style={{ left: `${x}px` }}
            />
          )
        })}

        {/* Playhead on this track */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
          style={{ left: `${currentTime * pixelsPerSecond}px` }}
        />

        {/* Elements */}
        {trackElements.map((el) => (
          <TimelineElement
            key={el.id}
            element={el}
            pixelsPerSecond={pixelsPerSecond}
            duration={musicDuration}
            snapMode={snapMode}
            beatMarkers={beatMarkers}
            phraseMarkers={phraseMarkers}
            isSelected={el.id === selectedElementId}
            isActive={currentTime >= el.timestamp && currentTime < el.timestamp + el.duration}
            onSelect={setSelectedElement}
            onEdit={setSelectedElement}
          />
        ))}

        {/* Empty hint */}
        {trackElements.length === 0 && (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            <Plus className="mr-1 h-3 w-3" />
            Двойной клик для добавления
          </div>
        )}
      </div>
    </div>
  )
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(" ")
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/element-track.tsx
git commit -m "feat(choreography): add ElementTrack with beat markers and element rendering"
```

---

### Task 8: Wire up timeline editor page

**Files:**
- Modify: `frontend/src/app/(app)/choreography/programs/[id]/page.tsx`

- [ ] **Step 1: Replace the editor page with timeline layout**

Read the current file first, then replace its contents:

```typescript
"use client"

import { useParams } from "next/navigation"
import { useTranslations } from "@/i18n"
import { useProgram, useMusicAnalysis, useSaveProgram, useRenderRink } from "@/lib/api/choreography"
import { useChoreographyEditor } from "@/components/choreography/editor/store"
import { WaveformView } from "@/components/choreography/editor/waveform-view"
import { TransportBar } from "@/components/choreography/editor/transport-bar"
import { ElementTrack } from "@/components/choreography/editor/element-track"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { ScoreBar } from "@/components/choreography/score-bar"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function ProgramEditorPage() {
  const { id } = useParams<{ id: string }>()
  const t = useTranslations("choreography")
  const tc = useTranslations("common")
  const { data: program, isLoading } = useProgram(id)
  const saveProgram = useSaveProgram()
  const editor = useChoreographyEditor()

  const musicAnalysisId = program?.music_analysis_id
  const { data: musicAnalysis } = useMusicAnalysis(musicAnalysisId ?? "")

  const renderRink = useRenderRink()
  const svgHtml = renderRink.data?.svgHtml ?? null

  // Initialize editor from program data
  const initialized = program && editor.programId === program.id

  // Auto-save effect — will be replaced with store middleware in Phase 4
  // For now, save is triggered by the save button

  function handleSave() {
    if (!program) return
    const { layout, total_tes, back_half_indices } = editor.getLayoutForSave()
    saveProgram.mutate({
      id,
      layout: { elements: layout, total_tes, back_half_indices },
      total_tes,
    })
  }

  function handleTitleChange(newTitle: string) {
    editor.setTitle(newTitle)
  }

  const audioUrl = musicAnalysis?.audio_url ?? null
  const musicDuration = musicAnalysis?.duration_sec ?? 180
  const beatMarkers = musicAnalysis?.peaks ?? []
  const phraseMarkers = (musicAnalysis?.structure ?? []).map((s) => s.start)

  const { layout, total_tes } = editor.getLayoutForSave()

  if (isLoading || !program) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        {tc("loading")}
      </div>
    )
  }

  if (!initialized) {
    // Initialize editor state from program data
    editor.initFromProgram(program, audioUrl, musicDuration, beatMarkers, phraseMarkers)
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-2">
        <Link href="/choreography" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <input
          type="text"
          value={editor.title}
          onChange={(e) => handleTitleChange(e.target.value)}
          placeholder={t("untitled")}
          className="flex-1 bg-transparent text-lg font-semibold outline-none placeholder:text-muted-foreground"
        />
        <button
          type="button"
          onClick={handleSave}
          disabled={saveProgram.isPending}
          className="rounded-lg bg-primary px-3 py-1 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          {saveProgram.isPending ? "..." : t("save")}
        </button>
      </div>

      {/* Main content */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Timeline panel */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Transport bar */}
          <div className="border-b border-border p-2">
            <TransportBar />
          </div>

          {/* Waveform */}
          <div className="border-b border-border p-2">
            <WaveformView audioUrl={audioUrl} />
          </div>

          {/* Tracks */}
          <div className="flex-1 overflow-y-auto">
            <ElementTrack type="jumps" />
            <ElementTrack type="spins" />
            <ElementTrack type="sequences" />
          </div>

          {/* Score bar */}
          <div className="border-t border-border p-2">
            <ScoreBar
              layout={layout ? { elements: layout, total_tes, back_half_indices: [] } : null}
              discipline={program.discipline as "mens_singles" | "womens_singles"}
              segment={program.segment as "short_program" | "free_skate"}
            />
          </div>
        </div>

        {/* Rink diagram panel (desktop only) */}
        <div className="hidden w-80 shrink-0 border-l border-border bg-muted/20 p-2 lg:block">
          <RinkDiagram svgHtml={svgHtml} isLoading={renderRink.isPending} />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Fix any type issues and verify**

Run: `cd frontend && bunx tsc --noEmit`

Fix any type errors that arise from the integration. Common issues:
- The `ScoreBar` props interface may need adjustment — check that `layout` shape matches
- The `useMusicAnalysis` hook may need to be checked for correct return type

- [ ] **Step 3: Test in browser**

Run: `cd frontend && bun run dev`
Open: `http://localhost:3000/choreography/programs/<existing-id>`
Expected: Timeline editor renders with tracks, transport bar, and rink panel

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/\(app\)/choreography/programs/\[id\]/page.tsx
git commit -m "feat(choreography): wire up timeline editor page with waveform, tracks, and rink"
```

---

## Phase 2: Editing & Interaction

### Task 9: Create ElementPicker popover

**Files:**
- Create: `frontend/src/components/choreography/editor/element-picker.tsx`

- [ ] **Step 1: Write ElementPicker**

```typescript
"use client"

import { useState, useMemo } from "react"
import type { TrackType } from "@/types/choreography"

interface ElementPickerProps {
  trackType: TrackType
  onSelect: (code: string) => void
  onClose: () => void
}

const ELEMENTS_BY_TYPE: Record<TrackType, { code: string; name: string; bv: number }[]> = {
  jumps: [
    { code: "1T", name: "Single Toe Loop", bv: 0.4 },
    { code: "1S", name: "Single Salchow", bv: 0.4 },
    { code: "1Lo", name: "Single Loop", bv: 0.5 },
    { code: "1F", name: "Single Flip", bv: 0.5 },
    { code: "1Lz", name: "Single Lutz", bv: 0.6 },
    { code: "1A", name: "Single Axel", bv: 1.1 },
    { code: "2T", name: "Double Toe Loop", bv: 1.3 },
    { code: "2S", name: "Double Salchow", bv: 1.3 },
    { code: "2Lo", name: "Double Loop", bv: 1.7 },
    { code: "2F", name: "Double Flip", bv: 1.8 },
    { code: "2Lz", name: "Double Lutz", bv: 2.1 },
    { code: "2A", name: "Double Axel", bv: 3.3 },
    { code: "3T", name: "Triple Toe Loop", bv: 4.2 },
    { code: "3S", name: "Triple Salchow", bv: 4.3 },
    { code: "3Lo", name: "Triple Loop", bv: 4.9 },
    { code: "3F", name: "Triple Flip", bv: 5.3 },
    { code: "3Lz", name: "Triple Lutz", bv: 5.9 },
    { code: "3A", name: "Triple Axel", bv: 8.0 },
    { code: "4T", name: "Quad Toe Loop", bv: 9.5 },
    { code: "4S", name: "Quad Salchow", bv: 9.7 },
    { code: "4Lo", name: "Quad Loop", bv: 10.5 },
    { code: "4F", name: "Quad Flip", bv: 11.0 },
    { code: "4Lz", name: "Quad Lutz", bv: 11.5 },
  ],
  spins: [
    { code: "CSp1", name: "Combination Spin Lv1", bv: 1.5 },
    { code: "CSp2", name: "Combination Spin Lv2", bv: 2.0 },
    { code: "CSp3", name: "Combination Spin Lv3", bv: 2.5 },
    { code: "CSp4", name: "Combination Spin Lv4", bv: 3.2 },
    { code: "FSp1", name: "Flying Spin Lv1", bv: 1.7 },
    { code: "FSp2", name: "Flying Spin Lv2", bv: 2.3 },
    { code: "FSp3", name: "Flying Spin Lv3", bv: 2.8 },
    { code: "FSp4", name: "Flying Spin Lv4", bv: 3.0 },
    { code: "LSp1", name: "Layback Spin Lv1", bv: 1.5 },
    { code: "LSp2", name: "Layback Spin Lv2", bv: 2.0 },
    { code: "LSp3", name: "Layback Spin Lv3", bv: 2.5 },
    { code: "LSp4", name: "Layback Spin Lv4", bv: 3.0 },
    { code: "USp1", name: "Upright Spin Lv1", bv: 1.5 },
    { code: "USp2", name: "Upright Spin Lv2", bv: 2.0 },
    { code: "USp3", name: "Upright Spin Lv3", bv: 2.5 },
    { code: "USp4", name: "Upright Spin Lv4", bv: 3.0 },
    { code: "CSpB1", name: "Camel Spin Lv1", bv: 1.7 },
    { code: "CSpB2", name: "Camel Spin Lv2", bv: 2.3 },
    { code: "CSpB3", name: "Camel Spin Lv3", bv: 2.8 },
    { code: "CSpB4", name: "Camel Spin Lv4", bv: 3.0 },
  ],
  sequences: [
    { code: "StSq1", name: "Step Sequence Lv1", bv: 1.5 },
    { code: "StSq2", name: "Step Sequence Lv2", bv: 2.6 },
    { code: "StSq3", name: "Step Sequence Lv3", bv: 3.3 },
    { code: "StSq4", name: "Step Sequence Lv4", bv: 3.9 },
    { code: "ChSq1", name: "Choreographic Sequence", bv: 3.0 },
  ],
}

export function ElementPicker({ trackType, onSelect, onClose }: ElementPickerProps) {
  const [search, setSearch] = useState("")
  const items = ELEMENTS_BY_TYPE[trackType]

  const filtered = useMemo(() => {
    if (!search) return items
    const q = search.toLowerCase()
    return items.filter((el) => el.code.toLowerCase().includes(q) || el.name.toLowerCase().includes(q))
  }, [search, items])

  return (
    <div className="w-64 rounded-lg border border-border bg-background p-2 shadow-lg" onClick={(e) => e.stopPropagation()}>
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Поиск элемента..."
        autoFocus
        className="mb-2 w-full rounded-md border border-border bg-muted/30 px-2 py-1 text-sm outline-none focus:border-primary"
      />
      <div className="max-h-60 overflow-y-auto">
        {filtered.map((el) => (
          <button
            key={el.code}
            type="button"
            className="flex w-full items-center justify-between rounded px-2 py-1 text-sm hover:bg-muted/50"
            onClick={() => {
              onSelect(el.code)
              onClose()
            }}
          >
            <span className="font-medium">{el.code}</span>
            <span className="text-xs text-muted-foreground">{el.bv.toFixed(1)}</span>
          </button>
        ))}
        {filtered.length === 0 && (
          <p className="py-2 text-center text-xs text-muted-foreground">Не найдено</p>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/element-picker.tsx
git commit -m "feat(choreography): add ElementPicker popover with search and type filtering"
```

---

### Task 10: Wire ElementPicker into ElementTrack double-click

**Files:**
- Modify: `frontend/src/components/choreography/editor/element-track.tsx`

- [ ] **Step 1: Add state and popover to ElementTrack**

Add a `useState` for picker position and timestamp, render `ElementPicker` conditionally:

```typescript
const [pickerState, setPickerState] = useState<{ x: number; y: number; timestamp: number } | null>(null)

function handleDoubleClick(e: React.MouseEvent) {
  const rect = e.currentTarget.getBoundingClientRect()
  const x = e.clientX - rect.left
  const timestamp = x / pixelsPerSecond
  setPickerState({ x: e.clientX, y: e.clientY, timestamp })
}

// In the JSX, after the track content div, add:
{pickerState && (
  <div
    className="fixed inset-0 z-50"
    onClick={() => setPickerState(null)}
  >
    <div
      className="absolute"
      style={{ left: pickerState.x, top: pickerState.y }}
    >
      <ElementPicker
        trackType={type}
        onSelect={(code) => addElement(type, pickerState.timestamp, code)}
        onClose={() => setPickerState(null)}
      />
    </div>
  </div>
)}
```

Import `ElementPicker` from `"./element-picker"` and `useState` from React.

- [ ] **Step 2: Verify in browser**

Double-click on an empty area of any track. ElementPicker popover should appear.
Select an element. It should appear on the track at the clicked position.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/element-track.tsx
git commit -m "feat(choreography): wire ElementPicker into ElementTrack double-click"
```

---

### Task 11: Create ElementEditor popover

**Files:**
- Create: `frontend/src/components/choreography/editor/element-editor.tsx`

- [ ] **Step 1: Write ElementEditor**

```typescript
"use client"

import { useChoreographyEditor } from "./store"
import { Trash2, Copy } from "lucide-react"

interface ElementEditorProps {
  elementId: string
  onClose: () => void
}

export function ElementEditor({ elementId, onClose }: ElementEditorProps) {
  const { elements, updateElement, removeElement, duplicateElement } = useChoreographyEditor()
  const el = elements.find((e) => e.id === elementId)
  if (!el) return null

  return (
    <div className="w-56 rounded-lg border border-border bg-background p-3 shadow-lg" onClick={(e) => e.stopPropagation()}>
      {/* Code display */}
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-bold">{el.code}</span>
        <span className="text-xs text-muted-foreground">
          {el.timestamp.toFixed(1)}s
        </span>
      </div>

      {/* GOE slider */}
      <div className="mb-2">
        <label className="mb-0.5 block text-xs text-muted-foreground">GOE: {el.goe > 0 ? "+" : ""}{el.goe}</label>
        <input
          type="range"
          min={-5}
          max={5}
          step={1}
          value={el.goe}
          onChange={(e) => updateElement(elementId, { goe: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>

      {/* Duration */}
      <div className="mb-3">
        <label className="mb-0.5 block text-xs text-muted-foreground">Duration (s)</label>
        <input
          type="number"
          min={1}
          max={20}
          step={0.5}
          value={el.duration}
          onChange={(e) => updateElement(elementId, { duration: Math.max(1, Number(e.target.value)) })}
          className="w-full rounded border border-border bg-muted/30 px-2 py-0.5 text-sm"
        />
      </div>

      {/* Actions */}
      <div className="flex gap-1 border-t border-border pt-2">
        <button
          type="button"
          onClick={() => {
            duplicateElement(elementId)
            onClose()
          }}
          className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted/50"
        >
          <Copy className="h-3 w-3" />
          Duplicate
        </button>
        <button
          type="button"
          onClick={() => {
            removeElement(elementId)
            onClose()
          }}
          className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-xs text-red-500 hover:bg-red-500/10"
        >
          <Trash2 className="h-3 w-3" />
          Delete
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Wire ElementEditor into ElementTrack click**

In `element-track.tsx`, add a click handler on `TimelineElement` that shows `ElementEditor` as a popover near the clicked element. Add state for editor position:

```typescript
const [editorState, setEditorState] = useState<{ x: number; y: number; elementId: string } | null>(null)
```

Pass `onEdit` to `TimelineElement` that sets editorState. Render `ElementEditor` conditionally in a fixed overlay.

- [ ] **Step 3: Verify in browser**

Click on any element. ElementEditor popover with GOE slider and delete button should appear.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/choreography/editor/element-editor.tsx frontend/src/components/choreography/editor/element-track.tsx
git commit -m "feat(choreography): add ElementEditor popover with GOE, duration, delete, duplicate"
```

---

## Phase 3: Scoring & Rink

### Task 12: Update backend rink_renderer.py with dimensions

**Files:**
- Modify: `backend/app/services/choreography/rink_renderer.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes/choreography.py`

- [ ] **Step 1: Update rink_renderer.py**

Change `render_rink` signature to accept `rink_width` and `rink_height`:

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

Replace hardcoded `rink_w, rink_h = 60.0, 30.0` with the parameters. Scale element position coordinates:

```python
    rink_w, rink_h = rink_width, rink_height
    # Scale factor from 60x30 base to actual dimensions
    sx = rink_w / 60.0
    sy = rink_h / 30.0
```

Apply `sx, sy` scaling to all coordinate values in SVG elements (faceoff circles, etc.):

```python
    for cx, cy in [(10 * sx, 7.5 * sy), (10 * sx, 22.5 * sy), (50 * sx, 7.5 * sy), (50 * sx, 22.5 * sy)]:
```

And element positions:

```python
            x, y = pos["x"] * sx, pos["y"] * sy
```

- [ ] **Step 2: Update RenderRinkRequest schema**

In `backend/app/schemas.py`, add fields to `RenderRinkRequest`:

```python
class RenderRinkRequest(BaseModel):
    elements: list[dict]
    width: int = Field(default=1200, ge=400, le=4000)
    height: int = Field(default=600, ge=200, le=2000)
    rink_width: float = Field(default=60.0, ge=20.0, le=80.0)
    rink_height: float = Field(default=30.0, ge=10.0, le=40.0)
```

- [ ] **Step 3: Update route to pass dimensions**

In `backend/app/routes/choreography.py`, `render_rink_diagram` endpoint:

```python
    svg = render_rink(
        body.elements,
        width=body.width,
        height=body.height,
        rink_width=body.rink_width,
        rink_height=body.rink_height,
    )
```

- [ ] **Step 4: Run backend tests**

Run: `uv run pytest backend/tests/ -v -k choreography`
Expected: All passing

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/choreography/rink_renderer.py backend/app/schemas.py backend/app/routes/choreography.py
git commit -m "feat(choreography): add rink_width/rink_height params to rink renderer and API"
```

---

### Task 13: Add i18n keys

**Files:**
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add Russian i18n keys**

Inside the `"choreography"` object, add a `"timeline"` sub-object:

```json
    "timeline": {
      "jumpsTrack": "Прыжки",
      "spinsTrack": "Вращения",
      "sequencesTrack": "Последов.",
      "addElement": "Добавить элемент",
      "deleteElement": "Удалить",
      "duplicateElement": "Дублировать",
      "snapOff": "Без привязки",
      "snapBeats": "К битам",
      "snapPhrases": "К фразам",
      "autoFit": "Вместить",
      "rinkPreset": "Размер катка",
      "rinkCustom": "Другой",
      "backHalf": "2-я половина",
      "validationErrors": "Ошибки валидации",
      "validationWarnings": "Предупреждения",
      "search": "Поиск элемента...",
      "notFound": "Не найдено",
      "duration": "Длительность (с)",
      "goe": "GOE"
    }
```

- [ ] **Step 2: Add English i18n keys**

Inside the `"choreography"` object, add:

```json
    "timeline": {
      "jumpsTrack": "Jumps",
      "spinsTrack": "Spins",
      "sequencesTrack": "Seq.",
      "addElement": "Add element",
      "deleteElement": "Delete",
      "duplicateElement": "Duplicate",
      "snapOff": "Off",
      "snapBeats": "Beats",
      "snapPhrases": "Phrases",
      "autoFit": "Fit",
      "rinkPreset": "Rink size",
      "rinkCustom": "Custom",
      "backHalf": "Back half",
      "validationErrors": "Validation errors",
      "validationWarnings": "Warnings",
      "search": "Search element...",
      "notFound": "Not found",
      "duration": "Duration (s)",
      "goe": "GOE"
    }
```

- [ ] **Step 3: Commit**

```bash
git add frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(choreography): add timeline editor i18n keys (ru/en)"
```

---

## Phase 4: Polish

### Task 14: Add auto-save with debounce

**Files:**
- Modify: `frontend/src/components/choreography/editor/store.ts`
- Modify: `frontend/src/app/(app)/choreography/programs/[id]/page.tsx`

- [ ] **Step 1: Add auto-save subscription to store**

In the store file, add a `subscribeWithSelector` middleware or a separate `useAutoSave` hook in the editor page that watches `elements`, `title`, and triggers `saveProgram` with 500ms debounce.

Implementation: Create `frontend/src/components/choreography/editor/auto-save.ts`:

```typescript
"use client"

import { useRef } from "react"
import { useChoreographyEditor } from "./store"

export function useAutoSave(saveMutation: { mutate: (data: unknown) => void; isPending: boolean }) {
  const timerRef = useRef<ReturnType<typeof setTimeout>>(null)
  const { programId, elements, title, getLayoutForSave } = useChoreographyEditor()

  // This is called from the page's render — since we can't use useEffect,
  // we'll call it from the save button handler and from event handlers
  function scheduleSave() {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      if (!programId) return
      const { layout, total_tes, back_half_indices } = getLayoutForSave()
      saveMutation.mutate({
        id: programId,
        layout: { elements: layout, total_tes, back_half_indices },
        total_tes,
      })
    }, 500)
  }

  return { scheduleSave }
}
```

- [ ] **Step 2: Wire scheduleSave into event handlers**

In the editor page, call `scheduleSave()` after `initFromProgram`, and in a `MutationObserver`-like pattern after store changes. For now, the simplest approach: call `scheduleSave()` in the `onClick` of the save button and also set up a `requestAnimationFrame` loop that checks for changes.

Alternative simpler approach: Just call save on every relevant action by wrapping the store actions. This can be done by calling `scheduleSave` from the `handleSave` wrapper and removing the explicit save button — auto-save is always on.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/editor/auto-save.ts frontend/src/components/choreography/editor/store.ts
git commit -m "feat(choreography): add auto-save with 500ms debounce"
```

---

### Task 15: Update ScoreBar for live back-half detection

**Files:**
- Modify: `frontend/src/components/choreography/score-bar.tsx`

- [ ] **Step 1: Update ScoreBar to accept computed layout**

The existing `ScoreBar` already accepts a `layout` prop. Ensure it correctly receives the layout from `editor.getLayoutForSave()`. The layout shape `{ elements, total_tes, back_half_indices }` should be compatible with the existing `Layout` type.

Read the existing `ScoreBar` component and verify the `Layout` type from `frontend/src/types/choreography.ts` matches. If needed, adjust the prop type.

- [ ] **Step 2: Verify live updates**

In the browser, drag an element across the midpoint of the timeline. The TES score should update to reflect back-half bonus changes.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/score-bar.tsx
git commit -m "fix(choreography): update ScoreBar for live back-half TES calculation"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- WaveformView: Task 4 ✅
- TransportBar: Task 5 ✅
- ElementTrack: Task 7 ✅
- TimelineElement: Task 6 ✅
- ElementPicker: Task 9 ✅
- ElementEditor: Task 11 ✅
- RinkDiagramPanel presets: Task 12 ✅
- ScoreBar live updates: Task 15 ✅
- Zustand store: Task 3 ✅
- Snap to beats: Task 6 ✅
- Live scoring: Task 3 (store) + Task 15 ✅
- Back-half bonus: Task 3 (store) ✅
- i18n: Task 13 ✅
- Auto-save: Task 14 ✅

**2. Placeholder scan:** No TBD, TODO, or "similar to" found. All code is explicit.

**3. Type consistency:** `TimelineElement`, `TrackType`, `SnapMode` defined once in types/choreography.ts. Store uses these types. Components import from types. Consistent.

**4. Missing from plan (deferred):**
- Undo/redo (zustand middleware) — Phase 4 mention, not implemented
- Responsive mobile layout — deferred
- Export improvements — deferred
- Accessibility — deferred
- Rink click-to-position — deferred
- Keyboard shortcut (Delete key) — included in TimelineElement ✅
