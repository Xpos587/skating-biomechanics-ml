"use client"

import { create } from "zustand"
import type {
  ChoreographyProgram,
  LayoutElement,
  RinkPreset,
  SnapMode,
  TimelineElement,
  TrackType,
} from "@/types/choreography"
import { DEFAULT_DURATIONS, layoutElementToTimeline } from "@/types/choreography"

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
  editorBpm: number

  // Rink
  rinkPreset: RinkPreset
  rinkWidth: number
  rinkHeight: number

  // Loading
  isLoading: boolean

  // Actions — initialization
  initFromProgram: (
    program: ChoreographyProgram,
    audioUrl: string | null,
    musicDuration: number,
    beatMarkers: number[],
    phraseMarkers: number[],
    bpm: number,
  ) => void

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
  updateElementPosition: (id: string, x: number, y: number) => void

  // Actions — save
  setTitle: (title: string) => void

  // Computed
  getElementsByTrack: (trackType: TrackType) => TimelineElement[]
  getLayoutForSave: () => {
    layout: LayoutElement[]
    total_tes: number
    back_half_indices: number[]
  }
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
  editorBpm: 0,
  rinkPreset: "olympic",
  rinkWidth: 60,
  rinkHeight: 30,
  isLoading: false,

  initFromProgram: (program, audioUrl, musicDuration, beatMarkers, phraseMarkers, bpm) => {
    idCounter = 0
    const layoutElements = program.layout?.elements ?? []
    const elements: TimelineElement[] = layoutElements.map(layoutElementToTimeline)
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
      editorBpm: bpm,
      elements,
      selectedElementId: null,
      currentTime: 0,
      isPlaying: false,
    })
  },

  addElement: (trackType, timestamp, code) => {
    const duration = DEFAULT_DURATIONS[trackType]
    const el: TimelineElement = {
      id: nextId(),
      code,
      trackType,
      timestamp,
      duration,
      goe: 0,
    }
    set(s => ({ elements: [...s.elements, el] }))
    get().setSelectedElement(el.id)
  },

  removeElement: id => {
    set(s => ({
      elements: s.elements.filter(e => e.id !== id),
      selectedElementId: s.selectedElementId === id ? null : s.selectedElementId,
    }))
  },

  moveElement: (id, newTimestamp) => {
    set(s => ({
      elements: s.elements.map(e =>
        e.id === id ? { ...e, timestamp: Math.max(0, newTimestamp) } : e,
      ),
    }))
  },

  resizeElement: (id, newDuration) => {
    set(s => ({
      elements: s.elements.map(e =>
        e.id === id ? { ...e, duration: Math.max(1, newDuration) } : e,
      ),
    }))
  },

  updateElement: (id, updates) => {
    set(s => ({
      elements: s.elements.map(e => (e.id === id ? { ...e, ...updates } : e)),
    }))
  },

  duplicateElement: id => {
    const el = get().elements.find(e => e.id === id)
    if (!el) return
    const newEl: TimelineElement = { ...el, id: nextId(), timestamp: el.timestamp + 2 }
    set(s => ({ elements: [...s.elements, newEl] }))
    get().setSelectedElement(newEl.id)
  },

  setSelectedElement: id => set({ selectedElementId: id }),

  setCurrentTime: time => set({ currentTime: time }),
  setIsPlaying: playing => set({ isPlaying: playing }),
  setPixelsPerSecond: pps => set({ pixelsPerSecond: Math.max(2, Math.min(60, pps)) }),
  setSnapMode: mode => set({ snapMode: mode }),

  setRinkPreset: (preset, width, height) => {
    const presets: Record<string, [number, number]> = {
      olympic: [60, 30],
      nhl: [61, 26],
      training: [56, 26],
    }
    const found = presets[preset]
    const [w, h] =
      width !== undefined && height !== undefined ? [width, height] : (found ?? [60, 30])
    set({ rinkPreset: preset, rinkWidth: w, rinkHeight: h })
  },

  setRinkDimensions: (width, height) =>
    set({ rinkWidth: width, rinkHeight: height, rinkPreset: "custom" }),

  updateElementPosition: (id, x, y) =>
    set(s => ({
      elements: s.elements.map(e => (e.id === id ? { ...e, position: { x, y } } : e)),
    })),

  setTitle: title => set({ title }),

  getElementsByTrack: trackType => get().elements.filter(e => e.trackType === trackType),

  getLayoutForSave: () => {
    const { elements, musicDuration } = get()
    const layoutElements: LayoutElement[] = elements.map(el => ({
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
    return { layout: layoutElements, total_tes, back_half_indices: backHalfIndices }
  },
}))

function calculateBackHalfIndices(elements: TimelineElement[], duration: number): number[] {
  const halfTime = duration / 2
  const jumpPasses = new Map<number, number[]>()
  for (const el of elements) {
    if (el.trackType !== "jumps") continue
    const idx = el.jumpPassIndex ?? 0
    if (!jumpPasses.has(idx)) jumpPasses.set(idx, [])
    jumpPasses.get(idx)?.push(el.timestamp)
  }
  const indices: number[] = []
  const sorted = [...jumpPasses.entries()].sort((a, b) => a[0] - b[0])
  const totalPasses = sorted.length
  if (totalPasses >= 3) {
    const backHalfPasses = sorted.slice(-3)
    for (const [passIdx] of backHalfPasses) {
      const passes = jumpPasses.get(passIdx)
      if (!passes) continue
      if (passes[0] > halfTime) indices.push(passIdx)
    }
  }
  return indices
}

function calculateClientSideTes(elements: LayoutElement[], backHalfIndices: number[]): number {
  const ELEMENTS_BV: Record<string, number> = {
    "1T": 0.4,
    "1S": 0.4,
    "1Lo": 0.5,
    "1F": 0.5,
    "1Lz": 0.6,
    "1A": 1.1,
    "2T": 1.3,
    "2S": 1.3,
    "2Lo": 1.7,
    "2F": 1.8,
    "2Lz": 2.1,
    "2A": 3.3,
    "3T": 4.2,
    "3S": 4.3,
    "3Lo": 4.9,
    "3F": 5.3,
    "3Lz": 5.9,
    "3A": 8.0,
    "4T": 9.5,
    "4S": 9.7,
    "4Lo": 10.5,
    "4F": 11.0,
    "4Lz": 11.5,
    "4A": 12.5,
    "1Eu": 0.5,
    CSp1: 1.5,
    CSp2: 2.0,
    CSp3: 2.5,
    CSp4: 3.2,
    FSp1: 1.7,
    FSp2: 2.3,
    FSp3: 2.8,
    FSp4: 3.0,
    LSp1: 1.5,
    LSp2: 2.0,
    LSp3: 2.5,
    LSp4: 3.0,
    USp1: 1.5,
    USp2: 2.0,
    USp3: 2.5,
    USp4: 3.0,
    CSpB1: 1.7,
    CSpB2: 2.3,
    CSpB3: 2.8,
    CSpB4: 3.0,
    StSq1: 1.5,
    StSq2: 2.6,
    StSq3: 3.3,
    StSq4: 3.9,
    ChSq1: 3.0,
  }
  function goeFactor(bv: number): number {
    if (bv < 2) return 0.5
    if (bv < 4) return 0.7
    return 1.0
  }
  let total = 0
  for (const el of elements) {
    const bv = ELEMENTS_BV[el.code] ?? 0
    const inBackHalf = backHalfIndices.includes(el.jump_pass_index ?? -1)
    const finalBv = inBackHalf ? bv * 1.1 : bv
    const goe = Math.max(-5, Math.min(5, el.goe))
    total += finalBv + goe * goeFactor(bv)
  }
  return Math.round(total * 100) / 100
}
