// ---------------------------------------------------------------------------
// Music Analysis
// ---------------------------------------------------------------------------

export interface MusicSegment {
  type: string
  start: number
  end: number
}

export interface EnergyCurve {
  timestamps: number[]
  values: number[]
}

export interface MusicAnalysis {
  id: string
  user_id: string
  filename: string
  audio_url: string
  duration_sec: number
  bpm: number | null
  meter: string | null
  structure: MusicSegment[] | null
  energy_curve: EnergyCurve | null
  downbeats: number[] | null
  peaks: number[] | null
  status: "pending" | "analyzing" | "completed" | "failed"
  created_at: string
  updated_at: string
}

export interface UploadMusicResponse {
  music_id: string
  filename: string
}

// ---------------------------------------------------------------------------
// Layout Generation
// ---------------------------------------------------------------------------

export interface LayoutElement {
  code: string
  goe: number
  timestamp: number
  position: { x: number; y: number } | null
  is_back_half: boolean
  is_jump_pass: boolean
  jump_pass_index: number | null
}

export interface Layout {
  elements: LayoutElement[]
  total_tes: number
  back_half_indices: number[]
}

export interface GenerateResponse {
  layouts: Layout[]
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  total_tes: number | null
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

export interface ProgramLayout {
  elements: LayoutElement[]
}

export interface ChoreographyProgram {
  id: string
  user_id: string
  music_analysis_id: string | null
  title: string | null
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"
  season: string
  layout: ProgramLayout | null
  total_tes: number | null
  estimated_goe: number | null
  estimated_pcs: number | null
  estimated_total: number | null
  is_valid: boolean | null
  validation_errors: string[] | null
  validation_warnings: string[] | null
  created_at: string
  updated_at: string
}

export interface ProgramListResponse {
  programs: ChoreographyProgram[]
  total: number
}

// ---------------------------------------------------------------------------
// Element Inventory
// ---------------------------------------------------------------------------

export interface Inventory {
  jumps: string[]
  spins: string[]
  combinations: string[]
}

// ---------------------------------------------------------------------------
// Timeline Editor
// ---------------------------------------------------------------------------

export type TrackType = "jumps" | "spins" | "sequences"
export type SnapMode = "beats" | "phrases" | "off"
export type RinkPreset = "olympic" | "nhl" | "training" | "custom"

export const TRACK_CONFIG: Record<
  TrackType,
  { maxElements: number; color: string; colorVar: string }
> = {
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
