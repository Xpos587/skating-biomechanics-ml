export interface PersonInfo {
  track_id: number
  hits: number
  bbox: number[] // [x1, y1, x2, y2] normalized
  mid_hip: number[] // [x, y] normalized
}

export interface PersonClick {
  x: number
  y: number
}

export interface DetectResponse {
  persons: PersonInfo[]
  preview_image: string // base64 PNG
  video_key: string
  auto_click: PersonClick | null
  status: string
}

export interface ProcessRequest {
  video_key: string
  person_click: PersonClick
  frame_skip: number
  layer: number
  tracking: string
  export: boolean
}

export interface ProcessStats {
  total_frames: number
  valid_frames: number
  fps: number
  resolution: string
}

export interface ProcessResponse {
  video_path: string
  poses_path: string | null
  csv_path: string | null
  stats: ProcessStats
  status: string
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export interface SessionMetric {
  id: string
  metric_name: string
  metric_value: number
  is_pr: boolean
  prev_best: number | null
  reference_value: number | null
  is_in_range: boolean | null
}

export interface Session {
  id: string
  user_id: string
  element_type: string
  video_url: string | null
  processed_video_url: string | null
  poses_url: string | null
  csv_url: string | null
  status: string
  error_message: string | null
  phases: Record<string, number> | null
  recommendations: string[] | null
  overall_score: number | null
  created_at: string
  processed_at: string | null
  metrics: SessionMetric[]
}

export interface SessionListResponse {
  sessions: Session[]
  total: number
}

// ---------------------------------------------------------------------------
// Metrics & Progress
// ---------------------------------------------------------------------------

export interface TrendDataPoint {
  date: string
  value: number
  session_id: string
  is_pr: boolean
}

export interface TrendResponse {
  metric_name: string
  element_type: string
  data_points: TrendDataPoint[]
  trend: "improving" | "stable" | "declining"
  current_pr: number | null
  reference_range: { min: number; max: number } | null
}

export interface DiagnosticsFinding {
  severity: "warning" | "info"
  element: string
  metric: string
  message: string
  detail: string
}

export interface DiagnosticsResponse {
  user_id: string
  findings: DiagnosticsFinding[]
}

export interface MetricDef {
  name: string
  label_ru: string
  unit: string
  format: string
  direction: "higher" | "lower"
  element_types: string[]
  ideal_range: [number, number]
}

// ---------------------------------------------------------------------------
// Relationships
// ---------------------------------------------------------------------------

export interface Relationship {
  id: string
  coach_id: string
  skater_id: string
  status: "invited" | "active" | "ended"
  initiated_by: string | null
  created_at: string
  ended_at: string | null
  coach_name: string | null
  skater_name: string | null
}

export interface RelationshipListResponse {
  relationships: Relationship[]
}
