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
  video_path: string
  auto_click: PersonClick | null
  status: string
}

export interface ProcessRequest {
  video_path: string
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
