import { z } from "zod"

// Mirror Pydantic schemas from src/backend/schemas.py

export const PersonInfoSchema = z.object({
  track_id: z.number().int().positive(),
  hits: z.number().int().nonnegative(),
  bbox: z.tuple([
    z.number().min(0).max(1), // x1
    z.number().min(0).max(1), // y1
    z.number().min(0).max(1), // x2
    z.number().min(0).max(1), // y2
  ]),
  mid_hip: z.tuple([
    z.number().min(0).max(1), // x
    z.number().min(0).max(1), // y
  ]),
})

export const PersonClickSchema = z.object({
  x: z.number().int().nonnegative(),
  y: z.number().int().nonnegative(),
})

export const DetectResponseSchema = z.object({
  persons: z.array(PersonInfoSchema),
  preview_image: z.string().min(1),
  video_key: z.string().min(1),
  auto_click: PersonClickSchema.nullable(),
  status: z.string(),
})

export const DetectQueueResponseSchema = z.object({
  task_id: z.string(),
  video_key: z.string(),
  status: z.string(),
})

export const DetectResultResponseSchema = z.object({
  persons: z.array(PersonInfoSchema),
  preview_image: z.string(),
  video_key: z.string(),
  auto_click: PersonClickSchema.nullable(),
  status: z.string(),
})

export const ProcessStatsSchema = z.object({
  total_frames: z.number().int().positive(),
  valid_frames: z.number().int().nonnegative(),
  fps: z.number().positive(),
  resolution: z.string().regex(/^\d+x\d+$/),
})

export const ProcessRequestSchema = z.object({
  video_key: z.string().min(1),
  person_click: PersonClickSchema,
  frame_skip: z.number().int().positive().default(1),
  layer: z.number().int().min(0).max(3).default(3),
  tracking: z.enum(["auto", "manual"]).default("auto"),
  export: z.boolean().default(true),
  depth: z.boolean().default(false),
  optical_flow: z.boolean().default(false),
  segment: z.boolean().default(false),
  foot_track: z.boolean().default(false),
  matting: z.boolean().default(false),
  inpainting: z.boolean().default(false),
})

export const ProcessResponseSchema = z.object({
  video_path: z.string().min(1),
  poses_path: z.string().nullable(),
  csv_path: z.string().nullable(),
  stats: ProcessStatsSchema,
  status: z.string(),
})

export const TaskStatusResponseSchema = z.object({
  task_id: z.string().min(1),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(1),
  message: z.string(),
  result: ProcessResponseSchema.nullable(),
  error: z.string().nullable(),
})

// Type inference
export type PersonInfo = z.infer<typeof PersonInfoSchema>
export type PersonClick = z.infer<typeof PersonClickSchema>
export type DetectResponse = z.infer<typeof DetectResponseSchema>
export type DetectQueueResponse = z.infer<typeof DetectQueueResponseSchema>
export type DetectResultResponse = z.infer<typeof DetectResultResponseSchema>
export type ProcessStats = z.infer<typeof ProcessStatsSchema>
export type ProcessRequest = z.infer<typeof ProcessRequestSchema>
export type ProcessResponse = z.infer<typeof ProcessResponseSchema>
