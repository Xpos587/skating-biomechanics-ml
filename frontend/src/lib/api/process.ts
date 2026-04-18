/**
 * Process and Detect API: enqueue ML jobs, poll status.
 */

import { z } from "zod"
import { apiFetch } from "@/lib/api-client"

// ---------------------------------------------------------------------------
// Detect
// ---------------------------------------------------------------------------

export const DetectQueueResponseSchema = z.object({
  task_id: z.string(),
  video_key: z.string(),
  status: z.string(),
})

export const PersonInfoSchema = z.object({
  track_id: z.number(),
  hits: z.number(),
  bbox: z.array(z.number()),
  mid_hip: z.array(z.number()),
})

export const DetectResultSchema = z.object({
  persons: z.array(PersonInfoSchema),
  preview_image: z.string(),
  video_key: z.string(),
  auto_click: z.object({ x: z.number(), y: z.number() }).nullable().optional(),
  status: z.string(),
})

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

export const QueueProcessResponseSchema = z.object({
  task_id: z.string(),
  status: z.string(),
})

export const ProcessStatsSchema = z.object({
  total_frames: z.number(),
  valid_frames: z.number(),
  fps: z.number(),
  resolution: z.string(),
})

export const ProcessResponseSchema = z.object({
  video_path: z.string(),
  poses_path: z.string().nullable(),
  csv_path: z.string().nullable(),
  stats: ProcessStatsSchema,
  status: z.string(),
})

export const TaskStatusResponseSchema = z.object({
  task_id: z.string(),
  status: z.string(),
  progress: z.number(),
  message: z.string(),
  result: ProcessResponseSchema.nullable().optional(),
  error: z.string().nullable().optional(),
})

export type TaskStatusResponse = z.infer<typeof TaskStatusResponseSchema>
export type DetectResult = z.infer<typeof DetectResultSchema>
export type PersonInfo = z.infer<typeof PersonInfoSchema>

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export async function enqueueDetect(videoKey: string, tracking = "auto") {
  return apiFetch(`/detect?tracking=${tracking}`, DetectQueueResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_key: videoKey }),
  })
}

export async function getDetectStatus(taskId: string) {
  return apiFetch(`/detect/${taskId}/status`, TaskStatusResponseSchema)
}

export async function getDetectResult(taskId: string) {
  return apiFetch(`/detect/${taskId}/result`, DetectResultSchema)
}

export async function enqueueProcess(params: {
  video_key: string
  person_click: { x: number; y: number }
  frame_skip?: number
  layer?: number
  tracking?: string
  session_id?: string
}) {
  return apiFetch("/process/queue", QueueProcessResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
}

export async function getProcessStatus(taskId: string) {
  return apiFetch(`/process/${taskId}/status`, TaskStatusResponseSchema)
}
