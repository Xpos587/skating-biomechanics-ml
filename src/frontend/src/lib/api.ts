/**
 * Non-auth API wrappers: models, process queue, detect, SSE streaming.
 */

import { z } from "zod"
import { API_BASE, apiFetch, ApiError, getAccessToken } from "@/lib/api-client"
import type { ProcessResponse, PersonInfo } from "@/lib/schemas"
import { DetectQueueResponseSchema, DetectResultResponseSchema, ProcessRequestSchema, ProcessResponseSchema } from "@/lib/schemas"

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface ModelStatus {
  id: string
  available: boolean
  size_mb: number | null
}

const ModelStatusListSchema = z.array(
  z.object({ id: z.string(), available: z.boolean(), size_mb: z.number().nullable() }),
)

export async function getModels(): Promise<ModelStatus[]> {
  return apiFetch("/models", ModelStatusListSchema, { auth: false })
}

// ---------------------------------------------------------------------------
// Process queue
// ---------------------------------------------------------------------------

const QueueResponseSchema = z.object({ task_id: z.string() })

const TaskStatusSchema = z.object({
  task_id: z.string(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number(),
  message: z.string(),
  result: ProcessResponseSchema.nullable(),
  error: z.string().nullable(),
})

export interface TaskStatusResponse {
  task_id: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  progress: number
  message: string
  result: ProcessResponse | null
  error: string | null
}

export async function enqueueProcess(
  request: Parameters<typeof ProcessRequestSchema.parse>[0],
): Promise<{ task_id: string }> {
  const validated = ProcessRequestSchema.parse(request)
  return apiFetch("/process/queue", QueueResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  })
}

export async function pollTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch(`/process/${taskId}/status`, TaskStatusSchema)
}

export async function cancelQueuedProcess(taskId: string): Promise<void> {
  await apiFetch(
    `/process/${taskId}/cancel`,
    z.object({ status: z.string(), task_id: z.string() }),
    {
      method: "POST",
    },
  )
}

// ---------------------------------------------------------------------------
// Detect (FormData — can't use JSON apiFetch)
// ---------------------------------------------------------------------------

export async function detectEnqueue(
  file: File,
  tracking = "auto",
): Promise<{ task_id: string; video_key: string }> {
  const form = new FormData()
  form.append("video", file)
  form.append("tracking", tracking)
  const token = getAccessToken()
  const res = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    body: form,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new ApiError((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`, res.status)
  return DetectQueueResponseSchema.parse(await res.json())
}

export async function detectStatus(
  taskId: string,
): Promise<{ task_id: string; status: string; progress: number; message: string; result: unknown | null; error: string | null }> {
  const data = await apiFetch(`/detect/${taskId}/status`, TaskStatusSchema)
  return data
}

export async function detectResult(
  taskId: string,
): Promise<{ persons: PersonInfo[]; preview_image: string; video_key: string; auto_click: { x: number; y: number } | null; status: string }> {
  const data = await apiFetch(`/detect/${taskId}/result`, DetectResultResponseSchema)
  return data
}
