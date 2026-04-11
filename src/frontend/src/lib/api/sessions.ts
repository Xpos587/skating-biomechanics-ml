// src/frontend/src/lib/api/sessions.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiDelete, apiFetch, apiPatch, apiPost } from "@/lib/api-client"

const SessionMetricSchema = z.object({
  id: z.string(), metric_name: z.string(), metric_value: z.number(),
  is_pr: z.boolean(), prev_best: z.number().nullable(), reference_value: z.number().nullable(),
  is_in_range: z.boolean().nullable(),
})

const SessionSchema = z.object({
  id: z.string(), user_id: z.string(), element_type: z.string(),
  video_url: z.string().nullable(), processed_video_url: z.string().nullable(),
  status: z.string(), error_message: z.string().nullable(),
  phases: z.record(z.number()).nullable(),
  recommendations: z.array(z.string()).nullable(),
  overall_score: z.number().nullable(),
  created_at: z.string(), processed_at: z.string().nullable(),
  metrics: z.array(SessionMetricSchema),
})

const SessionListSchema = z.object({ sessions: z.array(SessionSchema), total: z.number() })

export function useSessions(userId?: string, elementType?: string) {
  const params = new URLSearchParams()
  if (userId) params.set("user_id", userId)
  if (elementType) params.set("element_type", elementType)
  return useQuery({
    queryKey: ["sessions", userId, elementType],
    queryFn: () => apiFetch("/sessions?" + params.toString(), SessionListSchema),
  })
}

export function useSession(id: string) {
  return useQuery({
    queryKey: ["session", id],
    queryFn: () => apiFetch(`/sessions/${id}`, SessionSchema),
    enabled: !!id,
  })
}

export function useCreateSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { element_type: string }) =>
      apiPost("/sessions", SessionSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}

export function usePatchSession(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { element_type?: string }) =>
      apiPatch(`/sessions/${id}`, SessionSchema, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["session", id] }); qc.invalidateQueries({ queryKey: ["sessions"] }) },
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/sessions/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}
