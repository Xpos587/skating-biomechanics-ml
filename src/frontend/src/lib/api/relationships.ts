// src/frontend/src/lib/api/relationships.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch, apiPost } from "@/lib/api-client"

const RelationshipSchema = z.object({
  id: z.string(), coach_id: z.string(), skater_id: z.string(),
  status: z.enum(["invited", "active", "ended"]),
  initiated_by: z.string().nullable(), created_at: z.string(), ended_at: z.string().nullable(),
  coach_name: z.string().nullable(), skater_name: z.string().nullable(),
})

const RelationshipListSchema = z.object({ relationships: z.array(RelationshipSchema) })

export function useRelationships() {
  return useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })
}

export function usePendingInvites() {
  return useQuery({
    queryKey: ["relationships", "pending"],
    queryFn: () => apiFetch("/relationships/pending", RelationshipListSchema),
  })
}

export function useInvite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { skater_email: string }) =>
      apiPost("/relationships/invite", RelationshipSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}

export function useAcceptInvite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (relId: string) =>
      apiPost(`/relationships/${relId}/accept`, RelationshipSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}

export function useEndRelationship() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (relId: string) =>
      apiPost(`/relationships/${relId}/end`, RelationshipSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}
