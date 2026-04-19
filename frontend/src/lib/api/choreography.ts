import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { API_BASE, apiDelete, apiFetch, apiPatch, apiPost, getAccessToken } from "@/lib/api-client"

// ---------------------------------------------------------------------------
// Zod Schemas
// ---------------------------------------------------------------------------

const MusicSegmentSchema = z.object({
  type: z.string(),
  start: z.number(),
  end: z.number(),
})

const EnergyCurveSchema = z.object({
  timestamps: z.array(z.number()),
  values: z.array(z.number()),
})

export const MusicAnalysisSchema = z.object({
  id: z.string(),
  user_id: z.string(),
  filename: z.string(),
  audio_url: z.string(),
  duration_sec: z.number(),
  bpm: z.number().nullable(),
  meter: z.string().nullable(),
  structure: z.array(MusicSegmentSchema).nullable(),
  energy_curve: EnergyCurveSchema.nullable(),
  downbeats: z.array(z.number()).nullable(),
  peaks: z.array(z.number()).nullable(),
  status: z.enum(["pending", "analyzing", "completed", "failed"]),
  created_at: z.string(),
  updated_at: z.string(),
})

export const UploadMusicResponseSchema = z.object({
  music_id: z.string(),
  filename: z.string(),
})

const PositionSchema = z.object({
  x: z.number(),
  y: z.number(),
})

const LayoutElementSchema = z.object({
  code: z.string(),
  goe: z.number(),
  timestamp: z.number(),
  position: PositionSchema.nullable(),
  is_back_half: z.boolean(),
  is_jump_pass: z.boolean(),
  jump_pass_index: z.number().nullable(),
})

const LayoutSchema = z.object({
  elements: z.array(LayoutElementSchema),
  total_tes: z.number(),
  back_half_indices: z.array(z.number()),
})

export const GenerateResponseSchema = z.object({
  layouts: z.array(LayoutSchema),
})

export const ValidationResultSchema = z.object({
  is_valid: z.boolean(),
  errors: z.array(z.string()),
  warnings: z.array(z.string()),
  total_tes: z.number().nullable(),
})

const ProgramLayoutSchema = z.object({
  elements: z.array(LayoutElementSchema),
})

export const ChoreographyProgramSchema = z.object({
  id: z.string(),
  user_id: z.string(),
  music_analysis_id: z.string().nullable(),
  title: z.string().nullable(),
  discipline: z.enum(["mens_singles", "womens_singles"]),
  segment: z.enum(["short_program", "free_skate"]),
  season: z.string(),
  layout: ProgramLayoutSchema.nullable(),
  total_tes: z.number().nullable(),
  estimated_goe: z.number().nullable(),
  estimated_pcs: z.number().nullable(),
  estimated_total: z.number().nullable(),
  is_valid: z.boolean().nullable(),
  validation_errors: z.array(z.string()).nullable(),
  validation_warnings: z.array(z.string()).nullable(),
  created_at: z.string(),
  updated_at: z.string(),
})

const ProgramListResponseSchema = z.object({
  programs: z.array(ChoreographyProgramSchema),
  total: z.number(),
})

// ---------------------------------------------------------------------------
// Hooks — Music Analysis
// ---------------------------------------------------------------------------

export function useMusicAnalysis(musicId: string | undefined) {
  return useQuery({
    queryKey: ["music-analysis", musicId],
    queryFn: () => apiFetch(`/choreography/music/${musicId}/analysis`, MusicAnalysisSchema),
    enabled: !!musicId,
    refetchInterval: query => {
      const status = query.state.data?.status
      if (status === "pending" || status === "analyzing") return 3000
      return false
    },
  })
}

export function uploadMusicFile(
  file: File,
  token: string | null,
  onProgress?: (loaded: number, total: number) => void,
): Promise<z.infer<typeof UploadMusicResponseSchema>> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append("file", file)
    const xhr = new XMLHttpRequest()
    // Upload directly to backend to avoid Next.js 10MB body limit on proxy
    xhr.open("POST", "http://localhost:8000/api/v1/choreography/music/upload")
    xhr.setRequestHeader("Authorization", `Bearer ${token}`)
    xhr.upload.onprogress = e => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(UploadMusicResponseSchema.parse(JSON.parse(xhr.responseText)))
        } catch {
          reject(new Error("Invalid response"))
        }
      } else {
        let detail = `HTTP ${xhr.status}`
        try {
          const body = JSON.parse(xhr.responseText)
          if (body.detail) detail = body.detail
        } catch {
          // non-JSON error response
        }
        reject(new Error(detail))
      }
    }
    xhr.onerror = () => reject(new Error("Network error"))
    xhr.send(form)
  })
}

export function useUploadMusic() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => uploadMusicFile(file, getAccessToken()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["music-analysis"] }),
  })
}

// ---------------------------------------------------------------------------
// Hooks — Layout Generation & Validation
// ---------------------------------------------------------------------------

export function useGenerateLayouts() {
  return useMutation({
    mutationFn: (body: {
      music_analysis_id: string
      discipline: "mens_singles" | "womens_singles"
      segment: "short_program" | "free_skate"
      inventory: { jumps: string[]; spins: string[]; combinations: string[] }
      count?: number
    }) => apiPost("/choreography/generate", GenerateResponseSchema, {
      music_id: body.music_analysis_id,
      discipline: body.discipline,
      segment: body.segment,
      inventory: body.inventory,
    }),
  })
}

export function useValidateLayout() {
  return useMutation({
    mutationFn: (body: {
      discipline: "mens_singles" | "womens_singles"
      segment: "short_program" | "free_skate"
      layout: { elements: Array<{ code: string; timestamp: number; goe: number }> }
    }) => apiPost("/choreography/validate", ValidationResultSchema, {
      discipline: body.discipline,
      segment: body.segment,
      elements: body.layout.elements,
    }),
  })
}

// ---------------------------------------------------------------------------
// Hooks — Programs
// ---------------------------------------------------------------------------

export function usePrograms() {
  return useQuery({
    queryKey: ["programs"],
    queryFn: () => apiFetch("/choreography/programs", ProgramListResponseSchema),
  })
}

export function useProgram(id: string | undefined) {
  return useQuery({
    queryKey: ["program", id],
    queryFn: () => apiFetch(`/choreography/programs/${id}`, ChoreographyProgramSchema),
    enabled: !!id,
  })
}

export function useSaveProgram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: {
      id?: string
      music_analysis_id?: string
      title?: string
      discipline?: "mens_singles" | "womens_singles"
      segment?: "short_program" | "free_skate"
      season?: string
      layout?: { elements: Array<{ code: string; timestamp: number; goe: number }> }
    }) => {
      if (body.id) {
        const { id, ...rest } = body
        return apiPatch(`/choreography/programs/${id}`, ChoreographyProgramSchema, rest)
      }
      return apiPost("/choreography/programs", ChoreographyProgramSchema, body)
    },
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["programs"] })
      if (variables.id) {
        qc.invalidateQueries({ queryKey: ["program", variables.id] })
      }
    },
  })
}

export function useDeleteProgram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/choreography/programs/${id}`),
    onMutate: async id => {
      await qc.cancelQueries({ queryKey: ["programs"] })
      const previous = qc.getQueryData(["programs"])
      qc.setQueryData(
        ["programs"],
        (old: z.infer<typeof ProgramListResponseSchema> | undefined) => {
          if (!old) return old
          return {
            ...old,
            programs: old.programs.filter(p => p.id !== id),
            total: old.total - 1,
          }
        },
      )
      return { previous }
    },
    onError: (_err, _id, context) => {
      if (context?.previous) {
        qc.setQueryData(["programs"], context.previous)
      }
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["programs"] })
    },
  })
}
