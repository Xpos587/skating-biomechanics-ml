// src/frontend/src/lib/api/metrics.ts
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch } from "@/lib/api-client"

const TrendDataPointSchema = z.object({
  date: z.string(), value: z.number(), session_id: z.string(), is_pr: z.boolean(),
})

const TrendSchema = z.object({
  metric_name: z.string(), element_type: z.string(),
  data_points: z.array(TrendDataPointSchema),
  trend: z.enum(["improving", "stable", "declining"]),
  current_pr: z.number().nullable(),
  reference_range: z.object({ min: z.number(), max: z.number() }).nullable(),
})

const FindingSchema = z.object({
  severity: z.enum(["warning", "info"]), element: z.string(), metric: z.string(),
  message: z.string(), detail: z.string(),
})

const DiagnosticsSchema = z.object({
  user_id: z.string(), findings: z.array(FindingSchema),
})

const MetricDefSchema = z.object({
  name: z.string(), label_ru: z.string(), unit: z.string(), format: z.string(),
  direction: z.enum(["higher", "lower"]), element_types: z.array(z.string()),
  ideal_range: z.tuple([z.number(), z.number()]),
})

export function useTrend(userId: string | undefined, elementType: string, metricName: string, period: string = "30d") {
  const params = new URLSearchParams({ element_type: elementType, metric_name: metricName, period })
  if (userId) params.set("user_id", userId)
  return useQuery({
    queryKey: ["trend", userId, elementType, metricName, period],
    queryFn: () => apiFetch("/metrics/trend?" + params.toString(), TrendSchema),
    enabled: !!elementType && !!metricName,
  })
}

export function useDiagnostics(userId?: string) {
  const params = userId ? `?user_id=${userId}` : ""
  return useQuery({
    queryKey: ["diagnostics", userId],
    queryFn: () => apiFetch("/metrics/diagnostics" + params, DiagnosticsSchema),
  })
}

export function useMetricRegistry() {
  return useQuery({
    queryKey: ["metric-registry"],
    queryFn: () => apiFetch("/metrics/registry", z.record(z.any())),
    staleTime: Infinity,
  })
}

export type MetricDefType = z.infer<typeof MetricDefSchema>
