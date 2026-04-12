"use client"

import Link from "next/link"
import { useParams } from "next/navigation"
import { useState } from "react"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"
import { PeriodSelector } from "@/components/progress/period-selector"
import { TrendChart } from "@/components/progress/trend-chart"
import { useTranslations } from "@/i18n"
import { useDiagnostics, useTrend } from "@/lib/api/metrics"

const ELEMENT_IDS = [
  "three_turn",
  "waltz_jump",
  "toe_loop",
  "flip",
  "salchow",
  "loop",
  "lutz",
  "axel",
] as const

export default function StudentProfilePage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<"progress" | "diagnostics">("progress")
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")
  const te = useTranslations("elements")
  const tc = useTranslations("common")
  const ts = useTranslations("students")

  const { data: trend } = useTrend(id, element, metric, period)
  const { data: diag } = useDiagnostics(id)

  return (
    <div className="mx-auto max-w-2xl space-y-4 sm:max-w-3xl">
      <div className="flex gap-2">
        <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">
          &larr; {tc("back")}
        </Link>
      </div>

      <div className="flex gap-1 rounded-lg bg-muted p-1">
        <button
          type="button"
          onClick={() => setTab("progress")}
          className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "progress" ? "bg-background shadow-sm" : ""}`}
        >
          {ts("progress")}
        </button>
        <button
          type="button"
          onClick={() => setTab("diagnostics")}
          className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "diagnostics" ? "bg-background shadow-sm" : ""}`}
        >
          {ts("diagnostics")}
        </button>
      </div>

      {tab === "progress" && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
            {ELEMENT_IDS.map(elId => (
              <button
                type="button"
                key={elId}
                onClick={() => setElement(elId)}
                className={`truncate rounded-xl border p-1.5 text-center text-[11px] sm:p-2 sm:text-xs ${element === elId ? "border-primary bg-primary/10" : "border-border"}`}
              >
                {te(elId)}
              </button>
            ))}
          </div>
          <select
            value={metric}
            onChange={e => setMetric(e.target.value)}
            className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
          >
            <option value="max_height">{ts("metrics.max_height")}</option>
            <option value="airtime">{ts("metrics.airtime")}</option>
            <option value="landing_knee_stability">{ts("metrics.landing_knee_stability")}</option>
            <option value="rotation_speed">{ts("metrics.rotation_speed")}</option>
          </select>
          <PeriodSelector value={period} onChange={setPeriod} />
          {trend && <TrendChart data={trend} />}
        </div>
      )}

      {tab === "diagnostics" && diag && <DiagnosticsList findings={diag.findings} />}
    </div>
  )
}
