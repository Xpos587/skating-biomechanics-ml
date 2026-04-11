"use client"

import { useState } from "react"
import { useMetricRegistry, useTrend } from "@/lib/api/metrics"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"
import { ELEMENT_TYPE_KEYS } from "@/lib/constants"
import { useTranslations } from "@/i18n"

export default function ProgressPage() {
  const { data: registry } = useMetricRegistry()
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")
  const { data: trend } = useTrend(undefined, element, metric, period)
  const te = useTranslations("elements")
  const ELEMENTS = ELEMENT_TYPE_KEYS.map(id => ({ id, label: te(id) }))

  const availableMetrics = registry
    ? Object.entries(registry).filter(([, v]) => (v as any).element_types.includes(element))
    : []

  return (
    <div className="mx-auto max-w-2xl space-y-4 sm:max-w-3xl">
      <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
        {ELEMENTS.map((el) => (
          <button
            key={el.id}
            onClick={() => setElement(el.id)}
            className={`truncate rounded-xl border p-1.5 text-center text-[11px] sm:p-2 sm:text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}
          >
            {el.label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
        >
          {availableMetrics.map(([name, def]) => (
            <option key={name} value={name}>{(def as any).label_ru}</option>
          ))}
        </select>
        <PeriodSelector value={period} onChange={setPeriod} />
      </div>

      {trend && <TrendChart data={trend} />}
    </div>
  )
}
