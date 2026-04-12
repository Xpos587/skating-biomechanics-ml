"use client"

import { Trophy } from "lucide-react"
import Link from "next/link"
import { useTranslations } from "@/i18n"
import { useMetricRegistry, usePRs } from "@/lib/api/metrics"

export function PersonalRecords({ userId }: { userId?: string }) {
  const { data: prsData } = usePRs(userId)
  const { data: registry } = useMetricRegistry()
  const te = useTranslations("elements")
  const t = useTranslations("profile")

  if (!prsData || !registry) return null

  const prs = prsData.prs
  if (prs.length === 0) {
    return (
      <div className="rounded-xl border border-border p-6 text-center text-sm text-muted-foreground">
        {t("noRecordsHint")}
      </div>
    )
  }

  // Group PRs by element type
  const grouped = prs.reduce<Record<string, typeof prs>>((acc, pr) => {
    if (!acc[pr.element_type]) acc[pr.element_type] = []
    acc[pr.element_type].push(pr)
    return acc
  }, {})

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([elementType, elementPRs]) => (
        <div key={elementType}>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">{te(elementType)}</h3>
          <div className="space-y-1.5">
            {elementPRs.map(pr => {
              const mdef = registry[pr.metric_name]
              if (!mdef) return null
              const decimals = mdef.format?.replace(".", "") ?? "2"
              const formatted = pr.value.toFixed(Number(decimals))
              return (
                <Link
                  key={`${pr.metric_name}-${pr.session_id}`}
                  href={`/sessions/${pr.session_id}`}
                  className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5 transition-colors hover:bg-accent"
                >
                  <div className="flex items-center gap-2">
                    <Trophy className="h-4 w-4" style={{ color: "oklch(var(--score-mid))" }} />
                    <span className="text-sm">{mdef.label_ru ?? pr.metric_name}</span>
                  </div>
                  <span className="text-sm font-semibold">
                    {formatted} {mdef.unit ?? ""}
                  </span>
                </Link>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
