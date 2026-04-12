"use client"

import { useParams } from "next/navigation"
import { MetricRow } from "@/components/session/metric-row"
import { useTranslations } from "@/i18n"
import { useSession } from "@/lib/api/sessions"

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: session, isLoading } = useSession(id)
  const te = useTranslations("elements")
  const tc = useTranslations("common")
  const ts = useTranslations("sessions")

  if (isLoading)
    return <div className="py-20 text-center text-muted-foreground">{tc("loading")}</div>
  if (!session)
    return <div className="py-20 text-center text-muted-foreground">{ts("notFound")}</div>

  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
      <div>
        <h1 className="text-xl font-semibold">
          {te(session.element_type) ?? session.element_type}
        </h1>
        <p className="text-sm text-muted-foreground">
          {new Date(session.created_at).toLocaleDateString("ru-RU")}
        </p>
      </div>

      {session.processed_video_url && (
        <video src={session.processed_video_url} controls className="w-full rounded-xl">
          <track kind="captions" />
        </video>
      )}

      {session.metrics.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">{ts("metrics")}</h2>
          {session.metrics.map(m => (
            <MetricRow
              key={m.id}
              name={m.metric_name}
              label={m.metric_name}
              value={m.metric_value}
              unit={m.unit ?? (m.metric_name === "score" ? "" : m.metric_name === "deg" ? "°" : "")}
              isInRange={m.is_in_range}
              isPr={m.is_pr}
              prevBest={m.prev_best}
              refRange={m.reference_value ? [m.reference_value, m.reference_value + 1] : null}
            />
          ))}
        </div>
      )}

      {session.recommendations && session.recommendations.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">{ts("recommendations")}</h2>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {session.recommendations.map(r => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
