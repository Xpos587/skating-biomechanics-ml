"use client"

import type React from "react"
import Link from "next/link"
import { Award, Clock, Loader2 } from "lucide-react"
import type { Session } from "@/types"
import { useTranslations } from "@/i18n"

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "только что"
  if (mins < 60) return `${mins} мин назад`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.floor(hours / 24)
  return `${days} дн назад`
}

function scoreStyle(score: number | null): React.CSSProperties {
  if (score === null) return { color: "oklch(var(--muted-foreground))" }
  if (score >= 0.8) return { color: "oklch(var(--score-good))" }
  if (score >= 0.5) return { color: "oklch(var(--score-mid))" }
  return { color: "oklch(var(--score-bad))" }
}

export function SessionCard({ session }: { session: Session }) {
  const hasPR = session.metrics.some((m) => m.is_pr)
  const t = useTranslations("elements")

  return (
    <Link href={`/sessions/${session.id}`} className="block">
      <div className="rounded-2xl border border-border p-3 sm:p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="font-medium truncate">{t(session.element_type)}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3 shrink-0" />
              {relativeTime(session.created_at)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {hasPR && <Award className="h-4 w-4" style={{ color: "oklch(var(--accent-gold))" }} />}
            {session.overall_score !== null && (
              <span className="text-sm font-medium" style={scoreStyle(session.overall_score)}>
                {Math.round(session.overall_score * 100)}%
              </span>
            )}
          </div>
        </div>

        {session.status !== "done" ? (
          <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {session.status === "processing" ? "Анализ..." : "Загрузка..."}
          </div>
        ) : (
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            {session.metrics.slice(0, 3).map((m) => (
              <span key={m.metric_name}>{m.metric_name}: {m.metric_value.toFixed(2)}</span>
            ))}
          </div>
        )}
      </div>
    </Link>
  )
}
