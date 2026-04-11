"use client"

import Link from "next/link"
import type React from "react"
import { useSessions } from "@/lib/api/sessions"

const ELEMENT_LABELS: Record<string, string> = {
  waltz_jump: "Вальсовый",
  toe_loop: "Перекидной",
  flip: "Флип",
  salchow: "Сальхов",
  loop: "Петля",
  lutz: "Лютц",
  axel: "Аксель",
  three_turn: "Тройка",
}

function scoreStyle(score: number): React.CSSProperties {
  if (score >= 0.7) return { color: "oklch(var(--score-good))" }
  if (score >= 0.4) return { color: "oklch(var(--score-mid))" }
  return { color: "oklch(var(--score-bad))" }
}

export function RecentActivity({ userId }: { userId?: string }) {
  const { data: sessionsData } = useSessions(userId)

  if (!sessionsData) return null

  const sessions = sessionsData.sessions.slice(0, 5)
  if (sessions.length === 0) {
    return (
      <div className="rounded-xl border border-border p-6 text-center text-sm text-muted-foreground">
        Пока нет записей. Запишите первое видео!
      </div>
    )
  }

  return (
    <div className="space-y-1.5">
      {sessions.map((s) => {
        const date = new Date(s.created_at).toLocaleDateString("ru-RU", {
          day: "numeric", month: "short",
        })
        const score = s.overall_score != null
          ? `${Math.round(s.overall_score * 100)}%`
          : null
        return (
          <Link
            key={s.id}
            href={`/sessions/${s.id}`}
            className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5 transition-colors hover:bg-accent"
          >
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">
                {ELEMENT_LABELS[s.element_type] ?? s.element_type}
              </p>
              <p className="text-xs text-muted-foreground">{date}</p>
            </div>
            {score && (
              <span className="ml-2 text-sm font-semibold" style={scoreStyle(s.overall_score!)}>
                {score}
              </span>
            )}
          </Link>
        )
      })}
    </div>
  )
}
