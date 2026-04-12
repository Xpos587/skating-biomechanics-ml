"use client"

import { Trophy, Video } from "lucide-react"
import { useTranslations } from "@/i18n"
import { usePRs } from "@/lib/api/metrics"
import { useSessions } from "@/lib/api/sessions"

type Props = {
  userId?: string
}

export function StatsSummary({ userId }: Props) {
  const { data: sessionsData } = useSessions(userId)
  const { data: prsData } = usePRs(userId)
  const t = useTranslations("profile")

  const totalSessions = sessionsData?.total ?? 0
  const totalPRs = prsData?.prs.length ?? 0

  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="flex items-center gap-3 rounded-xl border border-border p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <Video className="h-5 w-5 text-primary" />
        </div>
        <div>
          <p className="text-2xl font-bold">{totalSessions}</p>
          <p className="text-xs text-muted-foreground">{t("totalSessions")}</p>
        </div>
      </div>
      <div className="flex items-center gap-3 rounded-xl border border-border p-4">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
          style={{ backgroundColor: "oklch(var(--accent-gold) / 0.1)" }}
        >
          <Trophy className="h-5 w-5" style={{ color: "oklch(var(--accent-gold))" }} />
        </div>
        <div>
          <p className="text-2xl font-bold">{totalPRs}</p>
          <p className="text-xs text-muted-foreground">{t("totalPRs")}</p>
        </div>
      </div>
    </div>
  )
}
