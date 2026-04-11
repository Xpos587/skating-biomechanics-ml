"use client"

import { Trophy, Video } from "lucide-react"
import { useSessions } from "@/lib/api/sessions"
import { usePRs } from "@/lib/api/metrics"

type Props = {
  userId?: string
}

export function StatsSummary({ userId }: Props) {
  const { data: sessionsData } = useSessions(userId)
  const { data: prsData } = usePRs(userId)

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
          <p className="text-xs text-muted-foreground">Тренировок</p>
        </div>
      </div>
      <div className="flex items-center gap-3 rounded-xl border border-border p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-amber-500/10">
          <Trophy className="h-5 w-5 text-amber-500" />
        </div>
        <div>
          <p className="text-2xl font-bold">{totalPRs}</p>
          <p className="text-xs text-muted-foreground">Рекордов</p>
        </div>
      </div>
    </div>
  )
}
