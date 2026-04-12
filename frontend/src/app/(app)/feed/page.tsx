"use client"

import Link from "next/link"
import { SessionCard } from "@/components/session/session-card"
import { useTranslations } from "@/i18n"
import { useSessions } from "@/lib/api/sessions"

export default function FeedPage() {
  const { data, isLoading } = useSessions()
  const tc = useTranslations("common")
  const tf = useTranslations("feed")

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        {tc("loading")}
      </div>
    )
  }

  if (!data?.sessions.length) {
    return (
      <div className="flex flex-col items-center gap-4 py-20">
        <p className="text-muted-foreground">{tf("noSessions")}</p>
        <p className="text-sm text-muted-foreground">{tf("noSessionsHint")}</p>
        <Link
          href="/upload"
          className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          {tf("recordVideo")}
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      {data.sessions.map(session => (
        <SessionCard key={session.id} session={session} />
      ))}
    </div>
  )
}
