"use client"

import Link from "next/link"
import { StudentCard } from "@/components/coach/student-card"
import { useTranslations } from "@/i18n"
import { useRelationships } from "@/lib/api/relationships"

export default function DashboardPage() {
  const { data, isLoading } = useRelationships()
  const tc = useTranslations("common")
  const ts = useTranslations("students")

  const students = (data?.relationships ?? []).filter(r => r.status === "active")

  if (isLoading)
    return <div className="py-20 text-center text-muted-foreground">{tc("loading")}</div>

  if (!students.length) {
    return (
      <div className="flex flex-col items-center gap-4 py-20">
        <p className="text-muted-foreground">{ts("noStudents")}</p>
        <p className="text-sm text-muted-foreground">{ts("noStudentsHint")}</p>
        <Link
          href="/connections"
          className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          {ts("inviteStudent")}
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      <h1 className="nike-h3">{ts("title")}</h1>
      {students.map(rel => (
        <StudentCard key={rel.id} rel={rel} />
      ))}
    </div>
  )
}
