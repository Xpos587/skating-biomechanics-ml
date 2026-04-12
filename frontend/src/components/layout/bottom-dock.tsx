"use client"

import { useQuery } from "@tanstack/react-query"
import { BarChart3, Camera, Newspaper, Users } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { z } from "zod"
import { useTranslations } from "@/i18n"
import { apiFetch } from "@/lib/api-client"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function BottomDock() {
  const pathname = usePathname()
  const t = useTranslations("nav")

  const { data: relsData } = useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })
  const hasStudents = (relsData?.relationships ?? []).some(r => r.status === "active")

  const tabs = [
    { href: "/feed", icon: Newspaper, label: t("feed") },
    { href: "/upload", icon: Camera, label: t("upload") },
    { href: "/progress", icon: BarChart3, label: t("progress") },
    ...(hasStudents ? [{ href: "/dashboard", icon: Users, label: t("students") }] : []),
  ] as const

  const isActive = (href: string) => pathname === href || pathname.startsWith(`${href}/`)

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background pb-[env(safe-area-inset-bottom)] lg:hidden">
      <div className="flex h-16 items-center justify-around px-2">
        {tabs.map(tab => {
          const Icon = tab.icon
          const active = isActive(tab.href)
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? "page" : undefined}
              aria-label={tab.label}
              className={`flex flex-col items-center gap-0.5 rounded-lg px-4 py-1.5 text-[10px] transition-colors ${
                active ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{tab.label}</span>
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
