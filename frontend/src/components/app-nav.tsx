"use client"

import { useQuery } from "@tanstack/react-query"
import { BarChart3, Camera, LogOut, Newspaper, User, Users } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { z } from "zod"
import { useAuth } from "@/components/auth-provider"
import { ThemeToggle } from "@/components/theme-toggle"
import { useTranslations } from "@/i18n"
import { apiFetch } from "@/lib/api-client"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function AppNav() {
  const pathname = usePathname()
  const router = useRouter()
  const { logout } = useAuth()
  const t = useTranslations("nav")
  const tp = useTranslations("profile")

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

  async function handleLogout() {
    await logout()
    document.cookie = "sb_auth=; path=/; max-age=0"
    router.push("/login")
  }

  return (
    <nav className="flex items-center gap-0.5">
      {/* Desktop tabs — hidden on mobile (bottom dock handles that) */}
      <div className="hidden items-center gap-0.5 md:flex">
        {tabs.map(tab => {
          const Icon = tab.icon
          const active = isActive(tab.href)
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? "page" : undefined}
              className={`flex shrink-0 items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors whitespace-nowrap ${
                active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </Link>
          )
        })}
      </div>

      {/* Right-side actions (always visible) */}
      <div className="ml-auto flex shrink-0 items-center gap-1">
        <ThemeToggle />
        <Link
          href="/profile"
          aria-label={t("profile")}
          className={`flex items-center gap-1.5 px-2 py-2 text-sm transition-colors hover:text-foreground ${
            isActive("/profile") ? "text-foreground" : "text-muted-foreground"
          }`}
        >
          <User className="h-4 w-4" />
        </Link>
        <button
          type="button"
          onClick={handleLogout}
          aria-label={tp("signOut")}
          className="flex items-center p-2 text-muted-foreground transition-colors hover:text-foreground"
        >
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </nav>
  )
}
