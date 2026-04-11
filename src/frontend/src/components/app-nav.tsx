"use client"

import { Activity, LogIn, LogOut, Trophy, User } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useAuth } from "@/components/auth-provider"
import { ThemeToggle } from "@/components/theme-toggle"
import { useTranslations } from "@/i18n"

export function AppNav() {
  const pathname = usePathname()
  const router = useRouter()
  const { isAuthenticated, user, logout } = useAuth()
  const t = useTranslations("nav")

  const navItems = [
    { href: "/", label: t("analysis"), icon: Activity },
    { href: "/training", label: t("training"), icon: Trophy },
  ] as const

  async function handleLogout() {
    await logout()
    router.push("/")
  }

  return (
    <nav className="flex items-center gap-1">
      {navItems.map(item => {
        const Icon = item.icon
        const isActive = pathname === item.href
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted ${isActive ? "bg-muted font-medium" : "text-muted-foreground"}`}
          >
            <Icon className="h-4 w-4" />
            <span className="hidden md:inline">{item.label}</span>
          </Link>
        )
      })}

      {isAuthenticated ? (
        <>
          <Link
            href="/profile"
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted ${pathname === "/profile" ? "bg-muted font-medium" : "text-muted-foreground"}`}
          >
            <User className="h-4 w-4" />
            <span className="hidden md:inline">{user?.display_name ?? t("profile")}</span>
          </Link>
          <button
            type="button"
            onClick={handleLogout}
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </>
      ) : (
        <Link
          href="/login"
          className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted text-muted-foreground"
        >
          <LogIn className="h-4 w-4" />
          <span className="hidden md:inline">{t("signIn")}</span>
        </Link>
      )}

      <ThemeToggle />
    </nav>
  )
}
