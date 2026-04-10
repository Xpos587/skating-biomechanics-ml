"use client"

import { Activity, Settings, Trophy } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"

const navItems = [
  { href: "/", label: "Анализ", icon: Activity },
  { href: "/training", label: "Тренировки", icon: Trophy },
  { href: "/settings", label: "Настройки", icon: Settings },
] as const

export function AppNav() {
  const pathname = usePathname()

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
      <ThemeToggle />
    </nav>
  )
}
