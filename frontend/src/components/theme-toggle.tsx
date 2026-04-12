"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme()
  const t = useTranslations("theme")

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="h-4 w-4 scale-100 dark:scale-0" />
      <Moon className="absolute h-4 w-4 scale-0 dark:scale-100" />
      <span className="sr-only">{t("toggle")}</span>
    </Button>
  )
}
