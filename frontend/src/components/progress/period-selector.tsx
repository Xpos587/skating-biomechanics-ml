"use client"

import { useTranslations } from "@/i18n"

const PERIOD_KEYS = ["7d", "30d", "90d", "all"] as const

export function PeriodSelector({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const t = useTranslations("progress")

  return (
    <div className="flex gap-1 rounded-lg bg-muted p-1">
      {PERIOD_KEYS.map(p => (
        <button
          type="button"
          key={p}
          onClick={() => onChange(p)}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === p ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {t(`periods.${p}`)}
        </button>
      ))}
    </div>
  )
}
