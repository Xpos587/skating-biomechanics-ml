"use client"

import { useTranslations } from "@/i18n"

const ELEMENT_IDS = [
  "three_turn",
  "waltz_jump",
  "toe_loop",
  "flip",
  "salchow",
  "loop",
  "lutz",
  "axel",
] as const

export function ElementPicker({
  value,
  onChange,
}: {
  value: string | null
  onChange: (id: string) => void
}) {
  const te = useTranslations("elements")

  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
      {ELEMENT_IDS.map(id => (
        <button
          type="button"
          key={id}
          onClick={() => onChange(id)}
          className={`truncate rounded-xl border px-2 py-2.5 text-center text-xs transition-colors sm:p-3 sm:text-sm ${
            value === id
              ? "border-primary bg-primary/10 text-primary"
              : "border-border hover:bg-accent/50"
          }`}
        >
          {te(id)}
        </button>
      ))}
    </div>
  )
}
