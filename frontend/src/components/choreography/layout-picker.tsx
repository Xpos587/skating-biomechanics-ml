"use client"

import { Check } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { Layout } from "@/types/choreography"

interface LayoutPickerProps {
  layouts: Layout[]
  selectedIndex: number | null
  onSelect: (index: number) => void
}

export function LayoutPicker({ layouts, selectedIndex, onSelect }: LayoutPickerProps) {
  const t = useTranslations("choreography")

  if (layouts.length === 0) return null

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium">{t("layout.title")}</h3>
      <div className="grid gap-3 sm:grid-cols-3">
        {layouts.map((layout, idx) => (
          <button
            // biome-ignore lint/suspicious/noArrayIndexKey: layout index is the natural identifier
            key={idx}
            type="button"
            onClick={() => onSelect(idx)}
            className={`relative rounded-2xl border p-4 text-left transition-colors ${
              selectedIndex === idx
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50 hover:bg-accent/30"
            }`}
          >
            {selectedIndex === idx && (
              <div className="absolute top-3 right-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary">
                <Check className="h-3 w-3 text-primary-foreground" />
              </div>
            )}

            <p className="text-xs text-muted-foreground">{t("layout.option", { n: idx + 1 })}</p>
            <p className="mt-1 text-lg font-semibold text-primary">{layout.total_tes.toFixed(2)}</p>

            <div className="mt-3 flex flex-wrap gap-1">
              {layout.elements.map((el, elIdx) => (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: element code is not guaranteed unique within layout
                  key={elIdx}
                  className={`inline-block rounded-md px-1.5 py-0.5 text-[10px] font-medium ${
                    el.is_jump_pass
                      ? "bg-primary/10 text-primary"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {el.code}
                </span>
              ))}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
