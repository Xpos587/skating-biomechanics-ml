"use client"

import { useTranslations } from "@/i18n"
import type { Inventory } from "@/types/choreography"

const ALL_JUMPS = [
  "3A",
  "4T",
  "4S",
  "4Lo",
  "4F",
  "4Lz",
  "3Lz",
  "3F",
  "3Lo",
  "3S",
  "3T",
  "2A",
  "2Lz",
  "2F",
  "2Lo",
  "2S",
  "2T",
  "1Eu",
] as const

const ALL_SPINS = ["CSp4", "CSp3", "FSp4", "FSp3", "LSp4", "LSp3", "USp4", "USp3", "CSpB4"] as const

const ALL_COMBOS = [
  "3Lz+3T",
  "3Lz+2T",
  "3F+3T",
  "3F+2T",
  "3Lo+2T",
  "3S+2T",
  "3T+2T",
  "2A+3T",
  "2A+2T",
  "3Lz+1Eu+2S",
  "3Lz+1Eu+3S",
  "3F+1Eu+2S",
] as const

interface InventoryEditorProps {
  value: Inventory
  onChange: (v: Inventory) => void
}

export function InventoryEditor({ value, onChange }: InventoryEditorProps) {
  const t = useTranslations("choreography")

  function toggle(category: "jumps" | "spins" | "combinations", code: string) {
    const current = value[category]
    const next = current.includes(code) ? current.filter(c => c !== code) : [...current, code]
    onChange({ ...value, [category]: next })
  }

  return (
    <div className="space-y-5">
      <section>
        <h3 className="mb-2 text-sm font-medium">{t("inventory.jumps")}</h3>
        <div className="flex flex-wrap gap-1.5">
          {ALL_JUMPS.map(code => (
            <Chip
              key={code}
              label={code}
              active={value.jumps.includes(code)}
              onClick={() => toggle("jumps", code)}
            />
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-medium">{t("inventory.spins")}</h3>
        <div className="flex flex-wrap gap-1.5">
          {ALL_SPINS.map(code => (
            <Chip
              key={code}
              label={code}
              active={value.spins.includes(code)}
              onClick={() => toggle("spins", code)}
            />
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-medium">{t("inventory.combinations")}</h3>
        <div className="flex flex-wrap gap-1.5">
          {ALL_COMBOS.map(code => (
            <Chip
              key={code}
              label={code}
              active={value.combinations.includes(code)}
              onClick={() => toggle("combinations", code)}
            />
          ))}
        </div>
      </section>
    </div>
  )
}

function Chip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "border-primary bg-primary/10 text-primary"
          : "border-border text-muted-foreground hover:bg-accent/50"
      }`}
    >
      {label}
    </button>
  )
}
