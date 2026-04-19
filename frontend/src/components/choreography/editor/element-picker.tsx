"use client"

import { useState, useMemo } from "react"
import type { TrackType } from "@/types/choreography"

interface ElementPickerProps {
  trackType: TrackType
  onSelect: (code: string) => void
  onClose: () => void
}

const ELEMENTS_BY_TYPE: Record<TrackType, { code: string; name: string; bv: number }[]> = {
  jumps: [
    { code: "1T", name: "Single Toe Loop", bv: 0.4 },
    { code: "1S", name: "Single Salchow", bv: 0.4 },
    { code: "1Lo", name: "Single Loop", bv: 0.5 },
    { code: "1F", name: "Single Flip", bv: 0.5 },
    { code: "1Lz", name: "Single Lutz", bv: 0.6 },
    { code: "1A", name: "Single Axel", bv: 1.1 },
    { code: "2T", name: "Double Toe Loop", bv: 1.3 },
    { code: "2S", name: "Double Salchow", bv: 1.3 },
    { code: "2Lo", name: "Double Loop", bv: 1.7 },
    { code: "2F", name: "Double Flip", bv: 1.8 },
    { code: "2Lz", name: "Double Lutz", bv: 2.1 },
    { code: "2A", name: "Double Axel", bv: 3.3 },
    { code: "3T", name: "Triple Toe Loop", bv: 4.2 },
    { code: "3S", name: "Triple Salchow", bv: 4.3 },
    { code: "3Lo", name: "Triple Loop", bv: 4.9 },
    { code: "3F", name: "Triple Flip", bv: 5.3 },
    { code: "3Lz", name: "Triple Lutz", bv: 5.9 },
    { code: "3A", name: "Triple Axel", bv: 8.0 },
    { code: "4T", name: "Quad Toe Loop", bv: 9.5 },
    { code: "4S", name: "Quad Salchow", bv: 9.7 },
    { code: "4Lo", name: "Quad Loop", bv: 10.5 },
    { code: "4F", name: "Quad Flip", bv: 11.0 },
    { code: "4Lz", name: "Quad Lutz", bv: 11.5 },
  ],
  spins: [
    { code: "CSp1", name: "Combination Spin Lv1", bv: 1.5 },
    { code: "CSp2", name: "Combination Spin Lv2", bv: 2.0 },
    { code: "CSp3", name: "Combination Spin Lv3", bv: 2.5 },
    { code: "CSp4", name: "Combination Spin Lv4", bv: 3.2 },
    { code: "FSp1", name: "Flying Spin Lv1", bv: 1.7 },
    { code: "FSp2", name: "Flying Spin Lv2", bv: 2.3 },
    { code: "FSp3", name: "Flying Spin Lv3", bv: 2.8 },
    { code: "FSp4", name: "Flying Spin Lv4", bv: 3.0 },
    { code: "LSp1", name: "Layback Spin Lv1", bv: 1.5 },
    { code: "LSp2", name: "Layback Spin Lv2", bv: 2.0 },
    { code: "LSp3", name: "Layback Spin Lv3", bv: 2.5 },
    { code: "LSp4", name: "Layback Spin Lv4", bv: 3.0 },
    { code: "USp1", name: "Upright Spin Lv1", bv: 1.5 },
    { code: "USp2", name: "Upright Spin Lv2", bv: 2.0 },
    { code: "USp3", name: "Upright Spin Lv3", bv: 2.5 },
    { code: "USp4", name: "Upright Spin Lv4", bv: 3.0 },
    { code: "CSpB1", name: "Camel Spin Lv1", bv: 1.7 },
    { code: "CSpB2", name: "Camel Spin Lv2", bv: 2.3 },
    { code: "CSpB3", name: "Camel Spin Lv3", bv: 2.8 },
    { code: "CSpB4", name: "Camel Spin Lv4", bv: 3.0 },
  ],
  sequences: [
    { code: "StSq1", name: "Step Sequence Lv1", bv: 1.5 },
    { code: "StSq2", name: "Step Sequence Lv2", bv: 2.6 },
    { code: "StSq3", name: "Step Sequence Lv3", bv: 3.3 },
    { code: "StSq4", name: "Step Sequence Lv4", bv: 3.9 },
    { code: "ChSq1", name: "Choreographic Sequence", bv: 3.0 },
  ],
}

export function ElementPicker({ trackType, onSelect, onClose }: ElementPickerProps) {
  const [search, setSearch] = useState("")
  const items = ELEMENTS_BY_TYPE[trackType]

  const filtered = useMemo(() => {
    if (!search) return items
    const q = search.toLowerCase()
    return items.filter((el) => el.code.toLowerCase().includes(q) || el.name.toLowerCase().includes(q))
  }, [search, items])

  return (
    <div className="w-64 rounded-lg border border-border bg-background p-2 shadow-lg" onClick={(e) => e.stopPropagation()}>
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Поиск элемента..."
        autoFocus
        className="mb-2 w-full rounded-md border border-border bg-muted/30 px-2 py-1 text-sm outline-none focus:border-primary"
      />
      <div className="max-h-60 overflow-y-auto">
        {filtered.map((el) => (
          <button
            key={el.code}
            type="button"
            className="flex w-full items-center justify-between rounded px-2 py-1 text-sm hover:bg-muted/50"
            onClick={() => {
              onSelect(el.code)
              onClose()
            }}
          >
            <span className="font-medium">{el.code}</span>
            <span className="text-xs text-muted-foreground">{el.bv.toFixed(1)}</span>
          </button>
        ))}
        {filtered.length === 0 && (
          <p className="py-2 text-center text-xs text-muted-foreground">Не найдено</p>
        )}
      </div>
    </div>
  )
}
