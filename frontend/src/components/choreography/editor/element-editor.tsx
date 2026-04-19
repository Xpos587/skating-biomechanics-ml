"use client"

import { Copy, Trash2 } from "lucide-react"
import { useChoreographyEditor } from "./store"

interface ElementEditorProps {
  elementId: string
  onClose: () => void
}

export function ElementEditor({ elementId, onClose }: ElementEditorProps) {
  const { elements, updateElement, removeElement, duplicateElement } = useChoreographyEditor()
  const el = elements.find(e => e.id === elementId)
  if (!el) return null

  return (
    <div
      role="dialog"
      aria-label={`Edit ${el.code}`}
      className="w-56 rounded-lg border border-border bg-background p-3 shadow-lg"
      onClick={e => e.stopPropagation()}
      onKeyDown={e => {
        if (e.key === "Escape") onClose()
      }}
    >
      {/* Code display */}
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-bold">{el.code}</span>
        <span className="text-xs text-muted-foreground">{el.timestamp.toFixed(1)}s</span>
      </div>

      {/* GOE slider */}
      <div className="mb-2">
        <label htmlFor={`goe-${elementId}`} className="mb-0.5 block text-xs text-muted-foreground">
          GOE: {el.goe > 0 ? "+" : ""}
          {el.goe}
        </label>
        <input
          id={`goe-${elementId}`}
          type="range"
          min={-5}
          max={5}
          step={1}
          value={el.goe}
          onChange={e => updateElement(elementId, { goe: Number(e.target.value) })}
          className="w-full accent-primary"
        />
      </div>

      {/* Duration */}
      <div className="mb-3">
        <label htmlFor={`dur-${elementId}`} className="mb-0.5 block text-xs text-muted-foreground">
          Duration (s)
        </label>
        <input
          id={`dur-${elementId}`}
          type="number"
          min={1}
          max={20}
          step={0.5}
          value={el.duration}
          onChange={e =>
            updateElement(elementId, { duration: Math.max(1, Number(e.target.value)) })
          }
          className="w-full rounded border border-border bg-muted/30 px-2 py-0.5 text-sm"
        />
      </div>

      {/* Actions */}
      <div className="flex gap-1 border-t border-border pt-2">
        <button
          type="button"
          onClick={() => {
            duplicateElement(elementId)
            onClose()
          }}
          className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted/50"
        >
          <Copy className="h-3 w-3" />
          Duplicate
        </button>
        <button
          type="button"
          onClick={() => {
            removeElement(elementId)
            onClose()
          }}
          className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-xs text-red-500 hover:bg-red-500/10"
        >
          <Trash2 className="h-3 w-3" />
          Delete
        </button>
      </div>
    </div>
  )
}
