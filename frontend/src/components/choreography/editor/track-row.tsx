"use client"

import { Plus } from "lucide-react"
import { useMemo, useRef, useState } from "react"
import { useTranslations } from "@/i18n"
import type { TrackType } from "@/types/choreography"
import { TRACK_CONFIG } from "@/types/choreography"
import { ElementEditor } from "./element-editor"
import { ElementPicker } from "./element-picker"
import { useChoreographyEditor } from "./store"
import { TrackElement } from "./track-element"

export function TrackRow({ type }: { type: TrackType }) {
  const t = useTranslations("choreography")
  const store = useChoreographyEditor()
  const {
    elements,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    beatMarkers,
    phraseMarkers,
    selectedElementId,
    editorBpm,
    addElement,
    setSelectedElement,
  } = store

  const [pickerPos, setPickerPos] = useState<{ x: number; y: number; ts: number } | null>(null)
  const [editorPos, setEditorPos] = useState<{ x: number; y: number; id: string } | null>(null)

  const trackEls = elements.filter(e => e.trackType === type)
  const cfg = TRACK_CONFIG[type]
  const label = t(`${type}Track` as Parameters<typeof t>[0])
  const totalW = musicDuration * pixelsPerSecond

  const grid = useMemo(() => {
    const lines: { x: number; major: boolean }[] = []
    if (editorBpm > 0) {
      const beat = 60 / editorBpm
      const measure = beat * 4
      for (let s = 0; s <= musicDuration; s += beat) {
        lines.push({
          x: s * pixelsPerSecond,
          major: Math.abs(s % measure) < 0.01 || Math.abs((s % measure) - measure) < 0.01,
        })
      }
    } else {
      for (const t of phraseMarkers) lines.push({ x: t * pixelsPerSecond, major: true })
      for (const t of beatMarkers) lines.push({ x: t * pixelsPerSecond, major: false })
    }
    return lines
  }, [editorBpm, musicDuration, pixelsPerSecond, beatMarkers, phraseMarkers])

  const ruler = useMemo(() => {
    const marks: { x: number; label: string }[] = []
    let step: number
    if (editorBpm > 0) {
      step = (60 / editorBpm) * 4
    } else {
      step = musicDuration > 120 ? 10 : 5
    }
    for (let s = 0; s <= musicDuration; s += step) {
      const m = Math.floor(s / 60)
      const sec = Math.floor(s % 60)
      marks.push({
        x: s * pixelsPerSecond,
        label: m > 0 ? `${m}:${sec.toString().padStart(2, "0")}` : `${sec}`,
      })
    }
    return marks
  }, [editorBpm, musicDuration, pixelsPerSecond])

  const scrollRef = useRef<HTMLDivElement>(null)

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const left = e.currentTarget.scrollLeft
    for (const el of document.querySelectorAll<HTMLElement>("[data-track-scroll]")) {
      if (el !== e.currentTarget) el.scrollLeft = left
    }
  }

  function handleDoubleClick(e: React.MouseEvent) {
    const rect = e.currentTarget.getBoundingClientRect()
    setPickerPos({ x: e.clientX, y: e.clientY, ts: (e.clientX - rect.left) / pixelsPerSecond })
  }

  return (
    <section className="flex min-h-[72px] flex-1" aria-label={label}>
      {/* ── Track label ── */}
      <div className="flex w-28 shrink-0 flex-col justify-center gap-1 border-r border-border px-3 py-2 bg-muted/20 sm:w-32 sm:gap-1.5 sm:px-4">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: cfg.hex }} />
          <span
            className="min-w-0 flex-1 truncate text-xs font-semibold tracking-tight"
            style={{ color: cfg.hex }}
          >
            {label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] tabular-nums text-muted-foreground/70">
            {trackEls.length}/{cfg.maxElements}
          </span>
          <button
            type="button"
            onClick={e => {
              e.stopPropagation()
              const ts =
                trackEls.length > 0
                  ? Math.max(...trackEls.map(el => el.timestamp + el.duration))
                  : 0
              const rect = (e.target as HTMLElement).closest("section")?.getBoundingClientRect()
              if (rect) setPickerPos({ x: rect.right - 8, y: rect.top + 4, ts })
            }}
            className="flex h-5 w-5 items-center justify-center rounded-md text-muted-foreground/40 transition-colors hover:bg-muted/60 hover:text-foreground active:bg-muted"
            aria-label={t("timeline.addElement")}
          >
            <Plus className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* ── Track canvas ── */}
      <div
        ref={scrollRef}
        role="application"
        data-track-scroll
        className="relative flex-1 overflow-x-auto overflow-y-hidden"
        style={{ backgroundColor: "oklch(var(--background))" }}
        onScroll={handleScroll}
        onDoubleClick={handleDoubleClick}
      >
        <div className="relative h-full" style={{ width: totalW }}>
          {/* Alternating measure shading */}
          {editorBpm > 0 &&
            (() => {
              const measure = (60 / editorBpm) * 4
              const measureW = measure * pixelsPerSecond
              const cols = []
              for (let i = 0; i * measure < musicDuration; i++) {
                if (i % 2 === 1) {
                  cols.push(
                    <div
                      key={`shade-${i}`}
                      className="pointer-events-none absolute inset-y-0"
                      style={{
                        left: `${i * measureW}px`,
                        width: `${measureW}px`,
                        backgroundColor: "oklch(var(--muted-foreground) / 0.04)",
                      }}
                    />,
                  )
                }
              }
              return cols
            })()}

          {/* Grid lines */}
          {grid.map(g => (
            <div
              key={`g-${g.x.toFixed(1)}`}
              className="pointer-events-none absolute inset-y-0"
              style={{
                left: `${g.x}px`,
                width: 1,
                backgroundColor: g.major ? "oklch(var(--border))" : "oklch(var(--border) / 0.2)",
              }}
            />
          ))}

          {/* Ruler labels */}
          {ruler.map(r => (
            <span
              key={`r-${r.x.toFixed(0)}`}
              className="pointer-events-none absolute top-0.5 select-none text-[9px] leading-none text-muted-foreground/50"
              style={{ left: `${r.x + 4}px` }}
            >
              {r.label}
            </span>
          ))}

          {/* Playhead */}
          <div
            className="pointer-events-none absolute inset-y-0 z-20 w-0.5"
            style={{
              left: `${currentTime * pixelsPerSecond}px`,
              backgroundColor: "oklch(0.65 0.25 25)",
            }}
          />

          {/* Elements */}
          {trackEls.map(el => (
            <TrackElement
              key={el.id}
              element={el}
              pixelsPerSecond={pixelsPerSecond}
              musicDuration={musicDuration}
              snapMode={snapMode}
              beatMarkers={beatMarkers}
              isSelected={el.id === selectedElementId}
              isActive={currentTime >= el.timestamp && currentTime < el.timestamp + el.duration}
              onSelect={setSelectedElement}
              onEdit={id => {
                const dom = document.querySelector(`[data-el-id="${id}"]`)
                const r = dom?.getBoundingClientRect()
                if (r) setEditorPos({ x: r.right + 4, y: r.top, id })
              }}
            />
          ))}

          {/* Empty hint */}
          {trackEls.length === 0 && (
            <div className="flex h-full items-center justify-center gap-1.5 text-[11px] text-muted-foreground/40">
              <Plus className="h-3 w-3" />
              {t("timeline.addElement")}
            </div>
          )}
        </div>
      </div>

      {/* Popovers */}
      {pickerPos && (
        <div
          role="dialog"
          className="fixed inset-0 z-50"
          onClick={() => setPickerPos(null)}
          onKeyDown={e => e.key === "Escape" && setPickerPos(null)}
        >
          <div className="absolute" style={{ left: pickerPos.x, top: pickerPos.y }}>
            <ElementPicker
              trackType={type}
              onSelect={code => {
                addElement(type, pickerPos.ts, code)
                setPickerPos(null)
              }}
              onClose={() => setPickerPos(null)}
            />
          </div>
        </div>
      )}
      {editorPos && (
        <div
          role="dialog"
          className="fixed inset-0 z-50"
          onClick={() => setEditorPos(null)}
          onKeyDown={e => e.key === "Escape" && setEditorPos(null)}
        >
          <div className="absolute" style={{ left: editorPos.x, top: editorPos.y }}>
            <ElementEditor elementId={editorPos.id} onClose={() => setEditorPos(null)} />
          </div>
        </div>
      )}
    </section>
  )
}
