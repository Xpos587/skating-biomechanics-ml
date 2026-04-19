"use client"

import { useParams } from "next/navigation"
import { useRef, useState } from "react"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { ScoreBar } from "@/components/choreography/score-bar"
import { useTranslations } from "@/i18n"
import { useProgram, useSaveProgram } from "@/lib/api/choreography"

export default function ProgramEditorPage() {
  const { id } = useParams<{ id: string }>()
  const tc = useTranslations("common")
  const t = useTranslations("choreography")
  const { data: program, isLoading } = useProgram(id)
  const saveProgram = useSaveProgram()
  const [title, setTitle] = useState<string | null>(null)
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null)

  const displayTitle = title ?? program?.title ?? ""

  const elements = program?.layout?.elements ?? []

  function handleTitleChange(newTitle: string) {
    setTitle(newTitle)
    clearTimeout(saveTimer.current ?? 0)
    saveTimer.current = setTimeout(() => {
      saveProgram.mutate({ id, title: newTitle })
    }, 500)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        {tc("loading")}
      </div>
    )
  }

  if (!program) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        {t("notFound")}
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col">
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Timeline panel */}
        <div className="flex flex-1 flex-col border-b border-border p-4 lg:border-b-0 lg:border-r">
          <input
            type="text"
            value={displayTitle}
            onChange={e => handleTitleChange(e.target.value)}
            placeholder={t("untitled")}
            className="mb-3 bg-transparent text-lg font-semibold outline-none placeholder:text-muted-foreground"
          />
          <div className="flex-1 space-y-1.5 overflow-y-auto">
            {elements.map((el, i) => (
              <div
                key={`${el.code}-${el.timestamp}`}
                className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-1.5"
              >
                <span className="w-5 text-xs text-muted-foreground">{i + 1}</span>
                <span className="flex-1 text-sm font-medium">{el.code}</span>
                <span className="text-xs text-muted-foreground">{el.timestamp.toFixed(1)}s</span>
              </div>
            ))}
            {elements.length === 0 && (
              <p className="py-8 text-center text-sm text-muted-foreground">{t("noElements")}</p>
            )}
          </div>
        </div>

        {/* Rink diagram panel */}
        <div className="flex-1 bg-muted/20 p-2">
          <RinkDiagram elements={elements} />
        </div>
      </div>

      {/* Score bar */}
      <ScoreBar
        layout={
          program.layout
            ? { ...program.layout, total_tes: program.total_tes ?? 0, back_half_indices: [] }
            : null
        }
        discipline={program.discipline}
        segment={program.segment}
      />
    </div>
  )
}
