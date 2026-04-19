"use client"

import { useRef } from "react"
import { useParams } from "next/navigation"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"
import { useTranslations } from "@/i18n"
import { useProgram, useMusicAnalysis, useSaveProgram } from "@/lib/api/choreography"
import { useChoreographyEditor } from "@/components/choreography/editor/store"
import { startAutoSave } from "@/components/choreography/editor/auto-save"
import { WaveformView } from "@/components/choreography/editor/waveform-view"
import { TransportBar } from "@/components/choreography/editor/transport-bar"
import { ElementTrack } from "@/components/choreography/editor/element-track"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { ScoreBar } from "@/components/choreography/score-bar"

export default function ProgramEditorPage() {
  const { id } = useParams<{ id: string }>()
  const tc = useTranslations("common")
  const t = useTranslations("choreography")
  const { data: program, isLoading } = useProgram(id)
  const saveProgram = useSaveProgram()
  const editor = useChoreographyEditor()
  const unsubRef = useRef<(() => void) | null>(null)

  const musicAnalysisId = program?.music_analysis_id
  const { data: musicAnalysis } = useMusicAnalysis(musicAnalysisId ?? undefined)

  // Initialize editor from program data
  const initialized = program && editor.programId === program.id

  const audioUrl = musicAnalysis?.audio_url ?? null
  const musicDuration = musicAnalysis?.duration_sec ?? 180
  const beatMarkers = musicAnalysis?.peaks ?? []
  const phraseMarkers = (musicAnalysis?.structure ?? []).map((s) => s.start)

  // Derive layout for ScoreBar from editor state
  const layout = initialized
    ? {
        elements: editor.elements.map((el) => ({
          code: el.code,
          goe: el.goe,
          timestamp: el.timestamp,
          position: el.position ?? null,
          is_back_half: false,
          is_jump_pass: el.trackType === "jumps",
          jump_pass_index: el.jumpPassIndex ?? null,
        })),
        total_tes: editor.getLayoutForSave().total_tes,
        back_half_indices: editor.getLayoutForSave().back_half_indices,
      }
    : null

  function handleSave() {
    if (!program) return
    if (unsubRef.current) unsubRef.current()
    const { layout: saveLayout } = editor.getLayoutForSave()
    saveProgram.mutate(
      {
        id,
        title: editor.title,
        layout: { elements: saveLayout },
      },
      {
        onSuccess: () => {
          unsubRef.current = startAutoSave(
            (data) => saveProgram.mutate(data),
            () => saveProgram.isPending,
          )
        },
      },
    )
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

  if (!initialized) {
    editor.initFromProgram(program, audioUrl, musicDuration, beatMarkers, phraseMarkers)
    if (unsubRef.current) unsubRef.current()
    unsubRef.current = startAutoSave(
      (data) => saveProgram.mutate(data),
      () => saveProgram.isPending,
    )
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-2">
        <Link href="/choreography" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <input
          type="text"
          value={editor.title}
          onChange={(e) => editor.setTitle(e.target.value)}
          placeholder={t("untitled")}
          className="flex-1 bg-transparent text-lg font-semibold outline-none placeholder:text-muted-foreground"
        />
        <button
          type="button"
          onClick={handleSave}
          disabled={saveProgram.isPending}
          className="rounded-lg bg-primary px-3 py-1 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          {saveProgram.isPending ? "..." : t("save")}
        </button>
      </div>

      {/* Main content */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Timeline panel */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Transport bar */}
          <div className="border-b border-border p-2">
            <TransportBar />
          </div>

          {/* Waveform */}
          <div className="border-b border-border p-2">
            <WaveformView audioUrl={audioUrl} />
          </div>

          {/* Tracks */}
          <div className="flex-1 overflow-y-auto">
            <ElementTrack type="jumps" />
            <ElementTrack type="spins" />
            <ElementTrack type="sequences" />
          </div>

          {/* Score bar */}
          <div className="border-t border-border p-2">
            <ScoreBar
              layout={layout}
              discipline={program.discipline}
              segment={program.segment}
            />
          </div>
        </div>

        {/* Rink diagram panel (desktop only) */}
        <div className="hidden w-80 shrink-0 border-l border-border bg-muted/20 p-2 lg:block">
          <RinkDiagram elements={layout?.elements ?? []} />
        </div>
      </div>
    </div>
  )
}
