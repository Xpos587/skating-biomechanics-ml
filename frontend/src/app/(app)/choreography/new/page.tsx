"use client"

import { ArrowLeft, Sparkles } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState } from "react"
import { InventoryEditor } from "@/components/choreography/inventory-editor"
import { LayoutPicker } from "@/components/choreography/layout-picker"
import { MusicUploader } from "@/components/choreography/music-uploader"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { ScoreBar } from "@/components/choreography/score-bar"
import { useTranslations } from "@/i18n"
import {
  uploadMusicFile,
  useGenerateLayouts,
  useMusicAnalysis,
  useSaveProgram,
} from "@/lib/api/choreography"
import { getAccessToken } from "@/lib/api-client"
import type { Inventory, Layout } from "@/types/choreography"

const DEFAULT_INVENTORY: Inventory = { jumps: [], spins: [], combinations: [] }

export default function NewProgramPage() {
  const t = useTranslations("choreography")
  const tc = useTranslations("common")
  const router = useRouter()

  const [step, setStep] = useState<"music" | "inventory" | "generate" | "result">("music")
  const [musicId, setMusicId] = useState<string | null>(null)
  const [discipline, setDiscipline] = useState<"mens_singles" | "womens_singles">("mens_singles")
  const [segment, setSegment] = useState<"short_program" | "free_skate">("short_program")
  const [inventory, setInventory] = useState<Inventory>(DEFAULT_INVENTORY)
  const [selectedLayout, setSelectedLayout] = useState<Layout | null>(null)

  const [uploading, setUploading] = useState(false)
  const { data: analysis } = useMusicAnalysis(musicId ?? undefined)
  const generateLayouts = useGenerateLayouts()
  const saveProgram = useSaveProgram()

  async function handleUpload(file: File, onProgress?: (loaded: number, total: number) => void) {
    const token = getAccessToken()
    if (!token) return
    setUploading(true)
    try {
      const res = await uploadMusicFile(file, token, onProgress)
      setMusicId(res.music_id)
    } finally {
      setUploading(false)
    }
  }

  function handleGenerate() {
    if (!musicId) return
    generateLayouts.mutate(
      {
        music_analysis_id: musicId,
        discipline,
        segment,
        inventory,
        count: 3,
      },
      {
        onSuccess: res => {
          if (res.layouts.length > 0) {
            setSelectedLayout(res.layouts[0])
            setStep("result")
          }
        },
      },
    )
  }

  function handleSave() {
    if (!selectedLayout) return
    saveProgram.mutate(
      {
        music_analysis_id: musicId ?? undefined,
        discipline,
        segment,
        layout: {
          elements: selectedLayout.elements.map(el => ({
            code: el.code,
            timestamp: el.timestamp,
            goe: el.goe,
          })),
        },
      },
      {
        onSuccess: program => {
          router.push(`/choreography/programs/${program.id}`)
        },
      },
    )
  }

  const musicReady = analysis?.status === "completed"
  const canGenerate = musicReady && (inventory.jumps.length > 0 || inventory.spins.length > 0)

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/choreography" className="rounded-lg p-1.5 hover:bg-accent/50">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <h1 className="nike-h2">{t("newProgram")}</h1>
      </div>

      {/* Step 1: Upload music */}
      <section>
        <h2 className="mb-3 nike-h3">{t("music.title")}</h2>
        <MusicUploader
          analysis={analysis ?? null}
          onUpload={handleUpload}
          isUploading={uploading}
        />
        {musicReady && (
          <div className="mt-4 flex flex-wrap gap-2">
            <div className="flex gap-2">
              {(["mens_singles", "womens_singles"] as const).map(d => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setDiscipline(d)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    discipline === d
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border hover:bg-accent/50"
                  }`}
                >
                  {t(d)}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              {(["short_program", "free_skate"] as const).map(s => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSegment(s)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    segment === s
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border hover:bg-accent/50"
                  }`}
                >
                  {t(s)}
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Step 2: Element inventory */}
      <section>
        <h2 className="mb-3 nike-h3">{t("inventory.title")}</h2>
        <InventoryEditor value={inventory} onChange={setInventory} />
      </section>

      {/* Step 3: Generate */}
      <section>
        <button
          type="button"
          onClick={handleGenerate}
          disabled={!canGenerate || generateLayouts.isPending}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          <Sparkles className="h-5 w-5" />
          {generateLayouts.isPending ? t("generating") : t("generate")}
        </button>
      </section>

      {/* Step 4: Pick layout + save */}
      {generateLayouts.data && step === "result" && (
        <section className="space-y-4">
          <h2 className="nike-h3">{t("selectLayout")}</h2>
          <LayoutPicker
            layouts={generateLayouts.data.layouts}
            selectedIndex={
              selectedLayout ? generateLayouts.data.layouts.indexOf(selectedLayout) : null
            }
            onSelect={idx => {
              setSelectedLayout(generateLayouts.data.layouts[idx])
            }}
          />

          {selectedLayout && (
            <>
              <div className="rounded-2xl bg-muted/20 p-2">
                <RinkDiagram elements={selectedLayout.elements} />
              </div>

              <ScoreBar layout={selectedLayout} discipline={discipline} segment={segment} />

              <button
                type="button"
                onClick={handleSave}
                disabled={saveProgram.isPending}
                className="flex w-full items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
              >
                {saveProgram.isPending ? tc("saving") : tc("save")}
              </button>
            </>
          )}
        </section>
      )}
    </div>
  )
}
