"use client"

import { Loader2, Music, Upload } from "lucide-react"
import { useRef, useState } from "react"
import { useTranslations } from "@/i18n"
import type { MusicAnalysis } from "@/types/choreography"

interface MusicUploaderProps {
  analysis: MusicAnalysis | null
  onUpload: (file: File, onProgress?: (loaded: number, total: number) => void) => void
  isUploading: boolean
}

export function MusicUploader({ analysis, onUpload, isUploading }: MusicUploaderProps) {
  const t = useTranslations("choreography")
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [progress, setProgress] = useState(0)

  const isPending = analysis?.status === "pending" || analysis?.status === "analyzing"

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onUpload(file, (loaded, total) => setProgress(Math.round((loaded / total) * 100)))
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) onUpload(file, (loaded, total) => setProgress(Math.round((loaded / total) * 100)))
  }

  if (isPending) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border p-8 text-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">{t("music.analyzing")}</p>
      </div>
    )
  }

  if (analysis && analysis.status === "completed") {
    return (
      <div className="flex items-center gap-3 rounded-2xl border border-border p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10">
          <Music className="h-5 w-5 text-primary" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{analysis.filename}</p>
          <p className="text-xs text-muted-foreground">
            {formatDuration(analysis.duration_sec)}
            {analysis.bpm ? ` \u00B7 ${analysis.bpm} BPM` : ""}
          </p>
        </div>
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={() => inputRef.current?.click()}
      onDragOver={e => {
        if (!isUploading) {
          e.preventDefault()
          setDragOver(true)
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => {
        if (!isUploading) handleDrop(e)
      }}
      disabled={isUploading}
      className={`flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
        dragOver
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-accent/30"
      } ${isUploading ? "pointer-events-none cursor-default" : ""}`}
    >
      {isUploading ? (
        <>
          <div className="w-full max-w-xs space-y-2">
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full transition-all duration-300 ${progress >= 100 ? "animate-pulse" : "bg-primary"}`}
                style={{ width: `${progress}%`, backgroundColor: progress >= 100 ? "oklch(var(--muted-foreground))" : undefined }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {progress >= 100 ? t("music.processing") : `${progress}%`}
            </p>
          </div>
        </>
      ) : (
        <>
          <Upload className="h-8 w-8 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium">{t("music.dropzone")}</p>
            <p className="mt-1 text-xs text-muted-foreground">{t("music.dropzoneHint")}</p>
          </div>
        </>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleFileChange}
        disabled={isUploading}
      />
    </button>
  )
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}
