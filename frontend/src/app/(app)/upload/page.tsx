"use client"

import { CheckCircle2, Film, Loader2, Upload } from "lucide-react"
import { useRef, useState } from "react"
import { toast } from "sonner"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ElementPicker } from "@/components/upload/element-picker"
import { useTranslations } from "@/i18n"
import { useCreateSession } from "@/lib/api/sessions"
import { ChunkedUploader } from "@/lib/api/uploads"

type Step = "ready" | "picked" | "uploading" | "done"

export default function UploadPage() {
  const createSession = useCreateSession()
  const t = useTranslations("upload")
  const tc = useTranslations("common")

  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [step, setStep] = useState<Step>("ready")
  const [elementType, setElementType] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  function handleFile(f: File) {
    setFile(f)
    setStep("picked")
  }

  async function handleUpload() {
    if (!file) return
    setStep("uploading")
    try {
      const uploader = new ChunkedUploader(file, (loaded, total) => {
        setProgress(Math.round((loaded / total) * 100))
      })
      await uploader.upload()
      await createSession.mutateAsync({
        element_type: elementType ?? "auto",
      })
      setStep("done")
      toast.success(t("videoUploaded"))
    } catch {
      toast.error(t("uploadError"))
      setStep("picked")
    }
  }

  // Done state — success, redirect after delay
  if (step === "done") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 px-4 py-20">
        <CheckCircle2 className="h-12 w-12 text-primary" />
        <p className="nike-h3">{t("videoUploaded")}</p>
        <p className="text-sm text-muted-foreground">{t("analyzingHint")}</p>
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // File picked — show preview + element picker + upload
  if (step === "picked" && file) {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
        {/* File preview */}
        <div className="flex items-center gap-3 rounded-xl border border-border p-3">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-muted">
            <Film className="h-5 w-5 text-muted-foreground" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted-foreground">
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </p>
          </div>
        </div>

        {/* Element picker */}
        <div>
          <p className="mb-2 text-sm font-medium">{t("selectElement")}</p>
          <ElementPicker value={elementType} onChange={setElementType} />
          <p className="mt-1.5 text-xs text-muted-foreground">{t("autoDetectHint")}</p>
        </div>

        {/* Upload button */}
        <button
          type="button"
          onClick={handleUpload}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          <Upload className="h-5 w-5" />
          {t("startUpload")}
        </button>
      </div>
    )
  }

  // Uploading state
  if (step === "uploading") {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-20">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 nike-h3">{t("uploadingVideo")}</p>
        </div>
        <div className="space-y-2">
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-center text-xs text-muted-foreground">{progress}%</p>
        </div>
      </div>
    )
  }

  // Initial state — camera + file upload option
  return (
    <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
      <CameraRecorder
        onRecorded={blob =>
          handleFile(new File([blob], `recording_${Date.now()}.webm`, { type: blob.type }))
        }
      />

      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs text-muted-foreground">{tc("or")}</span>
        <div className="h-px flex-1 bg-border" />
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="mx-auto flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-accent/50"
      >
        <Film className="h-4 w-4" />
        {t("chooseFile")}
      </button>
    </div>
  )
}
