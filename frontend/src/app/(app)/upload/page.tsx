"use client"

import { CheckCircle2, Film, Loader2, RotateCcw, Upload } from "lucide-react"
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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [step, setStep] = useState<Step>("ready")
  const [elementType, setElementType] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  function handleFile(f: File) {
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setStep("picked")
  }

  function handleRetake() {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setElementType(null)
    setStep("ready")
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

  // Done state
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

  // Uploading state
  if (step === "uploading") {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-20">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 nike-h3">{t("uploadingVideo")}</p>
        </div>
        <div className="space-y-2">
          <div className="h-2 overflow-hidden rounded-full bg-muted">
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

  // File picked — video preview + element picker + upload
  if (step === "picked" && file && previewUrl) {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
        {/* Video preview */}
        <div className="relative overflow-hidden rounded-2xl bg-black">
          {/* biome-ignore lint/a11y/useMediaCaption: user recording, no captions */}
          <video
            src={previewUrl}
            controls
            playsInline
            className="aspect-video w-full object-contain"
          />
        </div>

        {/* Element picker */}
        <div>
          <p className="mb-2 text-sm font-medium">{t("selectElement")}</p>
          <ElementPicker value={elementType} onChange={setElementType} />
          <p className="mt-1.5 text-xs text-muted-foreground">{t("autoDetectHint")}</p>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            type="button"
            onClick={handleRetake}
            className="flex flex-1 items-center justify-center gap-2 rounded-2xl border border-border px-4 py-3 font-medium text-muted-foreground transition-colors hover:bg-accent"
          >
            <RotateCcw className="h-4 w-4" />
            {t("retake")}
          </button>
          <button
            type="button"
            onClick={handleUpload}
            className="flex flex-[2] items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <Upload className="h-5 w-5" />
            {t("startUpload")}
          </button>
        </div>
      </div>
    )
  }

  // Initial state — camera fills width with negative margins
  return (
    <div className="-mx-4 -mt-4 sm:-mx-6 sm:-mt-6">
      <CameraRecorder
        onRecorded={blob =>
          handleFile(new File([blob], `recording_${Date.now()}.webm`, { type: blob.type }))
        }
      />

      {/* File upload + hint strip below camera */}
      <div className="flex items-center justify-between px-4 py-3 sm:px-6">
        <p className="text-xs text-muted-foreground">{t("recordHint")}</p>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{tc("or")}</span>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent"
          >
            <Film className="h-3.5 w-3.5" />
            {t("chooseFile")}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </div>
      </div>
    </div>
  )
}
