"use client"

import { CheckCircle2, Loader2, RotateCcw, Upload } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"
import { toast } from "sonner"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ElementPicker } from "@/components/upload/element-picker"
import { useTranslations } from "@/i18n"
import { enqueueProcess } from "@/lib/api/process"
import { useCreateSession } from "@/lib/api/sessions"
import { ChunkedUploader } from "@/lib/api/uploads"
import { useMountEffect } from "@/lib/useMountEffect"

type Step = "ready" | "picked" | "uploading" | "done"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const t = useTranslations("upload")

  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [step, setStep] = useState<Step>("ready")
  const [elementType, setElementType] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const uploaderRef = useRef<ChunkedUploader | null>(null)

  useMountEffect(() => {
    return () => {
      uploaderRef.current?.abort()
    }
  })

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
      uploaderRef.current = uploader
      const videoKey = await uploader.upload()
      const session = await createSession.mutateAsync({
        element_type: elementType ?? "auto",
        video_key: videoKey,
      })
      await enqueueProcess({
        video_key: videoKey,
        person_click: { x: -1, y: -1 },
        session_id: session.id,
      })
      setStep("done")
      toast.success(t("videoUploaded"))
      if (session?.id) {
        router.push(`/sessions/${session.id}`)
      }
    } catch {
      toast.error(t("uploadError"))
      setStep("picked")
    }
  }

  function handleFileUploadClick() {
    fileRef.current?.click()
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
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
        <div
          className="relative overflow-hidden rounded-2xl"
          style={{ backgroundColor: "oklch(var(--background))" }}
        >
          {/* biome-ignore lint/a11y/useMediaCaption: user recording, no captions */}
          <video
            src={previewUrl}
            controls
            playsInline
            className="aspect-video w-full object-contain"
          />
        </div>

        <div>
          <p className="mb-2 text-sm font-medium">{t("selectElement")}</p>
          <ElementPicker value={elementType} onChange={setElementType} />
          <p className="mt-1.5 text-xs text-muted-foreground">{t("autoDetectHint")}</p>
        </div>

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

  // Camera view — fills viewport, standard camera UX
  return (
    <div className="-mx-4 -mt-4 flex h-[calc(100dvh-52px-64px)] flex-col sm:-mx-6 sm:-mt-6">
      <CameraRecorder
        onRecorded={blob =>
          handleFile(new File([blob], `recording_${Date.now()}.webm`, { type: blob.type }))
        }
        onFileUpload={handleFileUploadClick}
        previewUrl={previewUrl}
      />
      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  )
}
