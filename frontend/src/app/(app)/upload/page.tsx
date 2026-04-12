"use client"

import { Upload } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"
import { toast } from "sonner"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ChunkedUploader } from "@/components/upload/chunked-uploader"
import { useTranslations } from "@/i18n"
import { useCreateSession } from "@/lib/api/sessions"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const t = useTranslations("toast")
  const tu = useTranslations("upload")
  const tc = useTranslations("common")

  const handleFile = (f: File) => {
    setFile(f)
  }

  const handleUploaded = async (_key: string) => {
    try {
      await createSession.mutateAsync({ element_type: "auto" })
      toast.success(t("videoUploaded"))
      router.push("/feed")
    } catch {
      toast.error(t("sessionCreateError"))
    }
  }

  if (file) {
    return (
      <div className="mx-auto max-w-lg space-y-4">
        <p className="text-sm text-muted-foreground">{tu("uploadingVideo")}</p>
        <ChunkedUploader file={file} onUploaded={handleUploaded} />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
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
        <Upload className="h-4 w-4" />
        {tu("uploadFile")}
      </button>
    </div>
  )
}
