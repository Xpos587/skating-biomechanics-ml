"use client"

import { useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { useCreateSession } from "@/lib/api/sessions"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ChunkedUploader } from "@/components/upload/chunked-uploader"
import { ElementPicker } from "@/components/upload/element-picker"

type Mode = "pick" | "record" | "uploading"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const [mode, setMode] = useState<Mode>("pick")
  const [elementType, setElementType] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setMode("uploading")
  }

  const handleRecorded = (blob: Blob) => {
    const f = new File([blob], `recording_${Date.now()}.webm`, { type: blob.type })
    setFile(f)
    setMode("uploading")
  }

  const handleUploaded = async (key: string) => {
    if (!elementType) return
    try {
      await createSession.mutateAsync({ element_type: elementType })
      toast.success("Видео загружено, анализ начат")
      router.push("/feed")
    } catch {
      toast.error("Ошибка создания сессии")
    }
  }

  return (
    <div className="max-w-lg mx-auto space-y-6">
      {mode !== "uploading" && (
        <div className="flex gap-3">
          <button onClick={() => setMode("pick")} className={`flex-1 rounded-xl border p-4 text-center text-sm ${mode === "pick" ? "border-primary bg-primary/10" : "border-border"}`}>
            Выбрать файл
          </button>
          <button onClick={() => setMode("record")} className={`flex-1 rounded-xl border p-4 text-center text-sm ${mode === "record" ? "border-primary bg-primary/10" : "border-border"}`}>
            Записать
          </button>
        </div>
      )}

      {mode === "pick" && (
        <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
      )}

      {mode === "record" && <CameraRecorder onRecorded={handleRecorded} />}

      {mode !== "uploading" && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Элемент:</p>
          <ElementPicker value={elementType} onChange={setElementType} />
        </div>
      )}

      {mode === "pick" && (
        <button onClick={() => fileRef.current?.click()} className="w-full rounded-xl bg-primary text-primary-foreground py-3 text-sm font-medium">
          Выбрать видео
        </button>
      )}

      {mode === "uploading" && file && <ChunkedUploader file={file} onUploaded={handleUploaded} />}
    </div>
  )
}
