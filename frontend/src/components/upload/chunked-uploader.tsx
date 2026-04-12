"use client"

import { useState } from "react"
import { ChunkedUploader as ChunkedUploaderClass } from "@/lib/api/uploads"

export function ChunkedUploader({
  file,
  onUploaded,
}: {
  file: File
  onUploaded: (key: string) => void
}) {
  const [progress, setProgress] = useState(0)

  const upload = async () => {
    const uploader = new ChunkedUploaderClass(file, (loaded, total) => {
      setProgress(Math.round((loaded / total) * 100))
    })
    const key = await uploader.upload()
    onUploaded(key)
  }

  return (
    <div className="space-y-2">
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground text-center">{progress}%</p>
      {progress === 0 && (
        <button
          type="button"
          onClick={upload}
          className="w-full rounded-xl bg-primary text-primary-foreground py-3 text-sm font-medium"
        >
          Загрузить
        </button>
      )}
    </div>
  )
}
