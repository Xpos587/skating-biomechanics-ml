"use client"

import { VideoOff } from "lucide-react"
import { useCallback, useRef, useState } from "react"
import { useTranslations } from "@/i18n"
import { useMountEffect } from "@/lib/useMountEffect"

const MIME_TYPES = ["video/webm; codecs=vp9", "video/mp4"]

function getSupportedMimeType(): string {
  for (const mime of MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return "video/webm"
}

export function CameraRecorder({ onRecorded }: { onRecorded: (blob: Blob) => void }) {
  const t = useTranslations("upload")
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [cameraReady, setCameraReady] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null)
  const streamRef = useRef<MediaStream | null>(null)

  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, frameRate: { ideal: 60 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setCameraReady(true)
    } catch {
      setCameraReady(false)
    }
  }

  const startRecording = useCallback(async () => {
    if (!streamRef.current) return
    const stream = streamRef.current
    const mimeType = getSupportedMimeType()
    const recorder = new MediaRecorder(stream, { mimeType })
    const chunks: Blob[] = []

    recorder.ondataavailable = e => chunks.push(e.data)
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType })
      onRecorded(blob)
    }

    mediaRecorderRef.current = recorder
    recorder.start()
    setRecording(true)
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000)
  }, [onRecorded])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`

  useMountEffect(() => {
    initCamera()
    return () => {
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop()
        }
      }
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  })

  return (
    <div className="relative -mx-4 -mt-4 aspect-video overflow-hidden bg-black sm:-mx-6 sm:-mt-6">
      <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" />

      {!cameraReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-muted">
          <VideoOff className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{t("cameraUnavailable")}</p>
        </div>
      )}

      {recording && (
        <div className="absolute left-3 top-3 flex items-center gap-2 rounded-xl bg-black/50 px-3 py-1.5 backdrop-blur-sm">
          <div className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
          <span className="font-mono text-sm text-white">{fmt(elapsed)}</span>
        </div>
      )}

      {cameraReady && (
        <div className="absolute inset-x-0 bottom-0 flex justify-center pb-5">
          {recording ? (
            <button
              type="button"
              onClick={stopRecording}
              className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-[6px] border-white bg-red-500 shadow-lg transition-transform hover:scale-95 active:scale-90"
              aria-label="Stop recording"
            >
              <div className="h-7 w-7 rounded-sm bg-white" />
            </button>
          ) : (
            <button
              type="button"
              onClick={startRecording}
              className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-[6px] border-white/40 bg-red-500/80 shadow-lg transition-transform hover:scale-105"
              aria-label="Start recording"
            >
              <div className="h-8 w-8 rounded-full bg-red-500" />
            </button>
          )}
        </div>
      )}
    </div>
  )
}
