"use client"

import { Image as LucideImage, RotateCcw, VideoOff } from "lucide-react"
import NextImage from "next/image"
import { usePathname } from "next/navigation"
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

/** Stop all tracks on a MediaStream */
function stopStream(stream: MediaStream | null) {
  if (!stream) return
  for (const track of stream.getTracks()) {
    track.stop()
  }
}

export function CameraRecorder({
  onRecorded,
  onFileUpload,
  previewUrl,
}: {
  onRecorded: (blob: Blob) => void
  onFileUpload?: () => void
  previewUrl?: string | null
}) {
  const t = useTranslations("upload")
  const pathname = usePathname()
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [cameraReady, setCameraReady] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const mountedRef = useRef(true)

  function stopCamera() {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
    stopStream(streamRef.current)
    streamRef.current = null
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (mountedRef.current) {
      setRecording(false)
      setCameraReady(false)
    }
  }

  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, frameRate: { ideal: 60 } },
        audio: false,
      })
      if (!mountedRef.current) {
        stopStream(stream)
        return
      }
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setCameraReady(true)
    } catch {
      if (mountedRef.current) setCameraReady(false)
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

  // Cleanup when navigating away (Next.js may not unmount on soft navigation)
  const prevPathRef = useRef(pathname)
  if (pathname !== prevPathRef.current) {
    prevPathRef.current = pathname
    stopCamera()
  }

  useMountEffect(() => {
    mountedRef.current = true
    initCamera()
    return () => {
      mountedRef.current = false
      stopCamera()
    }
  })

  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`

  return (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-black">
      <video ref={videoRef} autoPlay playsInline muted className="min-h-0 flex-1 object-cover" />

      {!cameraReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-muted">
          <VideoOff className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{t("cameraUnavailable")}</p>
        </div>
      )}

      {/* Recording indicator — top center */}
      {recording && (
        <div className="absolute left-1/2 top-4 flex -translate-x-1/2 items-center gap-2 rounded-full bg-black/50 px-4 py-1.5 backdrop-blur-sm">
          <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" />
          <span className="font-mono text-sm font-medium tabular-nums text-white">
            {fmt(elapsed)}
          </span>
        </div>
      )}

      {/* Bottom bar — floating over camera */}
      {cameraReady && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between px-6 pb-8">
          {/* Gallery / file upload — bottom left */}
          <button
            type="button"
            onClick={onFileUpload}
            className="pointer-events-auto flex h-12 w-12 items-center justify-center overflow-hidden rounded-xl border-2 border-white/30 bg-white/10 backdrop-blur-sm transition-colors active:scale-95"
            aria-label={t("chooseFile")}
          >
            {previewUrl ? (
              <NextImage
                src={previewUrl}
                alt=""
                className="h-full w-full object-cover"
                width={200}
                height={200}
              />
            ) : (
              <LucideImage className="h-5 w-5 text-white" />
            )}
          </button>

          {/* Shutter button — center */}
          <div className="pointer-events-auto">
            {recording ? (
              <button
                type="button"
                onClick={stopRecording}
                className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-[6px] border-white bg-red-500 shadow-lg transition-transform active:scale-90"
                aria-label="Stop"
              >
                <div className="h-7 w-7 rounded-sm bg-white" />
              </button>
            ) : (
              <button
                type="button"
                onClick={startRecording}
                className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-[6px] border-white/30 bg-red-500 shadow-lg transition-transform active:scale-90"
                aria-label="Record"
              >
                <div className="h-8 w-8 rounded-full bg-red-500" />
              </button>
            )}
          </div>

          {/* Retake / switch camera — bottom right */}
          <button
            type="button"
            onClick={() => {
              if (streamRef.current) {
                const track = streamRef.current.getVideoTracks()[0]
                if (track) {
                  const current = track.getSettings().facingMode
                  track.applyConstraints({
                    facingMode: current === "environment" ? "user" : "environment",
                  })
                }
              }
            }}
            className="pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full bg-white/10 backdrop-blur-sm transition-transform active:scale-90"
            aria-label="Flip camera"
          >
            <RotateCcw className="h-5 w-5 text-white" />
          </button>
        </div>
      )}
    </div>
  )
}
