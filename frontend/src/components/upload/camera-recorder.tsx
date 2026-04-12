"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const MIME_TYPES = ["video/webm; codecs=vp9", "video/mp4"]

function getSupportedMimeType(): string {
  for (const mime of MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return "video/webm"
}

export function CameraRecorder({ onRecorded }: { onRecorded: (blob: Blob) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, frameRate: { ideal: 60 } },
        audio: false,
      })
      if (videoRef.current) videoRef.current.srcObject = stream

      const mimeType = getSupportedMimeType()
      const recorder = new MediaRecorder(stream, { mimeType })
      const chunks: Blob[] = []

      recorder.ondataavailable = e => chunks.push(e.data)
      recorder.onstop = () => {
        for (const track of stream.getTracks()) track.stop()
        const blob = new Blob(chunks, { type: mimeType })
        onRecorded(blob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecording(true)
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000)
    } catch {
      // Camera not available — silently fail
    }
  }, [onRecorded])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`

  return (
    <div className="space-y-3">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full rounded-xl bg-black aspect-video"
      />
      <div className="flex items-center justify-center gap-4">
        {!recording ? (
          <button
            type="button"
            onClick={startRecording}
            className="rounded-full bg-red-500 p-4 text-white hover:bg-red-600 transition-colors"
          >
            <div className="h-6 w-6 rounded-full bg-white" />
          </button>
        ) : (
          <button
            type="button"
            onClick={stopRecording}
            className="rounded-full bg-red-500 p-4 text-white hover:bg-red-600 transition-colors"
          >
            <div className="h-6 w-6 rounded-sm bg-white" />
          </button>
        )}
        {recording && <span className="text-sm font-mono text-red-500">{fmt(elapsed)}</span>}
      </div>
    </div>
  )
}
