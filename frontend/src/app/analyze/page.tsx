"use client"

import { AlertCircle, Loader2 } from "lucide-react"
import { useRouter, useSearchParams } from "next/navigation"
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DownloadSection } from "@/components/dashboard/download-section"
import { StatsCards } from "@/components/dashboard/stats-cards"
import { VideoPlayer } from "@/components/dashboard/video-player"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useTranslations } from "@/i18n"
import { cancelQueuedProcess, enqueueProcess, pollTaskStatus } from "@/lib/api"
import { toastError, toastSuccess } from "@/lib/toast"
import type { PersonClick, ProcessResponse } from "@/types"

type Phase = "processing" | "done" | "error"

function AnalyzeContent() {
  const params = useSearchParams()
  const router = useRouter()
  const t = useTranslations("analyze")

  const [phase, setPhase] = useState<Phase>("processing")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState(t("starting"))
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState("")
  const [taskId, setTaskId] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const videoKey = params.get("video_key") || ""
  const clickParts = (params.get("person_click") || "0,0").split(",")
  const personClick: PersonClick = useMemo(
    () => ({
      x: Number(clickParts[0]),
      y: Number(clickParts[1]),
    }),
    [clickParts],
  )
  const frameSkip = Number(params.get("frame_skip") || 1)
  const layer = Number(params.get("layer") || 3)
  const tracking = params.get("tracking") || "auto"
  const doExport = params.get("export") !== "false"

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startProcessing = useCallback(async () => {
    setPhase("processing")
    setProgress(0)
    setMessage(t("preparing"))

    try {
      const { task_id } = await enqueueProcess({
        video_key: videoKey,
        person_click: personClick,
        frame_skip: frameSkip,
        layer: layer,
        tracking: tracking,
        export: doExport,
      })
      setTaskId(task_id)

      pollRef.current = setInterval(async () => {
        try {
          const status = await pollTaskStatus(task_id)

          if (status.progress) {
            setProgress(Math.round(status.progress * 100))
          }
          if (status.message) {
            setMessage(status.message)
          }

          if (status.status === "completed" && status.result) {
            stopPolling()
            setResult(status.result as ProcessResponse)
            setPhase("done")
            toastSuccess(t("complete"))
          } else if (status.status === "failed") {
            stopPolling()
            setError(status.error || "Processing failed")
            setPhase("error")
            toastError(status.error || "Processing failed")
          } else if (status.status === "cancelled") {
            stopPolling()
            setError("Cancelled")
            setPhase("error")
          }
        } catch (err) {
          console.error("Poll error:", err)
        }
      }, 1000)
    } catch (err) {
      setError(String(err))
      setPhase("error")
      toastError(String(err))
    }
  }, [videoKey, personClick, frameSkip, layer, tracking, doExport, t, stopPolling])

  const handleCancel = useCallback(async () => {
    stopPolling()
    if (taskId) {
      await cancelQueuedProcess(taskId).catch(() => {})
    }
    setError("Cancelled")
    setPhase("error")
  }, [taskId, stopPolling])

  useEffect(() => {
    if (videoKey) startProcessing()
    return () => stopPolling()
  }, [videoKey, startProcessing, stopPolling])

  const videoUrl = result ? `/api/v1/outputs/${result.video_path}` : ""
  const posesUrl = result?.poses_path ? `/api/v1/outputs/${result.poses_path}` : null
  const csvUrl = result?.csv_path ? `/api/v1/outputs/${result.csv_path}` : null

  return (
    <div>
      {/* Processing */}
      {phase === "processing" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <h2 className="nike-h2">{t("title")}</h2>
            <Progress value={progress} className="w-full max-w-md" />
            <p className="text-sm text-muted-foreground">{message}</p>
            <p className="text-xs text-muted-foreground">{progress}%</p>
            <Button variant="outline" size="sm" onClick={handleCancel}>
              Cancel
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Done — dashboard */}
      {phase === "done" && result && (
        <div className="space-y-6">
          <StatsCards stats={result.stats} />
          <VideoPlayer src={videoUrl} />
          <DownloadSection videoUrl={videoUrl} posesUrl={posesUrl} csvUrl={csvUrl} />
        </div>
      )}

      {/* Error */}
      {phase === "error" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <AlertCircle className="h-8 w-8 text-destructive" />
            <p className="text-destructive">{error}</p>
            <div className="flex gap-2">
              <Button onClick={startProcessing}>{t("retry")}</Button>
              <Button variant="outline" onClick={() => router.push("/feed")}>
                {t("back")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default function AnalyzePage() {
  return (
    <Suspense>
      <AnalyzeContent />
    </Suspense>
  )
}
