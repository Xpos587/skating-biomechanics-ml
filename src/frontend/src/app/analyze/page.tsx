"use client"

import { AlertCircle, Loader2 } from "lucide-react"
import { useRouter, useSearchParams } from "next/navigation"
import { Suspense, useCallback, useEffect, useMemo, useState } from "react"
import { DownloadSection } from "@/components/dashboard/download-section"
import { StatsCards } from "@/components/dashboard/stats-cards"
import { VideoPlayer } from "@/components/dashboard/video-player"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { processVideo } from "@/lib/api"
import { toastError, toastSuccess } from "@/lib/toast"
import type { PersonClick, ProcessResponse } from "@/types"

type Phase = "processing" | "done" | "error"

function AnalyzeContent() {
  const params = useSearchParams()
  const router = useRouter()

  const [phase, setPhase] = useState<Phase>("processing")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("Начинаем...")
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState("")

  const videoPath = params.get("video_path") || ""
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

  const startProcessing = useCallback(() => {
    setPhase("processing")
    setProgress(0)
    setMessage("Подготовка...")

    processVideo(
      {
        video_path: videoPath,
        person_click: personClick,
        frame_skip: frameSkip,
        layer: layer,
        tracking: tracking,
        export: doExport,
      },
      {
        onProgress(p, msg) {
          setProgress(Math.round(p * 100))
          setMessage(msg)
        },
        onResult(r) {
          setResult(r as ProcessResponse)
          setPhase("done")
          toastSuccess("Анализ завершён")
        },
        onError(err) {
          setError(err)
          setPhase("error")
          toastError(err)
        },
      },
    )
  }, [videoPath, personClick, frameSkip, layer, tracking, doExport])

  useEffect(() => {
    if (videoPath) startProcessing()
  }, [videoPath, startProcessing])

  const videoUrl = result ? `/api/outputs/${result.video_path}` : ""
  const posesUrl = result?.poses_path ? `/api/outputs/${result.poses_path}` : null
  const csvUrl = result?.csv_path ? `/api/outputs/${result.csv_path}` : null

  return (
    <div>
      {/* Processing */}
      {phase === "processing" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <h2 className="text-lg font-medium">Анализ видео</h2>
            <Progress value={progress} className="w-full max-w-md" />
            <p className="text-sm text-muted-foreground">{message}</p>
            <p className="text-xs text-muted-foreground">{progress}%</p>
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
              <Button onClick={startProcessing}>Повторить</Button>
              <Button variant="outline" onClick={() => router.push("/")}>
                Назад
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
