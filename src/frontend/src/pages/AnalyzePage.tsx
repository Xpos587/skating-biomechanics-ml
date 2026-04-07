import { AlertCircle, ArrowLeft, CheckCircle, Download, Loader2, X } from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cancelQueuedProcess, enqueueProcess, pollTaskStatus } from "@/lib/api"
import type { PersonClick, ProcessResponse } from "@/types"

type Phase = "processing" | "done" | "error" | "cancelled"

export default function AnalyzePage() {
  const [params] = useSearchParams()
  const navigate = useNavigate()

  const [phase, setPhase] = useState<Phase>("processing")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("Подготовка конвейера...")
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState("")

  // Guard: prevent double-call in StrictMode
  const startedRef = useRef(false)
  const taskIdRef = useRef<string | null>(null)

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
  const enableDepth = params.get("depth") !== "false"
  const enableOpticalFlow = params.get("optical_flow") !== "false"
  const enableSegment = params.get("segment") !== "false"
  const enableFootTrack = params.get("foot_track") !== "false"
  const enableMatting = params.get("matting") !== "false"
  const enableInpainting = params.get("inpainting") !== "false"

  // Stable process request — doesn't change between renders
  const processRequest = useMemo(
    () => ({
      video_path: videoPath,
      person_click: personClick,
      frame_skip: frameSkip,
      layer: layer,
      tracking: tracking,
      export: doExport,
      depth: enableDepth,
      optical_flow: enableOpticalFlow,
      segment: enableSegment,
      foot_track: enableFootTrack,
      matting: enableMatting,
      inpainting: enableInpainting,
    }),
    [
      videoPath,
      personClick,
      frameSkip,
      layer,
      tracking,
      doExport,
      enableDepth,
      enableOpticalFlow,
      enableSegment,
      enableFootTrack,
      enableMatting,
      enableInpainting,
    ],
  )

  const handleCancel = useCallback(async () => {
    try {
      if (taskIdRef.current) {
        await cancelQueuedProcess(taskIdRef.current)
      }
    } catch {
      // ignore
    }
    setPhase("cancelled")
    startedRef.current = false
  }, [])

  // Only the request object matters — callback identity doesn't affect behavior
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const startProcessing = useCallback(() => {
    if (startedRef.current) return
    startedRef.current = true

    setPhase("processing")
    setProgress(0)
    setMessage("Queuing analysis...")

    const cancelled = false

    enqueueProcess(processRequest)
      .then(res => {
        taskIdRef.current = res.task_id
        setMessage("Waiting for worker...")
        const poll = setInterval(async () => {
          if (cancelled) {
            clearInterval(poll)
            return
          }
          try {
            const status = await pollTaskStatus(res.task_id)
            setProgress(Math.round(status.progress * 100))
            setMessage(status.message)

            if (status.status === "completed" && status.result) {
              clearInterval(poll)
              setResult(status.result)
              setPhase("done")
            } else if (status.status === "failed") {
              clearInterval(poll)
              setError(status.error || "Unknown error")
              setPhase("error")
              startedRef.current = false
            } else if (status.status === "cancelled") {
              clearInterval(poll)
              setPhase("cancelled")
              startedRef.current = false
            }
          } catch {
            // Network error — keep polling
          }
        }, 1000)
      })
      .catch(err => {
        setError(err.message)
        setPhase("error")
        startedRef.current = false
      })
  }, [processRequest])

  useEffect(() => {
    if (videoPath) startProcessing()
  }, [videoPath, startProcessing])

  const videoUrl = result ? `/api/outputs/${result.video_path}` : ""
  const posesUrl = result?.poses_path ? `/api/outputs/${result.poses_path}` : null
  const csvUrl = result?.csv_path ? `/api/outputs/${result.csv_path}` : null

  return (
    <div className="mx-auto max-w-4xl p-6">
      <Button variant="ghost" onClick={() => navigate("/")} className="mb-4 gap-1">
        <ArrowLeft className="h-4 w-4" />
        Назад
      </Button>

      {/* Processing */}
      {phase === "processing" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <h2 className="text-lg font-medium">Анализ видео</h2>
            <Progress value={progress} className="w-full max-w-md" />
            <p className="text-sm text-muted-foreground">{message}</p>
            <div className="flex items-center gap-1.5">
              <div
                className={`h-2 w-2 rounded-full transition-colors ${progress > 0 ? "bg-primary" : "bg-muted"}`}
              />
              <div
                className={`h-2 w-2 rounded-full transition-colors ${progress > 25 ? "bg-primary" : "bg-muted"}`}
              />
              <div
                className={`h-2 w-2 rounded-full transition-colors ${progress > 50 ? "bg-primary" : "bg-muted"}`}
              />
              <div
                className={`h-2 w-2 rounded-full transition-colors ${progress > 75 ? "bg-primary" : "bg-muted"}`}
              />
            </div>
            <Button variant="outline" size="sm" onClick={handleCancel} className="gap-1">
              <X className="h-4 w-4" />
              Отменить
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Cancelled */}
      {phase === "cancelled" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <AlertCircle className="h-8 w-8 text-muted-foreground" />
            <h2 className="text-lg font-medium">Обработка отменена</h2>
            <div className="flex gap-2">
              <Button onClick={startProcessing}>Повторить</Button>
              <Button variant="outline" onClick={() => navigate("/")}>
                Назад
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Done */}
      {phase === "done" && result && (
        <div className="space-y-4">
          <Card>
            <CardContent className="p-4">
              <div className="mb-2 flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <h2 className="font-medium">{result.status}</h2>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground sm:grid-cols-4">
                <span>Кадров: {result.stats.total_frames}</span>
                <span>Валидных: {result.stats.valid_frames}</span>
                <span>FPS: {result.stats.fps}</span>
                <span>{result.stats.resolution}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              {/* biome-ignore lint/a11y/useMediaCaption: analysis output, not media */}
              <video src={videoUrl} controls className="w-full rounded border border-border" />
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" asChild>
              <a href={videoUrl} download>
                <Download className="mr-1 h-4 w-4" />
                Видео
              </a>
            </Button>
            {posesUrl && (
              <Button variant="outline" size="sm" asChild>
                <a href={posesUrl} download>
                  <Download className="mr-1 h-4 w-4" />
                  Позы (.npy)
                </a>
              </Button>
            )}
            {csvUrl && (
              <Button variant="outline" size="sm" asChild>
                <a href={csvUrl} download>
                  <Download className="mr-1 h-4 w-4" />
                  Биомеханика (.csv)
                </a>
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {phase === "error" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <AlertCircle className="h-8 h-8 text-destructive" />
            <p className="text-destructive">{error}</p>
            <div className="flex gap-2">
              <Button onClick={startProcessing}>Повторить</Button>
              <Button variant="outline" onClick={() => navigate("/")}>
                Назад
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
