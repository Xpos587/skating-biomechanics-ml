"use client"

import { Loader2 } from "lucide-react"
import { useParams } from "next/navigation"
import { PhaseTimeline } from "@/components/analysis/phase-timeline"
import { ThreeJSkeletonViewer } from "@/components/analysis/threejs-skeleton-viewer"
import { VideoWithSkeleton } from "@/components/analysis/video-with-skeleton"
import { MetricRow } from "@/components/session/metric-row"
import { useTranslations } from "@/i18n"
import { useSession } from "@/lib/api/sessions"
import { useAnalysisStore } from "@/stores/analysis"

const POLLING_STATUSES = new Set(["queued", "uploading", "running", "pending"])

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const isProcessing = POLLING_STATUSES.has(useSession(id).data?.status ?? "")
  const { data: session, isLoading } = useSession(id, {
    refetchInterval: isProcessing ? 3000 : false,
  })
  const te = useTranslations("elements")
  const tc = useTranslations("common")
  const ts = useTranslations("sessions")

  const totalFrames = session?.pose_data ? Math.max(...session.pose_data.frames) : 300

  if (isLoading)
    return <div className="py-20 text-center text-muted-foreground">{tc("loading")}</div>
  if (!session)
    return <div className="py-20 text-center text-muted-foreground">{ts("notFound")}</div>

  if (POLLING_STATUSES.has(session.status)) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 px-4 py-20">
        <Loader2 className="h-10 w-10 animate-spin text-primary" />
        <p className="nike-h3">{ts("analyzing")}</p>
        <p className="text-sm text-muted-foreground">{ts("analyzingHint")}</p>
      </div>
    )
  }

  if (session.status === "failed" || session.error_message) {
    return (
      <div className="mx-auto max-w-lg space-y-4 px-4 py-20 text-center">
        <p className="nike-h3 text-destructive">{ts("analysisFailed")}</p>
        <p className="text-sm text-muted-foreground">{session.error_message}</p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
      <div>
        <h1 className="text-xl font-semibold">
          {te(session.element_type) ?? session.element_type}
        </h1>
        <p className="text-sm text-muted-foreground">
          {new Date(session.created_at).toLocaleDateString("ru-RU")}
        </p>
      </div>

      {session.processed_video_url && session.pose_data && (
        <VideoWithSkeleton
          videoUrl={session.processed_video_url}
          poseData={session.pose_data}
          phases={session.phases ?? null}
          totalFrames={totalFrames}
          className="rounded-xl"
        />
      )}

      {session.pose_data && (
        <PhaseTimeline totalFrames={totalFrames} phases={session.phases ?? {}} />
      )}

      {session.processed_video_url && !session.pose_data && (
        <video src={session.processed_video_url} controls className="w-full rounded-xl">
          <track kind="captions" />
        </video>
      )}

      {!session.processed_video_url && session.video_url && (
        <video src={session.video_url} controls className="w-full rounded-xl">
          <track kind="captions" />
        </video>
      )}

      {session.pose_data && session.frame_metrics && (
        <ThreeJSkeletonViewer
          poseData={session.pose_data}
          frameMetrics={session.frame_metrics}
          className="rounded-xl"
        />
      )}

      {session.metrics.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">{ts("metrics")}</h2>
          {session.metrics.map(m => (
            <MetricRow
              key={m.id}
              name={m.metric_name}
              label={m.metric_name}
              value={m.metric_value}
              unit={m.unit ?? (m.metric_name === "score" ? "" : m.metric_name === "deg" ? "°" : "")}
              isInRange={m.is_in_range}
              isPr={m.is_pr}
              prevBest={m.prev_best}
              refRange={m.reference_value ? [m.reference_value, m.reference_value + 1] : null}
            />
          ))}
        </div>
      )}

      {session.recommendations && session.recommendations.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">{ts("recommendations")}</h2>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {session.recommendations.map(r => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
