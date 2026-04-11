import { Film, Frame, Grid3x3, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslations } from "@/i18n"
import type { ProcessStats } from "@/types"

interface StatsCardsProps {
  stats: ProcessStats
}

export function StatsCards({ stats }: StatsCardsProps) {
  const t = useTranslations("stats")

  const statItems = [
    {
      key: "total",
      label: t("frames"),
      icon: Film,
      getValue: (s: ProcessStats) => s.total_frames,
    },
    {
      key: "valid",
      label: t("validFrames"),
      icon: Grid3x3,
      getValue: (s: ProcessStats) => s.valid_frames,
    },
    {
      key: "fps",
      label: "FPS",
      icon: Zap,
      getValue: (s: ProcessStats) => s.fps,
    },
    {
      key: "resolution",
      label: t("resolution"),
      icon: Frame,
      getValue: (s: ProcessStats) => s.resolution,
    },
  ] as const

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {statItems.map(item => {
        const Icon = item.icon
        return (
          <Card key={item.key}>
            <CardHeader className="flex flex-row items-center gap-2 pb-2">
              <Icon className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-xs font-medium text-muted-foreground">
                {item.label}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-2xl font-medium">{item.getValue(stats)}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
