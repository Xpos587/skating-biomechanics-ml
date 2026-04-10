import { Film, Frame, Grid3x3, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProcessStats } from "@/types"

interface StatsCardsProps {
  stats: ProcessStats
}

const statItems = [
  {
    key: "total",
    label: "Кадров",
    icon: Film,
    getValue: (s: ProcessStats) => s.total_frames,
  },
  {
    key: "valid",
    label: "Валидных",
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
    label: "Разрешение",
    icon: Frame,
    getValue: (s: ProcessStats) => s.resolution,
  },
] as const

export function StatsCards({ stats }: StatsCardsProps) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
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
              <p className="text-2xl font-bold">{item.getValue(stats)}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
