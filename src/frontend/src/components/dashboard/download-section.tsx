import { Download, FileSpreadsheet, Table2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface DownloadItem {
  href: string
  label: string
  icon: typeof Download
}

interface DownloadSectionProps {
  videoUrl: string
  posesUrl: string | null
  csvUrl: string | null
}

export function DownloadSection({ videoUrl, posesUrl, csvUrl }: DownloadSectionProps) {
  const items: DownloadItem[] = [
    { href: videoUrl, label: "Видео", icon: Download },
    ...(posesUrl ? [{ href: posesUrl, label: "Позы (.npy)", icon: FileSpreadsheet }] : []),
    ...(csvUrl ? [{ href: csvUrl, label: "Биомеханика (.csv)", icon: Table2 }] : []),
  ]

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {items.map(item => {
        const Icon = item.icon
        return (
          <Button key={item.label} variant="outline" asChild>
            <a href={item.href} download>
              <Icon className="mr-1.5 h-4 w-4" />
              {item.label}
            </a>
          </Button>
        )
      })}
    </div>
  )
}
