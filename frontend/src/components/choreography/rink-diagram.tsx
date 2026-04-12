"use client"

import { Loader2 } from "lucide-react"
import { useTranslations } from "@/i18n"

interface RinkDiagramProps {
  svgHtml: string | null
  isLoading: boolean
}

export function RinkDiagram({ svgHtml, isLoading }: RinkDiagramProps) {
  const t = useTranslations("choreography")

  if (isLoading) {
    return (
      <div className="flex items-center justify-center rounded-2xl border border-border p-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">{t("rink.loading")}</span>
      </div>
    )
  }

  if (!svgHtml) {
    return (
      <div className="flex items-center justify-center rounded-2xl border border-dashed border-border p-8 text-sm text-muted-foreground">
        {t("rink.empty")}
      </div>
    )
  }

  return (
    <div
      className="rounded-2xl border border-border bg-white p-2 dark:bg-card"
      // biome-ignore lint/security/noDangerouslySetInnerHtml: SVG from trusted server-side renderer
      dangerouslySetInnerHTML={{ __html: svgHtml }}
    />
  )
}
