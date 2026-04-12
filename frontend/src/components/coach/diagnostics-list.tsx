"use client"

import { AlertTriangle, Info } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { DiagnosticsFinding } from "@/types"

export function DiagnosticsList({ findings }: { findings: DiagnosticsFinding[] }) {
  const t = useTranslations("coach")

  if (!findings.length) {
    return <p className="text-sm text-muted-foreground">{t("noProblems")}</p>
  }

  return (
    <div className="space-y-2">
      {findings.map(f => (
        <div
          key={`${f.severity}-${f.metric}-${f.message}`}
          className={`rounded-xl border p-3 ${
            f.severity === "warning" ? "border-border" : "border-border bg-muted/30"
          }`}
          style={
            f.severity === "warning"
              ? {
                  borderColor: "oklch(var(--score-mid) / 0.5)",
                  backgroundColor: "oklch(var(--score-mid) / 0.08)",
                }
              : undefined
          }
        >
          <div className="flex items-start gap-2">
            {f.severity === "warning" ? (
              <AlertTriangle
                className="h-4 w-4 mt-0.5 shrink-0"
                style={{ color: "oklch(var(--score-mid))" }}
              />
            ) : (
              <Info
                className="h-4 w-4 mt-0.5 shrink-0"
                style={{ color: "oklch(var(--primary))" }}
              />
            )}
            <div>
              <p className="text-sm font-medium">{f.message}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{f.detail}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
