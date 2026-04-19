"use client"

import { useMemo } from "react"
import { renderRink } from "@/lib/rink-renderer"
import type { LayoutElement } from "@/types/choreography"

interface RinkDiagramProps {
  elements: LayoutElement[]
  className?: string
}

export function RinkDiagram({ elements, className }: RinkDiagramProps) {
  const svgHtml = useMemo(() => {
    const rinkElements = elements
      .filter(el => el.position)
      .map(el => ({
        code: el.code,
        position: el.position,
      }))
    if (rinkElements.length === 0) return null
    return renderRink(rinkElements)
  }, [elements])

  if (!svgHtml) {
    return (
      <div
        className={`flex items-center justify-center rounded-2xl border border-dashed border-border p-8 text-sm text-muted-foreground ${className ?? ""}`}
      >
        No elements with positions
      </div>
    )
  }

  return (
    <div
      className={`rounded-2xl border border-border p-2 ${className ?? ""}`}
      style={{ backgroundColor: "oklch(var(--background))" }}
      // biome-ignore lint/security/noDangerouslySetInnerHtml: SVG from local renderer
      dangerouslySetInnerHTML={{ __html: svgHtml }}
    />
  )
}
