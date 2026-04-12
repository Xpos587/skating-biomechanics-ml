"use client"

import { Clock } from "lucide-react"
import Link from "next/link"
import type { Relationship } from "@/types"

export function StudentCard({ rel }: { rel: Relationship }) {
  return (
    <Link href={`/students/${rel.skater_id}`} className="block">
      <div className="rounded-2xl border border-border p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center text-sm font-medium">
            {(rel.skater_name ?? "?")[0].toUpperCase()}
          </div>
          <div>
            <p className="font-medium text-sm">{rel.skater_name ?? "Ученик"}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {new Date(rel.created_at).toLocaleDateString("ru-RU")}
            </p>
          </div>
        </div>
      </div>
    </Link>
  )
}
