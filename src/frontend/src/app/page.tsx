"use client"

import { useAuth } from "@/components/auth-provider"
import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { useRelationships } from "@/lib/api/relationships"

export default function HomePage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const { data: rels } = useRelationships()

  useEffect(() => {
    if (isLoading) return
    if (!user) {
      router.replace("/login")
      return
    }
    const hasStudents = (rels?.relationships ?? []).some((r) => r.status === "active")
    router.replace(hasStudents ? "/dashboard" : "/feed")
  }, [user, isLoading, rels, router])

  return null
}
