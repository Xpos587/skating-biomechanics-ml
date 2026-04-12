"use client"

import { Pencil } from "lucide-react"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormTextarea } from "@/components/form-field"
import { PersonalRecords } from "@/components/profile/personal-records"
import { RecentActivity } from "@/components/profile/recent-activity"
import { StatsSummary } from "@/components/profile/stats-summary"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()
  const t = useTranslations("profile")
  const tc = useTranslations("common")

  const [editing, setEditing] = useState(false)
  const [displayName, setDisplayName] = useState("")
  const [bio, setBio] = useState("")
  const [height, setHeight] = useState("")
  const [weight, setWeight] = useState("")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">{tc("loading")}</div>
  if (!user) return null

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
        display_name: displayName || undefined,
        bio: bio || undefined,
        height_cm: height ? Number.parseInt(height, 10) : undefined,
        weight_kg: weight ? Number.parseFloat(weight) : undefined,
      })
      toast.success(t("updateSuccess"))
      setEditing(false)
    } catch {
      toast.error(t("updateError"))
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    await logout()
    document.cookie = "sb_auth=; path=/; max-age=0"
    router.push("/login")
  }

  function startEditing() {
    if (!user) return
    setDisplayName(user.display_name ?? "")
    setBio(user.bio ?? "")
    setHeight(user.height_cm?.toString() ?? "")
    setWeight(user.weight_kg?.toString() ?? "")
    setEditing(true)
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="nike-h1">{t("title")}</h1>
        <button
          type="button"
          onClick={handleLogout}
          className="text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          {t("signOut")}
        </button>
      </div>

      {/* User info card — edit button lives here */}
      <div className="rounded-xl border border-border p-4">
        {!editing ? (
          <>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary/10 text-lg font-bold text-primary">
                  {(user.display_name ?? user.email)[0].toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-base font-semibold">
                    {user.display_name ?? user.email}
                  </p>
                  {user.bio && <p className="truncate text-sm text-muted-foreground">{user.bio}</p>}
                </div>
              </div>
              <button
                type="button"
                onClick={startEditing}
                className="flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <Pencil className="h-3.5 w-3.5" />
                {t("editProfile")}
              </button>
            </div>
            {(user.height_cm || user.weight_kg) && (
              <div className="mt-3 flex gap-4 text-sm text-muted-foreground">
                {user.height_cm && <span>{user.height_cm} см</span>}
                {user.weight_kg && <span>{user.weight_kg} кг</span>}
              </div>
            )}
          </>
        ) : (
          <form onSubmit={handleSave} className="space-y-3">
            <FormField
              label={t("name")}
              id="name"
              type="text"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
            />
            <FormTextarea
              label={t("bio")}
              id="bio"
              value={bio}
              onChange={e => setBio(e.target.value)}
              rows={3}
            />
            <div className="grid grid-cols-2 gap-3 sm:gap-4">
              <FormField
                label={t("height")}
                id="height"
                type="number"
                value={height}
                onChange={e => setHeight(e.target.value)}
                min={50}
                max={250}
              />
              <FormField
                label={t("weight")}
                id="weight"
                type="number"
                value={weight}
                onChange={e => setWeight(e.target.value)}
                min={20}
                max={300}
                step={0.1}
              />
            </div>
            <div className="flex justify-end gap-2 pt-1">
              <Button type="button" variant="ghost" size="sm" onClick={() => setEditing(false)}>
                {t("cancel")}
              </Button>
              <Button type="submit" size="sm" disabled={saving}>
                {saving ? tc("saving") : tc("save")}
              </Button>
            </div>
          </form>
        )}
      </div>

      {/* Stats summary */}
      <StatsSummary />

      {/* Personal Records */}
      <div>
        <h2 className="mb-3 text-sm font-medium">{t("personalRecords")}</h2>
        <PersonalRecords />
      </div>

      {/* Recent Activity */}
      <div>
        <h2 className="mb-3 text-sm font-medium">{t("recentActivity")}</h2>
        <RecentActivity />
      </div>
    </div>
  )
}
