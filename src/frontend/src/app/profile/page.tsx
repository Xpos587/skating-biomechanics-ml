"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormTextarea } from "@/components/form-field"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()
  const t = useTranslations("profile")
  const tc = useTranslations("common")

  const [displayName, setDisplayName] = useState("")
  const [bio, setBio] = useState("")
  const [height, setHeight] = useState("")
  const [weight, setWeight] = useState("")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">{tc("loading")}</div>
  if (!user) {
    router.push("/login")
    return null
  }

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
    } catch {
      toast.error(t("updateError"))
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    await logout()
    router.push("/")
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("title")}</h1>
        <button
          type="button"
          onClick={handleLogout}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          {t("signOut")}
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-4">
        <FormField label="Email" id="email" type="email" value={user.email} disabled />
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
        <div className="grid grid-cols-2 gap-4">
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
        <Button type="submit" disabled={saving}>
          {saving ? tc("saving") : tc("save")}
        </Button>
      </form>
    </div>
  )
}
