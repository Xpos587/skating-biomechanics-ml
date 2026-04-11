"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormTextarea } from "@/components/form-field"
import { Button } from "@/components/ui/button"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()

  const [displayName, setDisplayName] = useState("")
  const [bio, setBio] = useState("")
  const [height, setHeight] = useState("")
  const [weight, setWeight] = useState("")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">Загрузка...</div>
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
      toast.success("Профиль обновлён")
    } catch {
      toast.error("Ошибка сохранения")
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
        <h1 className="text-2xl font-bold">Профиль</h1>
        <button
          type="button"
          onClick={handleLogout}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Выйти
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-4">
        <FormField label="Email" id="email" type="email" value={user.email} disabled />
        <FormField
          label="Имя"
          id="name"
          type="text"
          value={displayName}
          onChange={e => setDisplayName(e.target.value)}
        />
        <FormTextarea
          label="О себе"
          id="bio"
          value={bio}
          onChange={e => setBio(e.target.value)}
          rows={3}
        />
        <div className="grid grid-cols-2 gap-4">
          <FormField
            label="Рост (см)"
            id="height"
            type="number"
            value={height}
            onChange={e => setHeight(e.target.value)}
            min={50}
            max={250}
          />
          <FormField
            label="Вес (кг)"
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
          {saving ? "Сохранение..." : "Сохранить"}
        </Button>
      </form>
    </div>
  )
}
