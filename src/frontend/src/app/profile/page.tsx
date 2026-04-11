"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()

  const [displayName, setDisplayName] = useState(user?.display_name ?? "")
  const [bio, setBio] = useState(user?.bio ?? "")
  const [height, setHeight] = useState(user?.height_cm?.toString() ?? "")
  const [weight, setWeight] = useState(user?.weight_kg?.toString() ?? "")
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
        <div className="space-y-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={user.email}
            disabled
            className="w-full rounded-md border border-input bg-muted px-3 py-2 text-sm"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="name" className="text-sm font-medium">
            Имя
          </label>
          <input
            id="name"
            type="text"
            value={displayName}
            onChange={e => setDisplayName(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="bio" className="text-sm font-medium">
            О себе
          </label>
          <textarea
            id="bio"
            value={bio}
            onChange={e => setBio(e.target.value)}
            rows={3}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label htmlFor="height" className="text-sm font-medium">
              Рост (см)
            </label>
            <input
              id="height"
              type="number"
              value={height}
              onChange={e => setHeight(e.target.value)}
              min={50}
              max={250}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="weight" className="text-sm font-medium">
              Вес (кг)
            </label>
            <input
              id="weight"
              type="number"
              value={weight}
              onChange={e => setWeight(e.target.value)}
              min={20}
              max={300}
              step={0.1}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "Сохранение..." : "Сохранить"}
        </button>
      </form>
    </div>
  )
}
