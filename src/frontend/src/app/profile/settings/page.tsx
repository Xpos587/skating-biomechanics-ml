"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

const LANGUAGES = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
]

const THEMES = [
  { value: "system", label: "Системная" },
  { value: "light", label: "Светлая" },
  { value: "dark", label: "Тёмная" },
]

export default function SettingsPage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()

  const [language, setLanguage] = useState(user?.language ?? "ru")
  const [timezone, setTimezone] = useState(user?.timezone ?? "Europe/Moscow")
  const [theme, setTheme] = useState(user?.theme ?? "system")
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
      const { updateSettings } = await import("@/lib/auth")
      await updateSettings({ language, timezone, theme })
      toast.success("Настройки сохранены")
    } catch {
      toast.error("Ошибка сохранения")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <h1 className="text-2xl font-bold">Настройки</h1>

      <form onSubmit={handleSave} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="language" className="text-sm font-medium">
            Язык
          </label>
          <select
            id="language"
            value={language}
            onChange={e => setLanguage(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {LANGUAGES.map(l => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label htmlFor="timezone" className="text-sm font-medium">
            Часовой пояс
          </label>
          <input
            id="timezone"
            type="text"
            value={timezone}
            onChange={e => setTimezone(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="Europe/Moscow"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="theme" className="text-sm font-medium">
            Тема
          </label>
          <div className="flex gap-2">
            {THEMES.map(t => (
              <button
                key={t.value}
                type="button"
                onClick={() => setTheme(t.value)}
                className={`rounded-md border px-4 py-2 text-sm ${theme === t.value ? "border-primary bg-primary/10" : "border-input"}`}
              >
                {t.label}
              </button>
            ))}
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
