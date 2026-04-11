"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormSelect } from "@/components/form-field"
import { Button } from "@/components/ui/button"

const LANGUAGES = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
]

const THEMES: { value: "light" | "dark" | "system"; label: string }[] = [
  { value: "system", label: "Системная" },
  { value: "light", label: "Светлая" },
  { value: "dark", label: "Тёмная" },
]

export default function SettingsPage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()

  const [language, setLanguage] = useState("")
  const [timezone, setTimezone] = useState("")
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system")
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
        <FormSelect
          label="Язык"
          id="language"
          value={language}
          onChange={e => setLanguage(e.target.value)}
        >
          {LANGUAGES.map(l => (
            <option key={l.value} value={l.value}>
              {l.label}
            </option>
          ))}
        </FormSelect>

        <FormField
          label="Часовой пояс"
          id="timezone"
          type="text"
          value={timezone}
          onChange={e => setTimezone(e.target.value)}
          placeholder="Europe/Moscow"
        />

        <div className="space-y-2">
          <span className="text-sm font-medium">Тема</span>
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

        <Button type="submit" disabled={saving}>
          {saving ? "Сохранение..." : "Сохранить"}
        </Button>
      </form>
    </div>
  )
}
