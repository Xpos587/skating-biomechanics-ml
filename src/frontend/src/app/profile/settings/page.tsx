"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormSelect } from "@/components/form-field"
import { Button } from "@/components/ui/button"
import { useLocale, useTranslations } from "@/i18n"
import { setLocale } from "@/i18n/actions"

const LANGUAGES = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
]

const THEMES: { value: "light" | "dark" | "system"; labelKey: "system" | "light" | "dark" }[] = [
  { value: "system", labelKey: "system" },
  { value: "light", labelKey: "light" },
  { value: "dark", labelKey: "dark" },
]

export default function SettingsPage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const currentLocale = useLocale()
  const t = useTranslations("settings")
  const tc = useTranslations("common")

  const [language, setLanguage] = useState("")
  const [timezone, setTimezone] = useState("")
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system")
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
      const { updateSettings } = await import("@/lib/auth")
      await updateSettings({ language, timezone, theme })

      // If language changed, update cookie via Server Action, then reload
      if (language && language !== currentLocale) {
        await setLocale(language as "ru" | "en")
        router.refresh()
        return
      }

      toast.success(t("saved"))
    } catch {
      toast.error(t("saveError"))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <h1 className="nike-h1">{t("title")}</h1>

      <form onSubmit={handleSave} className="space-y-4">
        <FormSelect
          label={t("language")}
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
          label={t("timezone")}
          id="timezone"
          type="text"
          value={timezone}
          onChange={e => setTimezone(e.target.value)}
          placeholder="Europe/Moscow"
        />

        <div className="space-y-2">
          <span className="text-sm font-medium">{t("theme")}</span>
          <div className="flex gap-2">
            {THEMES.map(th => (
              <button
                key={th.value}
                type="button"
                onClick={() => setTheme(th.value)}
                className={`rounded-[0.5rem] border-[1.5px] px-4 py-2 text-sm ${theme === th.value ? "border-foreground bg-secondary" : "border-input"}`}
              >
                {t(th.labelKey)}
              </button>
            ))}
          </div>
        </div>

        <Button type="submit" disabled={saving}>
          {saving ? tc("saving") : tc("save")}
        </Button>
      </form>
    </div>
  )
}
