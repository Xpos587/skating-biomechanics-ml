"use client"

import { LogOut, Settings, X } from "lucide-react"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField } from "@/components/form-field"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export function SettingsSheet() {
  const t = useTranslations("profile")
  const ts = useTranslations("settings")
  const { user, logout } = useAuth()
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [height, setHeight] = useState(user?.height_cm?.toString() ?? "")
  const [weight, setWeight] = useState(user?.weight_kg?.toString() ?? "")
  const [saving, setSaving] = useState(false)

  async function saveBody() {
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
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
    setOpen(false)
    await logout()
    router.push("/login")
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex w-full items-center justify-between rounded-xl border border-border px-4 py-3 text-sm text-muted-foreground transition-colors hover:bg-accent"
      >
        <span className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          {ts("title")}
        </span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end justify-center">
          <button
            type="button"
            className="absolute inset-0 bg-black/50"
            onClick={() => setOpen(false)}
            aria-label="Close settings modal"
          />
          <div className="relative w-full max-w-lg rounded-t-2xl border-t border-border bg-background p-5 pb-[calc(1.5rem+env(safe-area-inset-bottom))]">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="nike-h3">{ts("title")}</h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{ts("theme")}</span>
                <ThemeToggle />
              </div>

              <form
                onSubmit={(e: FormEvent) => {
                  e.preventDefault()
                  saveBody()
                }}
                className="space-y-3"
              >
                <p className="text-sm font-medium">{t("bodyMeasurements")}</p>
                <div className="grid grid-cols-2 gap-3">
                  <FormField
                    label={t("height")}
                    id="settings-height"
                    type="number"
                    value={height}
                    onChange={e => setHeight(e.target.value)}
                    min={50}
                    max={250}
                  />
                  <FormField
                    label={t("weight")}
                    id="settings-weight"
                    type="number"
                    value={weight}
                    onChange={e => setWeight(e.target.value)}
                    min={20}
                    max={300}
                    step={0.1}
                  />
                </div>
                <Button type="submit" size="sm" disabled={saving} className="w-full">
                  {saving ? t("saving") : ts("save")}
                </Button>
              </form>

              <button
                type="button"
                onClick={handleLogout}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-destructive/30 px-4 py-3 text-sm text-destructive transition-colors hover:bg-destructive/10"
              >
                <LogOut className="h-4 w-4" />
                {t("signOut")}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
