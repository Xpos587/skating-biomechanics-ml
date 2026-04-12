"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField } from "@/components/form-field"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()
  const t = useTranslations("auth")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      await login(email, password)
      toast.success(t("signInSuccess"))
      router.push("/feed")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("signInError"))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2 text-center">
        <h1 className="nike-h1">{t("signIn")}</h1>
        <p className="text-sm text-muted-foreground">{t("signInSubtitle")}</p>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <FormField
          label="Email"
          id="email"
          type="email"
          required
          value={email}
          onChange={e => setEmail(e.target.value)}
          placeholder="you@example.com"
        />
        <FormField
          label={t("password")}
          id="password"
          type="password"
          required
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="••••••••"
        />
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? t("signingIn") : t("signInBtn")}
        </Button>
      </form>
      <p className="text-center text-sm text-muted-foreground">
        {t("noAccount")}{" "}
        <Link href="/register" className="text-link hover:underline">
          {t("register")}
        </Link>
      </p>
    </div>
  )
}
