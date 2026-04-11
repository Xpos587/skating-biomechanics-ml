"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField } from "@/components/form-field"
import { Button } from "@/components/ui/button"

export default function RegisterPage() {
  const router = useRouter()
  const { register } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      await register(email, password, displayName || undefined)
      toast.success("Аккаунт создан")
      router.push("/")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Ошибка регистрации")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl font-bold">Регистрация</h1>
        <p className="text-sm text-muted-foreground">Создайте аккаунт для начала работы</p>
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
          label="Имя (необязательно)"
          id="name"
          type="text"
          value={displayName}
          onChange={e => setDisplayName(e.target.value)}
          placeholder="Ваше имя"
        />
        <FormField
          label="Пароль"
          id="password"
          type="password"
          required
          minLength={8}
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="Минимум 8 символов"
        />
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Создание..." : "Создать аккаунт"}
        </Button>
      </form>
      <p className="text-center text-sm text-muted-foreground">
        Уже есть аккаунт?{" "}
        <Link href="/login" className="text-primary hover:underline">
          Войти
        </Link>
      </p>
    </div>
  )
}
