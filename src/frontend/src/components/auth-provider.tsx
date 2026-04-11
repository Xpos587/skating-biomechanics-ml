"use client"

import { createContext, type ReactNode, useContext, useState } from "react"
import type { UserResponse } from "@/lib/auth"
import * as auth from "@/lib/auth"
import { useMountEffect } from "@/lib/useMountEffect"

interface AuthContextValue {
  user: UserResponse | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName?: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useMountEffect(() => {
    const token = auth.getAccessToken()
    if (!token) {
      setIsLoading(false)
      return
    }

    auth
      .fetchMe()
      .then(setUser)
      .catch(async () => {
        const refresh = auth.getRefreshToken()
        if (!refresh) {
          auth.clearTokens()
          window.location.href = "/login"
          return
        }
        try {
          const tokens = await auth.refreshToken(refresh)
          auth.setTokens(tokens.access_token, tokens.refresh_token)
          const u = await auth.fetchMe()
          setUser(u)
        } catch {
          auth.clearTokens()
          window.location.href = "/login"
        }
      })
      .finally(() => setIsLoading(false))
  })

  async function login(email: string, password: string) {
    const tokens = await auth.login({ email, password })
    auth.setTokens(tokens.access_token, tokens.refresh_token)
    const u = await auth.fetchMe()
    setUser(u)
  }

  async function register(email: string, password: string, displayName?: string) {
    const tokens = await auth.register({ email, password, display_name: displayName })
    auth.setTokens(tokens.access_token, tokens.refresh_token)
    const u = await auth.fetchMe()
    setUser(u)
  }

  async function logout() {
    await auth.logout()
    setUser(null)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
