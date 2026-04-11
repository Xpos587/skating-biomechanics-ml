import type { LoginRequest, RegisterRequest, TokenResponse, UserResponse } from "@/lib/auth-schemas"
import { TokenResponseSchema, UserResponseSchema } from "@/lib/auth-schemas"

const API_BASE = "/api/v1"

const TOKEN_KEY = "access_token"
const REFRESH_KEY = "refresh_token"

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(REFRESH_KEY)
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(TOKEN_KEY, access)
  localStorage.setItem(REFRESH_KEY, refresh)
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

export function authHeaders(): Record<string, string> {
  const token = getAccessToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function register(data: RegisterRequest): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }))
    throw new Error(err.detail)
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function login(data: LoginRequest): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    throw new Error("Неверный email или пароль")
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function refreshToken(refresh: string): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  })
  if (!res.ok) {
    clearTokens()
    throw new Error("Сессия истекла")
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function logout(): Promise<void> {
  const refresh = getRefreshToken()
  if (refresh) {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    }).catch(() => {})
  }
  clearTokens()
}

export async function fetchMe(): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me`, {
    headers: { ...authHeaders() },
  })
  if (!res.ok) throw new Error("Unauthorized")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}

export async function updateProfile(
  data: Partial<RegisterRequest> & { height_cm?: number; weight_kg?: number; bio?: string },
): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error("Update failed")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}

export async function updateSettings(data: {
  language?: string
  timezone?: string
  theme?: string
}): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error("Update failed")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}
