/**
 * Shared API infrastructure: base URL, token storage, typed fetch helper.
 */

import type { z } from "zod"

export const API_BASE = "/api/v1"

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

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
  // biome-ignore lint: sync auth cookie for SSR gating
  document.cookie = "sb_auth=1; path=/; max-age=31536000; SameSite=Lax"
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
  // biome-ignore lint: clear auth cookie on logout
  document.cookie = "sb_auth=; path=/; max-age=0"
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
  }
}

// ---------------------------------------------------------------------------
// Typed fetch
// ---------------------------------------------------------------------------

function authHeaders(): Record<string, string> {
  const token = getAccessToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function apiFetch<T>(
  path: string,
  schema: z.ZodSchema<T>,
  init?: RequestInit & { auth?: boolean },
): Promise<T> {
  const { auth = true, headers, ...rest } = init ?? {}

  const res = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: { ...(auth ? authHeaders() : {}), ...headers },
  })

  if (!res.ok) {
    if (res.status === 401) {
      clearTokens()
      window.location.href = "/login"
      throw new ApiError("Unauthorized", res.status)
    }
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new ApiError(body.detail, res.status)
  }

  if (res.status === 204) return undefined as T
  return schema.parse(await res.json())
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

export async function apiPost<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  })
}

export async function apiPatch<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, {
    method: "PATCH",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  })
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE", headers: authHeaders() })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new ApiError(body.detail, res.status)
  }
}
