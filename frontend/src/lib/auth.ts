/**
 * Auth API: schemas, token helpers, and endpoint wrappers.
 */

import { z } from "zod"
import { ApiError, apiFetch, clearTokens, getRefreshToken, setTokens } from "@/lib/api-client"

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

export const RegisterRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
  display_name: z.string().max(100).optional(),
})

export const LoginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

export const TokenResponseSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal("bearer"),
})

export const UserResponseSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  display_name: z.string().nullable(),
  avatar_url: z.string().nullable(),
  bio: z.string().nullable(),
  height_cm: z.number().int().nullable(),
  weight_kg: z.number().nullable(),
  language: z.string(),
  timezone: z.string(),
  theme: z.string(),
  is_active: z.boolean(),
  created_at: z.string(),
})

export const UpdateProfileRequestSchema = z.object({
  display_name: z.string().max(100).optional().nullable(),
  bio: z.string().optional().nullable(),
  height_cm: z.number().int().min(50).max(250).optional().nullable(),
  weight_kg: z.number().min(20).max(300).optional().nullable(),
})

export const UpdateSettingsRequestSchema = z.object({
  language: z.string().max(10).optional().nullable(),
  timezone: z.string().max(50).optional().nullable(),
  theme: z.enum(["light", "dark", "system"]).optional().nullable(),
})

export type RegisterRequest = z.infer<typeof RegisterRequestSchema>
export type LoginRequest = z.infer<typeof LoginRequestSchema>
export type TokenResponse = z.infer<typeof TokenResponseSchema>
export type UserResponse = z.infer<typeof UserResponseSchema>
export type UpdateProfileRequest = z.infer<typeof UpdateProfileRequestSchema>
export type UpdateSettingsRequest = z.infer<typeof UpdateSettingsRequestSchema>

// Re-export token helpers for consumers
export { clearTokens, getAccessToken, getRefreshToken, setTokens } from "@/lib/api-client"

// ---------------------------------------------------------------------------
// Auth API
// ---------------------------------------------------------------------------

const JSON_POST = { "Content-Type": "application/json" }

export async function register(data: RegisterRequest): Promise<TokenResponse> {
  return apiFetch("/auth/register", TokenResponseSchema, {
    method: "POST",
    auth: false,
    headers: JSON_POST,
    body: JSON.stringify(data),
  })
}

export async function login(data: LoginRequest): Promise<TokenResponse> {
  return apiFetch("/auth/login", TokenResponseSchema, {
    method: "POST",
    auth: false,
    headers: JSON_POST,
    body: JSON.stringify(data),
  })
}

export async function refreshToken(refresh: string): Promise<TokenResponse> {
  try {
    return await apiFetch("/auth/refresh", TokenResponseSchema, {
      method: "POST",
      auth: false,
      headers: JSON_POST,
      body: JSON.stringify({ refresh_token: refresh }),
    })
  } catch {
    clearTokens()
    throw new ApiError("Session expired", 401)
  }
}

export async function logout(): Promise<void> {
  const refresh = getRefreshToken()
  if (refresh) {
    await fetch("/api/v1/auth/logout", {
      method: "POST",
      headers: JSON_POST,
      body: JSON.stringify({ refresh_token: refresh }),
    }).catch(() => {})
  }
  clearTokens()
}

export async function fetchMe(): Promise<UserResponse> {
  return apiFetch("/users/me", UserResponseSchema)
}

export async function updateProfile(data: UpdateProfileRequest): Promise<UserResponse> {
  return apiFetch("/users/me", UserResponseSchema, {
    method: "PATCH",
    headers: JSON_POST,
    body: JSON.stringify(data),
  })
}

export async function updateSettings(data: UpdateSettingsRequest): Promise<UserResponse> {
  return apiFetch("/users/me/settings", UserResponseSchema, {
    method: "PATCH",
    headers: JSON_POST,
    body: JSON.stringify(data),
  })
}
