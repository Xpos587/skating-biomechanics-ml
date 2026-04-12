"use server"

import { cookies } from "next/headers"
import type { Locale } from "next-intl"

const COOKIE_NAME = "NEXT_LOCALE"
const COOKIE_MAX_AGE = 365 * 24 * 60 * 60 // 1 year

export async function setLocale(locale: Locale): Promise<void> {
  const store = await cookies()
  store.set(COOKIE_NAME, locale, {
    path: "/",
    maxAge: COOKIE_MAX_AGE,
    sameSite: "lax",
  })
}
