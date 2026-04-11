import { Activity } from "lucide-react"
import type { Metadata } from "next"
import Link from "next/link"
import { NextIntlClientProvider } from "next-intl"
import { getLocale, getMessages, getTranslations } from "next-intl/server"
import { AppNav } from "@/components/app-nav"
import { Toaster } from "@/components/ui/sonner"
import { Providers } from "./providers"
import "./globals.css"

export async function generateMetadata(): Promise<Metadata> {
  const t = await getTranslations("app")
  return {
    title: t("titleFull"),
    description: "ML-based AI coach for figure skating",
  }
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const locale = await getLocale()
  const messages = await getMessages()
  const t = await getTranslations("app")

  return (
    <html lang={locale} suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <NextIntlClientProvider messages={messages}>
          <Providers>
            <header className="sticky top-0 z-50 border-b border-border bg-background">
              <div className="mx-auto flex h-[60px] max-w-7xl items-center justify-between px-4">
                <Link href="/" className="flex items-center gap-2 font-semibold">
                  <Activity className="h-5 w-5" />
                  <span className="hidden sm:inline">{t("title")}</span>
                </Link>
                <AppNav />
              </div>
            </header>
            <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-8">{children}</main>
            <footer className="px-4 py-6 text-center text-xs text-muted-foreground">
              {t("footer")}
            </footer>
            <Toaster richColors position="bottom-right" />
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  )
}
