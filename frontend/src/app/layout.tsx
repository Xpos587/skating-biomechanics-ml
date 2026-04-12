import type { Metadata } from "next"
import { NextIntlClientProvider } from "next-intl"
import { getLocale, getMessages, getTranslations } from "next-intl/server"
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

  return (
    <html lang={locale} suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <NextIntlClientProvider messages={messages}>
          <Providers>
            {children}
            <Toaster richColors position="bottom-center" toastOptions={{ duration: 3000 }} />
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  )
}
