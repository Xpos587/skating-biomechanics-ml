import type { Metadata } from "next"
import { Providers } from "./providers"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Тренер — Фигурное катание",
  description: "ML-based AI coach for figure skating",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <Providers>
          <header className="border-b border-border px-6 py-3">
            <h1 className="text-lg font-semibold">AI Тренер — Фигурное катание</h1>
          </header>
          <main>{children}</main>
        </Providers>
      </body>
    </html>
  )
}
