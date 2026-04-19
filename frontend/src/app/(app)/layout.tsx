import { Activity } from "lucide-react"
import { cookies } from "next/headers"
import Link from "next/link"
import { redirect } from "next/navigation"
import { getTranslations } from "next-intl/server"
import { AppNav } from "@/components/app-nav"
import { BottomDock } from "@/components/layout/bottom-dock"

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const t = await getTranslations("app")

  // Server-side auth gate — skip when NEXT_PUBLIC_SKIP_AUTH=true
  const skipAuth = process.env.NEXT_PUBLIC_SKIP_AUTH === "true"
  if (!skipAuth) {
    const hasAuth = (await cookies()).get("sb_auth")?.value
    if (!hasAuth) redirect("/login")
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 border-b border-border bg-background">
        <div className="mx-auto flex h-[52px] max-w-5xl items-center justify-between px-4 sm:px-6">
          <Link href="/feed" className="flex items-center gap-2 font-semibold">
            <Activity className="h-5 w-5" />
            <span className="hidden sm:inline">{t("title")}</span>
          </Link>
          <AppNav />
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-4 pb-24 sm:px-6 sm:py-6 sm:pb-8">
        {children}
      </main>
      <BottomDock />
    </div>
  )
}
