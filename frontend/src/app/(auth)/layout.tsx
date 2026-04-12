import { Activity } from "lucide-react"
import { cookies } from "next/headers"
import Link from "next/link"
import { redirect } from "next/navigation"
import { getTranslations } from "next-intl/server"

export default async function AuthLayout({ children }: { children: React.ReactNode }) {
  const t = await getTranslations("app")

  // Already authenticated — redirect to app
  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (hasAuth) redirect("/feed")

  return (
    <div className="flex min-h-[dvh] flex-col">
      <header className="border-b border-border bg-background pt-[env(safe-area-inset-top)]">
        <div className="flex h-[52px] items-center px-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Activity className="h-5 w-5" />
            <span>{t("title")}</span>
          </Link>
        </div>
      </header>
      <div className="flex flex-1 items-center justify-center px-4 py-8">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  )
}
