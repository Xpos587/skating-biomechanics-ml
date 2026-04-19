import { cookies } from "next/headers"
import { redirect } from "next/navigation"

export default async function HomePage() {
  const skipAuth = process.env.NEXT_PUBLIC_SKIP_AUTH === "true"
  if (skipAuth) redirect("/feed")

  const hasAuth = (await cookies()).get("sb_auth")?.value
  redirect(hasAuth ? "/feed" : "/login")
}
