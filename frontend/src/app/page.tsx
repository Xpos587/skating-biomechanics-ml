import { cookies } from "next/headers"
import { redirect } from "next/navigation"

export default async function HomePage() {
  const hasAuth = (await cookies()).get("sb_auth")?.value
  redirect(hasAuth ? "/feed" : "/login")
}
