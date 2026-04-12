import { getTranslations } from "next-intl/server"
import { PlaceholderPage } from "@/components/placeholder-page"

export default async function SettingsPage() {
  const t = await getTranslations("settings")
  return <PlaceholderPage title={t("title")} />
}
