import { getTranslations } from "next-intl/server"
import { PlaceholderPage } from "@/components/placeholder-page"

export default async function TrainingPage() {
  const t = await getTranslations("training")
  return <PlaceholderPage title={t("title")} />
}
