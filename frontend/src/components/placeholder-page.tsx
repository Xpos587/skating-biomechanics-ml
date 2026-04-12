import { getTranslations } from "next-intl/server"

export async function PlaceholderPage({ title }: { title: string }) {
  const t = await getTranslations("placeholder")
  return (
    <div className="mx-auto max-w-4xl p-6">
      <h2 className="nike-h2 mb-4">{title}</h2>
      <p className="text-muted-foreground">{t("inDevelopment")}</p>
    </div>
  )
}
