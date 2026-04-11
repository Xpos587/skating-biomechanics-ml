export function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="mx-auto max-w-4xl p-6">
      <h2 className="mb-4 text-xl font-semibold">{title}</h2>
      <p className="text-muted-foreground">Раздел в разработке.</p>
    </div>
  )
}
