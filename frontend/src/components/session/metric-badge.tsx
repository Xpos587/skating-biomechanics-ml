export function MetricBadge({ text }: { text: string }) {
  return (
    <span
      className="ml-1.5 inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium"
      style={{
        backgroundColor: "oklch(var(--score-mid) / 0.15)",
        color: "oklch(var(--score-mid))",
      }}
    >
      PR {text}
    </span>
  )
}
