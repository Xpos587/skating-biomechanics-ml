import type { ComponentProps } from "react"

const inputClasses =
  "w-full rounded-[0.5rem] border border-input bg-secondary px-3 py-2 text-sm transition-colors duration-200 placeholder:text-muted-foreground focus-visible:border-foreground disabled:opacity-50"

export function FormField({
  label,
  id,
  ...props
}: { label: string; id: string } & ComponentProps<"input">) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium text-foreground">
        {label}
      </label>
      <input id={id} {...props} className={`${inputClasses} ${props.className ?? ""}`} />
    </div>
  )
}

export function FormSelect({
  label,
  id,
  children,
  ...props
}: { label: string; id: string; children: React.ReactNode } & ComponentProps<"select">) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium text-foreground">
        {label}
      </label>
      <select id={id} {...props} className={`${inputClasses} ${props.className ?? ""}`}>
        {children}
      </select>
    </div>
  )
}

export function FormTextarea({
  label,
  id,
  ...props
}: { label: string; id: string } & ComponentProps<"textarea">) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium text-foreground">
        {label}
      </label>
      <textarea id={id} {...props} className={`${inputClasses} ${props.className ?? ""}`} />
    </div>
  )
}
