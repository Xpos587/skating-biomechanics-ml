import { type RenderOptions, render } from "@testing-library/react"
import type { ReactElement } from "react"

// Custom render with providers (add QueryClientProvider later if needed)
export function renderWithProviders(ui: ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return render(ui, options)
}

// Re-export everything from RTL
export * from "@testing-library/react"
