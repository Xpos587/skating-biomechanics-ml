"use client"

import { useChoreographyEditor } from "./store"

/**
 * Subscribes to Zustand store changes and debounces save calls.
 * Uses store.subscribe() directly — not useEffect.
 * Returns an unsubscribing function.
 */
export function startAutoSave(
  save: (data: {
    id: string
    title: string
    layout: { elements: Array<{ code: string; timestamp: number; goe: number }> }
  }) => void,
  isPending: () => boolean,
  delay = 500,
): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null
  let lastSnapshot = ""

  return useChoreographyEditor.subscribe(state => {
    const snapshot = `${state.title}|${state.elements.map(e => `${e.id}:${e.timestamp}:${e.goe}`).join(",")}`
    if (snapshot === lastSnapshot) return
    const pid = state.programId
    if (!pid || isPending()) return

    lastSnapshot = snapshot
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      const { layout } = state.getLayoutForSave()
      save({
        id: pid,
        title: state.title,
        layout: {
          elements: layout.map(el => ({ code: el.code, timestamp: el.timestamp, goe: el.goe })),
        },
      })
    }, delay)
  })
}
