interface RinkElement {
  code: string
  position: { x: number; y: number } | null
  timestamp?: number
  is_jump_pass?: boolean
}

const _RINK_W = 60
const _RINK_H = 30

function elementColor(code: string): string {
  if (code.includes("Sp")) return "#7c3aed"
  if (code.includes("StSq")) return "#16a34a"
  if (code.includes("ChSq")) return "#2563eb"
  return "#ea580c"
}

function _elementLabel(code: string): string {
  if (code.includes("Sp")) return "Вращение"
  if (code.includes("StSq")) return "Шаговая"
  if (code.includes("ChSq")) return "Хорео"
  return "Прыжок"
}

function elementMarker(el: RinkElement, x: number, y: number): string {
  const color = elementColor(el.code)

  if (el.code.includes("Sp")) {
    return `<circle cx="${x}" cy="${y}" r="1.3" fill="${color}" opacity="0.25" stroke="${color}" stroke-width="0.15"/>`
  }
  if (el.code.includes("StSq")) {
    return `<rect x="${x - 1.2}" y="${y - 0.5}" width="2.4" height="1" fill="none" stroke="${color}" stroke-width="0.15" stroke-dasharray="0.4,0.2" rx="0.2"/>`
  }
  if (el.code.includes("ChSq")) {
    return `<polygon points="${x},${y - 0.9} ${x + 0.9},${y} ${x},${y + 0.9} ${x - 0.9},${y}" fill="${color}" opacity="0.2" stroke="${color}" stroke-width="0.15"/>`
  }
  // Jumps — filled circle
  return `<circle cx="${x}" cy="${y}" r="0.7" fill="${color}" opacity="0.85"/>`
}

export function renderRink(elements: RinkElement[], options?: { width?: number }): string {
  const maxW = options?.width ?? 1200

  const parts: string[] = []

  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 60 30" style="max-width:${maxW}px">`,
  )

  // Ice surface
  parts.push(`<rect x="0" y="0" width="60" height="30" fill="#e8f0fe" rx="1"/>`)
  parts.push(
    `<rect x="1" y="1" width="58" height="28" fill="none" stroke="#2563eb" stroke-width="0.15" rx="0.5"/>`,
  )

  // Center line
  parts.push(
    `<line x1="30" y1="1" x2="30" y2="29" stroke="#dc2626" stroke-width="0.08" stroke-dasharray="0.5,0.5"/>`,
  )

  // Center circle
  parts.push(`<circle cx="30" cy="15" r="4.5" fill="none" stroke="#dc2626" stroke-width="0.08"/>`)
  parts.push(`<circle cx="30" cy="15" r="0.15" fill="#dc2626"/>`)

  // Zone lines
  parts.push(`<line x1="5" y1="1" x2="5" y2="29" stroke="#2563eb" stroke-width="0.06"/>`)
  parts.push(`<line x1="55" y1="1" x2="55" y2="29" stroke="#2563eb" stroke-width="0.06"/>`)

  // Corner circles
  for (const [cx, cy] of [
    [10, 7.5],
    [10, 22.5],
    [50, 7.5],
    [50, 22.5],
  ]) {
    parts.push(
      `<circle cx="${cx}" cy="${cy}" r="3" fill="none" stroke="#2563eb" stroke-width="0.06"/>`,
    )
    parts.push(`<circle cx="${cx}" cy="${cy}" r="0.15" fill="#dc2626"/>`)
  }

  // Flow lines: connect elements in chronological order (by timestamp)
  const sorted = elements
    .filter(el => el.position)
    .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))

  for (let i = 0; i < sorted.length - 1; i++) {
    const from = sorted[i].position
    const to = sorted[i + 1].position
    if (!from || !to) continue
    // Only draw if distance > threshold (skip very close elements)
    const dx = to.x - from.x
    const dy = to.y - from.y
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < 2) continue

    parts.push(
      `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="#94a3b8" stroke-width="0.08" stroke-dasharray="0.6,0.4" opacity="0.5"/>`,
    )
    // Small arrow at midpoint
    const mx = (from.x + to.x) / 2
    const my = (from.y + to.y) / 2
    const angle = Math.atan2(dy, dx)
    const arrowLen = 0.6
    const ax1 = mx - arrowLen * Math.cos(angle - 0.5)
    const ay1 = my - arrowLen * Math.sin(angle - 0.5)
    const ax2 = mx - arrowLen * Math.cos(angle + 0.5)
    const ay2 = my - arrowLen * Math.sin(angle + 0.5)
    parts.push(
      `<polyline points="${ax1},${ay1} ${mx},${my} ${ax2},${ay2}" fill="none" stroke="#94a3b8" stroke-width="0.08" opacity="0.5"/>`,
    )
  }

  // Elements with labels
  for (let i = 0; i < sorted.length; i++) {
    const el = sorted[i]
    const pos = el.position
    if (!pos) continue
    const x = pos.x
    const y = pos.y
    const color = elementColor(el.code)
    const num = i + 1

    parts.push(elementMarker(el, x, y))

    // Number badge — white circle with number
    parts.push(
      `<circle cx="${x + 1.2}" cy="${y - 1.0}" r="0.7" fill="white" stroke="${color}" stroke-width="0.12"/>`,
    )
    parts.push(
      `<text x="${x + 1.2}" y="${y - 0.65}" text-anchor="middle" font-size="0.85" fill="${color}" font-weight="bold">${num}</text>`,
    )

    // Element code below marker
    parts.push(
      `<text x="${x}" y="${y + 1.8}" text-anchor="middle" font-size="0.9" fill="#334155" font-weight="600">${el.code}</text>`,
    )
  }

  // Legend
  const ly = 28.5
  const legendItems = [
    { code: "3Lz", label: "Прыжок" },
    { code: "CSp4", label: "Вращение" },
    { code: "StSq4", label: "Шаговая" },
    { code: "ChSq1", label: "Хорео" },
  ]
  let lx = 2
  for (const item of legendItems) {
    const _color = elementColor(item.code)
    parts.push(elementMarker({ code: item.code } as RinkElement, lx + 0.4, ly))
    parts.push(
      `<text x="${lx + 1.6}" y="${ly + 0.35}" font-size="0.7" fill="#64748b">${item.label}</text>`,
    )
    lx += 12
  }

  parts.push("</svg>")
  return parts.join("\n")
}
