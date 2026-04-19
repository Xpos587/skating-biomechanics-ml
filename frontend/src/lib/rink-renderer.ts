interface RinkElement {
  code: string
  position?: { x: number; y: number }
  timestamp?: number
}

const RINK_W = 60
const RINK_H = 30

function elementColor(code: string): string {
  if (code.includes("Sp")) return "#9333ea"
  if (code.includes("StSq")) return "#16a34a"
  if (code.includes("ChSq")) return "#2563eb"
  return "#ea580c"
}

function elementShape(el: RinkElement, x: number, y: number): string {
  const color = elementColor(el.code)

  if (el.code.includes("Sp")) {
    return `<circle cx="${x}" cy="${y}" r="1.2" fill="${color}" opacity="0.3" stroke="${color}" stroke-width="0.1"/>`
  }
  if (el.code.includes("StSq")) {
    return `<rect x="${x - 1}" y="${y - 0.5}" width="2" height="1" fill="none" stroke="${color}" stroke-width="0.1" stroke-dasharray="0.3,0.2"/>`
  }
  if (el.code.includes("ChSq")) {
    return `<polygon points="${x},${y - 0.8} ${x + 0.8},${y} ${x},${y + 0.8} ${x - 0.8},${y}" fill="${color}" opacity="0.3" stroke="${color}" stroke-width="0.1"/>`
  }
  return `<circle cx="${x}" cy="${y}" r="0.6" fill="${color}" opacity="0.8"/>`
}

export function renderRink(
  elements: RinkElement[],
  options?: { width?: number; height?: number },
): string {
  const { width = 1200, height = 600 } = options ?? {}

  const parts: string[] = []

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 60 30">`)
  parts.push(`<rect x="0" y="0" width="60" height="30" fill="#e8f0fe" rx="1"/>`)
  parts.push(`<rect x="1" y="1" width="58" height="28" fill="none" stroke="#2563eb" stroke-width="0.15" rx="0.5"/>`)
  parts.push(`<line x1="30" y1="1" x2="30" y2="29" stroke="#dc2626" stroke-width="0.1" stroke-dasharray="0.5,0.5"/>`)
  parts.push(`<circle cx="30" cy="15" r="4.5" fill="none" stroke="#dc2626" stroke-width="0.1"/>`)
  parts.push(`<circle cx="30" cy="15" r="0.15" fill="#dc2626"/>`)
  parts.push(`<line x1="5" y1="1" x2="5" y2="29" stroke="#2563eb" stroke-width="0.08"/>`)
  parts.push(`<line x1="55" y1="1" x2="55" y2="29" stroke="#2563eb" stroke-width="0.08"/>`)

  for (const [cx, cy] of [[10, 7.5], [10, 22.5], [50, 7.5], [50, 22.5]]) {
    parts.push(`<circle cx="${cx}" cy="${cy}" r="3" fill="none" stroke="#2563eb" stroke-width="0.08"/>`)
    parts.push(`<circle cx="${cx}" cy="${cy}" r="0.15" fill="#dc2626"/>`)
  }

  for (let i = 0; i < elements.length; i++) {
    const el = elements[i]
    const pos = el.position
    if (!pos) continue

    const x = pos.x
    const y = pos.y
    const color = elementColor(el.code)

    parts.push(elementShape(el, x, y))
    parts.push(`<text x="${x}" y="${y - 1.2}" text-anchor="middle" font-size="1.2" fill="${color}" font-weight="bold">${el.code}</text>`)
    parts.push(`<text x="${x}" y="${y + 0.3}" text-anchor="middle" font-size="0.7" fill="#666">${i + 1}</text>`)

    if (i < elements.length - 1) {
      const nextPos = elements[i + 1].position
      if (nextPos) {
        parts.push(`<line x1="${x}" y1="${y}" x2="${nextPos.x}" y2="${nextPos.y}" stroke="#94a3b8" stroke-width="0.06" stroke-dasharray="0.3,0.2" opacity="0.6"/>`)
      }
    }
  }

  parts.push("</svg>")
  return parts.join("\n")
}
