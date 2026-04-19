import { describe, expect, it } from "vitest"
import { renderRink } from "../rink-renderer"

describe("renderRink", () => {
  it("returns SVG string with rink outline", () => {
    const svg = renderRink([])
    expect(svg).toContain("<svg")
    expect(svg).toContain("</svg>")
    expect(svg).toContain('width="1200"')
    expect(svg).toContain('viewBox="0 0 60 30"')
  })

  it("renders jump elements", () => {
    const svg = renderRink([{ code: "3Lz", position: { x: 30, y: 15 } }])
    expect(svg).toContain("3Lz")
    expect(svg).toContain("#ea580c")
  })

  it("renders spin elements", () => {
    const svg = renderRink([{ code: "CSp4", position: { x: 20, y: 10 } }])
    expect(svg).toContain("CSp4")
    expect(svg).toContain("#9333ea")
  })

  it("renders step sequences", () => {
    const svg = renderRink([{ code: "StSq4", position: { x: 40, y: 10 } }])
    expect(svg).toContain("StSq4")
    expect(svg).toContain("#16a34a")
  })

  it("renders choreo sequences", () => {
    const svg = renderRink([{ code: "ChSq1", position: { x: 15, y: 20 } }])
    expect(svg).toContain("ChSq1")
    expect(svg).toContain("#2563eb")
  })

  it("draws connecting paths between elements", () => {
    const svg = renderRink([
      { code: "3Lz", position: { x: 20, y: 15 } },
      { code: "3F", position: { x: 40, y: 15 } },
    ])
    expect(svg).toContain("x1=\"20\"")
    expect(svg).toContain("x2=\"40\"")
  })

  it("skips elements without position", () => {
    const svg = renderRink([{ code: "3Lz" }])
    expect(svg).not.toContain("3Lz")
  })

  it("supports custom width/height", () => {
    const svg = renderRink([], { width: 600, height: 300 })
    expect(svg).toContain('width="600"')
    expect(svg).toContain('height="300"')
  })
})
