export const ELEMENT_TYPE_KEYS = [
  "three_turn",
  "waltz_jump",
  "toe_loop",
  "flip",
  "salchow",
  "loop",
  "lutz",
  "axel",
] as const

export type ElementType = (typeof ELEMENT_TYPE_KEYS)[number]
