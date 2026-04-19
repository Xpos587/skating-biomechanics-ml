"""SVG rink diagram renderer.

Generates a top-down orthographic view of a 60m x 30m ice rink
with element markers, labels, and connecting paths.
"""

from __future__ import annotations


def _f(v: float) -> str:
    """Format a float, stripping unnecessary trailing zeros."""
    return f"{v:g}"


def render_rink(
    elements: list[dict],
    *,
    width: int = 1200,
    height: int = 600,
    rink_width: float = 60.0,
    rink_height: float = 30.0,
) -> str:
    """Render a rink diagram as SVG string.

    Args:
        elements: list of dicts with "code", "position" ({x, y}), "timestamp".
        width: SVG width in pixels.
        height: SVG height in pixels.
        rink_width: Rink width in metres (default Olympic 60m).
        rink_height: Rink height in metres (default Olympic 30m).

    Returns:
        SVG string.
    """
    rink_w, rink_h = rink_width, rink_height
    sx = rink_w / 60.0
    sy = rink_h / 30.0
    parts: list[str] = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {_f(rink_w)} {_f(rink_h)}">'
    )
    parts.append(
        f'<rect x="0" y="0" width="{_f(rink_w)}" height="{_f(rink_h)}" fill="#e8f0fe" rx="{_f(1 * sx)}"/>'
    )
    parts.append(
        f'<rect x="{_f(1 * sx)}" y="{_f(1 * sy)}" width="{_f(58 * sx)}" height="{_f(28 * sy)}" fill="none" stroke="#2563eb" stroke-width="0.15" rx="{_f(0.5 * sx)}"/>'
    )
    parts.append(
        f'<line x1="{_f(30 * sx)}" y1="{_f(1 * sy)}" x2="{_f(30 * sx)}" y2="{_f(29 * sy)}" stroke="#dc2626" stroke-width="0.1" stroke-dasharray="{_f(0.5 * sx)},{_f(0.5 * sx)}"/>'
    )
    parts.append(
        f'<circle cx="{_f(30 * sx)}" cy="{_f(15 * sy)}" r="{_f(4.5 * sx)}" fill="none" stroke="#dc2626" stroke-width="0.1"/>'
    )
    parts.append(
        f'<circle cx="{_f(30 * sx)}" cy="{_f(15 * sy)}" r="{_f(0.15 * sx)}" fill="#dc2626"/>'
    )
    parts.append(
        f'<line x1="{_f(5 * sx)}" y1="{_f(1 * sy)}" x2="{_f(5 * sx)}" y2="{_f(29 * sy)}" stroke="#2563eb" stroke-width="0.08"/>'
    )
    parts.append(
        f'<line x1="{_f(55 * sx)}" y1="{_f(1 * sy)}" x2="{_f(55 * sx)}" y2="{_f(29 * sy)}" stroke="#2563eb" stroke-width="0.08"/>'
    )

    for cx, cy in [(10, 7.5), (10, 22.5), (50, 7.5), (50, 22.5)]:
        parts.append(
            f'<circle cx="{_f(cx * sx)}" cy="{_f(cy * sy)}" r="{_f(3 * sx)}" fill="none" stroke="#2563eb" stroke-width="0.08"/>'
        )
        parts.append(
            f'<circle cx="{_f(cx * sx)}" cy="{_f(cy * sy)}" r="{_f(0.15 * sx)}" fill="#dc2626"/>'
        )

    for i, el in enumerate(elements):
        pos = el.get("position")
        if not pos:
            continue
        x, y = pos["x"] * sx, pos["y"] * sy
        code = el.get("code", "")

        is_spin = "Sp" in code
        is_step = "StSq" in code
        is_choreo = "ChSq" in code

        if is_spin:
            parts.append(
                f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(1.2 * sx)}" fill="#9333ea" opacity="0.3" stroke="#9333ea" stroke-width="0.1"/>'
            )
            color = "#9333ea"
        elif is_step:
            parts.append(
                f'<rect x="{_f(x - 1 * sx)}" y="{_f(y - 0.5 * sy)}" width="{_f(2 * sx)}" height="{_f(1 * sy)}" fill="none" stroke="#16a34a" stroke-width="0.1" stroke-dasharray="{_f(0.3 * sx)},{_f(0.2 * sx)}"/>'
            )
            color = "#16a34a"
        elif is_choreo:
            parts.append(
                f'<polygon points="{_f(x)},{_f(y - 0.8 * sy)} {_f(x + 0.8 * sx)},{_f(y)} {_f(x)},{_f(y + 0.8 * sy)} {_f(x - 0.8 * sx)},{_f(y)}" fill="#2563eb" opacity="0.3" stroke="#2563eb" stroke-width="0.1"/>'
            )
            color = "#2563eb"
        else:
            parts.append(
                f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.6 * sx)}" fill="#ea580c" opacity="0.8"/>'
            )
            color = "#ea580c"

        parts.append(
            f'<text x="{_f(x)}" y="{_f(y - 1.2 * sy)}" text-anchor="middle" font-size="{_f(1.2 * sx)}" fill="{color}" font-weight="bold">{code}</text>'
        )
        parts.append(
            f'<text x="{_f(x)}" y="{_f(y + 0.3 * sy)}" text-anchor="middle" font-size="{_f(0.7 * sx)}" fill="#666">{i + 1}</text>'
        )

        if i < len(elements) - 1:
            next_pos = elements[i + 1].get("position")
            if next_pos:
                nx, ny = next_pos["x"] * sx, next_pos["y"] * sy
                parts.append(
                    f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(nx)}" y2="{_f(ny)}" '
                    f'stroke="#94a3b8" stroke-width="0.06" stroke-dasharray="{_f(0.3 * sx)},{_f(0.2 * sx)}" opacity="0.6"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts)
