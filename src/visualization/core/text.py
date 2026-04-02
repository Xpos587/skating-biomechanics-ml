"""Text rendering utilities with Cyrillic support.

Provides functions for:
- Rendering Cyrillic text using Pillow
- Drawing text boxes with backgrounds
- Measuring text size for layout
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from src.visualization.config import (
    font_color,
    font_path,
    font_scale,
    font_thickness,
    hud_bg_alpha,
    hud_bg_color,
    hud_padding,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
Position = tuple[int, int]  # (x, y) pixel coordinates


# =============================================================================
# FONT CACHE
# =============================================================================

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _get_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Get cached font or load new one.

    Args:
        font_path: Path to TTF font file.
        font_size: Font size in points.

    Returns:
        Pillow ImageFont object.
    """
    cache_key = (font_path, font_size)

    if cache_key not in _font_cache:
        try:
            _font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
        except OSError:
            # Fallback to default font
            _font_cache[cache_key] = ImageFont.load_default()

    return _font_cache[cache_key]


# =============================================================================
# TEXT MEASUREMENT
# =============================================================================


def measure_text_size_cv2(
    text: str,
    font_scale: float = font_scale,
    thickness: int = font_thickness,
) -> tuple[int, int]:
    """Measure text size using OpenCV.

    Args:
        text: Text string to measure.
        font_scale: Font scale factor.
        thickness: Text thickness.

    Returns:
        (width, height) in pixels.

    Note:
        Only works for ASCII text. Use measure_text_size_pillow() for Cyrillic.
    """
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    return (width, height + baseline)


def measure_text_size_pillow(
    text: str,
    font_path: str = font_path,
    font_size: int = 32,
) -> tuple[int, int]:
    """Measure text size using Pillow (supports Cyrillic).

    Args:
        text: Text string to measure.
        font_path: Path to TTF font file.
        font_size: Font size in points.

    Returns:
        (width, height) in pixels.

    Example:
        >>> measure_text_size_pillow("Привет", font_path, 32)
        (52, 32)
    """
    font = _get_font(font_path, font_size)

    # Create temporary image to measure text
    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Get bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return (width, height)


def measure_text_size(
    text: str,
    font_path: str = font_path,
    font_size: int = 32,
    use_cv2: bool = False,
) -> tuple[int, int]:
    """Measure text size, auto-detecting best method.

    Args:
        text: Text string to measure.
        font_path: Path to TTF font file.
        font_size: Font size in points.
        use_cv2: Force OpenCV measurement (ASCII only).

    Returns:
        (width, height) in pixels.
    """
    # Check for non-ASCII characters
    has_cyrillic = any(ord(c) > 127 for c in text)

    if use_cv2 or not has_cyrillic:
        # Use OpenCV for ASCII-only text
        return measure_text_size_cv2(text, font_scale, font_thickness)
    else:
        # Use Pillow for Cyrillic text
        return measure_text_size_pillow(text, font_path, font_size)


# =============================================================================
# TEXT RENDERING
# =============================================================================


def render_cyrillic_text(
    frame: Frame,
    text: str,
    position: Position,
    font_path: str = font_path,
    font_size: int = 32,
    color: tuple[int, int, int] = font_color,
    background: tuple[int, int, int] | None = None,
    background_alpha: float = hud_bg_alpha,
) -> Frame:
    """Render Cyrillic text on frame using Pillow.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        text: Text string to render (supports Cyrillic).
        position: (x, y) top-left position.
        font_path: Path to TTF font file.
        font_size: Font size in points.
        color: Text color (BGR).
        background: Background color (BGR), or None for transparent.
        background_alpha: Background transparency [0, 1].

    Returns:
        Frame with text rendered (modified in place).

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> render_cyrillic_text(frame, "Привет мир", (10, 30))
    """
    x, y = position

    # Convert BGR to RGB for Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Get font
    font = _get_font(font_path, font_size)

    # Create drawing context
    draw = ImageDraw.Draw(pil_img)

    # Convert BGR to RGB
    color_rgb = (color[2], color[1], color[0])

    # Draw background if specified
    if background is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        bg_bbox = (
            bbox[0] - hud_padding,
            bbox[1] - hud_padding,
            bbox[2] + hud_padding,
            bbox[3] + hud_padding,
        )

        # Create semi-transparent overlay
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        bg_rgb = (background[2], background[1], background[0])
        alpha_int = int(background_alpha * 255)
        overlay_draw.rectangle(bg_bbox, fill=(*bg_rgb, alpha_int))

        # Composite overlay onto image
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        pil_img = pil_img.convert("RGB")
        draw = ImageDraw.Draw(pil_img)

    # Draw text
    draw.text((x, y), text, font=font, fill=color_rgb)

    # Convert back to BGR
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return frame


def draw_text_box(
    frame: Frame,
    text: str,
    position: Position,
    font_scale: float = font_scale,
    thickness: int = font_thickness,
    color: tuple[int, int, int] = font_color,
    bg_color: tuple[int, int, int] = hud_bg_color,
    bg_alpha: float = hud_bg_alpha,
    padding: int = hud_padding,
    corner_radius: int = 5,
) -> tuple[Frame, Position]:
    """Draw text with background box.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        text: Text string to render.
        position: (x, y) top-left position.
        font_scale: Font scale factor for OpenCV.
        thickness: Text thickness.
        color: Text color (BGR).
        bg_color: Background color (BGR).
        bg_alpha: Background transparency [0, 1].
        padding: Padding around text.
        corner_radius: Corner radius for rounded rectangle.

    Returns:
        (frame, new_position) where new_position is the position for next text below.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_text_box(frame, "FPS: 30", (10, 30))
        (frame, (10, 55))  # Next text position
    """
    x, y = position

    # Measure text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    # Calculate box dimensions
    box_width = text_width + 2 * padding
    box_height = text_height + baseline + 2 * padding

    # Draw semi-transparent background
    if bg_alpha > 0:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            bg_color,
            -1,
            cv2.LINE_AA,
        )
        frame[:] = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)

    # Draw text
    cv2.putText(
        frame,
        text,
        (x + padding, y + padding + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # Calculate next position (below current box)
    new_position = (x, y + box_height + 5)

    return (frame, new_position)


def draw_text_multiline(
    frame: Frame,
    lines: list[str],
    position: Position,
    font_scale: float = font_scale,
    thickness: int = font_thickness,
    color: tuple[int, int, int] = font_color,
    bg_color: tuple[int, int, int] = hud_bg_color,
    bg_alpha: float = hud_bg_alpha,
    line_spacing: int = 5,
) -> Position:
    """Draw multiple text lines with background.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        lines: List of text strings.
        position: (x, y) top-left position.
        font_scale: Font scale factor.
        thickness: Text thickness.
        color: Text color (BGR).
        bg_color: Background color (BGR).
        bg_alpha: Background transparency [0, 1].
        line_spacing: Spacing between lines.

    Returns:
        Next position below all lines.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_text_multiline(frame, ["Line 1", "Line 2"], (10, 30))
        (10, 85)  # Next position
    """
    x, y = position

    for line in lines:
        frame, (x, y) = draw_text_box(
            frame,
            line,
            (x, y),
            font_scale,
            thickness,
            color,
            bg_color,
            bg_alpha,
        )
        y += line_spacing

    return (x, y)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def truncate_text(
    text: str,
    max_width: int,
    font_path: str = font_path,
    font_size: int = 32,
    ellipsis: str = "...",
) -> str:
    """Truncate text to fit within max width.

    Args:
        text: Text string to truncate.
        max_width: Maximum width in pixels.
        font_path: Path to TTF font file.
        font_size: Font size in points.
        ellipsis: String to append when truncated.

    Returns:
        Truncated text string.

    Example:
        >>> truncate_text("This is a very long text", 100)
        "This is..."
    """
    font = _get_font(font_path, font_size)

    # Create temporary image to measure text
    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Check if full text fits
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]

    if width <= max_width:
        return text

    # Binary search for truncation point
    left, right = 0, len(text)

    while left < right:
        mid = (left + right) // 2
        truncated = text[:mid] + ellipsis

        bbox = draw.textbbox((0, 0), truncated, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            left = mid + 1
        else:
            right = mid

    return text[:left] + ellipsis


def wrap_text(
    text: str,
    max_width: int,
    font_path: str = font_path,
    font_size: int = 32,
) -> list[str]:
    """Wrap text into multiple lines to fit within max width.

    Args:
        text: Text string to wrap.
        max_width: Maximum width in pixels.
        font_path: Path to TTF font file.
        font_size: Font size in points.

    Returns:
        List of text lines.

    Example:
        >>> wrap_text("This is a long text that needs wrapping", 100)
        ["This is a long", "text that needs", "wrapping"]
    """
    font = _get_font(font_path, font_size)

    # Create temporary image to measure text
    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)

    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word

        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines
