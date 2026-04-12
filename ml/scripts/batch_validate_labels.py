#!/usr/bin/env python3
"""Generate diverse labeled HALPE26 visualizations for human validation.

Bubble-style labels: line from keypoint to text on black background rectangle.
Uses Pillow for Cyrillic text rendering (cv2.putText doesn't support Unicode).
Saves to /tmp/halpe26_labeled_*.jpg.

Usage:
    uv run python scripts/batch_validate_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skating_ml.datasets.coco_builder import merge_coco_foot_keypoints
from skating_ml.datasets.projector import project_foot_frame, validate_foot_projection

DATA_ROOT = Path("data/datasets/athletepose3d")

FONT_PATH = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_SIZE = 22

HALPE26_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (16, 20),
    (17, 18),
    (18, 19),
    (20, 21),
    (21, 22),
    (23, 24),
    (24, 25),
]

HALPE26_NAMES = [
    "нос",
    "Л. глаз",
    "П. глаз",
    "Л. ухо",
    "П. ухо",
    "Л. плечо",
    "П. плечо",
    "Л. локоть",
    "П. локоть",
    "Л. кисть",
    "П. кисть",
    "Л. таз",
    "П. таз",
    "Л. колено",
    "П. колено",
    "Л. голеност.",
    "П. голеност.",
    "Л. пятка",
    "Л. бол. палец",
    "Л. мал. палец",
    "П. пятка",
    "П. бол. палец",
    "П. мал. палец",
    "Л. внутр. глаз",
    "П. внутр. глаз",
    "рот",
]

# Color per group: green=COCO 17, orange=foot 6, magenta=face 3
KP_COLORS = [(0, 255, 0)] * 17 + [(0, 165, 255)] * 6 + [(255, 0, 255)] * 3

# Label offset direction: push labels away from body center
LABEL_OFFSETS = {
    # Head — push up
    0: (0, -1),
    1: (-0.5, -1),
    2: (0.5, -1),
    3: (-1, -0.7),
    4: (1, -0.7),
    # Upper body — push outward
    5: (-1, -0.5),
    6: (1, -0.5),
    7: (-1, 0),
    8: (1, 0),
    9: (-1, 0.3),
    10: (1, 0.3),
    # Torso — push sideways
    11: (-0.7, 0.3),
    12: (0.7, 0.3),
    # Legs — push sideways
    13: (-0.8, 0),
    14: (0.8, 0),
    15: (-0.8, 0.3),
    16: (0.8, 0.3),
    # Feet — push outward/down
    17: (-1, 0.5),
    18: (-1, 0.8),
    19: (-0.7, 1),
    20: (1, 0.5),
    21: (1, 0.8),
    22: (0.7, 1),
    # Face dupes — push up
    23: (0, -1.2),
    24: (0.3, -1.2),
    25: (-0.3, -1.2),
}

LABEL_OFFSET_LEN = 60


def _load_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, FONT_SIZE)


def find_sequence(split: str, name_prefix: str, cam: int) -> Path | None:
    base = DATA_ROOT / "videos" / split
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        npy = d / f"{name_prefix}_cam_{cam}.npy"
        coco_npy = d / f"{name_prefix}_cam_{cam}_coco.npy"
        mp4 = d / f"{name_prefix}_cam_{cam}.mp4"
        json_f = d / f"{name_prefix}_cam_{cam}.json"
        if npy.exists() and coco_npy.exists() and mp4.exists() and json_f.exists():
            return npy
    return None


def _resolve_overlaps(
    labels: list[tuple[int, int, int, int]],
    max_w: int,
    max_h: int,
) -> list[tuple[int, int, int, int]]:
    """Push overlapping label rectangles apart.

    labels: list of (lx, ly, tw, th) — lx,ly is top-left of text
    Returns adjusted list.
    """
    placed: list[tuple[int, int, int, int]] = []
    pad = 6

    for lx, ly, tw, th in labels:
        cur_lx, cur_ly = lx, ly
        rect = (cur_lx - pad, cur_ly - th - pad, tw + 2 * pad, th + 2 * pad + 1)
        for _iter in range(20):
            collision = False
            for plx, ply, ptw, pth in placed:
                prect = (plx - pad, ply - pth - pad, ptw + 2 * pad, pth + 2 * pad + 1)
                if _rects_overlap(rect, prect):
                    collision = True
                    cx_a = rect[0] + rect[2] / 2
                    cy_a = rect[1] + rect[3] / 2
                    cx_b = prect[0] + prect[2] / 2
                    cy_b = prect[1] + prect[3] / 2
                    dx = cx_a - cx_b
                    dy = cy_a - cy_b
                    dist = max((dx * dx + dy * dy) ** 0.5, 1.0)
                    shift = 20
                    cur_lx = int(cur_lx + dx / dist * shift)
                    cur_ly = int(cur_ly + dy / dist * shift)
                    cur_lx = max(4, min(max_w - tw - 4, cur_lx))
                    cur_ly = max(th + 4, min(max_h - 4, cur_ly))
                    rect = (cur_lx - pad, cur_ly - th - pad, tw + 2 * pad, th + 2 * pad + 1)
                    break
            if not collision:
                break
        placed.append((cur_lx, cur_ly, tw, th))
    return placed


def _rects_overlap(a: tuple, b: tuple) -> bool:
    return not (
        a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1]
    )


def draw_bubble_labels(frame_bgr: np.ndarray, pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Draw keypoints with Russian bubble labels using PIL for text rendering."""
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]
    font = _load_font()

    # Draw skeleton (OpenCV, behind everything)
    for a, b in HALPE26_SKELETON:
        if vis[a] > 0.1 and vis[b] > 0.1:
            cv2.line(
                overlay,
                (int(pts[a, 0]), int(pts[a, 1])),
                (int(pts[b, 0]), int(pts[b, 1])),
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

    # Collect visible keypoints
    kp_vis = []
    for i in range(26):
        if vis[i] < 0.1:
            continue
        px, py = int(pts[i, 0]), int(pts[i, 1])
        kp_vis.append((i, px, py))

    # Compute initial label positions + sizes via PIL
    label_entries = []  # (i, px, py, lx, ly, tw, th, label)
    for i, px, py in kp_vis:
        dx, dy = LABEL_OFFSETS.get(i, (0.5, -0.8))
        norm = (dx**2 + dy**2) ** 0.5
        dx, dy = dx / norm, dy / norm
        offset = LABEL_OFFSET_LEN

        lx = int(px + dx * offset)
        ly = int(py + dy * offset)
        lx = max(10, min(w - 10, lx))
        ly = max(10, min(h - 10, ly))

        label = HALPE26_NAMES[i]
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Anchor text: left-align for right-going, right-align for left-going
        if dx > 0.3:
            pass  # lx stays as left edge
        elif dx < -0.3:
            lx = lx - tw
        else:
            lx = lx - tw // 2
        ly = ly + th // 2

        lx = max(4, min(w - tw - 4, lx))
        ly = max(th + 4, min(h - 4, ly))

        label_entries.append((i, px, py, lx, ly, tw, th, label))

    # Resolve overlaps
    label_rects = [(lx, ly, tw, th) for _, _, _, lx, ly, tw, th, _ in label_entries]
    resolved = _resolve_overlaps(label_rects, w, h)

    # Draw lines from keypoints to labels (cv2, before PIL conversion)
    for entry, (rlx, rly, rtw, rth) in zip(label_entries, resolved, strict=False):
        i, px, py = entry[0], entry[1], entry[2]
        label_cx = rlx + rtw // 2
        label_cy = rly - rth // 2
        cv2.line(overlay, (px, py), (label_cx, label_cy), (200, 200, 200), 1, cv2.LINE_AA)

    # Draw semi-transparent small dots via alpha blending
    dot_layer = np.zeros_like(overlay)
    for i, px, py in kp_vis:
        color = KP_COLORS[i]
        radius = 3 if i < 17 else 4
        cv2.circle(dot_layer, (px, py), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(dot_layer, 0.6, overlay, 1.0, 0, overlay)

    # Draw text labels with PIL (for Cyrillic support)
    pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for entry, (rlx, rly, rtw, rth) in zip(label_entries, resolved, strict=False):
        _, _, _, _, _, _, _, label = entry

        # Black background rectangle
        pad = 4
        draw.rectangle(
            [rlx - pad, rly - rth - pad, rlx + rtw + pad, rly + pad + 1],
            fill=(0, 0, 0),
        )

        # White text
        draw.text((rlx, rly - rth), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_one(seq_name: str, cam: int, frame_idx: int, split: str, out_path: str) -> bool:
    npy_path = find_sequence(split, seq_name, cam)
    if npy_path is None:
        print(f"  SKIP: {seq_name} cam_{cam} not found")
        return False

    kp3d = np.load(npy_path)
    coco_kps = np.load(npy_path.parent / f"{npy_path.stem}_coco.npy")

    if frame_idx >= len(kp3d):
        print(f"  SKIP: {seq_name} frame {frame_idx} out of range (0-{len(kp3d) - 1})")
        return False

    json_path = npy_path.with_suffix(".json")
    with json_path.open() as f:
        meta = json.load(f)

    with (DATA_ROOT / "cam_param.json").open() as f:
        cam_params = json.load(f)

    cam_key = meta["cam"]
    if cam_key not in cam_params:
        print(f"  SKIP: {seq_name} cam key {cam_key} not found")
        return False
    cam = cam_params[cam_key]

    foot_2d = project_foot_frame(kp3d[frame_idx], cam)
    coco_2d = coco_kps[frame_idx]

    # Validate foot projections (sets invalid points to NaN in-place)
    pre_valid = int(np.sum(~np.isnan(foot_2d[:, 0])))
    validate_foot_projection(foot_2d, coco_2d)
    post_valid = int(np.sum(~np.isnan(foot_2d[:, 0])))
    foot_rejected = pre_valid - post_valid

    pts, vis = merge_coco_foot_keypoints(coco_2d, foot_2d)

    mp4_path = npy_path.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  SKIP: {seq_name} cannot read frame {frame_idx}")
        return False

    overlay = draw_bubble_labels(frame, pts, vis)
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    valid = int((vis > 0.1).sum())
    foot_valid = int((vis[17:23] > 0.1).sum())
    print(
        f"  OK: {out_path} ({valid}/26 visible, foot: {foot_valid}/6 valid, {foot_rejected} rejected)"
    )
    return True


def main():
    configs = [
        # (seq_name, cam, frame_idx, split, output_suffix)  # noqa: ERA001
        ("Axel_1", 1, 80, "train_set", "axel1_cam1_f80"),
        ("Axel_3", 5, 120, "train_set", "axel3_cam5_f120"),
        ("Axel_10", 8, 50, "train_set", "axel10_cam8_f50"),
        ("Lutz_1", 2, 90, "train_set", "lutz1_cam2_f90"),
        ("Lutz_3", 7, 100, "train_set", "lutz3_cam7_f100"),
        ("Salchow_1", 1, 70, "train_set", "salchow1_cam1_f70"),
        ("Salchow_2", 4, 40, "train_set", "salchow2_cam4_f40"),
        ("Flip_1", 3, 85, "train_set", "flip1_cam3_f85"),
        ("Flip_2", 6, 60, "train_set", "flip2_cam6_f60"),
        ("Toeloop_1", 2, 75, "train_set", "toeloop1_cam2_f75"),
        ("Toeloop_3", 9, 95, "train_set", "toeloop3_cam9_f95"),
        ("Loop_1", 1, 65, "train_set", "loop1_cam1_f65"),
        ("Loop_2", 5, 100, "train_set", "loop2_cam5_f100"),
        ("Axel_5", 3, 130, "train_set", "axel5_cam3_f130"),
        ("Lutz_4", 10, 80, "train_set", "lutz4_cam10_f80"),
    ]

    print(f"Generating {len(configs)} labeled images (PIL/Cyrillic)...\n")

    ok = 0
    for seq, cam, frame, split, suffix in configs:
        out = f"/tmp/halpe26_labeled_{suffix}.jpg"
        if process_one(seq, cam, frame, split, out):
            ok += 1

    print(f"\nDone: {ok}/{len(configs)} images saved to /tmp/halpe26_labeled_*.jpg")


if __name__ == "__main__":
    main()
