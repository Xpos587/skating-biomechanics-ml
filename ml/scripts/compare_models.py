#!/usr/bin/env python3
"""Compare pose estimation backends on the same video.

Runs each backend sequentially, renders skeleton overlays side-by-side,
and outputs per-frame quality metrics to CSV.

Usage:
    python scripts/compare_models.py video.mp4 --backends rtmlib,yolo
    python scripts/compare_models.py video.mp4 --backends rtmlib,yolo --no-viz
    python scripts/compare_models.py video.mp4 --backends rtmlib --select-person
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skating_ml.device import DeviceConfig

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TYPE_CHECKING

from skating_ml.pose_estimation import H36M_SKELETON_EDGES, H36Key
from skating_ml.types import PersonClick
from skating_ml.utils.geometry import angle_3pt
from skating_ml.utils.video_writer import H264Writer
from skating_ml.visualization.skeleton.drawer import draw_skeleton

if TYPE_CHECKING:
    from skating_ml.pose_estimation.rtmlib_extractor import TrackedExtraction

# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

BACKEND_LABELS: dict[str, str] = {
    "rtmlib": "RTMPose (rtmlib, HALPE26 26kp)",
}

BACKEND_CLI_CHOICES: dict[str, str] = {}


def _create_extractor(backend: str, conf_threshold: float = 0.3):
    """Create a pose extractor for the given backend name."""
    if backend == "rtmlib":
        from skating_ml.pose_estimation.rtmlib_extractor import RTMPoseExtractor

        return RTMPoseExtractor(
            output_format="normalized",
            conf_threshold=conf_threshold,
            det_frequency=1,
            device=DeviceConfig.default().device,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Quality metrics (no ground truth needed)
# ---------------------------------------------------------------------------


def compute_bone_lengths(poses: np.ndarray, valid: np.ndarray) -> dict[int, np.ndarray]:
    """Compute bone lengths for each edge across valid frames.

    Returns:
        {edge_index: array of lengths (M,)}
    """
    lengths: dict[int, np.ndarray] = {}
    for idx, (a, b) in enumerate(H36M_SKELETON_EDGES):
        pa = poses[valid, a, :2]
        pb = poses[valid, b, :2]
        dists = np.linalg.norm(pa - pb, axis=1)
        lengths[idx] = dists
    return lengths


def compute_jitter(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean per-joint displacement between consecutive valid frames.

    Returns:
        Array of jitter values (M-1,) in normalized coords.
    """
    valid_poses = poses[valid, :, :2]  # (M, 17, 2)
    if len(valid_poses) < 2:
        return np.array([0.0])
    diffs = np.diff(valid_poses, axis=0)  # (M-1, 17, 2)
    return np.mean(np.linalg.norm(diffs, axis=2), axis=1)  # (M-1,)


def count_angle_violations(poses: np.ndarray, valid: np.ndarray) -> int:
    """Count frames where joint angles are anatomically implausible.

    Checks: knee 0-180, elbow 0-160, hip 30-180.
    """
    violations = 0
    valid_poses = poses[valid]
    checks = [
        (H36Key.RHIP, H36Key.RKNEE, H36Key.RFOOT, 0, 180),
        (H36Key.LHIP, H36Key.LKNEE, H36Key.LFOOT, 0, 180),
        (H36Key.RSHOULDER, H36Key.RELBOW, H36Key.RWRIST, 0, 160),
        (H36Key.LSHOULDER, H36Key.LELBOW, H36Key.LWRIST, 0, 160),
        (H36Key.SPINE, H36Key.RHIP, H36Key.RKNEE, 30, 180),
        (H36Key.SPINE, H36Key.LHIP, H36Key.LKNEE, 30, 180),
    ]
    for a, v, c, lo, hi in checks:
        for i in range(len(valid_poses)):
            pa = valid_poses[i, a, :2].astype(np.float64)
            pv = valid_poses[i, v, :2].astype(np.float64)
            pc = valid_poses[i, c, :2].astype(np.float64)
            if np.isnan(pa).any() or np.isnan(pv).any() or np.isnan(pc).any():
                continue
            angle = angle_3pt(pa, pv, pc)
            if np.isnan(angle) or angle < lo or angle > hi:
                violations += 1
    return violations


def compute_pose_quality(
    poses: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """Compute quality metrics for a pose sequence.

    Args:
        poses: (N, 17, 3) pose array with NaN for missing frames.
        valid_mask: (N,) boolean mask.

    Returns:
        Dict with coverage, bone_length_std, mean_jitter, angle_violations.
    """
    valid = valid_mask
    n_total = len(valid_mask)
    n_valid = valid.sum()

    coverage = n_valid / n_total if n_total > 0 else 0.0

    if n_valid < 2:
        return {
            "coverage": coverage,
            "bone_length_std": 0.0,
            "mean_jitter": 0.0,
            "angle_violations": 0,
        }

    bone_lengths = compute_bone_lengths(poses, valid)
    bone_stds = [np.std(b) for b in bone_lengths.values() if len(b) > 1]
    bone_length_std = float(np.mean(bone_stds)) if bone_stds else 0.0

    jitter = compute_jitter(poses, valid)
    mean_jitter = float(np.mean(jitter))

    angle_violations = count_angle_violations(poses, valid)

    return {
        "coverage": coverage,
        "bone_length_std": bone_length_std,
        "mean_jitter": mean_jitter,
        "angle_violations": angle_violations,
    }


# ---------------------------------------------------------------------------
# Person selection
# ---------------------------------------------------------------------------


def _select_person_interactive(
    backend: str,
    video_path: str,
) -> PersonClick | None:
    """Show detected persons and let user choose."""
    extractor = _create_extractor(backend)
    persons, _ = extractor.preview_persons(video_path, num_frames=30)

    if not persons:
        print(f"  [{backend}] No persons detected.")
        return None

    if len(persons) == 1:
        p = persons[0]
        print(f"  [{backend}] Only 1 person detected (track #{p['track_id']}). Auto-selecting.")
        return PersonClick(x=int(p["mid_hip"][0]), y=int(p["mid_hip"][1]))

    print(f"\n  [{backend}] Detected {len(persons)} persons:")
    for i, p in enumerate(persons):
        print(
            f"    {i + 1}. Track #{p['track_id']} — mid_hip=({p['mid_hip'][0]:.0f}, {p['mid_hip'][1]:.0f})"
        )

    while True:
        try:
            choice = int(input(f"  [{backend}] Select person (1-{len(persons)}): "))
            if 1 <= choice <= len(persons):
                break
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(persons)}")

    p = persons[choice - 1]
    print(f"  Selected person #{choice} (track #{p['track_id']})")
    return PersonClick(x=int(p["mid_hip"][0]), y=int(p["mid_hip"][1]))


# ---------------------------------------------------------------------------
# Per-frame CSV metrics
# ---------------------------------------------------------------------------


def compute_per_frame_metrics(
    poses: np.ndarray,
    valid_mask: np.ndarray,
    backend: str,
) -> list[dict[str, str | float | int]]:
    """Compute per-frame quality metrics for CSV output.

    Returns:
        List of dicts, one per frame.
    """
    rows = []
    bone_lengths = compute_bone_lengths(poses, valid_mask) if valid_mask.sum() >= 2 else {}

    # Compute mean bone length from valid frames for relative std
    mean_bone_lengths = {}
    if bone_lengths:
        for idx, dists in bone_lengths.items():
            if len(dists) > 1:
                mean_bone_lengths[idx] = np.mean(dists)

    prev_valid_idx = None
    for i in range(len(poses)):
        is_valid = valid_mask[i]
        row = {
            "frame": int(i),
            "backend": backend,
            "valid": int(is_valid),
        }

        if is_valid:
            # Per-frame bone length std (relative to sequence mean)
            bl_stds = []
            for idx, (a, b) in enumerate(H36M_SKELETON_EDGES):
                dist = np.linalg.norm(poses[i, a, :2] - poses[i, b, :2])
                if idx in mean_bone_lengths and mean_bone_lengths[idx] > 0:
                    bl_stds.append(abs(dist - mean_bone_lengths[idx]) / mean_bone_lengths[idx])
            row["bone_length_dev"] = float(np.mean(bl_stds)) if bl_stds else 0.0

            # Per-frame jitter (displacement from previous valid frame)
            if prev_valid_idx is not None:
                disp = np.mean(
                    np.linalg.norm(poses[i, :, :2] - poses[prev_valid_idx, :, :2], axis=1)
                )
                row["jitter"] = float(disp)
            else:
                row["jitter"] = 0.0

            prev_valid_idx = i
        else:
            row["bone_length_dev"] = ""
            row["jitter"] = ""

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_backend_frames(
    video_path: str,
    extraction,
    backend: str,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Render skeleton overlay frames for one backend.

    Returns:
        (frames, metrics_summary)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    poses = extraction.poses
    valid_mask = extraction.valid_mask()
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute summary metrics
    metrics = compute_pose_quality(poses, valid_mask)

    label = BACKEND_LABELS.get(backend, backend)
    metric_text = (
        f"Coverage: {metrics['coverage']:.0%} | "
        f"Jitter: {metrics['mean_jitter']:.4f} | "
        f"Angle violations: {metrics['angle_violations']}"
    )

    frames = []
    pbar = tqdm(total=num_frames, desc=f"  Rendering {label}", unit="frame", ncols=100)

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(poses) and valid_mask[frame_idx]:
            pose_2d = poses[frame_idx, :, :2]  # (17, 2) normalized
            frame = draw_skeleton(frame, pose_2d, h, w)

            # Draw foot keypoints (rtmlib only, 6 extra points)
            if extraction.foot_keypoints is not None and frame_idx < len(extraction.foot_keypoints):
                foot = extraction.foot_keypoints[frame_idx]  # (6, 3) normalized
                if not np.isnan(foot).any():
                    foot_labels = ["L.Heel", "L.BigT", "L.SmT", "R.Heel", "R.BigT", "R.SmT"]
                    foot_colors = [
                        (255, 100, 100),
                        (255, 150, 50),
                        (255, 200, 0),  # left: red tones
                        (100, 100, 255),
                        (50, 150, 255),
                        (0, 200, 255),  # right: blue tones
                    ]
                    for k in range(6):
                        fx, fy, fc = foot[k]
                        if fc < 0.3:
                            continue
                        px = int(fx * w)
                        py = int(fy * h)
                        if 50 <= px < w - 50 and 50 <= py < h - 50:
                            cv2.circle(frame, (px, py), 5, foot_colors[k], -1, cv2.LINE_AA)
                            cv2.putText(
                                frame,
                                foot_labels[k],
                                (px + 7, py - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                foot_colors[k],
                                1,
                                cv2.LINE_AA,
                            )

        # Backend label (top-left)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Metrics summary (top-right)
        cv2.putText(
            frame, metric_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )

        frames.append(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    return frames, metrics


def layout_frames(
    all_frames: dict[str, list[np.ndarray]],
    max_width: int = 3840,
) -> list[np.ndarray]:
    """Arrange multiple backend frames into a grid layout.

    2 backends → side-by-side (1 row, 2 cols)
    3+ backends → auto grid
    """
    backends = list(all_frames.keys())
    n = len(backends)
    num_frames = min(len(f) for f in all_frames.values())

    if n == 1:
        return all_frames[backends[0]][:num_frames]

    # Determine grid layout
    cols = 2 if n <= 2 else min(n, 3)
    rows = (n + cols - 1) // cols

    sample_h, sample_w = all_frames[backends[0]][0].shape[:2]

    # Calculate per-cell size to fit max_width
    cell_w = min(sample_w, max_width // cols)
    cell_h = int(sample_h * (cell_w / sample_w))

    out_frames = []
    for fi in range(num_frames):
        row_imgs = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                idx = r * cols + c
                if idx < n:
                    f = all_frames[backends[idx]][fi]
                    f_resized = cv2.resize(f, (cell_w, cell_h))
                else:
                    f_resized = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cells.append(f_resized)
            row_imgs.append(np.hstack(cells))
        out_frames.append(np.vstack(row_imgs))

    return out_frames


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    output_path: str,
    all_rows: list[dict[str, str | float | int]],
    all_summaries: dict[str, dict[str, float]],
) -> None:
    """Write per-frame metrics + summary to CSV."""
    fieldnames = ["frame", "backend", "valid", "bone_length_dev", "jitter"]

    with Path(output_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Append summary section
    with Path(output_path).open("a") as f:
        f.write("\n\n# Summary\n")
        f.write("backend,coverage,bone_length_std,mean_jitter,angle_violations\n")
        for backend, metrics in all_summaries.items():
            f.write(
                f"{backend},{metrics['coverage']:.4f},{metrics['bone_length_std']:.6f},"
                f"{metrics['mean_jitter']:.6f},{metrics['angle_violations']}\n"
            )

    print(f"\nMetrics saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pose estimation backends on the same video."
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--backends",
        default="rtmlib",
        help="Comma-separated backends: rtmlib (RTMPose HALPE26 26kp). Default: rtmlib",
    )
    parser.add_argument("--output", help="Output video path (default: {stem}_compare.mp4)")
    parser.add_argument(
        "--person-click",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Pixel coordinates to select target person",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for all backends (default: 0.3)",
    )
    parser.add_argument("--select-person", action="store_true", help="Interactive person selection")
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip video rendering, only produce CSV"
    )
    return _run(parser.parse_args())


def _run(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    backends_raw = [b.strip() for b in args.backends.split(",")]
    backends = []
    for b in backends_raw:
        if b in BACKEND_LABELS:
            backends.append(b)
        elif b in BACKEND_CLI_CHOICES:
            backends.append(BACKEND_CLI_CHOICES[b])
        else:
            print(
                f"Error: unknown backend '{b}'. Available: {list(BACKEND_LABELS.keys())}",
                file=sys.stderr,
            )
            return 1
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_backends: list[str] = []
    for b in backends:
        if b not in seen:
            seen.add(b)
            unique_backends.append(b)
    backends = unique_backends

    stem = video_path.stem
    output_path = args.output or str(video_path.parent / f"{stem}_compare.mp4")
    csv_path = str(Path(output_path).with_suffix(".csv"))

    print(f"Comparing {len(backends)} backends on: {video_path.name}")
    print(f"Backends: {', '.join(BACKEND_LABELS.get(b, b) for b in backends)}")
    print()

    # Person selection (use first backend for selection)
    person_click: PersonClick | None = None
    if args.person_click:
        person_click = PersonClick(x=args.person_click[0], y=args.person_click[1])
        print(f"Using person click: ({person_click.x}, {person_click.y})")
    elif args.select_person:
        person_click = _select_person_interactive(backends[0], str(video_path))

    # Extract poses for each backend
    all_extractions: dict[str, object] = {}
    all_summaries: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, str | float | int]] = []
    all_frames: dict[str, list[np.ndarray]] = {}

    for backend in backends:
        label = BACKEND_LABELS.get(backend, backend)
        print(f"\n{'=' * 60}")
        print(f"Backend: {label}")
        print(f"{'=' * 60}")

        extractor = _create_extractor(backend, conf_threshold=args.conf_threshold)

        print("Extracting poses...")

        extraction: TrackedExtraction = extractor.extract_video_tracked(
            str(video_path), person_click=person_click
        )

        n_valid = extraction.valid_mask().sum()
        n_total = len(extraction.valid_mask())
        print(f"Valid poses: {n_valid}/{n_total} ({n_valid / n_total:.0%})")

        all_extractions[backend] = extraction

        # Compute summary metrics
        metrics = compute_pose_quality(extraction.poses, extraction.valid_mask())
        all_summaries[backend] = metrics
        print(f"  Coverage: {metrics['coverage']:.0%}")
        print(f"  Bone length std: {metrics['bone_length_std']:.6f}")
        print(f"  Mean jitter: {metrics['mean_jitter']:.6f}")
        print(f"  Angle violations: {metrics['angle_violations']}")

        # Per-frame metrics for CSV
        rows = compute_per_frame_metrics(extraction.poses, extraction.valid_mask(), backend)
        all_rows.extend(rows)

        # Render frames
        if not args.no_viz:
            frames, _ = render_backend_frames(str(video_path), extraction, backend)
            all_frames[backend] = frames

    # Print comparison summary
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Backend':<25} {'Coverage':>10} {'Jitter':>10} {'Bone std':>10} {'Violations':>12}"
    print(header)
    print("-" * len(header))
    for backend in backends:
        m = all_summaries[backend]
        label = BACKEND_LABELS.get(backend, backend)
        print(
            f"{label:<25} {m['coverage']:>9.0%} {m['mean_jitter']:>10.4f} {m['bone_length_std']:>10.6f} {m['angle_violations']:>12d}"
        )

    # Write CSV
    write_csv(csv_path, all_rows, all_summaries)

    # Write video
    if not args.no_viz and all_frames:
        print("\nRendering comparison video...")
        combined = layout_frames(all_frames)

        if not combined:
            print("No frames to render.")
            return 1

        h, w = combined[0].shape[:2]
        first_ext = all_extractions[backends[0]]
        fps = first_ext.fps

        writer = H264Writer(output_path, w, h, fps)

        for frame in tqdm(combined, desc="Writing video", unit="frame", ncols=100):
            writer.write(frame)

        writer.close()
        print(f"Video saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
