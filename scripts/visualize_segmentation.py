#!/usr/bin/env python3
"""Visualize segmentation results on video."""

import argparse
from pathlib import Path

import cv2

from src.utils.video_writer import H264Writer


def main():
    parser = argparse.ArgumentParser(description="Visualize element segmentation on video")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--segments", type=Path, help="Path to segmentation JSON")
    parser.add_argument("--output", type=Path, help="Output video path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Default segments for video1 if no JSON provided
    if args.segments and args.segments.exists():
        import json

        with args.segments.open() as f:
            data = json.load(f)
        segments = [(s["start"], s["end"], s["element_type"]) for s in data["segments"]]
    # Default segments for the test videos
    elif "video1" in args.video.name:
        segments = [(8, 159, "three_turn"), (451, 500, "three_turn")]
    elif "video2" in args.video.name:
        segments = [
            (40, 133, "three_turn"),
            (162, 565, "three_turn"),
            (597, 669, "three_turn"),
            (701, 763, "three_turn"),
            (791, 1019, "three_turn"),
            (1121, 1246, "three_turn"),
            (1308, 1411, "three_turn"),
        ]
    elif "video3" in args.video.name:
        segments = [
            (47, 145, "three_turn"),
            (157, 585, "three_turn"),
            (607, 636, "three_turn"),
            (740, 785, "unknown"),
            (818, 936, "three_turn"),
            (960, 1209, "three_turn"),
            (1226, 1279, "unknown"),
            (1292, 1334, "unknown"),
            (1356, 1375, "unknown"),
            (1400, 1423, "three_turn"),
        ]
    else:
        segments = []

    # Colors for different element types
    colors = {
        "three_turn": (0, 255, 0),  # Green
        "waltz_jump": (255, 0, 0),  # Blue (BGR format)
        "toe_loop": (0, 0, 255),  # Red
        "flip": (255, 255, 0),  # Cyan
        "unknown": (128, 128, 128),  # Gray
    }

    # Setup output video
    output_path = args.output or args.video.parent / f"{args.video.stem}_segmented.mp4"
    writer = H264Writer(output_path, width, height, fps)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find active segment for this frame
        active_segment = None
        for start, end, elem_type in segments:
            if start <= frame_idx <= end:
                active_segment = (start, end, elem_type)
                break

        # Draw overlay
        if active_segment:
            start, end, elem_type = active_segment
            color = colors.get(elem_type, (255, 255, 255))

            # Element label background
            label = f"{elem_type} [{start}:{end}]"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (10 + w + 20, 50 + h), color, -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # Element label text
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Frame counter
            cv2.putText(
                frame,
                f"Frame: {frame_idx}/{total_frames}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Boundary markers
            if frame_idx == start:
                cv2.line(frame, (0, 0), (0, height), (0, 255, 0), 5)
                cv2.putText(
                    frame, "START", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            if frame_idx == end:
                cv2.line(frame, (width - 1, 0), (width - 1, height), (0, 0, 255), 5)
                cv2.putText(
                    frame,
                    "END",
                    (width - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.close()

    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
