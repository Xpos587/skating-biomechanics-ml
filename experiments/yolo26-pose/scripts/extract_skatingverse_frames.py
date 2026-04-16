#!/usr/bin/env python3
"""
Extract frames from SkatingVerse videos for pseudo-labeling.

Usage:
    python extract_skatingverse_frames.py \
        --video-dir /root/data/datasets/skatingverse/test_videos \
        --output-dir /root/data/datasets/skatingverse/frames \
        --fps 10 \
        --workers 8
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2


def extract_frames(video_path: Path, output_dir: Path, fps: int = 10) -> dict:
    """Extract frames from a single video using OpenCV.

    Args:
        video_path: Path to video file
        output_dir: Base output directory
        fps: Frames per second to extract

    Returns:
        Dict with video_id, num_frames, status
    """
    video_id = video_path.stem
    frames_dir = output_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                "video_id": video_id,
                "num_frames": 0,
                "status": "error",
                "error": "Cannot open video"
            }

        # Get video FPS
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            video_fps = 30  # Default assumption

        # Calculate frame interval
        frame_interval = int(video_fps / fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every Nth frame
            if frame_count % frame_interval == 0:
                output_path = frames_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1

            frame_count += 1

        cap.release()

        return {
            "video_id": video_id,
            "num_frames": extracted_count,
            "status": "success",
            "error": None
        }

    except Exception as e:
        return {
            "video_id": video_id,
            "num_frames": 0,
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from SkatingVerse videos"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Path to SkatingVerse test_videos directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output frames directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second to extract (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos for testing"
    )

    args = parser.parse_args()

    # Find all videos
    video_paths = sorted(args.video_dir.glob("*.mp4"))

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    print(f"Found {len(video_paths)} videos")
    print(f"Extracting at {args.fps} FPS")
    print(f"Using {args.workers} workers")

    # Extract frames in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extract_frames,
                video_path,
                args.output_dir,
                args.fps
            ): video_path
            for video_path in video_paths
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting frames"
        ):
            result = future.result()
            results.append(result)

            # Log errors
            if result["status"] != "success":
                print(f"\nError: {result['video_id']} - {result['error']}")

    # Summary statistics
    total_frames = sum(r["num_frames"] for r in results)
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    print(f"\n=== Summary ===")
    print(f"Videos processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total frames extracted: {total_frames:,}")
    if success_count > 0:
        print(f"Average frames per video: {total_frames / success_count:.1f}")

    # Save results to JSON
    output_json = args.output_dir / "extraction_results.json"
    with open(output_json, "w") as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "summary": {
                "total_videos": len(results),
                "successful": success_count,
                "errors": error_count,
                "total_frames": total_frames,
                "avg_frames_per_video": total_frames / success_count if success_count > 0 else 0
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
