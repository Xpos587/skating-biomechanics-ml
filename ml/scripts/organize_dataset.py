#!/usr/bin/env python3
"""Organize skating videos into a structured dataset."""

import json
import shutil
from pathlib import Path

from skating_ml.pipeline import AnalysisPipeline
from skating_ml.utils.video import get_video_meta

# Dataset configuration
DATASET_DIR = Path("data/dataset")
RAW_DIR = Path("data/raw")

# Video sources with metadata
VIDEO_SOURCES = [
    {
        "id": "video1",
        "filename": "skating_video.mp4",
        "url": "https://youtu.be/7wjiqGGv_RY?si=LRZSYUcTUVF8neaS",
        "title": "Лига фигурного катания - тренировка по скольжению",
        "description": "Обучающее видео о скольжении, тройках и базовых упражнениях",
    },
    {
        "id": "video2",
        "filename": "skating_video2.mp4",
        "url": "https://youtu.be/y9fwL-3OCpk?si=ikG9xrTVRNzuq0Hd",
        "title": "Фигурное катание - техника",
        "description": "Видео о технике фигурного катания",
    },
    {
        "id": "video3",
        "filename": "skating_video3.mp4",
        "url": "https://youtu.be/_29Z2EJ3nkI?si=N39e4W6NKD3BugeX",
        "title": "Фигурное катание - тренировка",
        "description": "Тренировочное видео по фигурному катанию",
    },
]

# Elements to analyze for each video
ELEMENTS_TO_ANALYZE = ["three_turn", "waltz_jump"]


def create_video_metadata(video_info: dict, video_path: Path) -> dict:
    """Create metadata entry for a video."""
    meta = get_video_meta(video_path)

    return {
        "id": video_info["id"],
        "filename": video_path.name,
        "source_url": video_info["url"],
        "title": video_info.get("title", ""),
        "description": video_info.get("description", ""),
        "video_metadata": {
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "num_frames": meta.num_frames,
            "duration_sec": meta.num_frames / meta.fps if meta.fps > 0 else 0,
        },
        "subtitle_file": f"{video_info['id']}.ru.vtt",
        "analysis": {},
    }


def analyze_video(video_path: Path, video_id: str) -> dict:  # noqa: ARG001
    """Run analysis on video and return results."""
    pipeline = AnalysisPipeline(reference_store=None)

    analysis_results = {}
    for element in ELEMENTS_TO_ANALYZE:
        try:
            report = pipeline.analyze(video_path, element)
            analysis_results[element] = {
                "overall_score": report.overall_score,
                "dtw_distance": report.dtw_distance,
                "metrics": [
                    {
                        "name": m.name,
                        "value": float(m.value),
                        "unit": m.unit,
                        "is_good": m.is_good,
                        "reference_range": m.reference_range,
                    }
                    for m in report.metrics
                ],
                "recommendations": report.recommendations,
            }
        except Exception as e:
            analysis_results[element] = {"error": str(e)}

    return analysis_results


def main():
    """Organize dataset."""
    print("Organizing skating dataset...")

    # Create dataset directory structure
    (DATASET_DIR / "videos").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "subtitles").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "analysis").mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "version": "1.0",
        "description": "Figure skating biomechanics dataset",
        "elements": ELEMENTS_TO_ANALYZE,
        "videos": [],
    }

    for video_info in VIDEO_SOURCES:
        print(f"\nProcessing {video_info['id']}...")

        raw_video = RAW_DIR / video_info["filename"]
        raw_subtitle = RAW_DIR / f"{video_info['filename'].split('.')[0]}.ru.vtt"

        if not raw_video.exists():
            print(f"  Warning: Video not found: {raw_video}")
            continue

        # Copy video
        dest_video = DATASET_DIR / "videos" / f"{video_info['id']}.mp4"
        shutil.copy(raw_video, dest_video)
        print(f"  Copied video: {dest_video.name}")

        # Copy subtitle
        if raw_subtitle.exists():
            dest_subtitle = DATASET_DIR / "subtitles" / f"{video_info['id']}.ru.vtt"
            shutil.copy(raw_subtitle, dest_subtitle)
            print(f"  Copied subtitle: {dest_subtitle.name}")
        else:
            print(f"  Warning: Subtitle not found: {raw_subtitle}")

        # Create metadata
        video_meta = create_video_metadata(video_info, dest_video)

        # Run analysis
        print("  Running analysis...")
        try:
            analysis_results = analyze_video(dest_video, video_info["id"])
            video_meta["analysis"] = analysis_results

            # Save analysis results
            analysis_file = DATASET_DIR / "analysis" / f"{video_info['id']}.json"
            with analysis_file.open("w") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved analysis: {analysis_file.name}")

            # Print summary
            for element, result in analysis_results.items():
                if "error" in result:
                    print(f"    {element}: ERROR - {result['error']}")
                else:
                    print(f"    {element}: score={result['overall_score']:.1f}")
        except Exception as e:
            print(f"  Analysis failed: {e}")
            video_meta["analysis"] = {"error": str(e)}

        # Save video metadata
        meta_file = DATASET_DIR / "metadata" / f"{video_info['id']}.json"
        with meta_file.open("w") as f:
            json.dump(video_meta, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: {meta_file.name}")

        dataset_metadata["videos"].append(video_info)

    # Save dataset index
    index_file = DATASET_DIR / "index.json"
    with index_file.open("w") as f:
        json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)

    print("\nDataset organized successfully!")
    print(f"  Location: {DATASET_DIR}")
    print(f"  Videos: {len(dataset_metadata['videos'])}")
    print(f"  Index: {index_file}")


if __name__ == "__main__":
    main()
