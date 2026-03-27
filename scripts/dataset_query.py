#!/usr/bin/env python3
"""Dataset query and management utilities."""

import json
from pathlib import Path

DATASET_DIR = Path("data/dataset")


def load_index():
    """Load dataset index."""
    with open(DATASET_DIR / "index.json") as f:
        return json.load(f)


def list_videos():
    """List all videos in dataset."""
    index = load_index()
    print(f"\n=== Dataset: {index['description']} ===")
    print(f"Version: {index['version']}")
    print(f"Elements: {', '.join(index['elements'])}")
    print(f"\nVideos ({len(index['videos'])}):")

    for video_info in index["videos"]:
        meta_path = DATASET_DIR / "metadata" / f"{video_info['id']}.json"
        with open(meta_path) as f:
            meta = json.load(f)

        vm = meta["video_metadata"]
        print(f"  [{video_info['id']}] {meta['title']}")
        print(f"      URL: {meta['source_url']}")
        print(f"      Duration: {vm['duration_sec']:.1f}s @ {vm['fps']}fps")
        print(f"      Resolution: {vm['width']}x{vm['height']}")
        if meta.get("analysis"):
            print(f"      Analysis: {list(meta['analysis'].keys())}")
        print()


def get_analysis(video_id: str, element: str = None):
    """Get analysis results for a video."""
    analysis_path = DATASET_DIR / "analysis" / f"{video_id}.json"
    with open(analysis_path) as f:
        analysis = json.load(f)

    if element:
        if element not in analysis:
            print(f"Element '{element}' not found in analysis")
            return
        results = {element: analysis[element]}
    else:
        results = analysis

    print(f"\n=== Analysis for {video_id} ===")
    for elem, data in results.items():
        if "error" in data:
            print(f"\n{elem}: ERROR - {data['error']}")
            continue

        print(f"\n{elem.upper()}:")
        print(f"  Overall Score: {data['overall_score']:.1f} / 10")

        print("\n  Metrics:")
        for m in data["metrics"]:
            status = "✓" if m["is_good"] else "✗"
            print(f"    {status} {m['name']}: {m['value']:.2f} {m['unit']} "
                  f"(ref: {m['reference_range'][0]}-{m['reference_range'][1]})")

        if data.get("recommendations"):
            print("\n  Recommendations:")
            for i, rec in enumerate(data["recommendations"], 1):
                print(f"    {i}. {rec}")


def search_by_score(min_score: float = 0.0):
    """Find videos with scores above threshold."""
    index = load_index()
    results = []

    for video_info in index["videos"]:
        analysis_path = DATASET_DIR / "analysis" / f"{video_info['id']}.json"
        with open(analysis_path) as f:
            analysis = json.load(f)

        for element, data in analysis.items():
            if "error" not in data and data.get("overall_score", 0) >= min_score:
                results.append({
                    "video_id": video_info['id'],
                    "element": element,
                    "score": data["overall_score"]
                })

    print(f"\n=== Results (score >= {min_score}) ===")
    for r in results:
        print(f"  {r['video_id']} - {r['element']}: {r['score']:.1f}/10")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset query utility")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list command
    subparsers.add_parser("list", help="List all videos")

    # analysis command
    analysis_parser = subparsers.add_parser("analysis", help="Get analysis results")
    analysis_parser.add_argument("video_id", help="Video ID (e.g., video1)")
    analysis_parser.add_argument("--element", help="Element type (e.g., three_turn)")

    # search command
    search_parser = subparsers.add_parser("search", help="Search by score")
    search_parser.add_argument("--min-score", type=float, default=0.0,
                              help="Minimum score threshold")

    args = parser.parse_args()

    if args.command == "list":
        list_videos()
    elif args.command == "analysis":
        get_analysis(args.video_id, args.element)
    elif args.command == "search":
        search_by_score(args.min_score)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
