"""E2E test for OOFSkate proxy metrics."""

import json
import sys
from pathlib import Path

import numpy as np

from src.pipeline import AnalysisPipeline

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/michael/Downloads/FineFS/data/video/1.mp4"


def main():
    pipeline = AnalysisPipeline()

    print(f"Analyzing: {VIDEO_PATH}")
    result = pipeline.analyze(VIDEO_PATH, element_type="waltz_jump")

    print("\n=== METRICS ===")
    for metric in result.get("metrics", []):
        name = metric.get("name", "?")
        value = metric.get("value", 0.0)
        unit = metric.get("unit", "")
        is_good = metric.get("is_good", False)
        ref = metric.get("reference_range", (0, 0))
        marker = "✅" if is_good else "❌"
        print(f"  {marker} {name}: {value:.3f} {unit} (ref: {ref[0]:.1f}-{ref[1]:.1f})")

    print("\n=== RECOMMENDATIONS ===")
    for rec in result.get("recommendations", []):
        print(f"  • {rec}")

    print("\n=== GOE SCORE ===")
    goe = next((m for m in result.get("metrics", []) if m["name"] == "goe_score"), None)
    if goe:
        print(f"  GOE Proxy: {goe['value']:.1f} / 10.0")
    else:
        print("  GOE score not found!")

    out_path = Path(VIDEO_PATH).with_suffix(".analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull result saved to: {out_path}")


if __name__ == "__main__":
    main()
