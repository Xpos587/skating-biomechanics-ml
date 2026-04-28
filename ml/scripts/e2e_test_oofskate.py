"""E2E test for OOFSkate proxy metrics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.pipeline import AnalysisPipeline


def main() -> None:
    video_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/home/michael/Downloads/FineFS/data/video/1.mp4")
    )
    element_type = sys.argv[2] if len(sys.argv) > 2 else "waltz_jump"

    pipeline = AnalysisPipeline()

    print(f"Analyzing: {video_path} (element: {element_type})")
    result = pipeline.analyze(video_path, element_type=element_type)

    print("\n=== METRICS ===")
    for metric in result.metrics:
        marker = "✅" if metric.is_good else "❌"
        print(
            f"  {marker} {metric.name}: {metric.value:.3f} {metric.unit} "
            f"(ref: {metric.reference_range[0]:.1f}-{metric.reference_range[1]:.1f})"
        )

    print("\n=== RECOMMENDATIONS ===")
    for rec in result.recommendations:
        print(f"  • {rec}")

    print("\n=== GOE SCORE ===")
    goe = next((m for m in result.metrics if m.name == "goe_score"), None)
    if goe:
        print(f"  GOE Proxy: {goe.value:.1f} / 10.0")
    else:
        print("  GOE score not found!")

    out_path = video_path.with_suffix(".analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "element_type": result.element_type,
                "phases": {
                    "start": result.phases.start,
                    "takeoff": result.phases.takeoff,
                    "peak": result.phases.peak,
                    "landing": result.phases.landing,
                    "end": result.phases.end,
                },
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "is_good": m.is_good,
                        "reference_range": m.reference_range,
                    }
                    for m in result.metrics
                ],
                "recommendations": result.recommendations,
                "overall_score": result.overall_score,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull result saved to: {out_path}")


if __name__ == "__main__":
    main()
