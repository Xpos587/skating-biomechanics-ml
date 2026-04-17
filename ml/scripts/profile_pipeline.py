"""Profile ML pipeline to find optimization targets."""

import cProfile
import pstats
import io
from pathlib import Path


def profile_extraction(video_path: str, device: str = "cuda"):
    """Profile pose extraction."""
    print(f"Profiling pose extraction on {device}...")

    from skating_ml.pose_estimation import PoseExtractor

    profiler = cProfile.Profile()
    profiler.enable()

    extractor = PoseExtractor(device=device)
    poses = extractor.extract_video(video_path)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())
    return poses


def profile_analysis(poses, fps: float = 30.0):
    """Profile biomechanics analysis."""
    print("\nProfiling biomechanics analysis...")

    from skating_ml.analysis import BiomechanicsAnalyzer

    profiler = cProfile.Profile()
    profiler.enable()

    analyzer = BiomechanicsAnalyzer()
    results = analyzer.analyze(poses, fps=fps)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: profile_pipeline.py <video_path> [device]")
        print("  video_path: Path to video file for profiling")
        print("  device: 'cuda' or 'cpu' (default: cuda)")
        sys.exit(1)

    video_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    poses = profile_extraction(video_path, device)
    profile_analysis(poses)
