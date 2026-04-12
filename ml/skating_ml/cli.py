#!/usr/bin/env python3
"""CLI for figure skating biomechanics analysis.

H3.6M Architecture:
    Uses H3.6M 17-keypoint format as the primary format.
    2D extraction: RTMPoseExtractor (rtmlib BodyWithFeet)

Usage:
    python -m skating_ml.cli analyze video.mp4 --element waltz_jump
    python -m skating_ml.cli build-ref expert.mp4 --element waltz_jump --takeoff 1.0 --landing 1.5
    python -m skating_ml.cli segment tutorial.mp4 --export-dir data/references
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from .analysis import element_defs
from .pipeline import AnalysisPipeline
from .pose_estimation import normalizer
from .references import reference_builder, reference_store
from .types import ElementPhase, PersonClick, ReferenceData
from .utils.video import get_video_meta

PoseNormalizer = normalizer.PoseNormalizer
ReferenceBuilder = reference_builder.ReferenceBuilder
ReferenceStore = reference_store.ReferenceStore
get_element_def = element_defs.get_element_def


def _get_extractor(**kwargs):
    """Create an RTMPoseExtractor.

    Args:
        **kwargs: Forwarded to RTMPoseExtractor constructor (including tracking_mode).

    Returns:
        RTMPoseExtractor instance.
    """
    from .pose_estimation.rtmlib_extractor import RTMPoseExtractor

    return RTMPoseExtractor(**kwargs)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute analyze command."""
    # Validate video file
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Initialize pipeline
    reference_store = None
    if args.reference_dir:
        # Create builder with RTMPoseExtractor
        extractor = _get_extractor(output_format="normalized", tracking_mode=args.tracking)
        norm = PoseNormalizer(target_spine_length=0.4)
        builder = ReferenceBuilder(extractor, norm)
        reference_store = ReferenceStore(args.reference_dir)
        reference_store.set_builder(builder)

    person_click = None
    if args.person_click:
        person_click = PersonClick(x=args.person_click[0], y=args.person_click[1])
    elif args.select_person:
        from .pose_estimation.person_selector import select_persons_interactive

        extractor = _get_extractor(output_format="normalized", tracking_mode=args.tracking)
        persons, _preview_path = extractor.preview_persons(args.video)
        if not persons:
            print("No persons detected in the first seconds of the video.")
            return 1
        if len(persons) == 1:
            print(f"Only 1 person detected (track #{persons[0]['track_id']}). Auto-selecting.")
            mid_hip = persons[0]["mid_hip"]
            meta = get_video_meta(args.video)
            person_click = PersonClick(
                x=int(mid_hip[0] * meta.width),
                y=int(mid_hip[1] * meta.height),
            )
        else:
            # Convert person data to pose array for matplotlib selector
            import numpy as np

            meta = get_video_meta(args.video)
            poses_array = np.zeros((len(persons), 17, 2), dtype=np.float32)
            bboxes = []
            for i, p in enumerate(persons):
                x1, y1, x2, y2 = p["bbox"]
                # Convert normalized bbox to pixel coordinates
                bboxes.append(
                    (
                        int(x1 * meta.width),
                        int(y1 * meta.height),
                        int(x2 * meta.width),
                        int(y2 * meta.height),
                    )
                )
                # Use mid_hip as a representative point (will be expanded to bbox in GUI)
                poses_array[i, 0] = [
                    p["mid_hip"][0] * meta.width,
                    p["mid_hip"][1] * meta.height,
                ]

            # Interactive matplotlib selection
            selected_indices = select_persons_interactive(args.video, poses_array, bboxes=bboxes)

            if not selected_indices:
                print("No person selected. Canceling.")
                return 1

            choice_idx = selected_indices[0]
            mid_hip = persons[choice_idx]["mid_hip"]
            person_click = PersonClick(
                x=int(mid_hip[0] * meta.width),
                y=int(mid_hip[1] * meta.height),
            )
            print(f"Selected person #{choice_idx} (track_id={persons[choice_idx]['track_id']})")

    pipeline = AnalysisPipeline(
        reference_store=reference_store,
        person_click=person_click,
        reestimate_camera=args.moving_camera,
    )

    element_type = args.element

    print(f"Analyzing: {args.video}")
    if element_type:
        print(f"Element: {element_type}")
    else:
        print("Element: not specified (poses + visualization only)")

    try:
        # Run analysis
        report = pipeline.analyze(args.video, element_type, reference_path=args.reference)

        # Format output
        if args.json:
            # JSON output
            output = {
                "element_type": report.element_type,
                "overall_score": report.overall_score,
                "dtw_distance": report.dtw_distance,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "is_good": m.is_good,
                        "reference_range": m.reference_range,
                    }
                    for m in report.metrics
                ],
                "recommendations": report.recommendations,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # Text output
            print(pipeline.format_report(report))

        # Save to file if requested
        output = None
        if args.output:
            if args.json:
                with Path(args.output).open("w") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
            else:
                with Path(args.output).open("w") as f:
                    f.write(pipeline.format_report(report))
            print(f"\nReport saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_build_ref(args: argparse.Namespace) -> int:
    """Execute build-ref command."""
    # Validate video file
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Initialize components (H3.6M format)
    pose_extractor = _get_extractor(output_format="normalized", tracking_mode=args.tracking)
    normalizer = PoseNormalizer(target_spine_length=0.4)
    builder = ReferenceBuilder(pose_extractor, normalizer)

    # Get video metadata
    meta = get_video_meta(args.video)
    fps = args.fps or meta.fps

    if fps <= 0:
        print(f"Error: Invalid FPS: {fps}")
        return 1

    # Convert timestamps to frame indices
    start_frame = int(args.start * fps) if args.start is not None else 0
    takeoff_frame = int(args.takeoff * fps)
    peak_frame = int(args.peak * fps)
    landing_frame = int(args.landing * fps)
    end_frame = int(args.end * fps) if args.end is not None else meta.num_frames - 1

    # Create phase boundaries
    phases = ElementPhase(
        name=args.element,
        start=start_frame,
        takeoff=takeoff_frame,
        peak=peak_frame,
        landing=landing_frame,
        end=end_frame,
    )

    print(f"Building reference for: {args.element}")
    print(f"Video: {args.video} ({meta.width}x{meta.height} @ {fps:.2f} fps)")
    print(
        f"Phases: start={start_frame}, takeoff={takeoff_frame}, "
        f"peak={peak_frame}, landing={landing_frame}, end={end_frame}"
    )

    try:
        # Build reference
        ref = builder.build_from_video(args.video, args.element, phases)
        print(f"Extracted {ref.poses.shape[0]} frames")

        # Save to store
        output_dir = args.output_dir or Path("data/references")
        store = ReferenceStore(output_dir)
        store.set_builder(builder)
        output_path = store.add(ref)

        print(f"Saved reference to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error building reference: {e}")
        traceback.print_exc()
        return 1


def cmd_segment(args: argparse.Namespace) -> int:
    """Execute segment command."""
    # Validate video file
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Initialize pipeline
    pipeline = AnalysisPipeline(
        enable_smoothing=True,
    )

    print(f"Segmenting: {args.video}")

    try:
        # Run segmentation
        result = pipeline.segment_video(args.video)

        # Print timeline
        print(result.get_timeline())
        print()

        # Export JSON if requested
        if args.output:
            json_path = args.output
            if not json_path.suffix:
                json_path = json_path / "segmentation_results.json"
            result.export_segments_json(json_path)
            print(f"Segmentation exported to: {json_path}")

        # Export segments as references if output-dir specified
        if args.export_dir:
            # Use RTMPoseExtractor for export extraction
            extractor = _get_extractor(output_format="normalized", tracking_mode=args.tracking)
            norm = PoseNormalizer(target_spine_length=0.4)

            # Extract poses in H3.6M format with tracking
            extraction = extractor.extract_video_tracked(args.video)
            normalized = norm.normalize(extraction.poses)

            export_dir = args.export_dir
            export_dir.mkdir(parents=True, exist_ok=True)

            for i, segment in enumerate(result.segments, 1):
                # Skip unknown elements
                if segment.element_type == "unknown":
                    print(f"Skipping segment {i}: unknown element type")
                    continue

                # Extract segment poses
                segment_poses = normalized[segment.start : segment.end]

                # Create phases if not available
                if segment.phases is None:
                    from .types import ElementPhase

                    segment_phases = ElementPhase(
                        name=segment.element_type,
                        start=0,
                        takeoff=0,
                        peak=segment.end - segment.start if segment.end > segment.start else 0,
                        landing=0,
                        end=segment.end - segment.start,
                    )
                else:
                    segment_phases = segment.phases

                # Create reference data
                ref = ReferenceData(
                    element_type=segment.element_type,
                    name=f"{args.video.stem}_{i:03d}",
                    poses=segment_poses,
                    phases=segment_phases,
                    fps=result.video_meta.fps,
                    meta=result.video_meta,
                    source=f"Auto-segmented from {args.video.name}",
                )

                # Save to element-specific directory
                element_dir = export_dir / segment.element_type
                element_dir.mkdir(parents=True, exist_ok=True)

                output_path = element_dir / f"{args.video.stem}_{i:03d}.npz"
                ref.save(output_path)

                print(f"Exported: {output_path}")

        return 0

    except Exception as e:
        print(f"Error during segmentation: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command — dual-video comparison with overlays."""
    if not args.athlete.exists():
        print(f"Error: Athlete video not found: {args.athlete}")
        return 1
    if not args.reference.exists():
        print(f"Error: Reference video not found: {args.reference}")
        return 1

    output_path = args.output or args.athlete.with_name(
        f"{args.athlete.stem}_compare{args.athlete.suffix}"
    )

    from .visualization.comparison import ComparisonConfig, ComparisonMode, ComparisonRenderer

    overlays = [o.strip() for o in args.overlays.split(",")]
    valid = {"skeleton", "axis", "angles", "timer"}
    overlays = [o for o in overlays if o in valid]
    if not overlays:
        overlays = ["skeleton"]

    config = ComparisonConfig(
        mode=ComparisonMode(args.mode),
        overlays=overlays,
        resize_width=args.resize,
        fps=args.fps,
        max_frames=args.max_frames,
        device=args.device,
        no_cache=args.no_cache,
    )

    renderer = ComparisonRenderer(config)
    print(f"Athlete:    {args.athlete}")
    print(f"Reference:  {args.reference}")
    print(f"Overlays:   {overlays}")
    print(f"Mode:       {args.mode}")
    print()

    try:
        renderer.process(args.athlete, args.reference, output_path)
        print(f"\nDone! Output: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, "verbose") else False:
            traceback.print_exc()
        return 1


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-тренер по фигурному катанию — анализ техники",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Анализировать видео \u0441 тройкой
  %(prog)s analyze my_skating.mp4 --element three_turn

  # Анализировать прыжок \u0441 сохранением отчёта
  %(prog)s analyze jump.mp4 --element waltz_jump --output report.txt

  # Создать референс из экспертного видео
  %(prog)s build-ref expert.mp4 --element waltz_jump --takeoff 1.0 --peak 1.2 --landing 1.4

  # Разбить обучающее видео на элементы и экспортировать
  %(prog)s segment coach_tutorial.mp4 --export-dir data/references

Архитектура:
  2D Pose: RTMPose (rtmlib, HALPE26 26kp) → H3.6M 17-keypoint format
  3D Pose: MotionAGFormer-S (59MB) or TCPFormer (422MB)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Анализировать видео фигурного катания",
    )
    analyze_parser.add_argument(
        "video",
        type=Path,
        help="Путь к видеофайлу",
    )
    analyze_parser.add_argument(
        "--element",
        type=str,
        default=None,
        choices=["three_turn", "waltz_jump", "toe_loop", "flip"],
        help="Тип элемента (опционально — без него только позы + визуализация)",
    )
    analyze_parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Путь к референсному видео (опционально)",
    )
    analyze_parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Директория с референсами (data/references)",
    )
    analyze_parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Пропустить детекцию (для видео с одним человеком)",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Сохранить отчёт в файл",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести в JSON формате",
    )
    analyze_parser.add_argument(
        "--person-click",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Click point to select target person (pixel coordinates)",
    )
    analyze_parser.add_argument(
        "--select-person",
        action="store_true",
        help="Интерактивный выбор персоны: показать превью, выбрать номер",
    )
    analyze_parser.add_argument(
        "--moving-camera",
        action="store_true",
        help="Enable per-frame camera re-estimation for moving cameras",
    )
    analyze_parser.add_argument(
        "--tracking",
        choices=["auto", "sports2d", "deepsort"],
        default="auto",
        help="Tracking mode: auto (built-in), sports2d (rtmlib Sports2D), deepsort (external DeepSORT)",
    )
    analyze_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод ошибок",
    )

    # build-ref command
    ref_parser = subparsers.add_parser(
        "build-ref",
        help="Создать референс из экспертного видео",
    )
    ref_parser.add_argument(
        "video",
        type=Path,
        help="Путь к видеофайлу",
    )
    ref_parser.add_argument(
        "--element",
        type=str,
        required=True,
        choices=["three_turn", "waltz_jump", "toe_loop", "flip"],
        help="Тип элемента",
    )
    ref_parser.add_argument(
        "--takeoff",
        type=float,
        required=True,
        help="Отрыв (секунды)",
    )
    ref_parser.add_argument(
        "--peak",
        type=float,
        required=True,
        help="Пик высоты (секунды)",
    )
    ref_parser.add_argument(
        "--landing",
        type=float,
        required=True,
        help="Приземление (секунды)",
    )
    ref_parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Начало элемента (секунды, по умолчанию 0)",
    )
    ref_parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Конец элемента (секунды, по умолчанию конец видео)",
    )
    ref_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS видео (по умолчанию автоопределение)",
    )
    ref_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Директория для сохранения (по умолчанию data/references)",
    )
    ref_parser.add_argument(
        "--name",
        type=str,
        default="expert",
        help="Имя референса (по умолчанию 'expert')",
    )
    ref_parser.add_argument(
        "--tracking",
        choices=["auto", "sports2d", "deepsort"],
        default="auto",
        help="Tracking mode: auto (built-in), sports2d (rtmlib Sports2D), deepsort (external DeepSORT)",
    )

    # segment command
    segment_parser = subparsers.add_parser(
        "segment",
        help="Разбить видео на отдельные элементы",
    )
    segment_parser.add_argument(
        "video",
        type=Path,
        help="Путь к видеофайлу",
    )
    segment_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Сохранить результаты сегментации в JSON",
    )
    segment_parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Директория для экспорта сегментов как референсов",
    )
    segment_parser.add_argument(
        "--tracking",
        choices=["auto", "sports2d", "deepsort"],
        default="auto",
        help="Tracking mode: auto (built-in), sports2d (rtmlib Sports2D), deepsort (external DeepSORT)",
    )
    segment_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод ошибок",
    )

    # compare command - dual-video comparison for training
    compare_parser = subparsers.add_parser(
        "compare",
        help="Сравнить видео спортсмена с эталоном (Kinovea-style)",
    )
    compare_parser.add_argument(
        "athlete",
        type=Path,
        help="Видео спортсмена",
    )
    compare_parser.add_argument(
        "reference",
        type=Path,
        help="Эталонное видео (профессионал)",
    )
    compare_parser.add_argument(
        "--overlays",
        type=str,
        default="skeleton,axis,angles,timer",
        help="Оверлеи: skeleton,axis,angles,timer (default: all)",
    )
    compare_parser.add_argument(
        "--mode",
        type=str,
        choices=["side-by-side", "overlay"],
        default="side-by-side",
        help="Режим сравнения (default: side-by-side)",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Выходной файл (default: <athlete>_compare.mp4)",
    )
    compare_parser.add_argument(
        "--resize",
        type=int,
        default=1280,
        help="Ширина кадра (default: 1280)",
    )
    compare_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Пересчитать позы (игнорировать кеш)",
    )
    compare_parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="FPS выхода (0 = авто)",
    )
    compare_parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Макс кадров (0 = все)",
    )
    compare_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Устройство: 'auto' (default), 'cuda', 'cpu', или GPU index '0'",
    )

    args = parser.parse_args()

    # Dispatch command
    if args.command == "analyze":
        sys.exit(cmd_analyze(args))
    elif args.command == "build-ref":
        sys.exit(cmd_build_ref(args))
    elif args.command == "segment":
        sys.exit(cmd_segment(args))
    elif args.command == "compare":
        sys.exit(cmd_compare(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
