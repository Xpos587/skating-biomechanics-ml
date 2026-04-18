"""Multi-GPU pose extraction for parallel processing.

Distributes video chunks across multiple GPUs for faster processing.
Each GPU processes a separate chunk of the video in parallel.

Usage:
    from skating_ml.device import MultiGPUConfig
    from skating_ml.pose_estimation import MultiGPUPoseExtractor

    config = MultiGPUConfig()  # Auto-detect all GPUs
    extractor = MultiGPUPoseExtractor(config=config)
    poses = extractor.extract_video_tracked(video_path)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..device import MultiGPUConfig
from ..types import PersonClick, TrackedExtraction
from ..utils.video import get_video_meta

logger = logging.getLogger(__name__)


class MultiGPUPoseExtractor:
    """Distribute pose extraction across multiple GPUs.

    Splits video into chunks and processes each chunk on a separate GPU
    in parallel using multiprocessing. Results are merged back together.

    Args:
        config: Multi-GPU configuration.
        output_format: Output coordinate format ("normalized" or "pixels").
        conf_threshold: Confidence threshold for poses.
        mode: RTMO model mode ("lightweight", "balanced", "performance").
    """

    def __init__(
        self,
        config: MultiGPUConfig | None = None,
        output_format: str = "normalized",
        conf_threshold: float = 0.5,
        mode: str = "balanced",
    ) -> None:
        """Initialize multi-GPU pose extractor.

        Args:
            config: Multi-GPU configuration.
            output_format: Output coordinate format.
            conf_threshold: Confidence threshold for poses.
            mode: RTMO model mode.
        """
        self.config = config or MultiGPUConfig()
        self.output_format = output_format
        self.conf_threshold = conf_threshold
        self.mode = mode

    def extract_video_tracked(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
    ) -> TrackedExtraction:
        """Extract poses from video, distributing work across GPUs.

        Strategy: Split video into chunks, process each chunk on separate GPU.

        Args:
            video_path: Path to video file.
            person_click: Optional person selection (only works for chunk 0).

        Returns:
            TrackedExtraction with merged poses from all chunks.
        """
        video_path = Path(video_path)
        meta = get_video_meta(video_path)
        total_frames = meta.num_frames

        # Single GPU or CPU fallback
        if len(self.config.enabled_gpus) <= 1:
            logger.info("Single GPU or CPU mode, using sequential extraction")
            return self._extract_single_gpu(video_path, person_click)

        # Split into chunks (one per GPU)
        num_gpus = len(self.config.enabled_gpus)
        chunk_size = total_frames // num_gpus

        chunks = []
        for i in range(num_gpus):
            start_frame = i * chunk_size
            end_frame = total_frames if i == num_gpus - 1 else (i + 1) * chunk_size
            chunks.append((i, start_frame, end_frame))

        # Process chunks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    self._extract_chunk,
                    str(video_path),
                    gpu_idx,
                    start_frame,
                    end_frame,
                    self.config.enabled_gpus[gpu_idx].device_id,
                    self.output_format,
                    self.conf_threshold,
                    self.mode,
                ): (gpu_idx, start_frame, end_frame)
                for gpu_idx, start_frame, end_frame in chunks
            }

            for future in as_completed(futures):
                gpu_idx, start_frame, end_frame = futures[future]
                try:
                    result = future.result()
                    results.append((start_frame, result))
                except Exception as e:
                    logger.error(f"Chunk {start_frame}-{end_frame} failed on GPU {gpu_idx}: {e}")
                    raise

        # Sort results by frame order
        results.sort(key=lambda x: x[0])

        # Merge results
        return self._merge_chunks(results, meta, person_click)

    def _extract_single_gpu(
        self,
        video_path: Path,
        person_click: PersonClick | None,
    ) -> TrackedExtraction:
        """Fallback: extract on single GPU or CPU."""
        from .pose_extractor import PoseExtractor

        device = self.config.get_device_for_worker(0)
        extractor = PoseExtractor(
            output_format=self.output_format,
            device=device,
            conf_threshold=self.conf_threshold,
            mode=self.mode,
        )
        return extractor.extract_video_tracked(video_path, person_click=person_click)

    @staticmethod
    def _extract_chunk(
        video_path: str,
        _gpu_idx: int,
        start_frame: int,
        end_frame: int,
        device_id: int,
        output_format: str,
        conf_threshold: float,
        mode: str,
    ) -> dict:
        """Extract poses from a video chunk on specific GPU.

        This runs in a separate process.

        Args:
            video_path: Path to video file.
            _gpu_idx: Chunk index (for logging, unused).
            start_frame: First frame to process (inclusive).
            end_frame: Last frame to process (exclusive).
            device_id: GPU device ID.
            output_format: Output coordinate format.
            conf_threshold: Confidence threshold.
            mode: RTMO model mode.

        Returns:
            Dict with extraction results.
        """
        import os

        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        from .pose_extractor import PoseExtractor

        extractor = PoseExtractor(
            output_format=output_format,
            device="cuda",  # Will use CUDA_VISIBLE_DEVICES
            conf_threshold=conf_threshold,
            mode=mode,
        )

        # Extract from video
        video_path_obj = Path(video_path)
        result = extractor.extract_video_tracked(video_path_obj, person_click=None)

        # Slice to chunk range
        poses_chunk = result.poses[start_frame:end_frame]
        frame_indices_chunk = result.frame_indices[start_frame:end_frame]

        return {
            "poses": poses_chunk,
            "frame_indices": frame_indices_chunk,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    def _merge_chunks(
        self,
        results: list[tuple[int, dict]],
        meta,
        person_click: PersonClick | None,
    ) -> TrackedExtraction:
        """Merge results from multiple chunks.

        Args:
            results: List of (start_frame, result_dict) tuples.
            meta: Video metadata.
            person_click: Optional person selection.

        Returns:
            Merged TrackedExtraction.
        """
        total_frames = meta.num_frames

        # Pre-allocate merged array
        merged_poses = np.full((total_frames, 17, 3), np.nan, dtype=np.float32)

        # Copy each chunk into merged array
        for start_frame, result in results:
            poses = result["poses"]
            end_frame = result["end_frame"]
            merged_poses[start_frame:end_frame] = poses

        # Find first valid frame
        valid_mask = ~np.isnan(merged_poses[:, 0, 0])
        if not np.any(valid_mask):
            raise ValueError("No valid pose detected in video")

        first_detection_frame = int(np.argmax(valid_mask))

        return TrackedExtraction(
            poses=merged_poses,
            frame_indices=np.arange(total_frames),
            first_detection_frame=first_detection_frame,
            target_track_id=None,  # Track IDs not preserved across chunks
            fps=meta.fps,
            video_meta=meta,
        )
