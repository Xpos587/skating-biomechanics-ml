"""Spatial reference detection for camera pose estimation.

Detects camera orientation relative to the ice rink and compensates
pose measurements to be independent of camera tilt.

Key concepts:
- Horizon line detection using Hough transform
- Camera pose estimation (roll, pitch, yaw)
- Pose compensation using rotation matrices
- XYZ axes visualization

Based on research from:
- Guo et al. (2022) "A Fast and Simple Method for Absolute Orientation"
- Kashany & Pourreza (2010) "Camera pose estimation in soccer scenes"
- Exa/Gemini spatial reference research (2026-03-28)
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..types import PixelPose


@dataclass
class CameraPose:
    """Camera orientation relative to the ice rink.

    All angles in degrees.
    - roll: rotation around Z axis (tilt left/right). Positive = tilted right.
    - pitch: rotation around X axis (tilt forward/backward). Positive = tilted down.
    - yaw: rotation around Y axis (pan left/right). Positive = panned right.
    - confidence: 0-1, how reliable the estimate is.
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    source: Literal["imu", "horizon", "gravity"] = "gravity"

    def as_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix.

        Returns:
            R such that v_world = R @ v_camera
        """
        # Create rotation from euler angles (XYZ convention)
        # Note: We want the inverse rotation to compensate camera tilt
        r = Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw], degrees=True)
        return r.as_matrix()

    def inverse_matrix(self) -> np.ndarray:
        """Get inverse rotation matrix for compensating poses.

        Returns:
            R_inv such that v_compensated = R_inv @ v_camera
        """
        r = Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw], degrees=True)
        return r.inv().as_matrix()


class SpatialReferenceDetector:
    """Detect camera pose and compensate pose measurements.

    Uses a hierarchy of methods:
    1. IMU data (if available) - most accurate
    2. Horizon line detection (Hough transform)
    3. Gravity prior (assume level camera) - fallback

    Example:
        detector = SpatialReferenceDetector()
        camera_pose = detector.estimate_pose(frame)
        poses_compensated = detector.compensate_poses(poses, camera_pose)
        detector.draw_axes(frame, camera_pose)
    """

    def __init__(
        self,
        hough_threshold: int = 100,
        hough_min_line_length: int = 100,
        hough_max_line_gap: int = 10,
        horizon_angle_tolerance: float = 5.0,
    ):
        """Initialize spatial reference detector.

        Args:
            hough_threshold: Minimum votes for Hough line detection
            hough_min_line_length: Minimum line length in pixels
            hough_max_line_gap: Maximum gap between line segments
            horizon_angle_tolerance: Expected horizon angle range (degrees)
        """
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.horizon_angle_tolerance = horizon_angle_tolerance

        # Smoothing for temporal consistency
        self._roll_history: list[float] = []
        self._pitch_history: list[float] = []
        self._smoothing_window = 5

    def estimate_pose(
        self, frame: np.ndarray, imu_data: dict[str, float] | None = None
    ) -> CameraPose:
        """Estimate camera pose from frame or IMU data.

        Args:
            frame: Video frame (BGR format)
            imu_data: Optional dict with 'roll', 'pitch', 'yaw' in degrees

        Returns:
            CameraPose with estimated orientation
        """
        # Method 1: IMU data (most accurate)
        if imu_data is not None:
            roll = imu_data.get("roll", 0.0)
            pitch = imu_data.get("pitch", 0.0)
            yaw = imu_data.get("yaw", 0.0)
            return CameraPose(roll=roll, pitch=pitch, yaw=yaw, confidence=1.0, source="imu")

        # Method 2: Horizon line detection
        horizon_pose = self._estimate_from_horizon(frame)
        if horizon_pose.confidence > 0.3:
            # Apply temporal smoothing
            horizon_pose.roll = self._smooth_value(horizon_pose.roll, self._roll_history)
            horizon_pose.pitch = self._smooth_value(horizon_pose.pitch, self._pitch_history)
            return horizon_pose

        # Method 3: Gravity prior (fallback)
        return CameraPose(roll=0.0, pitch=0.0, yaw=0.0, confidence=0.0, source="gravity")

    def _estimate_from_horizon(self, frame: np.ndarray) -> CameraPose:
        """Estimate camera roll from horizon line using Hough transform.

        The horizon line is assumed to be horizontal (angle = 0).
        Any deviation indicates camera roll.

        Args:
            frame: Video frame (BGR format)

        Returns:
            CameraPose with estimated roll (pitch, yaw = 0)
        """
        h, w = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only look at top 1/3 of frame for horizon (ignore ice/skater)
        horizon_region = gray[: h // 3, :]

        # Edge detection (Canny) - lower thresholds for more edges
        edges = cv2.Canny(horizon_region, 30, 100, apertureSize=3)

        # Try multiple Hough parameter combinations for robustness
        all_angles = []

        # Configuration 1: Medium lines (most reliable for horizon)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=w // 4,  # At least 1/4 of frame width
            maxLineGap=20,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle
                dx = x2 - x1
                dy = y2 - y1
                angle = np.arctan2(dy, dx) * 180 / np.pi

                # Normalize to [-90, 90]
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                # Filter for near-horizontal lines
                if abs(angle) < 10:
                    all_angles.append(angle)

        if not all_angles:
            return CameraPose(roll=0.0, pitch=0.0, yaw=0.0, confidence=0.0)

        # Use mean instead of median to detect actual tilt
        # Median gives 0 for symmetric distributions
        roll = float(np.mean(all_angles))

        # Confidence based on number of horizontal lines found
        confidence = min(len(all_angles) / 3.0, 1.0)

        return CameraPose(roll=roll, pitch=0.0, yaw=0.0, confidence=confidence, source="horizon")

    def _smooth_value(self, value: float, history: list[float]) -> float:
        """Apply temporal smoothing to value.

        Args:
            value: Current value
            history: List of previous values

        Returns:
            Smoothed value
        """
        history.append(value)
        if len(history) > self._smoothing_window:
            history.pop(0)
        return float(np.mean(history))

    def compensate_poses(self, poses: PixelPose, camera_pose: CameraPose) -> PixelPose:
        """Compensate poses for camera tilt.

        Rotates keypoints so that measurements are relative to true
        vertical/horizontal instead of camera frame.

        Args:
            poses: Pose array (N, 33, 2) in pixel coordinates
            camera_pose: Estimated camera orientation

        Returns:
            Compensated poses with same shape as input
        """
        if camera_pose.confidence < 0.1:
            # Low confidence, skip compensation
            return poses

        # For 2D poses, we only compensate for roll (rotation around Z axis)
        # This is equivalent to rotating in the image plane
        roll_rad = np.deg2rad(camera_pose.roll)
        cos_roll = np.cos(-roll_rad)  # Negative for compensation
        sin_roll = np.sin(-roll_rad)

        # Rotation matrix for 2D (around Z axis)
        R_2d = np.array([[cos_roll, -sin_roll], [sin_roll, cos_roll]])

        # Center of rotation (image center)
        # Assuming poses are already centered or we compensate in-place
        poses_compensated = poses.copy()

        # Rotate each keypoint
        for i in range(len(poses)):
            for j in range(poses.shape[1]):
                if poses[i, j, 0] < 0 or poses[i, j, 1] < 0:
                    # Invalid keypoint (x, y = -1)
                    continue
                point = poses[i, j, :2]
                rotated = R_2d @ point
                poses_compensated[i, j, 0] = rotated[0]
                poses_compensated[i, j, 1] = rotated[1]

        return poses_compensated

    def draw_axes(
        self,
        frame: np.ndarray,
        camera_pose: CameraPose,
        origin: tuple[int, int] = (50, 50),
        length: int = 40,
        font_scale: float = 0.4,
    ) -> np.ndarray:
        """Draw XYZ axes on frame to visualize spatial reference.

        Color coding:
        - X axis (red): parallel to ice, horizontal
        - Y axis (green): forward direction (depth)
        - Z axis (blue): vertical (up)

        Args:
            frame: Video frame (BGR format)
            camera_pose: Estimated camera orientation
            origin: Pixel position for axes origin (x, y)
            length: Length of each axis in pixels
            font_scale: Font scale for labels

        Returns:
            Frame with axes drawn
        """
        frame = frame.copy()

        # Rotation matrix from camera pose
        R = camera_pose.as_rotation_matrix()

        # Axis directions in world space
        axes_world = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Rotate axes by camera pose (to show how they appear from camera perspective)
        axes_camera = R @ axes_world.T

        # Project to 2D (simple orthographic projection)
        # We only show X and Z since Y is depth
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=B, Y=G, Z=R
        labels = ["X", "Y", "Z"]

        for i, (axis, color, label) in enumerate(zip(axes_camera.T, colors, labels, strict=False)):
            # Project 3D to 2D (simple orthographic)
            # For visualization, we use X and Z components
            if i == 1:  # Y axis (depth) - draw at angle
                end_x = int(origin[0] + axis[0] * length * 0.5)
                end_y = int(origin[1] - axis[2] * length * 0.5)
            else:  # X and Z axes
                end_x = int(origin[0] + axis[0] * length)
                end_y = int(origin[1] - axis[2] * length)

            # Draw axis line
            cv2.line(frame, origin, (end_x, end_y), color, 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (end_x + 5, end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
            )

        # Draw pose info text
        info_text = [
            f"Roll: {camera_pose.roll:.1f}°",
            f"Pitch: {camera_pose.pitch:.1f}°",
            f"Source: {camera_pose.source}",
            f"Conf: {camera_pose.confidence:.2f}",
        ]

        y_offset = origin[1] + length + 20
        for line in info_text:
            cv2.putText(
                frame,
                line,
                (origin[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )
            y_offset += 15

        return frame


def detect_horizon_angle(
    frame: np.ndarray,
    threshold: int = 100,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    angle_tolerance: float = 5.0,
) -> tuple[float, float]:
    """Detect horizon line angle in frame.

    Convenience function for quick horizon detection.

    Args:
        frame: Video frame (BGR format)
        threshold: Hough transform threshold
        min_line_length: Minimum line length
        max_line_gap: Maximum gap between segments
        angle_tolerance: Expected horizon angle range

    Returns:
        (angle, confidence) tuple where angle is in degrees
    """
    detector = SpatialReferenceDetector(
        hough_threshold=threshold,
        hough_min_line_length=min_line_length,
        hough_max_line_gap=max_line_gap,
        horizon_angle_tolerance=angle_tolerance,
    )
    pose = detector._estimate_from_horizon(frame)
    return pose.roll, pose.confidence


def compensate_angles_for_camera(
    angles: np.ndarray,
    camera_roll: float,
) -> np.ndarray:
    """Compensate angle measurements for camera roll.

    Args:
        angles: Array of angles in degrees
        camera_roll: Camera roll angle in degrees

    Returns:
        Compensated angles (true angles relative to vertical)
    """
    return angles - camera_roll


def estimate_pose_sequence(
    video_path: str,
    interval: int = 30,
    fps: float = 30.0,
) -> list[tuple[int, CameraPose]]:
    """Estimate camera pose at regular intervals throughout a video.

    Opens the video, samples frames at the given cadence, and runs
    SpatialReferenceDetector.estimate_pose() on each sampled frame.
    Camera pose estimates are smoothed with a One-Euro Filter for
    temporal consistency.

    Args:
        video_path: Path to the video file.
        interval: Sample every N-th frame.
        fps: Video frame rate (used for One-Euro Filter timing).

    Returns:
        List of (frame_idx, CameraPose) tuples. Only includes frames
        where person detection confidence > 0.1.
    """
    from ..utils.smoothing import OneEuroFilter

    detector = SpatialReferenceDetector()

    # One-Euro Filters for roll/pitch/yaw smoothing
    roll_filter = OneEuroFilter(freq=fps / interval, min_cutoff=1.0, beta=0.01)
    pitch_filter = OneEuroFilter(freq=fps / interval, min_cutoff=1.0, beta=0.01)
    yaw_filter = OneEuroFilter(freq=fps / interval, min_cutoff=1.0, beta=0.01)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    results: list[tuple[int, CameraPose]] = []
    frame_idx = 0
    sample_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                pose = detector.estimate_pose(frame)

                if pose.confidence > 0.1:
                    # Apply One-Euro smoothing
                    t = sample_idx / (fps / interval) if fps > 0 else float(sample_idx)
                    smoothed_roll = roll_filter.filter_sample(t, pose.roll)
                    smoothed_pitch = pitch_filter.filter_sample(t, pose.pitch)
                    smoothed_yaw = yaw_filter.filter_sample(t, pose.yaw)

                    results.append(
                        (
                            frame_idx,
                            CameraPose(
                                roll=smoothed_roll,
                                pitch=smoothed_pitch,
                                yaw=smoothed_yaw,
                                confidence=pose.confidence,
                                source=pose.source,
                            ),
                        )
                    )
                sample_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    return results


def compensate_poses_per_frame(
    poses: np.ndarray,
    camera_poses: list[tuple[int, CameraPose]],
    frame_indices: np.ndarray | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
) -> np.ndarray:
    """Compensate normalized poses using per-frame camera estimates.

    For each frame, the nearest camera pose estimate is used to apply
    roll compensation.  Camera poses are linearly interpolated between
    sparse samples for smooth transitions.

    Args:
        poses: Normalized poses (N, 17, 3) with x, y, confidence.
            May contain NaN values for missing frames.
        camera_poses: List of (frame_idx, CameraPose) from
            estimate_pose_sequence.
        frame_indices: Optional array of frame indices corresponding to
            each pose.  If None, assumes sequential 0..N-1.
        video_width: Video width in pixels (unused, kept for API compat).
        video_height: Video height in pixels (unused, kept for API compat).

    Returns:
        Compensated poses in normalized coordinates, same shape as input.
        NaN frames are preserved unchanged.
    """
    if not camera_poses:
        return poses.copy()

    num_frames = poses.shape[0]
    if frame_indices is None:
        frame_indices = np.arange(num_frames)

    # Build interpolated roll array for all frames
    sample_indices = np.array([cp[0] for cp in camera_poses], dtype=float)
    sample_rolls = np.array([cp[1].roll for cp in camera_poses], dtype=float)

    # Linear interpolation of roll across all frames
    if frame_indices is not None:
        target_indices = frame_indices.astype(float)
    else:
        target_indices = np.array([], dtype=float)
    if len(sample_indices) >= 2:
        interpolated_rolls = np.interp(
            target_indices,
            sample_indices,
            sample_rolls,
            left=sample_rolls[0],
            right=sample_rolls[-1],
        )
    else:
        # Only one sample -- use it everywhere
        interpolated_rolls = np.full(num_frames, sample_rolls[0])

    # Apply roll compensation per frame
    compensated = poses.copy()
    for i in range(num_frames):
        roll_deg = interpolated_rolls[i]
        if abs(roll_deg) < 0.01:
            continue  # Skip near-identity transforms

        roll_rad = np.deg2rad(roll_deg)
        cos_r = np.cos(-roll_rad)
        sin_r = np.sin(-roll_rad)

        for j in range(poses.shape[1]):
            x, y = compensated[i, j, 0], compensated[i, j, 1]
            # Skip NaN keypoints
            if np.isnan(x) or np.isnan(y):
                continue
            # Rotate around center (0.5, 0.5) in normalized space
            dx = x - 0.5
            dy = y - 0.5
            compensated[i, j, 0] = cos_r * dx - sin_r * dy + 0.5
            compensated[i, j, 1] = sin_r * dx + cos_r * dy + 0.5

    return compensated
