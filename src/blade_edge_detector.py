"""Blade edge detection for figure skating analysis (2D/BlazePose 33kp format).

.. DEPRECATED::
    This module is deprecated for H3.6M 17kp 3D format.
    Use `blade_edge_detector_3d.py` with `BladeEdgeDetector3D` class instead.

    The 2D detector relies on heel and foot_index keypoints that don't exist
    in H3.6M format. For 3D poses, use BladeEdgeDetector3D which uses 3D
    foot velocity and body lean angles for edge detection.

Based on the BDA (Blade Discrimination Algorithm) research:
"Automated Blade Type Discrimination Algorithm for Figure Skating Based on MediaPipe"

The algorithm uses foot angle, ankle angle, knee angle, and vertical acceleration
to classify blade state: inside edge, outside edge, flat, or toe pick.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import BKey, BladeType, NormalizedPose


def angle_with_horizontal(x: float, y: float) -> float:
    """Calculate angle of vector with horizontal axis in degrees.

    Positive = counterclockwise from horizontal (pointing right).
    Negative = clockwise from horizontal.

    Args:
        x: Horizontal component.
        y: Vertical component (up is positive in normalized coords).

    Returns:
        Angle in degrees [-180, 180].
    """
    angle = float(np.degrees(np.arctan2(y, x)))
    # For biomechanics, we want pure left/right to be unambiguous
    # Convert 180° to -180° for consistent classification
    if angle == 180:
        angle = -180
    return angle


def calculate_foot_vector(
    poses: NormalizedPose,
    frame_idx: int,
    foot: str = "left",
) -> NDArray[np.float32]:
    """Calculate foot vector (from ankle to toe).

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.
        foot: Either "left" or "right".

    Returns:
        Foot vector [dx, dy].
    """
    if foot == "left":
        ankle_idx, toe_idx = BKey.LEFT_ANKLE, BKey.LEFT_FOOT_INDEX
    else:
        ankle_idx, toe_idx = BKey.RIGHT_ANKLE, BKey.RIGHT_FOOT_INDEX

    ankle = poses[frame_idx, ankle_idx]
    toe = poses[frame_idx, toe_idx]

    return toe - ankle


def calculate_lower_leg_vector(
    poses: NormalizedPose,
    frame_idx: int,
    leg: str = "left",
) -> NDArray[np.float32]:
    """Calculate lower leg vector (from knee to ankle).

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.
        leg: Either "left" or "right".

    Returns:
        Lower leg vector [dx, dy].
    """
    if leg == "left":
        knee_idx, ankle_idx = BKey.LEFT_KNEE, BKey.LEFT_ANKLE
    else:
        knee_idx, ankle_idx = BKey.RIGHT_KNEE, BKey.RIGHT_ANKLE

    knee = poses[frame_idx, knee_idx]
    ankle = poses[frame_idx, ankle_idx]

    return ankle - knee


def calculate_motion_direction(
    poses: NormalizedPose,
    frame_idx: int,
    window: int = 5,
) -> NDArray[np.float32]:
    """Calculate motion direction from trajectory of mid-hip.

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Center frame index.
        window: Number of frames to average (must be odd).

    Returns:
        Motion direction vector [dx, dy] (normalized).
    """
    if window % 2 == 0:
        window += 1

    half_window = window // 2
    start = max(0, frame_idx - half_window)
    end = min(len(poses), frame_idx + half_window + 1)

    # Calculate mid-hip trajectory
    mid_hips = (poses[start:end, BKey.LEFT_HIP] + poses[start:end, BKey.RIGHT_HIP]) / 2

    if len(mid_hips) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    # Velocity is last - first position
    velocity = mid_hips[-1] - mid_hips[0]

    # Normalize
    norm = np.linalg.norm(velocity)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)

    return velocity / norm


def calculate_foot_angle(
    poses: NormalizedPose,
    frame_idx: int,
    foot: str = "left",
    motion_window: int = 5,
) -> float:
    """Calculate foot angle relative to motion direction.

    Positive angle = foot pointing outward (right of motion).
    Negative angle = foot pointing inward (left of motion).

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.
        foot: Either "left" or "right".
        motion_window: Window for motion direction calculation.

    Returns:
        Foot angle in degrees. Positive = outward, Negative = inward.
    """
    foot_vector = calculate_foot_vector(poses, frame_idx, foot)
    motion_dir = calculate_motion_direction(poses, frame_idx, motion_window)

    # Angle between foot vector and motion direction
    # Use cross product sign to determine direction
    foot_angle = angle_with_horizontal(foot_vector[0], foot_vector[1])
    motion_angle = angle_with_horizontal(motion_dir[0], motion_dir[1])

    relative_angle = foot_angle - motion_angle

    # Normalize to [-180, 180]
    if relative_angle > 180:
        relative_angle -= 360
    elif relative_angle < -180:
        relative_angle += 360

    return float(relative_angle)


def calculate_ankle_angle(
    poses: NormalizedPose,
    frame_idx: int,
    leg: str = "left",
) -> float:
    """Calculate ankle flexion angle (knee-ankle-toe).

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.
        leg: Either "left" or "right".

    Returns:
        Ankle angle in degrees [0, 180].
    """
    from .geometry import angle_3pt

    if leg == "left":
        knee_idx, ankle_idx, toe_idx = BKey.LEFT_KNEE, BKey.LEFT_ANKLE, BKey.LEFT_FOOT_INDEX
    else:
        knee_idx, ankle_idx, toe_idx = BKey.RIGHT_KNEE, BKey.RIGHT_ANKLE, BKey.RIGHT_FOOT_INDEX

    knee = poses[frame_idx, knee_idx]
    ankle = poses[frame_idx, ankle_idx]
    toe = poses[frame_idx, toe_idx]

    return angle_3pt(knee, ankle, toe)


def calculate_vertical_acceleration(
    poses: NormalizedPose,
    fps: float,
    frame_idx: int,
    leg: str = "left",
) -> float:
    """Calculate vertical acceleration of the foot.

    Positive = upward acceleration (takeoff).
    Negative = downward acceleration (landing).

    Args:
        poses: Normalized poses (N, 33, 2).
        fps: Frame rate.
        frame_idx: Frame index.
        leg: Either "left" or "right".

    Returns:
        Vertical acceleration in normalized units/s².
    """
    if frame_idx < 2 or frame_idx >= len(poses) - 2:
        return 0.0

    # Use central difference for velocity
    # y is inverted in normalized coords (0 = top, 1 = bottom)
    # So we negate to get "up" as positive
    if leg == "left":
        foot_idx = BKey.LEFT_FOOT_INDEX
    else:
        foot_idx = BKey.RIGHT_FOOT_INDEX

    # Calculate velocities
    v_prev = -poses[frame_idx - 1, foot_idx, 1]  # Negate for up=positive
    v_curr = -poses[frame_idx, foot_idx, 1]
    v_next = -poses[frame_idx + 1, foot_idx, 1]

    # Acceleration using central difference
    accel = (v_next - 2 * v_curr + v_prev) * (fps**2)
    return float(accel)


def detect_supporting_foot(
    poses: NormalizedPose,
    frame_idx: int,
) -> str | None:
    """Detect which foot is supporting weight.

    The supporting foot typically has:
    - Lower hip position (more weight on it)
    - Straighter knee (supporting vs swinging)

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.

    Returns:
        "left", "right", or None if unclear.
    """
    # Hip Y positions (higher value = lower in frame)
    left_hip_y = poses[frame_idx, BKey.LEFT_HIP, 1]
    right_hip_y = poses[frame_idx, BKey.RIGHT_HIP, 1]

    # Calculate knee angles (straighter = more support)
    from .geometry import angle_3pt

    left_knee_angle = angle_3pt(
        poses[frame_idx, BKey.LEFT_HIP],
        poses[frame_idx, BKey.LEFT_KNEE],
        poses[frame_idx, BKey.LEFT_ANKLE],
    )
    right_knee_angle = angle_3pt(
        poses[frame_idx, BKey.RIGHT_HIP],
        poses[frame_idx, BKey.RIGHT_KNEE],
        poses[frame_idx, BKey.RIGHT_ANKLE],
    )

    # Supporting foot: lower hip AND straighter knee
    # Lower hip means higher Y value in normalized coords
    hip_diff = left_hip_y - right_hip_y  # Positive = left hip lower

    # Knee angle: closer to 180° = straighter
    knee_diff = left_knee_angle - right_knee_angle  # Positive = left knee straighter

    # Combined score
    # If both indicators agree, confident
    # Lower thresholds for better detection
    if hip_diff > 0.01 and knee_diff > 5:
        return "left"
    elif hip_diff < -0.01 and knee_diff < -5:
        return "right"

    # Fallback: use hip position only (more reliable)
    if hip_diff > 0.02:
        return "left"
    elif hip_diff < -0.02:
        return "right"

    # If indicators disagree or difference is small, unclear
    return None


def calculate_path_curvature(
    poses: NormalizedPose,
    frame_idx: int,
    window: int = 10,
) -> float:
    """Calculate path curvature from mid-hip trajectory.

    Positive curvature = turning left (counter-clockwise)
    Negative curvature = turning right (clockwise)

    For edge detection:
    - Turning left + inside edge = left foot inside edge
    - Turning left + outside edge = right foot outside edge

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Center frame index.
        window: Number of frames before/after to use.

    Returns:
        Curvature (1/radius of curvature).
    """
    start = max(0, frame_idx - window)
    end = min(len(poses), frame_idx + window + 1)

    if end - start < 3:
        return 0.0

    # Get mid-hip trajectory
    mid_hips = []
    for i in range(start, end):
        mid_hip = (poses[i, BKey.LEFT_HIP] + poses[i, BKey.RIGHT_HIP]) / 2
        mid_hips.append(mid_hip)

    mid_hips = np.array(mid_hips)

    # Calculate curvature using discrete approach
    # Cross product of velocity vectors gives turning direction
    if len(mid_hips) < 3:
        return 0.0

    # Velocity at start and end of window
    v_start = mid_hips[1] - mid_hips[0]
    v_end = mid_hips[-1] - mid_hips[-2]

    # Cross product (2D): v1_x * v2_y - v1_y * v2_x
    cross = v_start[0] * v_end[1] - v_start[1] * v_end[0]

    # Normalize by magnitude
    v_start_mag = np.linalg.norm(v_start)
    v_end_mag = np.linalg.norm(v_end)

    if v_start_mag < 1e-6 or v_end_mag < 1e-6:
        return 0.0

    curvature = cross / (v_start_mag * v_end_mag)
    return float(curvature)


def calculate_body_lean_angle(
    poses: NormalizedPose,
    frame_idx: int,
) -> float:
    """Calculate body lean angle relative to vertical.

    Positive = leaning right (right shoulder lower)
    Negative = leaning left (left shoulder lower)

    For blade detection:
    - Turning left + leaning left = inside edge
    - Turning left + leaning right = outside edge

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.

    Returns:
        Lean angle in degrees. Positive = right, Negative = left.
    """
    # Shoulder positions
    left_shoulder = poses[frame_idx, BKey.LEFT_SHOULDER]
    right_shoulder = poses[frame_idx, BKey.RIGHT_SHOULDER]

    # Hip positions
    left_hip = poses[frame_idx, BKey.LEFT_HIP]
    right_hip = poses[frame_idx, BKey.RIGHT_HIP]

    # Midpoints
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    # Torso vector (from hip to shoulder)
    torso = mid_shoulder - mid_hip

    # Vertical vector (pointing up, so negative Y)
    vertical = np.array([0, -1])

    # Angle between torso and vertical
    # Use atan2 to get signed angle
    torso_angle = np.arctan2(torso[1], torso[0])
    vertical_angle = np.arctan2(vertical[1], vertical[0])

    lean_angle = float(np.degrees(torso_angle - vertical_angle))

    # Normalize to [-180, 180]
    if lean_angle > 180:
        lean_angle -= 360
    elif lean_angle < -180:
        lean_angle += 360

    return lean_angle


def classify_blade_from_lean_and_curvature(
    poses: NormalizedPose,
    frame_idx: int,
    curvature_window: int = 10,
) -> tuple[BladeType, float]:
    """Classify blade edge using body lean and path curvature.

    Key insight: During a turn, skater leans OPPOSITE to centrifugal force.
    - Inside edge: body leans INTO the turn
    - Outside edge: body leans OUT of the turn

    Examples:
    - Turning left (curvature > 0) + leaning left (lean < 0) = inside edge
    - Turning left (curvature > 0) + leaning right (lean > 0) = outside edge
    - Turning right (curvature < 0) + leaning right (lean > 0) = inside edge
    - Turning right (curvature < 0) + leaning left (lean < 0) = outside edge

    Args:
        poses: Normalized poses (N, 33, 2).
        frame_idx: Frame index.
        curvature_window: Window for path curvature calculation.

    Returns:
        (blade_type, confidence) tuple.
    """
    curvature = calculate_path_curvature(poses, frame_idx, curvature_window)
    lean_angle = calculate_body_lean_angle(poses, frame_idx)

    # Thresholds
    curvature_threshold = 0.01  # Minimum curvature to consider as turning
    lean_threshold = 2.0  # Minimum lean angle (degrees)

    # Check if we're actually turning
    if abs(curvature) < curvature_threshold:
        # Going straight - can't determine edge from lean
        return BladeType.FLAT, 0.5

    # Check if we have significant lean
    if abs(lean_angle) < lean_threshold:
        # Not leaning much - probably flat
        return BladeType.FLAT, 0.6

    # Determine edge from relationship between curvature and lean
    # Inside edge: lean and curvature have OPPOSITE signs
    # Outside edge: lean and curvature have SAME sign

    # curvature > 0 = turning left
    # lean_angle < 0 = leaning left (body goes left, torso points left)
    # If turning left AND leaning left = inside edge (opposite signs)

    if curvature * lean_angle < 0:
        # Opposite signs = inside edge
        blade_type = BladeType.INSIDE
        # Confidence based on magnitudes
        confidence = min(1.0, (abs(curvature) / 0.05) * (abs(lean_angle) / 10.0))
    else:
        # Same signs = outside edge
        blade_type = BladeType.OUTSIDE
        confidence = min(1.0, (abs(curvature) / 0.05) * (abs(lean_angle) / 10.0))

    return blade_type, max(0.3, min(1.0, confidence))

    y_positions = -poses[frame_idx - 2 : frame_idx + 3, foot_idx, 1]  # Negate for up=positive

    # Fit quadratic to get acceleration
    if len(y_positions) >= 3:
        # a = 2 * (second difference)
        accel = (y_positions[0] - 2 * y_positions[2] + y_positions[4]) * (fps**2) / 4
    else:
        accel = 0.0

    return float(accel)


@dataclass(frozen=True)
class BladeState:
    """Blade state at a single frame.

    Attributes:
        blade_type: Detected blade edge type.
        foot_angle: Foot angle relative to motion (degrees).
        ankle_angle: Ankle flexion angle (degrees).
        vertical_accel: Vertical acceleration (norm units/s²).
        confidence: Detection confidence [0, 1].
    """

    blade_type: BladeType
    foot_angle: float
    ankle_angle: float
    vertical_accel: float
    confidence: float


class BladeEdgeDetector:
    """Blade edge detection using improved physics-based approach (2D/BlazePose 33kp).

    .. DEPRECATED::
        Use `BladeEdgeDetector3D` from `blade_edge_detector_3d.py` for H3.6M 17kp format.
        This class requires heel and foot_index keypoints that don't exist in H3.6M.

    Uses multiple signals for robust classification:
    1. Body lean angle relative to turn direction (NEW - more accurate)
    2. Path curvature analysis
    3. Foot angle (original BDA algorithm - fallback)
    4. Toe pick detection via vertical acceleration

    Key insight: During a turn, skater leans OPPOSITE to centrifugal force.
    - Inside edge: body leans INTO the turn (lean opposite to curvature)
    - Outside edge: body leans OUT of the turn (lean same direction as curvature)

    Attributes:
        inside_threshold: Angle threshold for inside edge (negative, degrees).
        outside_threshold: Angle threshold for outside edge (positive, degrees).
        toe_pick_accel_threshold: Vertical acceleration threshold for toe pick.
        smoothing_window: Frames to smooth results over.
        use_lean_method: Use improved lean+curvature method (default: True).
    """

    def __init__(
        self,
        inside_threshold: float = -15.0,
        outside_threshold: float = 15.0,
        toe_pick_accel_threshold: float = 5.0,
        smoothing_window: int = 3,
        use_lean_method: bool = True,
    ):
        """Initialize blade edge detector.

        .. DEPRECATED::
            Use `BladeEdgeDetector3D` from `blade_edge_detector_3d.py` for H3.6M 17kp format.

        Args:
            inside_threshold: Foot angle threshold for inside edge (negative degrees).
            outside_threshold: Foot angle threshold for outside edge (positive degrees).
            toe_pick_accel_threshold: Vertical acceleration threshold for toe pick detection.
            smoothing_window: Number of frames for temporal smoothing.
            use_lean_method: Use improved lean+curvature classification (default: True).
        """
        import warnings

        warnings.warn(
            "BladeEdgeDetector is deprecated for H3.6M 17kp format. "
            "Use BladeEdgeDetector3D from blade_edge_detector_3d.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.inside_threshold = inside_threshold
        self.outside_threshold = outside_threshold
        self.toe_pick_accel_threshold = toe_pick_accel_threshold
        self.smoothing_window = smoothing_window
        self.use_lean_method = use_lean_method

    def classify_frame(
        self,
        poses: NormalizedPose,
        frame_idx: int,
        fps: float,
        foot: str = "left",
        check_supporting: bool = False,
    ) -> BladeState:
        """Classify blade state for a single frame.

        Args:
            poses: Normalized poses (N, 33, 2).
            frame_idx: Frame index.
            fps: Frame rate.
            foot: Either "left" or "right".
            check_supporting: If True, return UNKNOWN if foot is not supporting weight.

        Returns:
            BladeState with classification and intermediate values.
        """
        # Check if this foot is supporting weight
        if check_supporting:
            supporting = detect_supporting_foot(poses, frame_idx)
            if supporting != foot:
                # This foot is not supporting - return UNKNOWN with low confidence
                return BladeState(
                    blade_type=BladeType.UNKNOWN,
                    foot_angle=0.0,
                    ankle_angle=0.0,
                    vertical_accel=0.0,
                    confidence=0.0,
                )

        foot_angle = calculate_foot_angle(poses, frame_idx, foot)
        ankle_angle = calculate_ankle_angle(poses, frame_idx, foot)
        vertical_accel = calculate_vertical_acceleration(poses, fps, frame_idx, foot)

        # Toe pick detection: vertical acceleration spike
        if vertical_accel > self.toe_pick_accel_threshold:
            return BladeState(
                blade_type=BladeType.TOE_PICK,
                foot_angle=foot_angle,
                ankle_angle=ankle_angle,
                vertical_accel=vertical_accel,
                confidence=0.8,
            )

        # Try improved lean+curvature method first
        if self.use_lean_method:
            blade_type_lean, confidence_lean = classify_blade_from_lean_and_curvature(
                poses, frame_idx, curvature_window=10
            )

            # If lean method is confident, use it
            if confidence_lean > 0.5:
                return BladeState(
                    blade_type=blade_type_lean,
                    foot_angle=foot_angle,
                    ankle_angle=ankle_angle,
                    vertical_accel=vertical_accel,
                    confidence=confidence_lean,
                )

        # Fallback to foot angle method (original BDA)
        if foot_angle < self.inside_threshold:
            blade_type = BladeType.INSIDE
            confidence = min(1.0, abs(foot_angle) / 30.0)  # Higher confidence for stronger angles
        elif foot_angle > self.outside_threshold:
            blade_type = BladeType.OUTSIDE
            confidence = min(1.0, abs(foot_angle) / 30.0)
        else:
            blade_type = BladeType.FLAT
            # Confidence is highest near 0 degrees (true flat)
            confidence = 1.0 - (
                abs(foot_angle) / max(abs(self.inside_threshold), abs(self.outside_threshold))
            )

        return BladeState(
            blade_type=blade_type,
            foot_angle=foot_angle,
            ankle_angle=ankle_angle,
            vertical_accel=vertical_accel,
            confidence=confidence,
        )

    def detect_sequence(
        self,
        poses: NormalizedPose,
        fps: float,
        foot: str = "left",
        check_supporting: bool = False,
    ) -> list[BladeState]:
        """Classify blade state for entire pose sequence.

        Args:
            poses: Normalized poses (N, 33, 2).
            fps: Frame rate.
            foot: Either "left" or "right".
            check_supporting: If True, only return blade states for frames where
                this foot is supporting weight.

        Returns:
            List of BladeState for each frame.
        """
        num_frames = len(poses)
        states = []

        for frame_idx in range(num_frames):
            state = self.classify_frame(
                poses, frame_idx, fps, foot, check_supporting=check_supporting
            )
            states.append(state)

        # Apply temporal smoothing
        if self.smoothing_window > 1:
            states = self._smooth_states(states)

        return states

    def _smooth_states(self, states: list[BladeState]) -> list[BladeState]:
        """Apply temporal smoothing to blade states.

        Uses majority voting within sliding window to reduce flickering.

        Args:
            states: Raw blade states.

        Returns:
            Smoothed blade states.
        """
        if not states:
            return states

        window = self.smoothing_window
        smoothed = []

        for i in range(len(states)):
            start = max(0, i - window // 2)
            end = min(len(states), i + window // 2 + 1)
            window_states = states[start:end]

            # Filter out UNKNOWN states for majority vote
            valid_states = [s for s in window_states if s.blade_type != BladeType.UNKNOWN]

            if not valid_states:
                # All UNKNOWN - keep as UNKNOWN
                smoothed.append(states[i])
                continue

            # Count occurrences of each blade type (excluding UNKNOWN)
            type_counts: dict[BladeType, int] = {}
            conf_sum = 0.0
            foot_angle_sum = 0.0
            ankle_angle_sum = 0.0
            accel_sum = 0.0

            for s in valid_states:
                type_counts[s.blade_type] = type_counts.get(s.blade_type, 0) + 1
                conf_sum += s.confidence
                foot_angle_sum += s.foot_angle
                ankle_angle_sum += s.ankle_angle
                accel_sum += s.vertical_accel

            # Majority vote (excluding UNKNOWN)
            dominant_type = max(type_counts, key=type_counts.get)

            # Average values (only from valid states)
            n = len(valid_states)
            smoothed.append(
                BladeState(
                    blade_type=dominant_type,
                    foot_angle=foot_angle_sum / n,
                    ankle_angle=ankle_angle_sum / n,
                    vertical_accel=accel_sum / n,
                    confidence=conf_sum / n,
                )
            )

        return smoothed

    def detect_takeoff_landing(
        self,
        states: list[BladeState],
        fps: float,
    ) -> tuple[int | None, int | None]:
        """Detect takeoff and landing frames from blade state sequence.

        Takeoff: transition from edge to toe_pick or strong upward acceleration.
        Landing: transition from toe_pick/air to edge.

        Args:
            states: Blade state sequence.
            fps: Frame rate.

        Returns:
            Tuple of (takeoff_frame, landing_frame). None if not detected.
        """
        takeoff = None
        landing = None

        # Find takeoff: first transition to toe_pick or strong accel
        for i in range(len(states)):
            if states[i].blade_type == BladeType.TOE_PICK:
                takeoff = i
                break
            elif states[i].vertical_accel > self.toe_pick_accel_threshold * 0.8:
                takeoff = i
                break

        # Find landing: return to edge after takeoff
        if takeoff is not None:
            for i in range(takeoff + 1, len(states)):
                if (
                    states[i].blade_type in (BladeType.INSIDE, BladeType.OUTSIDE, BladeType.FLAT)
                    and states[i].vertical_accel < 0  # Downward acceleration
                ):
                    landing = i
                    break

        return takeoff, landing

    def get_blade_summary(
        self,
        states: list[BladeState],
    ) -> dict[str, any]:
        """Get summary statistics for blade state sequence.

        Args:
            states: Blade state sequence.

        Returns:
            Dictionary with summary statistics.
        """
        if not states:
            return {}

        type_counts: dict[BladeType, int] = {}
        total_confidence = 0.0

        for s in states:
            type_counts[s.blade_type] = type_counts.get(s.blade_type, 0) + 1
            total_confidence += s.confidence

        # Calculate percentages
        n = len(states)
        type_percentages = {bt.value: count / n * 100 for bt, count in type_counts.items()}

        return {
            "total_frames": n,
            "type_percentages": type_percentages,
            "average_confidence": total_confidence / n,
            "dominant_edge": max(type_counts, key=type_counts.get).value
            if type_counts
            else "unknown",
        }


__all__ = [
    "BladeState",
    "BladeEdgeDetector",
    "calculate_foot_vector",
    "calculate_foot_angle",
    "calculate_ankle_angle",
    "calculate_vertical_acceleration",
    "calculate_motion_direction",
    "angle_with_horizontal",
    "detect_supporting_foot",
    "calculate_path_curvature",
    "calculate_body_lean_angle",
    "classify_blade_from_lean_and_curvature",
]
