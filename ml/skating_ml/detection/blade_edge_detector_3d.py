"""Enhanced 3D-aware blade edge detection.

Uses 3D pose data from MotionAGFormer for improved blade state detection.
Based on JudgeAI-LutzEdge research (83% accuracy with 3D pose + IMU).

Key improvements over 2D detection:
1. Motion direction from 3D velocity vectors
2. Ice surface tracking (x, z plane)
3. Weight distribution from knee angles
4. Blade zone detection (toe pick, rocker, heel)
5. Enhanced edge classification (inside/outside/flat)
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..types import BladeState3D, BladeType, IceTrace, MotionDirection


class BladeZone(Enum):
    """Blade zones for detailed detection.

    Skate blade has 3 main zones:
    - Toe pick: Front 1/3, used for jumps/toe loops
    - Rocker: Middle 1/3, used for gliding/edges
    - Heel: Back 1/3, rarely used (mainly for landing stability)
    """

    TOE_PICK = "toe_pick"  # 0-33% of blade length
    ROCKER_FRONT = "rocker_front"  # 33-50% (front rocker)
    ROCKER_MIDDLE = "rocker_middle"  # 50-67% (middle)
    ROCKER_BACK = "rocker_back"  # 67-83% (back rocker)
    HEEL = "heel"  # 83-100% of blade length


@dataclass
class DetectionConfig:
    """Configuration for 3D blade detection."""

    # Edge angle thresholds (degrees)
    inside_threshold: float = -15.0  # < -15° = inside edge
    outside_threshold: float = 15.0  # > 15° = outside edge
    flat_tolerance: float = 10.0  # -10° to 10° = flat

    # Velocity thresholds for motion detection
    min_velocity: float = 0.01  # m/s below this = stationary
    rotation_threshold: float = 0.5  # angular velocity for rotation detection

    # Weight distribution thresholds
    weight_on_toe_threshold: float = 0.6  # knee angle < 60° = weight on toe
    weight_on_heel_threshold: float = 0.4  # knee angle > 80° = weight distributed

    # Smoothing
    velocity_window: int = 5  # frames for velocity smoothing
    direction_window: int = 10  # frames for direction smoothing


class BladeEdgeDetector3D:
    """3D-aware blade edge detector.

    Uses 3D pose data to detect:
    - Blade edge (inside/outside/flat)
    - Blade zone (toe pick/rocker/heel)
    - Motion direction (forward/backward/left/right/rotation)
    - Ice trace (path on ice surface)
    """

    def __init__(self, config: DetectionConfig | None = None, fps: float = 30.0):
        """Initialize the 3D blade detector.

        Args:
            config: Detection configuration (default: DetectionConfig())
            fps: Frame rate for velocity calculations
        """
        self.config = config or DetectionConfig()
        self.fps = fps

        # History buffers for smoothing
        self._velocity_history: dict[str, deque] = {
            "left": deque(maxlen=self.config.velocity_window),
            "right": deque(maxlen=self.config.velocity_window),
        }
        self._position_history: dict[str, deque] = {
            "left": deque(maxlen=100),  # For ice trace
            "right": deque(maxlen=100),
        }
        self._direction_history: dict[str, deque] = {
            "left": deque(maxlen=self.config.direction_window),
            "right": deque(maxlen=self.config.direction_window),
        }

    def detect_frame(
        self,
        pose_3d: NDArray[np.float32],  # (17, 3) H3.6M format
        frame_idx: int,
        foot: str = "left",
    ) -> BladeState3D:
        """Detect blade state for a single frame using 3D pose.

        Args:
            pose_3d: 3D pose in H3.6M format (17 joints, 3 coords)
            frame_idx: Current frame index
            foot: "left" or "right"

        Returns:
            BladeState3D with full 3D information
        """
        # Map H3.6M indices to our keypoints
        if foot == "left":
            hip_idx, knee_idx, ankle_idx, foot_idx = 0, 1, 2, 3  # Left leg in H3.6M
        else:
            hip_idx, knee_idx, ankle_idx, foot_idx = 6, 7, 8, 9  # Right leg in H3.6M

        # Get 3D positions
        hip = pose_3d[hip_idx]
        knee = pose_3d[knee_idx]
        ankle = pose_3d[ankle_idx]
        foot_pos = pose_3d[foot_idx]

        # Calculate foot angle (relative to vertical)
        # Vector from ankle to foot (pointing down/toe direction)
        foot_vector = foot_pos - ankle
        foot_angle = np.degrees(np.arctan2(foot_vector[0], foot_vector[1]))  # Angle in X-Z plane

        # Calculate ankle flexion angle
        # Vector from knee to ankle
        shin_vector = ankle - knee
        shin_angle = np.degrees(np.arctan2(shin_vector[0], shin_vector[1]))

        # Calculate knee flexion angle (for weight distribution)
        knee_angle = self._calculate_knee_angle(hip, knee, ankle)

        # Calculate velocity
        velocity = self._calculate_velocity(foot_pos, foot)
        velocity_magnitude = np.linalg.norm(velocity)

        # Detect motion direction
        motion_direction = self._detect_motion_direction(velocity, foot)

        # Detect blade type (edge + zone)
        blade_type = self._detect_blade_type(
            foot_angle, knee_angle, float(velocity_magnitude), vertical_accel=0
        )

        # Store in history
        self._velocity_history[foot].append(velocity)
        self._position_history[foot].append(foot_pos)

        return BladeState3D(
            blade_type=blade_type,
            foot=foot,
            motion_direction=motion_direction,
            foot_angle=foot_angle,
            ankle_angle=shin_angle,
            knee_angle=knee_angle,
            vertical_accel=0.0,  # Would need temporal data
            position_3d=tuple(foot_pos),
            velocity_3d=tuple(velocity),
            confidence=0.8,  # TODO: Calculate from pose confidence
            frame_idx=frame_idx,
        )

    def _calculate_knee_angle(self, hip: NDArray, knee: NDArray, ankle: NDArray) -> float:
        """Calculate knee flexion angle.

        Returns angle in degrees (0° = fully extended, >90° = deeply flexed).
        """
        # Vectors from knee
        thigh = knee - hip  # Hip to knee
        shin = knee - ankle  # Knee to ankle

        # Angle between thigh and shin
        cos_angle = np.dot(thigh, shin) / (np.linalg.norm(thigh) * np.linalg.norm(shin))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def _calculate_velocity(self, current_pos: NDArray, foot: str) -> NDArray[np.float32]:
        """Calculate velocity from position history.

        Returns 3D velocity vector (vx, vy, vz).
        """
        history = self._position_history[foot]

        if len(history) < 2:
            return np.zeros(3, dtype=np.float32)

        # Get previous position from position history
        prev_pos = history[-1]
        velocity = current_pos - prev_pos

        # Scale to reasonable range (3D poses may be in arbitrary units)
        # Clip to prevent overflow
        velocity = np.clip(velocity, -10.0, 10.0)

        return velocity * self.fps  # Convert to units/s

    def _detect_motion_direction(self, velocity: NDArray, _foot: str) -> MotionDirection:  # noqa: PLR0911
        """Detect motion direction from velocity vector.

        Uses smoothed velocity from history.
        """
        velocity_mag = np.linalg.norm(velocity)

        # Below threshold = stationary
        if velocity_mag < self.config.min_velocity:
            return MotionDirection.STATIONARY

        # Get velocity components
        vx, vz = velocity[0], velocity[2]  # X-Z plane (ice surface)

        # Determine direction (assume facing +Z initially)
        angle = np.degrees(np.arctan2(vx, vz))

        # Classify direction using angle ranges
        # Format: (min_angle, max_angle): direction
        # Using ranges ordered to reduce early returns
        abs_angle = abs(angle)

        # Forward/backward (highest priority)
        if abs_angle < 22.5:
            return MotionDirection.FORWARD
        if abs_angle > 157.5:
            return MotionDirection.BACKWARD

        # Rotations
        if 67.5 <= angle <= 112.5:
            return MotionDirection.ROTATION_LEFT
        if -112.5 <= angle <= -67.5:
            return MotionDirection.ROTATION_RIGHT

        # Cardinal directions
        if 22.5 < angle < 67.5:
            return MotionDirection.LEFT
        if -67.5 < angle < -22.5:
            return MotionDirection.RIGHT

        # Diagonal fallback
        if angle > 0:
            return MotionDirection.DIAGONAL_LEFT
        return MotionDirection.DIAGONAL_RIGHT

    def _detect_blade_type(
        self, foot_angle: float, knee_angle: float, velocity_mag: float, vertical_accel: float
    ) -> BladeType:
        """Detect blade type from angles and motion state.

        Detection logic:
        1. Check for toe pick (high vertical accel + flexed knee)
        2. Check for heel (extended knee + low velocity)
        3. Otherwise, classify edge from foot angle
        """
        # Toe pick detection
        if vertical_accel > 0.5 or (knee_angle < 60 and velocity_mag > 0.1):
            return BladeType.TOE_PICK

        # Heel detection (rare, mostly for landing stability)
        if knee_angle > 80 and velocity_mag < 0.05:
            return BladeType.HEEL

        # Edge classification from foot angle
        if foot_angle < self.config.inside_threshold:
            return BladeType.INSIDE
        elif foot_angle > self.config.outside_threshold:
            return BladeType.OUTSIDE
        else:
            return BladeType.FLAT

    def get_ice_trace(self, foot: str) -> IceTrace:
        """Get ice trace for a foot.

        Returns the path the blade has taken on the ice surface.
        """
        positions = list(self._position_history[foot])

        # TODO: Add blade types and timestamps
        return IceTrace(
            foot=foot,
            points=[tuple(p) for p in positions],
            timestamps=[],  # TODO: Track timestamps
            blade_types=[],  # TODO: Track blade types
        )

    def reset(self):
        """Reset all history buffers."""
        for foot in ["left", "right"]:
            self._velocity_history[foot].clear()
            self._position_history[foot].clear()
            self._direction_history[foot].clear()
