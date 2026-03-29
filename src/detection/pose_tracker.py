"""Multi-person pose tracker with OC-SORT and biometric Re-ID.

Implements a tracking system that handles:
- Nonlinear skating trajectories using Kalman prediction
- Re-identification after occlusion using anatomical ratios
- Identical black clothing via pose biometrics

Based on:
- OC-SORT: Occlusion-resistant Optical Flow Sort (CAI 2023)
- Pose2Sim personAssociation.py: Biometric-based Re-ID
"""

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine


@dataclass
class Track:
    """Represents a tracked person over time.

    Attributes:
        id: Unique track identifier.
        age: Number of frames since track was created.
        hits: Total number of detections associated.
        time_since_update: Frames since last detection update.
        state: Kalman filter state [x, y, vx, vy, a_x, a_y] for mid-hip.
        biometrics: Scale-invariant anatomical ratios for Re-ID.
        hit streak: Consecutive detections (for track initialization).
    """

    id: int
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: np.ndarray | None = None
    biometrics: dict[str, float] | None = None
    hit_streak: int = 0


class PoseTracker:
    """Multi-person pose tracker with OC-SORT + biometrics.

    Uses a constant acceleration Kalman filter for motion prediction
    and anatomical ratios for re-identification after occlusion.

    Based on:
        - OC-SORT: "Occlusion-resistant Optical Flow Sort" (CAI 2023)
        - Pose2Sim: Biometric Re-ID for identical clothing

    Attributes:
        max_disappeared: Max frames without detection before deletion.
        min_hits: Min detections before track is confirmed.
        next_id: Counter for new track IDs.
        tracks: List of active Track objects.
        kf: KalmanFilter for motion prediction (6-state).
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        min_hits: int = 3,
        fps: float = 30.0,
    ) -> None:
        """Initialize pose tracker.

        Args:
            max_disappeared: Max frames without detection before deletion.
            min_hits: Min detections before track is confirmed.
            fps: Frame rate for Kalman dt calculation.
        """
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.next_id = 0
        self.tracks: list[Track] = []
        self.fps = fps
        self.dt = 1.0 / fps

        # Initialize Kalman filter [x, y, vx, vy, ax, ay]
        self.kf = self._init_kalman_filter()

    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize 6-state Kalman filter for constant acceleration model.

        State vector: [x, y, vx, vy, ax, ay]
        - x, y: Mid-hip position (normalized coordinates)
        - vx, vy: Velocity
        - ax, ay: Acceleration

        Returns:
            Initialized KalmanFilter.
        """
        kf = KalmanFilter(dim_x=6, dim_z=2)

        # State transition matrix (constant acceleration)
        # x(t+dt) = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2
        dt = self.dt
        kf.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],  # x
                [0, 1, 0, dt, 0, 0.5 * dt**2],  # y
                [0, 0, 1, 0, dt, 0],  # vx
                [0, 0, 0, 1, 0, dt],  # vy
                [0, 0, 0, 0, 1, 0],  # ax
                [0, 0, 0, 0, 0, 1],  # ay
            ]
        )

        # Measurement matrix (observe x, y only)
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # Observe x
                [0, 1, 0, 0, 0, 0],  # Observe y
            ]
        )

        # Noise covariance
        kf.Q = np.eye(6) * 0.01  # Process noise
        kf.R = np.eye(2) * 0.1  # Measurement noise

        # Initial state covariance
        kf.P = np.eye(6) * 1.0

        return kf

    def update(
        self,
        poses: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> list[int]:
        """Update tracks with new frame detections.

        Args:
            poses: (N, 33, 2) detected poses in normalized coordinates.
            confidences: (N,) detection confidence values (optional).

        Returns:
            List of track IDs corresponding to each input pose.
            Unmatched poses get new IDs.
        """
        if len(poses) == 0:
            # No detections: age all tracks
            for track in self.tracks:
                track.time_since_update += 1
                track.age += 1
            self._remove_lost_tracks()
            return []

        # Extract mid-hip positions for association
        mid_hips = self._get_mid_hips(poses)

        # Predict all track positions
        predicted_positions = []
        for track in self.tracks:
            if track.state is not None:
                self.kf.x = track.state.copy()
                self.kf.predict()
                track.state = self.kf.x.copy()
                # Extract x, y from column vector
                predicted_positions.append([track.state[0, 0], track.state[1, 0]])
            else:
                predicted_positions.append(np.zeros(2))

        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            poses, mid_hips, predicted_positions
        )

        # Update matched tracks
        track_ids = []
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            mid_hip = mid_hips[det_idx]

            # Update Kalman filter
            if track.state is None:
                # Initialize state [x, y, vx, vy, ax, ay]
                self.kf.x = np.array([[mid_hip[0]], [mid_hip[1]], [0], [0], [0], [0]])
            else:
                self.kf.x = track.state.copy()

            self.kf.update(np.array([[mid_hip[0]], [mid_hip[1]]]))
            track.state = self.kf.x.copy()

            # Update track info
            track.hits += 1
            track.time_since_update = 0
            track.hit_streak += 1
            track.age += 1

            # Update biometrics (first time or periodically)
            if track.biometrics is None or track.hits % 30 == 0:
                track.biometrics = self._extract_biometrics(poses[det_idx])

            track_ids.append(track.id)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                id=self.next_id,
                age=1,
                hits=1,
                time_since_update=0,
                biometrics=self._extract_biometrics(poses[det_idx]),
                hit_streak=1,
            )
            # Initialize state [x, y, vx, vy, ax, ay]
            mid_hip = mid_hips[det_idx]
            self.kf.x = np.array([[mid_hip[0]], [mid_hip[1]], [0], [0], [0], [0]])
            new_track.state = self.kf.x.copy()

            self.tracks.append(new_track)
            track_ids.append(self.next_id)
            self.next_id += 1

        # Age unmatched tracks
        for track_idx in unmatched_trks:
            track = self.tracks[track_idx]
            track.time_since_update += 1
            track.hit_streak = 0
            track.age += 1

        # Remove lost tracks
        self._remove_lost_tracks()

        return track_ids

    def _get_mid_hips(self, poses: np.ndarray) -> np.ndarray:
        """Calculate mid-hip position for each pose.

        Args:
            poses: (N, 33, 2) pose array.

        Returns:
            (N, 2) mid-hip positions.
        """
        # BlazePose: LEFT_HIP=23, RIGHT_HIP=24
        left_hip = poses[:, 23, :]
        right_hip = poses[:, 24, :]
        return (left_hip + right_hip) / 2

    def _associate(
        self,
        poses: np.ndarray,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections to tracks using IoU + biometrics.

        Args:
            poses: (N, 33, 2) detected poses.
            detections: (N, 2) mid-hip positions.
            predictions: (M, 2) predicted track positions.

        Returns:
            Tuple of (matched_pairs, unmatched_det_indices, unmatched_trk_indices).
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Calculate cost matrix
        cost_matrix = np.full((len(self.tracks), len(detections)), 1e6)

        for i, track in enumerate(self.tracks):
            for j, (detection, pose) in enumerate(zip(detections, poses)):
                # IoU distance (simplified as L2 distance for normalized coords)
                iou_dist = float(np.linalg.norm(predictions[i] - detection))

                # Biometric distance (if available)
                bio_dist = 0.0
                if track.biometrics is not None:
                    current_bio = self._extract_biometrics(pose)
                    bio_dist = self._biometric_distance(track.biometrics, current_bio)

                # Combined cost (weighted sum)
                # Higher weight on biometrics for identical clothing case
                cost_matrix[i, j] = 0.4 * iou_dist + 0.6 * bio_dist

        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        # Filter by threshold
        matches = []
        unmatched_tracks = list(set(range(len(self.tracks))) - set(track_indices))
        unmatched_dets = list(set(range(len(detections))) - set(det_indices))

        for t, d in zip(track_indices, det_indices):
            # Threshold: combined cost < 0.3 for valid match
            if cost_matrix[t, d] < 0.3:
                matches.append((t, d))
            else:
                unmatched_tracks.append(t)
                unmatched_dets.append(d)

        return matches, unmatched_dets, unmatched_tracks

    def _extract_biometrics(self, pose: np.ndarray) -> dict[str, float]:
        """Extract scale-invariant anatomical ratios.

        These ratios are unique per person and stable across poses,
        enabling re-identification even with identical clothing.

        Based on Pose2Sim personAssociation.py biometric extraction.

        Args:
            pose: (33, 2) single pose array.

        Returns:
            Dictionary of anatomical ratios.
        """
        # BlazePose 33kp indices
        # Shoulder: 11 (left), 12 (right)
        # Hip: 23 (left), 24 (right)
        # Knee: 25 (left), 26 (right)
        # Ankle: 27 (left), 28 (right)
        # Wrist: 15 (left), 16 (right)

        # Calculate bone lengths
        shoulder_width = np.linalg.norm(pose[11] - pose[12])
        mid_hip = (pose[23] + pose[24]) / 2
        mid_shoulder = (pose[11] + pose[12]) / 2
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)

        femur_length = np.linalg.norm(pose[23] - pose[25])  # Left hip-knee
        tibia_length = np.linalg.norm(pose[25] - pose[27])  # Left knee-ankle
        arm_span = np.linalg.norm(pose[15] - pose[16])  # Left-right wrist

        # Total height estimate (from visible joints)
        height = torso_length + femur_length + tibia_length

        # Scale-invariant ratios (avoid division by zero)
        eps = 1e-8
        return {
            "shoulder_width/torso": shoulder_width / (torso_length + eps),
            "femur/tibia": femur_length / (tibia_length + eps),
            "arm_span/height": arm_span / (height + eps),
            "torso/height": torso_length / (height + eps),
            "shoulder_width/height": shoulder_width / (height + eps),
        }

    def _biometric_distance(
        self,
        bio1: dict[str, float],
        bio2: dict[str, float],
    ) -> float:
        """Calculate cosine distance between biometric signatures.

        Args:
            bio1: First biometric dictionary.
            bio2: Second biometric dictionary.

        Returns:
            Distance in [0, 1] where 0 = identical.
        """
        if not bio1 or not bio2:
            return 1.0

        # Compare all ratios
        keys = [
            "shoulder_width/torso",
            "femur/tibia",
            "arm_span/height",
            "torso/height",
            "shoulder_width/height",
        ]

        vec1 = np.array([bio1.get(k, 0) for k in keys])
        vec2 = np.array([bio2.get(k, 0) for k in keys])

        # Cosine distance
        return float(cosine(vec1, vec2))

    def _remove_lost_tracks(self) -> None:
        """Remove tracks that haven't been detected for too long."""
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_disappeared]

    def get_confirmed_tracks(self) -> list[Track]:
        """Get tracks that have enough detections to be confirmed.

        Returns:
            List of confirmed Track objects.
        """
        return [t for t in self.tracks if t.hits >= self.min_hits]
