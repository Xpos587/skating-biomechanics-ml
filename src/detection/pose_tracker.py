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
        covariance: Kalman filter covariance matrix (6x6).
        biometrics: Scale-invariant anatomical ratios for Re-ID.
        hit_streak: Consecutive detections (for track initialization).
    """

    id: int
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: np.ndarray | None = None
    covariance: np.ndarray | None = None
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
        F: State transition matrix (shared, immutable).
        H: Observation matrix (shared, immutable).
        Q: Process noise covariance (shared, immutable).
        R: Measurement noise covariance (shared, immutable).
        P0: Initial state covariance for new tracks.
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

        # Kalman filter parameters (shared, read-only during predict/update)
        self.F, self.H, self.Q, self.R, self.P0 = self._init_kalman_params()

    def _init_kalman_params(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize Kalman filter matrices for constant acceleration model.

        State vector: [x, y, vx, vy, ax, ay]
        - x, y: Mid-hip position (normalized coordinates)
        - vx, vy: Velocity
        - ax, ay: Acceleration

        Returns:
            Tuple of (F, H, Q, R, P0) matrices.
        """
        dt = self.dt

        # State transition matrix (constant acceleration)
        F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement matrix (observe x, y only)
        H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )

        # Noise covariances
        Q = np.eye(6) * 0.01  # Process noise
        R = np.eye(2) * 0.1  # Measurement noise
        P0 = np.eye(6) * 1.0  # Initial state covariance

        return F, H, Q, R, P0

    @staticmethod
    def _kalman_predict(
        x: np.ndarray,
        P: np.ndarray,
        F: np.ndarray,
        Q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman predict step on per-track state and covariance.

        Args:
            x: State vector (6, 1).
            P: Covariance matrix (6, 6).
            F: State transition matrix (6, 6).
            Q: Process noise covariance (6, 6).

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    @staticmethod
    def _kalman_update(
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman update step on per-track state and covariance.

        Args:
            x: Predicted state vector (6, 1).
            P: Predicted covariance matrix (6, 6).
            z: Measurement vector (2, 1).
            H: Observation matrix (2, 6).
            R: Measurement noise covariance (2, 2).

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        # Innovation
        y = z - H @ x
        # Innovation covariance
        S = H @ P @ H.T + R
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        # Updated state
        x_upd = x + K @ y
        # Updated covariance (Joseph form for numerical stability)
        I_KH = np.eye(len(x)) - K @ H
        P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T
        return x_upd, P_upd

    def update(
        self,
        poses: np.ndarray,
        _confidences: np.ndarray | None = None,
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
                track.state, track.covariance = self._kalman_predict(
                    track.state, track.covariance, self.F, self.Q
                )
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
            z = np.array([[mid_hip[0]], [mid_hip[1]]])

            if track.state is None:
                # Initialize state [x, y, vx, vy, ax, ay]
                track.state = np.array([[mid_hip[0]], [mid_hip[1]], [0], [0], [0], [0]])
                track.covariance = self.P0.copy()
            else:
                track.state, track.covariance = self._kalman_update(
                    track.state, track.covariance, z, self.H, self.R
                )

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
            mid_hip = mid_hips[det_idx]
            new_track = Track(
                id=self.next_id,
                age=1,
                hits=1,
                time_since_update=0,
                state=np.array([[mid_hip[0]], [mid_hip[1]], [0], [0], [0], [0]]),
                covariance=self.P0.copy(),
                biometrics=self._extract_biometrics(poses[det_idx]),
                hit_streak=1,
            )

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
            poses: (N, 17, 2) pose array - H3.6M format.

        Returns:
            (N, 2) mid-hip positions.
        """
        from ..types import H36Key

        # H3.6M: LHIP=4, RHIP=1
        left_hip = poses[:, H36Key.LHIP, :]
        right_hip = poses[:, H36Key.RHIP, :]
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
            for j, (detection, pose) in enumerate(zip(detections, poses, strict=False)):
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

        for t, d in zip(track_indices, det_indices, strict=False):
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
            pose: (17, 2) single pose array - H3.6M format.

        Returns:
            Dictionary of anatomical ratios.
        """
        from ..types import H36Key

        # H3.6M 17kp indices
        # Shoulder: 11 (left), 14 (right)
        # Hip: 4 (left), 1 (right)
        # Knee: 5 (left), 2 (right)
        # Foot: 6 (left), 3 (right)
        # Wrist: 13 (left), 16 (right)

        # Calculate bone lengths
        shoulder_width = np.linalg.norm(pose[H36Key.LSHOULDER] - pose[H36Key.RSHOULDER])
        mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
        mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)

        femur_length = np.linalg.norm(pose[H36Key.LHIP] - pose[H36Key.LKNEE])  # Left hip-knee
        tibia_length = np.linalg.norm(pose[H36Key.LKNEE] - pose[H36Key.LFOOT])  # Left knee-foot
        arm_span = np.linalg.norm(pose[H36Key.LWRIST] - pose[H36Key.RWRIST])  # Left-right wrist

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
