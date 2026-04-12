"""Tests for COCO JSON generation with HALPE26 26kp format."""

import numpy as np

from skating_ml.datasets.coco_builder import (
    DEFAULT_CATEGORY,
    HALPE26_KEYPOINT_NAMES,
    HALPE26_SKELETON,
    build_coco_json,
    format_keypoints,
    merge_coco_foot_keypoints,
)


class TestCocoBuilder:
    def test_keypoint_names_count(self):
        assert len(HALPE26_KEYPOINT_NAMES) == 26

    def test_category_is_person(self):
        assert DEFAULT_CATEGORY["name"] == "person"
        assert len(DEFAULT_CATEGORY["keypoint_names"]) == 26
        assert DEFAULT_CATEGORY["skeleton"] is not None

    def test_skeleton_edges(self):
        """Skeleton should connect foot keypoints to ankles."""
        skel = HALPE26_SKELETON
        # left_heel(17) should connect to left_ankle(15)
        assert [15, 17] in skel or [17, 15] in skel
        # right_heel(20) should connect to right_ankle(16)
        assert [16, 20] in skel or [20, 16] in skel

    def test_merge_produces_26kp(self):
        """Merging 17kp COCO + 6 foot points + 3 face dupes = 26kp."""
        coco_kp = np.random.rand(17, 2) * 1000
        foot_kp = np.random.rand(6, 2) * 1000
        merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        assert merged.shape == (26, 2)
        assert vis.shape == (26,)
        assert np.all(vis[:17] > 0)  # COCO 17kp from _coco.npy are valid
        assert np.all(vis[17:23] > 0)  # foot keypoints
        assert np.all(vis[23:] > 0)  # face dupes

    def test_merge_preserves_coco_coords(self):
        """COCO 17kp coordinates should be unchanged."""
        coco_kp = np.array([[100.0, 200.0], [150.0, 250.0]] + [[0.0, 0.0]] * 15)
        foot_kp = np.random.rand(6, 2) * 1000
        merged, _ = merge_coco_foot_keypoints(coco_kp, foot_kp)

        np.testing.assert_array_almost_equal(merged[0], [100.0, 200.0])
        np.testing.assert_array_almost_equal(merged[1], [150.0, 250.0])

    def test_face_dupes_copy_from_existing(self):
        """Face dupes (23-25) should copy from idx 0, 1, 2."""
        coco_kp = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]] + [[0.0, 0.0]] * 14)
        foot_kp = np.random.rand(6, 2) * 1000
        merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        # idx 23 (left_eye_inner) = copy of idx 1 (left_eye)
        np.testing.assert_array_almost_equal(merged[23], [110.0, 210.0])
        # idx 24 (right_eye_inner) = copy of idx 2 (right_eye)
        np.testing.assert_array_almost_equal(merged[24], [120.0, 220.0])
        # idx 25 (mouth) = copy of idx 0 (nose)
        np.testing.assert_array_almost_equal(merged[25], [100.0, 200.0])
        # Face dupes should have low visibility
        assert vis[23] == 0.3
        assert vis[24] == 0.3
        assert vis[25] == 0.3

    def test_nan_foot_sets_zero_vis(self):
        """NaN foot keypoints should get zero visibility."""
        coco_kp = np.ones((17, 2))
        foot_kp = np.array(
            [
                [np.nan, np.nan],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [np.nan, np.nan],
                [7.0, 8.0],
            ]
        )
        _merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        assert vis[17] == 0.0  # NaN foot
        assert vis[18] == 2.0  # valid foot (COCO visible)
        assert vis[21] == 0.0  # NaN foot
        assert vis[22] == 2.0  # valid foot (COCO visible)

    def test_format_keypoints_flat(self):
        """format_keypoints should produce flat [x,y,v,x,y,v,...] list."""
        pts = np.random.rand(26, 2)
        vis = np.ones(26)
        kp = format_keypoints(pts, vis)

        assert isinstance(kp, list)
        assert len(kp) == 26 * 3
        assert kp[2] == 1.0  # first visibility

    def test_build_coco_json_structure(self):
        images = [{"file_name": "test.jpg", "id": 1, "width": 1280, "height": 768}]
        annotations = [
            {
                "image_id": 1,
                "id": 1,
                "keypoints": [0.0] * 78,
                "num_keypoints": 26,
                "bbox": [0, 0, 100, 100],
            }
        ]
        result = build_coco_json(images, annotations)

        assert "images" in result
        assert "annotations" in result
        assert "categories" in result
        assert result["categories"][0]["name"] == "person"
        assert len(result["categories"][0]["keypoint_names"]) == 26
