"""Integration tests for ML pipeline (mocked ONNX sessions)."""

import numpy as np


class TestMLPipelineIntegration:
    """Test that ML models integrate correctly with the visualization pipeline."""

    def test_registry_with_all_models_registered(self):
        """All 6 models can be registered within VRAM budget."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu", vram_budget_mb=1000)
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")
        reg.register("optical_flow", vram_mb=80, path="/tmp/flow.onnx")
        reg.register("segment_anything", vram_mb=200, path="/tmp/sam.onnx")
        reg.register("foot_tracker", vram_mb=30, path="/tmp/foot.onnx")
        reg.register("video_matting", vram_mb=40, path="/tmp/rvm.onnx")
        reg.register("lama", vram_mb=300, path="/tmp/lama.onnx")

        assert reg.vram_used_mb == 0  # Nothing loaded yet
        assert reg.list_models() == [
            "depth_anything",
            "optical_flow",
            "segment_anything",
            "foot_tracker",
            "video_matting",
            "lama",
        ]

    def test_depth_layer_context_flow(self):
        """Depth map flows from estimator through LayerContext to layer."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer(opacity=0.5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.linspace(0, 1, 100 * 100).reshape(100, 100).astype(np.float32)

        ctx = LayerContext(frame_width=100, frame_height=100)
        ctx.custom_data["depth_map"] = depth

        result = layer.render(frame, ctx)
        assert result.shape == (100, 100, 3)
        assert not np.all(result == 0)

    def test_multiple_layers_compose(self):
        """Depth + flow + segmentation layers compose correctly."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer
        from src.visualization.layers.optical_flow_layer import OpticalFlowLayer
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layers = [
            DepthMapLayer(opacity=0.3),
            OpticalFlowLayer(opacity=0.4),
            SegmentationMaskLayer(opacity=0.2),
        ]

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=100, frame_height=100)
        ctx.custom_data["depth_map"] = np.random.rand(100, 100).astype(np.float32)
        ctx.custom_data["flow_field"] = np.random.rand(100, 100, 2).astype(np.float32) * 2 - 1
        ctx.custom_data["seg_mask"] = np.zeros((100, 100), dtype=bool)
        ctx.custom_data["seg_mask"][20:80, 20:80] = True

        for layer in layers:
            frame = layer.render(frame, ctx)

        assert frame.shape == (100, 100, 3)

    def test_process_request_schema_accepts_ml_flags(self):
        """ProcessRequest schema accepts ML boolean flags."""
        from src.backend.schemas import ProcessRequest

        req = ProcessRequest(
            video_path="/tmp/test.mp4",
            person_click={"x": 100, "y": 200},
            depth=True,
            optical_flow=True,
            segment=True,
            foot_track=True,
            matting=True,
        )
        assert req.depth is True
        assert req.optical_flow is True
        assert req.segment is True
        assert req.foot_track is True
        assert req.matting is True

    def test_process_request_defaults_ml_flags_false(self):
        """ML flags default to False."""
        from src.backend.schemas import ProcessRequest

        req = ProcessRequest(
            video_path="/tmp/test.mp4",
            person_click={"x": 100, "y": 200},
        )
        assert req.depth is False
        assert req.optical_flow is False
        assert req.segment is False
        assert req.foot_track is False
        assert req.matting is False
