#!/usr/bin/env python3
"""Validate feature distillation implementation without GPU.

Tests:
1. FeatureExtractorMogaNet hook registration
2. TeacherFeatureLoader HDF5 reading
3. MSE loss computation with channel/spatial mismatch handling
4. Integration with distill_trainer.py
"""

import sys
import tempfile
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F


def test_feature_extractor_hooks():
    """Test forward hook registration for feature extraction."""
    print("=" * 60)
    print("Test 1: FeatureExtractorMogaNet Hooks")
    print("=" * 60)

    # Mock MogaNet backbone
    class MockBlock(torch.nn.Module):
        def __init__(self, out_channels, idx):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, out_channels, 3, padding=1)
            self.idx = idx

        def forward(self, x):
            return self.conv(x)

    class MockBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    MockBlock(64, 0),
                    MockBlock(64, 1),
                    MockBlock(64, 2),
                    MockBlock(64, 3),
                    MockBlock(64, 4),  # Extract here
                    MockBlock(160, 5),
                    MockBlock(160, 6),  # Extract here
                    MockBlock(160, 7),
                    MockBlock(160, 8),  # Extract here
                ]
            )

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

        def get_block(self, idx):
            if idx < len(self.blocks):
                return 0, 0, self.blocks[idx]
            raise ValueError(f"Block {idx} out of range")

    # Import FeatureExtractorMogaNet (will fail if not defined)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_teacher_features import FeatureExtractorMogaNet

        backbone = MockBackbone()
        extractor = FeatureExtractorMogaNet(backbone, extract_layers=[4, 6, 8])

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        features = extractor(x)

        assert len(features) == 3, f"Expected 3 features, got {len(features)}"
        assert features[0].shape == (2, 64, 32, 32), f"Layer 4 shape mismatch: {features[0].shape}"
        assert features[1].shape == (2, 160, 32, 32), f"Layer 6 shape mismatch: {features[1].shape}"
        assert features[2].shape == (2, 160, 32, 32), f"Layer 8 shape mismatch: {features[2].shape}"

        print("✓ FeatureExtractorMogaNet hooks working correctly")
        print(f"  Layer 4: {features[0].shape}")
        print(f"  Layer 6: {features[1].shape}")
        print(f"  Layer 8: {features[2].shape}")

        extractor.remove_hooks()
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  (This is OK if running without MogaNet dependencies)")
        return False


def test_teacher_feature_loader():
    """Test TeacherFeatureLoader with mock HDF5 file."""
    print("\n" + "=" * 60)
    print("Test 2: TeacherFeatureLoader HDF5 Reading")
    print("=" * 60)

    # Create mock HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write mock data
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("layer4", shape=(10, 64, 72, 96), dtype="float16")
            f.create_dataset("layer6", shape=(10, 160, 36, 48), dtype="float16")
            f.create_dataset("layer8", shape=(10, 160, 36, 48), dtype="float16")

            # Add index mapping
            import json

            idx_map = {
                "img1.jpg": 0,
                "img2.jpg": 1,
                "img3.jpg": 2,
            }
            f.attrs["indices"] = json.dumps(idx_map)

        # Test loader
        from distill_trainer import TeacherFeatureLoader

        loader = TeacherFeatureLoader(tmp_path, feature_layers=[4, 6, 8])

        # Load features
        features = loader.load(["img1.jpg", "img2.jpg", "img3.jpg", "unknown.jpg"])

        assert features is not None, "Failed to load features"
        assert 4 in features, "Layer 4 not in features"
        assert 6 in features, "Layer 6 not in features"
        assert 8 in features, "Layer 8 not in features"

        assert features[4].shape == (4, 64, 72, 96), f"Layer 4 shape mismatch: {features[4].shape}"
        assert features[6].shape == (4, 160, 36, 48), f"Layer 6 shape mismatch: {features[6].shape}"
        assert features[8].shape == (4, 160, 36, 48), f"Layer 8 shape mismatch: {features[8].shape}"

        # Check that unknown.jpg is padded with zeros
        assert torch.allclose(features[4][3], torch.zeros(64, 72, 96)), (
            "Unknown image not zero-padded"
        )

        print("✓ TeacherFeatureLoader working correctly")
        print(f"  Layer 4: {features[4].shape}")
        print(f"  Layer 6: {features[6].shape}")
        print(f"  Layer 8: {features[8].shape}")
        print(f"  Unknown image padded: {torch.allclose(features[4][3], torch.zeros(64, 72, 96))}")

        loader.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        Path(tmp_path + ".json").unlink(missing_ok=True)


def test_mse_loss_computation():
    """Test MSE loss with channel and spatial mismatch handling."""
    print("\n" + "=" * 60)
    print("Test 3: MSE Loss Computation")
    print("=" * 60)

    try:
        # Test 1: Perfect match (no mismatch)
        teacher = torch.randn(2, 64, 72, 96)
        student = teacher.clone()
        loss = F.mse_loss(student, teacher)
        assert loss.item() < 1e-6, f"Perfect match should have ~0 loss, got {loss.item()}"
        print("✓ Test 1: Perfect match - MSE = 0.0")

        # Test 2: Spatial mismatch (student larger)
        teacher = torch.randn(2, 64, 36, 48)
        student = torch.randn(2, 64, 72, 96)
        student_resized = F.interpolate(
            student, size=(36, 48), mode="bilinear", align_corners=False
        )
        loss = F.mse_loss(student_resized, teacher)
        assert not torch.isnan(loss), "MSE loss should not be NaN"
        print(f"✓ Test 2: Spatial mismatch (72x96 → 36x48) - MSE = {loss.item():.4f}")

        # Test 3: Channel mismatch (student has more channels)
        teacher = torch.randn(2, 64, 36, 48)
        student = torch.randn(2, 128, 36, 48)
        min_c = min(64, 128)
        student_proj = student[:, :min_c, :, :]
        teacher_proj = teacher[:, :min_c, :, :]
        loss = F.mse_loss(student_proj, teacher_proj)
        assert not torch.isnan(loss), "MSE loss should not be NaN"
        print(f"✓ Test 3: Channel mismatch (128 → 64) - MSE = {loss.item():.4f}")

        # Test 4: Both mismatches
        teacher = torch.randn(2, 64, 36, 48)
        student = torch.randn(2, 128, 72, 96)
        # Resize spatial
        student_resized = F.interpolate(
            student, size=(36, 48), mode="bilinear", align_corners=False
        )
        # Project channels
        min_c = min(64, 128)
        student_proj = student_resized[:, :min_c, :, :]
        teacher_proj = teacher[:, :min_c, :, :]
        loss = F.mse_loss(student_proj, teacher_proj)
        assert not torch.isnan(loss), "MSE loss should not be NaN"
        print(f"✓ Test 4: Both mismatches (128x72x96 → 64x36x48) - MSE = {loss.item():.4f}")

        # Test 5: Multi-layer averaging
        layers = [
            (torch.randn(2, 64, 72, 96), torch.randn(2, 64, 72, 96)),
            (torch.randn(2, 160, 36, 48), torch.randn(2, 160, 36, 48)),
            (torch.randn(2, 160, 36, 48), torch.randn(2, 160, 36, 48)),
        ]
        losses = []
        for i, (t, s) in enumerate(layers):
            loss = F.mse_loss(s, t)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        assert not torch.isnan(avg_loss), "Average loss should not be NaN"
        print(f"✓ Test 5: Multi-layer averaging - MSE = {avg_loss.item():.4f}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test integration with distill_trainer.py."""
    print("\n" + "=" * 60)
    print("Test 4: Integration with distill_trainer")
    print("=" * 60)

    try:
        from distill_trainer import DistilPoseTrainer

        # Test DistilPoseTrainer initialization
        trainer = DistilPoseTrainer(
            teacher_hm_path=None,
            teacher_feat_path=None,
            alpha=0.00005,
            beta=0.1,
            feature_layers=[4, 6, 8],
        )

        assert trainer.alpha == 0.00005, "Alpha mismatch"
        assert trainer.beta == 0.1, "Beta mismatch"
        assert trainer.feature_layers == [4, 6, 8], "Feature layers mismatch"

        print("✓ DistilPoseTrainer initialization working")

        # Test weight decay computation
        # Formula: w_kd = 1 - (epoch-1) / max_epochs
        trainer.set_max_epochs(210)
        trainer.set_epoch(1)
        w_kd = trainer.compute_kd_weight()
        assert abs(w_kd - 1.0) < 1e-6, f"Epoch 1: expected w_kd=1.0, got {w_kd}"

        trainer.set_epoch(105)
        w_kd = trainer.compute_kd_weight()
        expected_105 = 1 - (105 - 1) / 210
        assert abs(w_kd - expected_105) < 1e-6, (
            f"Epoch 105: expected w_kd={expected_105}, got {w_kd}"
        )

        trainer.set_epoch(210)
        w_kd = trainer.compute_kd_weight()
        expected_210 = 1 - (210 - 1) / 210
        assert abs(w_kd - expected_210) < 1e-6, (
            f"Epoch 210: expected w_kd={expected_210}, got {w_kd}"
        )

        print("✓ Weight decay computation working")
        print("  Epoch 1: w_kd = 1.0")
        print(f"  Epoch 105: w_kd = {expected_105:.4f}")
        print(f"  Epoch 210: w_kd = {expected_210:.4f}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("Feature Distillation Validation Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("FeatureExtractorMogaNet Hooks", test_feature_extractor_hooks()))
    results.append(("TeacherFeatureLoader HDF5", test_teacher_feature_loader()))
    results.append(("MSE Loss Computation", test_mse_loss_computation()))
    results.append(("Integration with distill_trainer", test_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Feature distillation is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
