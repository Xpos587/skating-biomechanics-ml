#!/usr/bin/env python3
"""Test FeatureAdapter shapes and projection logic.

Tests that 1x1 conv adapters correctly project teacher features to student channel space.
"""

import sys
from pathlib import Path

import torch

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from distill_trainer import FeatureAdapter


def test_adapter_shapes():
    """Test that adapters produce correct output shapes."""
    print("Testing FeatureAdapter shape transformations...")

    # Test case 1: MogaNet-B Layer 4 (160 channels) → YOLO26n (64 channels)
    print("\n1. MogaNet Layer 4 (160) → YOLO26n (64)")
    adapter1 = FeatureAdapter(in_channels=160, out_channels=64)
    x1 = torch.randn(2, 160, 72, 96)  # Batch=2, H=72, W=96
    y1 = adapter1(x1)
    assert y1.shape == (2, 64, 72, 96), f"Expected (2,64,72,96), got {y1.shape}"
    print(f"   Input: {x1.shape} → Output: {y1.shape} ✓")

    # Test case 2: MogaNet-B Layer 6 (320 channels) → YOLO26n (128 channels)
    print("\n2. MogaNet Layer 6 (320) → YOLO26n (128)")
    adapter2 = FeatureAdapter(in_channels=320, out_channels=128)
    x2 = torch.randn(2, 320, 36, 48)  # Downsampled spatial
    y2 = adapter2(x2)
    assert y2.shape == (2, 128, 36, 48), f"Expected (2,128,36,48), got {y2.shape}"
    print(f"   Input: {x2.shape} → Output: {y2.shape} ✓")

    # Test case 3: MogaNet-B Layer 8 (512 channels) → YOLO26n (256 channels)
    print("\n3. MogaNet Layer 8 (512) → YOLO26n (256)")
    adapter3 = FeatureAdapter(in_channels=512, out_channels=256)
    x3 = torch.randn(2, 512, 18, 24)  # Further downsampled
    y3 = adapter3(x3)
    assert y3.shape == (2, 256, 18, 24), f"Expected (2,256,18,24), got {y3.shape}"
    print(f"   Input: {x3.shape} → Output: {y3.shape} ✓")

    # Test case 4: Verify gradients flow through adapter
    print("\n4. Gradient flow test")
    adapter4 = FeatureAdapter(in_channels=160, out_channels=64)
    x4 = torch.randn(1, 160, 72, 96, requires_grad=True)
    y4 = adapter4(x4)
    loss = y4.sum()
    loss.backward()
    assert x4.grad is not None, "Input gradients should be computed"
    assert adapter4.projection.weight.grad is not None, (
        "Adapter weight gradients should be computed"
    )
    print(f"   Input gradients: {x4.grad.shape} ✓")
    print(f"   Weight gradients: {adapter4.projection.weight.grad.shape} ✓")

    # Test case 5: Compare with old min_channels approach
    print("\n5. Compare with old min_channels slicing")
    adapter5 = FeatureAdapter(in_channels=160, out_channels=64)
    teacher_feat = torch.randn(2, 160, 72, 96)
    student_feat = torch.randn(2, 64, 72, 96)

    # New approach: use all 160 channels via adapter
    teacher_aligned = adapter5(teacher_feat)

    # Old approach: discard 96 channels (60% of features!)
    min_c = min(160, 64)
    teacher_old = teacher_feat[:, :min_c, :, :]

    print(f"   New approach: preserves {teacher_aligned.shape[1]} channels")
    print(f"   Old approach: discards {160 - min_c} channels ({100 * (160 - min_c) / 160:.1f}%)")
    print(f"   Information preserved: 100% (new) vs {100 * min_c / 160:.1f}% (old) ✓")

    print("\n✅ All FeatureAdapter tests passed!")
    return True


def test_adapter_integration():
    """Test adapter integration with DistilPoseTrainer."""
    print("\nTesting DistilPoseTrainer adapter integration...")

    from distill_trainer import DistilPoseTrainer

    # Create trainer
    trainer = DistilPoseTrainer(
        teacher_hm_path=None,
        feature_layers=[4, 6, 8],
    )

    # Mock device
    device = torch.device("cpu")

    # Test adapter creation
    print("\n1. Creating adapters for each layer")
    layer_configs = [
        (4, 160, 64),  # MogaNet Layer 4 → YOLO26n
        (6, 320, 128),  # MogaNet Layer 6 → YOLO26n
        (8, 512, 256),  # MogaNet Layer 8 → YOLO26n
    ]

    for layer_idx, teacher_ch, student_ch in layer_configs:
        adapter = trainer._get_or_create_adapter(layer_idx, teacher_ch, student_ch, device)
        assert adapter.projection.in_channels == teacher_ch
        assert adapter.projection.out_channels == student_ch
        print(f"   Layer {layer_idx}: {teacher_ch} → {student_ch} channels ✓")

    # Verify adapters are cached
    print("\n2. Verifying adapter caching")
    assert 4 in trainer.adapters
    assert 6 in trainer.adapters
    assert 8 in trainer.adapters
    print(f"   Cached adapters: {list(trainer.adapters.keys())} ✓")

    # Test forward pass through adapter
    print("\n3. Testing forward projection")
    adapter = trainer.adapters[4]
    teacher_feat = torch.randn(2, 160, 72, 96)
    projected = adapter(teacher_feat)
    assert projected.shape == (2, 64, 72, 96)
    print(f"   Teacher {teacher_feat.shape} → Student {projected.shape} ✓")

    print("\n✅ All integration tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_adapter_shapes()
        test_adapter_integration()
        print("\n" + "=" * 60)
        print("SUCCESS: All adapter tests passed!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
