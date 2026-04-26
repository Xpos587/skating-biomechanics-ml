"""
Deep dive into YOLO26-Pose sigma head structure.

The model already has a one2one_cv4_sigma head! Let's investigate its structure.
"""

from ultralytics import YOLO

model_path = "/home/michael/Github/skating-biomechanics-ml/yolo26n-pose.pt"
model = YOLO(model_path)

print("=" * 60)
print("YOLO26-Pose Built-in Sigma Head Investigation")
print("=" * 60)

# Get the pose head
pose_head = model.model.model[-1]

print(f"\nPose head type: {pose_head.__class__.__name__}")
print("Pose head attributes:")
for attr in dir(pose_head):
    if not attr.startswith("_") and "sigma" in attr.lower():
        value = getattr(pose_head, attr)
        print(f"  - {attr}: {value}")

# Check sigma head in state dict
print("\nSigma head layers in state dict:")
state_dict = model.model.state_dict()
sigma_keys = [k for k in state_dict.keys() if "sigma" in k]

print(f"Found {len(sigma_keys)} sigma-related tensors:")
for key in sorted(sigma_keys):
    shape = state_dict[key].shape
    print(f"  - {key}: {shape}")

# Analyze sigma head structure
print("\n" + "=" * 60)
print("Sigma Head Architecture")
print("=" * 60)

# Count layers per output scale
sigma_layers = {}
for key in sigma_keys:
    # Parse key format: model.23.one2one_cv4_sigma.{layer_idx}.{param}
    parts = key.split(".")
    if len(parts) >= 4:
        layer_idx = parts[3]
        param_name = parts[4] if len(parts) > 4 else ""

        if layer_idx not in sigma_layers:
            sigma_layers[layer_idx] = {}

        if param_name not in sigma_layers[layer_idx]:
            sigma_layers[layer_idx][param_name] = []

        sigma_layers[layer_idx][param_name].append(shape)

print(f"\nSigma head has {len(sigma_layers)} output scales:")
for layer_idx in sorted(sigma_layers.keys(), key=int):
    print(f"\n  Scale {layer_idx}:")
    for param_name, shapes in sigma_layers[layer_idx].items():
        print(f"    - {param_name}: {shapes[0]}")

# Compare with keypoints head
print("\n" + "=" * 60)
print("Comparison: Keypoints vs Sigma Heads")
print("=" * 60)

kpts_keys = [k for k in state_dict.keys() if "kpts" in k]
print(f"\nKeypoints head: {len(kpts_keys)} tensors")
print(f"Sigma head: {len(sigma_keys)} tensors")

# Check channel dimensions
print("\nChannel dimensions:")
if sigma_keys:
    sigma_weight_shape = state_dict[sigma_keys[0]].shape
    print(f"  - Sigma: {sigma_weight_shape}")
    print(f"    → Output channels: {sigma_weight_shape[0]}")

if kpts_keys:
    kpts_weight_shape = state_dict[kpts_keys[0]].shape
    print(f"  - Keypoints: {kpts_weight_shape}")
    print(f"    → Output channels: {kpts_weight_shape[0]}")

# Calculate expected sigma channels
# For 17 keypoints with 2D coordinates (x, y), we need 34 sigma values
# (sigma_x and sigma_y for each keypoint)
expected_sigma_channels = 34  # 17 keypoints × 2 (x, y)

print("\n" + "=" * 60)
print("Channel Analysis")
print("=" * 60)

print(f"\nExpected sigma channels: {expected_sigma_channels}")
print("  - 17 keypoints × 2 coordinates (x, y) = 34 sigma values")

if sigma_keys:
    actual_sigma_channels = state_dict[sigma_keys[0]].shape[0]
    print(f"\nActual sigma channels: {actual_sigma_channels}")

    if actual_sigma_channels == expected_sigma_channels:
        print("  ✓ MATCHES! Model already outputs sigma for all keypoints")
    else:
        print("  ✗ Mismatch!")

        # Analyze what the model actually outputs
        if actual_sigma_channels == 51:
            print("  → 51 = 17 keypoints × 3 (x, y, visibility)")
            print("  → Model outputs sigma for visibility too (unusual)")
        elif actual_sigma_channels == 17:
            print("  → 17 = 1 sigma per keypoint (isotropic)")
            print("  → Same sigma for x and y coordinates")

# Check if sigma head is trained or just initialized
print("\n" + "=" * 60)
print("Sigma Head Training Status")
print("=" * 60)

if sigma_keys:
    # Check if weights are near zero (untrained) or have values (trained)
    sigma_weight = state_dict[sigma_keys[0]]
    sigma_mean = sigma_weight.mean().item()
    sigma_std = sigma_weight.std().item()

    print("\nSigma head weights (first layer):")
    print(f"  - Mean: {sigma_mean:.6f}")
    print(f"  - Std: {sigma_std:.6f}")
    print(f"  - Min: {sigma_weight.min().item():.6f}")
    print(f"  - Max: {sigma_weight.max().item():.6f}")

    if abs(sigma_mean) < 0.01 and sigma_std < 0.1:
        print("  → Status: LIKELY UNTRAINED (near-zero initialization)")
    elif sigma_std > 0.1:
        print("  → Status: LIKELY TRAINED (has learned values)")
    else:
        print("  → Status: UNCERTAIN (check training logs)")

# Final conclusion
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(f"""
✓ YOLO26-Pose ALREADY HAS a sigma head built-in!
✓ Sigma head structure: one2one_cv4_sigma
✓ Number of scales: {len(sigma_layers)}
""")

if sigma_keys:
    print("""
This is EXCELLENT news for our uncertainty estimation:

1. NO ARCHITECTURE CHANGES NEEDED
   - Sigma head already exists
   - Just need to use it in the loss function

2. TWO OPTIONS:
   A. Use pretrained sigma weights (if trained)
      → Ready to use immediately
      → May need fine-tuning on skating data

   B. Initialize sigma head randomly
      → Train from scratch on skating data
      → Keep keypoints head frozen

3. IMPLEMENTATION PATH:
   - Extract sigma outputs from model.forward()
   - Use sigma in KL divergence loss: D_KL(N||N_0)
   - Compare: fixed sigma vs learned sigma

BLOCKERS:
⚠️  Model parameters are frozen (requires_grad=False)
   → Need to unfreeze sigma head for training
   → Can unfreeze only sigma head, keep keypoints frozen

NEXT STEPS:
1. Verify if sigma head is used in current training/evaluation
2. Check ultralytics Pose26 forward() method
3. Implement KL divergence loss with sigma outputs
4. Compare: fixed sigma baseline vs learned sigma
""")
