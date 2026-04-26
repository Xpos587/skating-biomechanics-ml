#!/usr/bin/env python3
"""
Simulate heatmaps from student keypoints + sigma using MSRA unbiased encoding.

For DistilPose KD: compares simulated student heatmaps against teacher heatmaps via MSE.

Usage:
    python simulate_heatmap.py --test
"""

import argparse
import sys

import torch
import torch.nn.functional as F


def keypoints_to_heatmap(
    kpts: torch.Tensor,
    sigma: torch.Tensor,
    visibility: torch.Tensor,
    hm_shape: tuple[int, int, int] = (17, 72, 96),
) -> torch.Tensor:
    """Generate 2D Gaussian heatmaps using MSRA unbiased encoding.

    Args:
        kpts: (B, K, 2) normalized [0,1] keypoint coordinates.
        sigma: (B, K, 2) per-keypoint sigma from YOLO26 cv4_sigma, normalized [0,1].
        visibility: (B, K) keypoint visibility flags (0 or 1).
        hm_shape: (K, H, W) output heatmap spatial dimensions.

    Returns:
        heatmap: (B, K, H, W) float32 heatmaps, fully differentiable.
    """
    _, _, H, W = 0, 0, hm_shape[1], hm_shape[2]

    mu_x = kpts[..., 0] * W
    mu_y = kpts[..., 1] * H
    sigma_x = sigma[..., 0] * W
    sigma_y = sigma[..., 1] * H

    sigma_x_sq = (sigma_x**2 - 0.5).clamp(min=0.01)
    sigma_y_sq = (sigma_y**2 - 0.5).clamp(min=0.01)

    y, x = torch.meshgrid(
        torch.arange(H, device=kpts.device, dtype=kpts.dtype),
        torch.arange(W, device=kpts.device, dtype=kpts.dtype),
        indexing="ij",
    )

    dx = x[None, None] - mu_x[..., None, None]
    dy = y[None, None] - mu_y[..., None, None]

    g = torch.exp(
        -0.5 * (dx**2 / sigma_x_sq[..., None, None] + dy**2 / sigma_y_sq[..., None, None])
    )
    g = g * visibility[..., None, None]

    return g


def extract_teacher_value(
    teacher_heatmap: torch.Tensor,
    kpts: torch.Tensor,
) -> torch.Tensor:
    """Bilinear-interpolate teacher heatmap at predicted keypoint locations.

    Used for ScoreLoss in DistilPose: teacher confidence at student predictions.

    Args:
        teacher_heatmap: (B, K, H, W) float32 teacher heatmaps.
        kpts: (B, K, 2) normalized [0,1] predicted keypoint coordinates.

    Returns:
        values: (B, K) teacher heatmap values at keypoint locations.
    """
    B, K, H, W = teacher_heatmap.shape

    grid_x = kpts[..., 0] * 2 - 1
    grid_y = kpts[..., 1] * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(2)

    hm_reshaped = teacher_heatmap.reshape(B * K, 1, H, W)
    grid_reshaped = grid.reshape(B * K, 1, 1, 2)
    sampled = F.grid_sample(
        hm_reshaped, grid_reshaped, mode="bilinear", align_corners=True, padding_mode="zeros"
    )

    return sampled.reshape(B, K)


def _run_tests() -> bool:
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    print("=== keypoints_to_heatmap tests ===\n")

    B, K, H, W = 2, 17, 72, 96
    hm_shape = (K, H, W)

    kpts = torch.zeros(B, K, 2)
    kpts[0, 0] = torch.tensor([0.5, 0.5])
    sigma = torch.full((B, K, 2), 0.05)
    visibility = torch.ones(B, K)

    hm = keypoints_to_heatmap(kpts, sigma, visibility, hm_shape)
    check("output shape", hm.shape == (B, K, H, W), f"got {hm.shape}")

    peak = hm[0, 0, H // 2, W // 2]
    check("peak at center keypoint", peak.item() > 0.99, f"got {peak.item():.4f}")

    off_center = hm[0, 0, 0, 0]
    check("near-zero far from keypoint", off_center.item() < 0.01, f"got {off_center.item():.6f}")

    vis_zero = torch.zeros_like(visibility)
    hm_invis = keypoints_to_heatmap(kpts, sigma, vis_zero, hm_shape)
    check("visibility=0 zeros heatmap", hm_invis[0, 0].sum().item() == 0.0)

    kpts_grad = kpts.clone().requires_grad_(True)
    sigma_grad = sigma.clone().requires_grad_(True)
    hm_grad = keypoints_to_heatmap(kpts_grad, sigma_grad, visibility, hm_shape)
    hm_grad.sum().backward()
    check(
        "differentiable through kpts", kpts_grad.grad is not None and kpts_grad.grad.abs().sum() > 0
    )
    check(
        "differentiable through sigma",
        sigma_grad.grad is not None and sigma_grad.grad.abs().sum() > 0,
    )

    corner_kpts = torch.zeros(B, K, 2)
    corner_kpts[0, 0] = torch.tensor([0.0, 0.0])
    hm_corner = keypoints_to_heatmap(corner_kpts, sigma, visibility, hm_shape)
    corner_peak = hm_corner[0, 0, 0, 0]
    check(
        "peak at top-left corner (0,0)", corner_peak.item() > 0.99, f"got {corner_peak.item():.4f}"
    )

    corner_kpts2 = torch.zeros(B, K, 2)
    corner_kpts2[0, 0] = torch.tensor([1.0, 1.0])
    hm_corner2 = keypoints_to_heatmap(corner_kpts2, sigma, visibility, hm_shape)
    corner_peak2 = hm_corner2[0, 0, H - 1, W - 1]
    check(
        "high value near bottom-right corner (1,1)",
        corner_peak2.item() > 0.93,
        f"got {corner_peak2.item():.4f}",
    )

    print("\n=== extract_teacher_value tests ===\n")

    teacher_hm = torch.zeros(B, K, H, W)
    cy, cx = H // 2, W // 2
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            teacher_hm[0, 0, cy + dy, cx + dx] = 1.0
    teacher_hm[0, 1, 0, 0] = 1.0
    teacher_hm[0, 1, 0, 1] = 0.5
    teacher_hm[0, 1, 1, 0] = 0.5

    pred_kpts = torch.zeros(B, K, 2)
    pred_kpts[0, 0] = torch.tensor([0.5, 0.5])
    pred_kpts[0, 1] = torch.tensor([0.0, 0.0])

    vals = extract_teacher_value(teacher_hm, pred_kpts)
    check("extract shape", vals.shape == (B, K), f"got {vals.shape}")
    check("extract at peak", abs(vals[0, 0].item() - 1.0) < 0.01, f"got {vals[0, 0].item():.4f}")
    check("extract at corner", abs(vals[0, 1].item() - 1.0) < 0.01, f"got {vals[0, 1].item():.4f}")

    pred_grad = pred_kpts.clone().requires_grad_(True)
    vals_grad = extract_teacher_value(teacher_hm, pred_grad)
    vals_grad.sum().backward()
    check("extract differentiable", pred_grad.grad is not None and pred_grad.grad.abs().sum() > 0)

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="MSRA heatmap simulation for DistilPose KD")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.test:
        ok = _run_tests()
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
