#!/usr/bin/env python3
"""
Pseudo-label SkatingVerse frames using MogaNet-B.

Output format: COCO JSON compatible with YOLO training.

Usage:
    python pseudo_label_skatingverse.py \
        --frames-dir /root/data/datasets/skatingverse/frames \
        --output-dir /root/data/datasets/skatingverse_pseudo \
        --model-path /root/data/models/athletepose3d/moganet_b_ap2d_384x288.pth \
        --conf-thresh 0.5 \
        --workers 4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add root to path for imports
sys.path.insert(0, "/root")

# Mock xtcocotools before mmpose imports
import pycocotools.mask as cocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class XtcocoTools:
    class coco:
        COCO = COCO

    class cocoeval:
        COCOeval = COCOeval

    class mask:
        encode = cocomask.encode
        decode = cocomask.decode


sys.modules["xtcocotools"] = XtcocoTools
sys.modules["xtcocotools.coco"] = XtcocoTools.coco
sys.modules["xtcocotools.cocoeval"] = XtcocoTools.cocoeval
sys.modules["xtcocotools.mask"] = XtcocoTools.mask

from moganet_official import MogaNet_feat

# COCO Keypoint mapping (17 keypoints)
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton connections for visualization
COCO_SKELETON = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # Head
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],  # Arms
    [5, 11],
    [6, 12],
    [11, 12],  # Torso
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],  # Legs
]


class DeconvHead(torch.nn.Module):
    """Deconvolution head for pose estimation."""

    def __init__(self):
        super().__init__()
        # 3 deconv layers: 512 -> 256 -> 256 -> 256
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.final = torch.nn.Conv2d(256, 17, 1, 1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final(x)
        return x


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian (DarkPose)."""
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps."""
    K, H, W = heatmaps.shape
    heatmaps_flatten = heatmaps.reshape(K, -1)
    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1
    return locs, vals


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Refine keypoints using UDP DarkPose algorithm."""
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # Modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()
    return keypoints


class MogaNetInference:
    """MogaNet-B inference wrapper for pseudo-labeling."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device

        # Load model
        print(f"Loading MogaNet-B from {model_path}")
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Initialize model
        self.backbone = MogaNet_feat(arch="base", out_indices=(3,)).to(device)
        self.head = DeconvHead().to(device)

        # Load weights (with prefix stripping)
        backbone_state = {}
        head_state = {}

        for k, v in state_dict.items():
            if "backbone" in k:
                new_k = k.replace("module.", "")
                backbone_state[new_k] = v
            elif "keypoint_head" in k:
                if "deconv_layers" in k:
                    suffix = k.replace("keypoint_head.deconv_layers.", "")
                    parts = suffix.split(".")
                    idx = int(parts[0])
                    rest = ".".join(parts[1:])
                    if idx == 0:
                        new_k = f"deconv1.0.{rest}"
                    elif idx == 1:
                        new_k = f"deconv1.1.{rest}"
                    elif idx == 3:
                        new_k = f"deconv2.0.{rest}"
                    elif idx == 4:
                        new_k = f"deconv2.1.{rest}"
                    elif idx == 6:
                        new_k = f"deconv3.0.{rest}"
                    elif idx == 7:
                        new_k = f"deconv3.1.{rest}"
                    else:
                        continue
                elif "final_layer" in k:
                    new_k = k.replace("keypoint_head.final_layer", "final")
                else:
                    continue
                head_state[new_k] = v

        self.backbone.load_state_dict(backbone_state)
        self.head.load_state_dict(head_state)

        self.backbone.eval()
        self.head.eval()

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print("Model loaded successfully")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for MogaNet-B."""
        # Resize to 384x288 (width x height)
        img_resized = cv2.resize(image, (384, 288))

        # Convert RGB to BGR if needed
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_resized

        # Normalize
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - torch.tensor(self.mean).view(1, 3, 1, 1)) / torch.tensor(
            self.std
        ).view(1, 3, 1, 1)

        return img_tensor.unsqueeze(0)  # Add batch dimension

    def postprocess(self, heatmaps: torch.Tensor, orig_h: int, orig_w: int) -> dict:
        """Convert heatmaps to COCO keypoints format."""
        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()  # (17, 72, 96)

        # Get keypoint locations
        keypoints, scores = get_heatmap_maximum(heatmaps_np)

        # Refine with DarkPose
        keypoints = keypoints[None]  # Add instance dimension
        keypoints = refine_keypoints_dark_udp(keypoints, heatmaps_np, blur_kernel_size=11)

        # Normalize to input size (384, 288)
        W, H = 96, 72  # Heatmap size
        keypoints = keypoints / [W - 1, H - 1] * [384, 288]

        # Scale to original image size
        scale_x = orig_w / 384
        scale_y = orig_h / 288
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y

        # Flatten to COCO format: [x1, y1, v1, x2, y2, v2, ...]
        keypoints_flat = []
        for i in range(17):
            x, y = keypoints[0, i]
            v = 2.0 if scores[i] > 0.5 else (1.0 if scores[i] > 0.3 else 0.0)
            keypoints_flat.extend([float(x), float(y), v])

        return {
            "keypoints": keypoints_flat,  # 51 values (17 * 3)
            "score": float(np.mean(scores)),  # Average confidence
        }

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """Predict pose for a single image."""
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        img_tensor = self.preprocess(image).to(self.device)

        # Forward pass
        features = self.backbone(img_tensor)
        heatmaps = self.head(features[-1])

        # Postprocess
        result = self.postprocess(heatmaps, orig_h, orig_w)

        return result


def create_coco_dict(images_info: list, annotations: list, categories: list) -> dict:
    """Create COCO format dictionary."""
    return {"images": images_info, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label SkatingVerse frames with MogaNet-B")
    parser.add_argument(
        "--frames-dir", type=Path, required=True, help="Path to extracted frames directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Path to output pseudo-labels directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/data/models/athletepose3d/moganet_b_ap2d_384x288.pth",
        help="Path to MogaNet-B checkpoint",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering (default: 0.5)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument(
        "--max-videos", type=int, default=None, help="Limit number of videos for testing"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotations").mkdir(exist_ok=True)
    (args.output_dir / "images").mkdir(exist_ok=True)

    # Initialize model
    model = MogaNetInference(args.model_path, args.device)

    # Find all frame directories
    frame_dirs = sorted([d for d in args.frames_dir.iterdir() if d.is_dir()])

    if args.max_videos:
        frame_dirs = frame_dirs[: args.max_videos]

    print(f"Found {len(frame_dirs)} video frame directories")

    # COCO categories (COCO 17 keypoints)
    categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": COCO_KEYPOINTS,
            "skeleton": COCO_SKELETON,
        }
    ]

    # Process frames
    all_images = []
    all_annotations = []

    annotation_id = 1

    for video_dir in tqdm(frame_dirs, desc="Processing videos"):
        video_id = video_dir.name
        frame_files = sorted(video_dir.glob("frame_*.jpg"))

        for frame_path in frame_files:
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            h, w = image.shape[:2]

            # Create image info
            image_id = len(all_images) + 1
            image_info = {
                "id": image_id,
                "file_name": str(frame_path.relative_to(args.frames_dir)),
                "width": w,
                "height": h,
            }
            all_images.append(image_info)

            # Predict pose
            result = model.predict(image)

            # Filter by confidence
            if result["score"] < args.conf_thresh:
                continue

            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": result["keypoints"],
                "score": result["score"],
                "num_keypoints": sum(1 for i in range(0, 51, 3) if result["keypoints"][i + 2] > 0),
                "area": 0,  # TODO: compute bbox area
                "iscrowd": 0,
                "bbox": [0, 0, w, h],  # TODO: compute bbox from keypoints
            }
            all_annotations.append(annotation)
            annotation_id += 1

    print(f"\nTotal images: {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")

    # Split train/val
    val_size = int(len(all_images) * args.val_split)

    import random

    random.seed(42)
    val_indices = set(random.sample(range(len(all_images)), val_size))

    train_images = [img for i, img in enumerate(all_images) if i not in val_indices]
    val_images = [img for i, img in enumerate(all_images) if i in val_indices]

    train_annotations = [ann for ann in all_annotations if ann["image_id"] - 1 not in val_indices]
    val_annotations = [ann for ann in all_annotations if ann["image_id"] - 1 in val_indices]

    # Remap image IDs
    val_id_map = {old_id: i + 1 for i, old_id in enumerate([img["id"] for img in val_images])}
    train_id_map = {old_id: i + 1 for i, old_id in enumerate([img["id"] for img in train_images])}

    for ann in val_annotations:
        ann["image_id"] = val_id_map[ann["image_id"]]
    for ann in train_annotations:
        ann["image_id"] = train_id_map[ann["image_id"]]

    for img in val_images:
        img["id"] = val_id_map[img["id"]]
    for img in train_images:
        img["id"] = train_id_map[img["id"]]

    # Save COCO JSON files
    train_json = args.output_dir / "annotations" / "train.json"
    val_json = args.output_dir / "annotations" / "val.json"

    with open(train_json, "w") as f:
        json.dump(create_coco_dict(train_images, train_annotations, categories), f)

    with open(val_json, "w") as f:
        json.dump(create_coco_dict(val_images, val_annotations, categories), f)

    print("\nSaved:")
    print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"  {train_json}")
    print(f"  {val_json}")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_path": args.model_path,
        "conf_thresh": args.conf_thresh,
        "val_split": args.val_split,
        "total_images": len(all_images),
        "total_annotations": len(all_annotations),
        "train": {"images": len(train_images), "annotations": len(train_annotations)},
        "val": {"images": len(val_images), "annotations": len(val_annotations)},
    }

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
