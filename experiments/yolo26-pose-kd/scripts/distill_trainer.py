#!/usr/bin/env python3
"""DistilPoseTrainer: Knowledge Distillation for YOLO26-Pose.

Teacher: MogaNet-B (pre-computed coordinates via soft argmax on heatmaps)
Student: YOLO26-Pose (coordinate regression)

v35c Architecture (2026-04-25):
    Coordinate KD only: MSE(student_kpts, teacher_kpts_global) with confidence weighting
    Feature KD: REMOVED (spatial misalignment between crop and full-image features)
    Progressive unfreeze: epoch 1-7 head only, epoch 8+ backbone with differential LR
    Progressive KD schedule: 0 during warmup, ramp 0->1 over 15 epochs, sustain at 1.0

Fixes applied from code review (2026-04-25):
    v35b: D2, C3/T1, D5, C1/D7, C4, T4, C2, F2
    v35c: BUG-1 PoseLoss26 decode formula (no *2.0, no -0.5)
    v35c: BUG-2 KD weight schedule inverted (progressive growth, not decay)
    v35c: BUG-3 cos_lr=True, lr0=0.001, warmup=3, mixup=0.1, patience=30, rect=True
    v35c: BUG-5 Anchor fallback uses median of centers (not K/2)
    v35c: BUG-6 teacher_conf.clamp(min=0.1) prevents inverse weights
    v35c: ARCH-1 Letterboxing: use batch['img_shape'] for normalization
    v35c: ARCH-3 Per-kp biomechanical weights, normalize by visible count
    v35c: ARCH-4 Optimizer split: weight (wd) vs bias/bn (no wd) groups

Usage:
    python3 distill_trainer.py train \\
        --model yolo26s-pose.yaml \\
        --data data.yaml \\
        --teacher-coords data/teacher_coords.h5 \\
        --epochs 210 \\
        --batch 128 \\
        --coord-alpha 0.05 \\
        --unfreeze-epoch 8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Teacher coordinate loader (HDF5)
# ---------------------------------------------------------------------------


class TeacherCoordLoader:
    """Load pre-computed teacher coordinates + confidence + crop params from HDF5.

    HDF5 structure:
        /coords: (N, 17, 2) float32 — keypoint (x, y) in normalized [0,1] crop space
        /confidence: (N, 17) float32 — max heatmap value per keypoint
        /crop_params: (N, 6) float32 — (x1, y1, crop_w, crop_h, img_w, img_h) in original pixels
        attrs["index"]: JSON mapping image_path -> row index
    """

    def __init__(self, h5_path: str | Path):
        self.path = Path(h5_path)
        self.file = None
        self.index: dict[str, int] = {}
        self._crop_params_ds = None
        print(f"Coord loader: {self.path}")

    def _open(self):
        if self.file is None:
            self.file = h5py.File(str(self.path), "r")
            self.index = json.loads(self.file.attrs.get("index", "{}"))
            self._crop_params_ds = self.file["crop_params"]
            print(f"Coord loader opened: {len(self.index)} entries")

    def load(
        self, im_files: list[str], device: torch.device = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Load teacher coords, confidence, crop_params for a batch.

        Returns: (coords, confidence, crop_params) or None
            coords: (B, 17, 2) float32 — crop-space coordinates [0,1]
            confidence: (B, 17) float32
            crop_params: (B, 6) float32 — (x1, y1, crop_w, crop_h, img_w, img_h)
        """
        import numpy as np

        self._open()
        coords_list, conf_list, cp_list = [], [], []

        for img_path in im_files:
            idx = self.index.get(img_path)
            if idx is None:
                idx = self.index.get(Path(img_path).name)
                if idx is None:
                    p = Path(img_path)
                    parts = p.parts
                    try:
                        data_idx = parts.index("data")
                        rel = str(Path(*parts[data_idx:]))
                        idx = self.index.get(rel)
                    except ValueError:
                        pass

            if idx is not None:
                coords_list.append(self.file["coords"][idx])
                conf_list.append(self.file["confidence"][idx])
                cp_list.append(self._crop_params_ds[idx])
            else:
                coords_list.append(np.zeros((17, 2), dtype=np.float32))
                conf_list.append(np.zeros(17, dtype=np.float32))
                cp_list.append(np.full(6, -1.0, dtype=np.float32))

        if not coords_list:
            return None

        coords = torch.from_numpy(np.stack(coords_list))
        conf = torch.from_numpy(np.stack(conf_list))
        crop_params = torch.from_numpy(np.stack(cp_list))

        if device is not None:
            coords = coords.to(device)
            conf = conf.to(device)
            crop_params = crop_params.to(device)

        return coords, conf, crop_params

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self._crop_params_ds = None


# ---------------------------------------------------------------------------
# DistilPoseTrainer — Coordinate-only KD
# ---------------------------------------------------------------------------


class DistilPoseTrainer:
    """v35b KD Trainer: Coordinate distillation with progressive unfreeze."""

    _CACHED_ORIG_LOSS = None  # class-level cache for original loss (survives deepcopy)

    def __init__(
        self,
        teacher_coords_path: str | Path | None = None,
        coord_alpha: float = 0.05,
        warmup_epochs: int = 3,
        unfreeze_epoch: int = 8,
        freeze_backbone: bool = False,
        stage2: bool = False,
    ):
        self.teacher_coords_path = teacher_coords_path
        self.coord_alpha = coord_alpha
        self.warmup_epochs = warmup_epochs
        self.unfreeze_epoch = unfreeze_epoch
        self.freeze_backbone = freeze_backbone
        self.stage2 = stage2

        self._current_epoch = 1
        self._max_epochs = 210
        self._original_loss = None
        self._model = None
        self._coord_loader: TeacherCoordLoader | None = None
        self._backbone_unfrozen = False
        self._base_lr = 0.002
        self._trainer_ref = None
        self._last_logged_epoch = None  # For diagnostic logging

    def __getstate__(self):
        """Exclude non-picklable objects (h5py, model) for deepcopy/EMA."""
        state = self.__dict__.copy()
        state["_coord_loader"] = None
        state["_model"] = None
        state["_original_loss"] = None
        state["_trainer_ref"] = None
        state["_last_logged_epoch"] = None
        return state

    def __setstate__(self, state):
        """Restore state and re-open HDF5 on unpickle."""
        self.__dict__.update(state)
        # Re-acquire original loss from class cache after unpickle
        if self._original_loss is None and self.teacher_coords_path:
            self._original_loss = type(self)._CACHED_ORIG_LOSS

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def set_max_epochs(self, epochs: int):
        self._max_epochs = epochs

    @property
    def loader(self):
        return self._coord_loader

    def compute_kd_weight(self) -> float:
        """Progressive KD weight: 0 during warmup, ramp 0->1 over 15 epochs, then sustain at 1.0."""
        if self._max_epochs <= 1:
            return 1.0
        post_warmup = self._current_epoch - self.warmup_epochs
        if post_warmup <= 0:
            return 0.0
        growth_epochs = 15
        if post_warmup >= growth_epochs:
            return 1.0
        return post_warmup / growth_epochs

    def setup_model(self, model: nn.Module):
        """Patch model: init coord loader, replace loss."""
        if self.teacher_coords_path:
            self._coord_loader = TeacherCoordLoader(self.teacher_coords_path)

        # Cache original loss at class level (survives deepcopy/EMA)
        self._original_loss = model.loss
        if type(self)._CACHED_ORIG_LOSS is None:
            type(self)._CACHED_ORIG_LOSS = model.loss

        # Stage 2: freeze backbone completely
        if self.stage2:
            for name, param in model.named_parameters():
                if "cv4" in name or "detect" in name or "sigma" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.freeze_backbone:
            for name, param in model.named_parameters():
                if "sigma" in name or "detect" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self._model = model
        model.loss = self.kd_loss  # type: ignore[assignment]

    def _decode_student_kpts(self, preds, B, K):
        """Decode student keypoints to pixel space.

        Matches PoseLoss26.kpts_decode + stride multiplication:
            raw + anchor  =>  anchor-relative space
            * stride      =>  pixel space

        Note: YOLO26's RealNVP flow replaced the v8 *2.0 - 0.5 scaling.
        PoseLoss26.calculate_keypoints_loss() divides GT by stride (line ~933),
        so kpts_decode returns anchor-relative coords. Adding stride gives pixel coords.

        Args:
            preds: dict from model.forward() during training
            B: batch size
            K: number of keypoints (17)

        Returns:
            decoded_kpts: (B, N_anchors, K, 3) in image pixel space
            stride_tensor: (N_anchors, 1)
        """
        from ultralytics.utils.tal import make_anchors

        src_preds = preds
        if isinstance(src_preds, dict) and "one2many" in src_preds:
            src_preds = src_preds["one2many"]

        kpts_raw = src_preds.get("kpts")
        if kpts_raw is None:
            return None, None

        head = self._model.model[-1]
        feats = src_preds.get("feats")
        if feats is None:
            return None, None

        anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

        # (B, nk, N_anchors) -> (B, N_anchors, K, 3)
        kpts = kpts_raw.permute(0, 2, 1).contiguous()
        kpts = kpts.view(B, -1, K, 3)  # FIX C3/T1: assign, don't discard

        sigma_perm = src_preds.get("kpts_sigma")
        if sigma_perm is None:
            return None, None

        # Decode: match PoseLoss26.kpts_decode + stride
        # PoseLoss26: y[..., 0] += anchor; y[..., 1] += anchor (no *2.0, no -0.5)
        # RealNVP flow replaced old v8 scaling. Then multiply by stride for pixel space.
        decoded = kpts.clone()
        decoded[..., 0] = (decoded[..., 0] + anchor_points[:, [0]]) * stride_tensor[:, [0]]
        decoded[..., 1] = (decoded[..., 1] + anchor_points[:, [1]]) * stride_tensor[:, [0]]
        decoded[..., 2] = decoded[..., 2].sigmoid()

        return decoded, stride_tensor

    def _inverse_affine_transform(self, crop_coords, crop_params):
        """Transform teacher coords from crop [0,1] to full image [0,1].

        Uses actual crop parameters (x1, y1, crop_w, crop_h, img_w, img_h)
        which account for 1.4x padding and edge clamping.

        Args:
            crop_coords: (B, K, 2) — teacher coords in crop [0,1] space
            crop_params: (B, 6) — (x1, y1, crop_w, crop_h, img_w, img_h) in original pixels

        Returns:
            global_coords: (B, K, 2) — coords in full image [0,1] space
        """
        # Crop dimensions (teacher uses 384x288)
        crop_w_teacher = 384  # MOGANET_W
        crop_h_teacher = 288  # MOGANET_H

        # Convert crop [0,1] to crop pixels
        crop_px = crop_coords[..., 0] * crop_w_teacher  # (B, K)
        crop_py = crop_coords[..., 1] * crop_h_teacher  # (B, K)

        # Actual crop parameters in original image pixels
        x1 = crop_params[:, 0]  # (B,)
        y1 = crop_params[:, 1]
        actual_cw = crop_params[:, 2]
        actual_ch = crop_params[:, 3]
        img_w = crop_params[:, 4]
        img_h = crop_params[:, 5]

        # Scale factors: crop pixels -> original image pixels
        sx = actual_cw / crop_w_teacher  # (B,)
        sy = actual_ch / crop_h_teacher  # (B,)

        # Map crop pixels to original image pixels
        global_x = x1.unsqueeze(1) + crop_px * sx.unsqueeze(1)  # (B, K)
        global_y = y1.unsqueeze(1) + crop_py * sy.unsqueeze(1)  # (B, K)

        # Normalize to [0,1]
        global_x = global_x / img_w.unsqueeze(1)
        global_y = global_y / img_h.unsqueeze(1)

        return torch.stack([global_x, global_y], dim=-1)  # (B, K, 2)

    def _select_best_anchor(self, student_kpts, batch_idx, gt_kpts, B, K):
        """Select best anchor per image using IoU matching.

        Student predicts (B, N_anchors, K, 3). We need to pick one anchor per
        image that best matches the teacher's person (via GT bbox).

        Uses bbox IoU between student detection and GT bbox.

        Args:
            student_kpts: (B, N_anchors, K, 3) in pixel space
            batch_idx: (N_objects,) which image each GT object belongs to
            gt_kpts: (N_objects, K, 3) GT keypoints
            B: batch size
            K: number of keypoints

        Returns:
            selected: (B, K, 3) — best anchor's keypoints per image
        """
        # Build per-image GT info
        batch_idx_flat = batch_idx.long().flatten()

        # For each image, get GT kpts (take first person per image)
        gt_per_img = {}
        for obj_i in range(len(batch_idx_flat)):
            img_i = batch_idx_flat[obj_i].item()
            if img_i not in gt_per_img:
                gt_per_img[img_i] = gt_kpts[obj_i]

        # Build GT bboxes from GT keypoints (visible only)
        gt_bboxes = torch.zeros(B, 4, device=student_kpts.device)
        for img_i, kpts in gt_per_img.items():
            vis = kpts[..., 2] > 0
            if vis.any():
                xs = kpts[vis, 0]
                ys = kpts[vis, 1]
                gt_bboxes[img_i, 0] = xs.min()  # x1
                gt_bboxes[img_i, 1] = ys.min()  # y1
                gt_bboxes[img_i, 2] = xs.max()  # x2
                gt_bboxes[img_i, 3] = ys.max()  # y2
            else:
                gt_bboxes[img_i] = -1  # no valid GT

        # Build student detection bboxes from keypoints
        # student_kpts: (B, N_anchors, K, 3)
        # For each anchor, compute bbox from all keypoints
        # Use all keypoints regardless of visibility for bbox (student may not have vis flag right)
        sx = student_kpts[..., 0]  # (B, N_anchors, K)
        sy = student_kpts[..., 1]  # (B, N_anchors, K)
        student_x1 = sx.min(dim=-1).values  # (B, N_anchors)
        student_y1 = sy.min(dim=-1).values
        student_x2 = sx.max(dim=-1).values
        student_y2 = sy.max(dim=-1).values

        selected = torch.zeros(B, K, 3, device=student_kpts.device)
        for img_i in range(B):
            if gt_bboxes[img_i, 0] < 0:
                # No valid GT — take anchor nearest to median of all anchor centers
                cx = (student_x1[img_i] + student_x2[img_i]) / 2
                cy = (student_y1[img_i] + student_y2[img_i]) / 2
                median_cx = cx.median()
                median_cy = cy.median()
                dist = (cx - median_cx).abs() + (cy - median_cy).abs()
                best = dist.argmin()
                selected[img_i] = student_kpts[img_i, best]
                continue

            # Compute IoU between all student anchors and GT bbox
            inter_x1 = torch.max(student_x1[img_i], gt_bboxes[img_i, 0])
            inter_y1 = torch.max(student_y1[img_i], gt_bboxes[img_i, 1])
            inter_x2 = torch.min(student_x2[img_i], gt_bboxes[img_i, 2])
            inter_y2 = torch.min(student_y2[img_i], gt_bboxes[img_i, 3])

            inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            student_area = (student_x2[img_i] - student_x1[img_i]) * (
                student_y2[img_i] - student_y1[img_i]
            )
            gt_area = (gt_bboxes[img_i, 2] - gt_bboxes[img_i, 0]) * (
                gt_bboxes[img_i, 3] - gt_bboxes[img_i, 1]
            )
            union_area = student_area + gt_area - inter_area
            iou = inter_area / union_area.clamp(min=1e-6)

            best = iou.argmax()
            selected[img_i] = student_kpts[img_i, best]

        return selected

    def kd_loss(self, batch: dict[str, Any], preds=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v35b KD loss: L_gt + coord_alpha * L_coord * batch_size."""
        model = self._model

        # FIX: Fallback epoch tracking — if _current_epoch wasn't updated via preprocess_batch
        if (
            self._trainer_ref is not None
            and hasattr(self._trainer_ref, "epoch")
            and self._current_epoch != self._trainer_ref.epoch
        ):
            self._current_epoch = self._trainer_ref.epoch

        if model is None:
            # After deepcopy (EMA), _model is None — just compute GT loss
            if self._original_loss is None:
                self._original_loss = type(self)._CACHED_ORIG_LOSS
            gt_loss, loss_items = self._original_loss(batch, preds)
            kd_pad = torch.zeros(2, device=gt_loss.device)
            return gt_loss, torch.cat([loss_items, kd_pad])

        # FIX: Diagnostic logging — log once per epoch
        if self._last_logged_epoch != self._current_epoch:
            w = self.compute_kd_weight()
            print(
                f"[KD] Epoch {self._current_epoch}, warmup={self.warmup_epochs}, kd_weight={w:.4f}"
            )
            self._last_logged_epoch = self._current_epoch

        # Progressive unfreeze at unfreeze_epoch
        if (
            not self._backbone_unfrozen
            and not self.freeze_backbone
            and not self.stage2
            and self._current_epoch >= self.unfreeze_epoch
        ):
            print(f"\n>>> Epoch {self._current_epoch}: Unfreezing backbone with differential LR")
            for _name, param in model.named_parameters():
                param.requires_grad = True
            self._backbone_unfrozen = True
            # Rebuild optimizer with differential LR
            self._rebuild_optimizer(model)

        # Forward pass
        if preds is None:
            preds = model.forward(batch["img"])

        preds_for_loss = preds
        if isinstance(preds_for_loss, tuple):
            preds_for_loss = preds_for_loss[1] if len(preds_for_loss) > 1 else preds_for_loss

        # Standard GT loss
        gt_loss, loss_items = self._original_loss(batch, preds)

        # KD pad: 6 base + 2 KD (coord, weight)
        kd_pad = torch.zeros(2, device=gt_loss.device)
        if not model.training:
            return gt_loss, torch.cat([loss_items, kd_pad])
        if self._current_epoch < self.warmup_epochs:
            return gt_loss, torch.cat([loss_items, kd_pad])

        w_kd = self.compute_kd_weight()
        coord_loss = torch.tensor(0.0, device=gt_loss.device)

        # --- Coordinate KD ---
        if self._coord_loader is not None:
            im_files = batch.get("im_file", [])
            if isinstance(im_files, (list, tuple)):
                teacher_data = self._coord_loader.load(list(im_files), device=gt_loss.device)

            if teacher_data is not None:
                teacher_coords, teacher_conf, teacher_crop_params = teacher_data
                B = teacher_coords.shape[0]
                K = teacher_coords.shape[1]

                # Decode student predictions
                student_kpts, _stride = self._decode_student_kpts(preds_for_loss, B, K)

                if student_kpts is not None:
                    gt_kpts = batch.get("keypoints")
                    batch_idx = batch.get("batch_idx")

                    if gt_kpts is not None and batch_idx is not None:
                        # Get image dimensions — prefer original sizes (before letterboxing)
                        if "img_shape" in batch:
                            # batch["img_shape"]: (B, 2) with original (h, w) per image
                            img_shapes = batch["img_shape"]  # (B, 2)
                            img_h = img_shapes[0, 0].item()  # Use first image's original height
                            img_w = img_shapes[0, 1].item()  # Use first image's original width
                        else:
                            # Fallback: letterboxed size (less accurate for non-square images)
                            img_h, img_w = batch["img"].shape[2], batch["img"].shape[3]

                        # Select best anchor per image via IoU matching
                        student_selected = self._select_best_anchor(
                            student_kpts, batch_idx, gt_kpts, B, K
                        )  # (B, K, 3) in pixel space

                        # Normalize student coords to [0,1]
                        student_xy_norm = student_selected[..., :2].clone()
                        student_xy_norm[..., 0] /= img_w
                        student_xy_norm[..., 1] /= img_h

                        # Transform teacher coords from crop to global [0,1]
                        valid_cp = teacher_crop_params[:, 0] >= 0  # (B,) has crop params
                        teacher_global = self._inverse_affine_transform(
                            teacher_coords, teacher_crop_params
                        )

                        # Visibility mask from GT (primary person, first GT per image)
                        batch_idx_flat = batch_idx.long().flatten()
                        max_objects = (
                            max((batch_idx_flat == i).sum().item() for i in range(B))
                            if B > 0
                            else 0
                        )
                        vis_mask = torch.ones(B, K, device=gt_loss.device)
                        if max_objects > 0:
                            gt_kpts_per_img = torch.zeros(
                                B, max_objects, K, 3, device=gt_loss.device, dtype=gt_kpts.dtype
                            )
                            offsets = torch.zeros(B + 1, dtype=torch.long, device=gt_loss.device)
                            offsets.scatter_add_(
                                0, batch_idx_flat + 1, torch.ones_like(batch_idx_flat)
                            )
                            offsets = offsets.cumsum(0)
                            within_idx = (
                                torch.arange(len(batch_idx_flat), device=gt_loss.device)
                                - offsets[batch_idx_flat]
                            )
                            gt_kpts_dev = gt_kpts.to(gt_loss.device)
                            gt_kpts_per_img[batch_idx_flat, within_idx] = gt_kpts_dev
                            primary_gt = gt_kpts_per_img[:, 0, :, :]  # (B, K, 3)
                            vis_mask = (primary_gt[..., 2] > 0).float()

                        # Confidence-weighted MSE in [0,1] space
                        # FIX T4: multiply by batch_size to match PoseLoss26 scaling
                        # FIX BUG-6: clamp teacher_conf to prevent negative/inverse weights
                        teacher_conf_clamped = teacher_conf.clamp(min=0.1)
                        weight = teacher_conf_clamped * vis_mask * valid_cp.unsqueeze(1).float()

                        # ARCH-3: Per-keypoint biomechanical importance weights (H3.6M 17kp order)
                        # HIP_CENTER(0), RHIP(1), RKNEE(2), RFOOT(3), LHIP(4), LKNEE(5),
                        # LFOOT(6), SPINE(7), THORAX(8), NECK(9), HEAD(10), LSHOULDER(11),
                        # LELBOW(12), LWRIST(13), RSHOULDER(14), RELBOW(15), RWRIST(16)
                        kp_weights = torch.tensor(
                            [
                                3.0,  # HIP_CENTER
                                2.5,  # RHIP
                                2.5,  # RKNEE
                                2.0,  # RFOOT
                                2.0,  # LHIP
                                2.0,  # LKNEE
                                2.0,  # LFOOT
                                2.0,  # SPINE
                                2.0,  # THORAX
                                1.5,  # NECK
                                1.0,  # HEAD
                                1.0,  # LSHOULDER
                                0.8,  # LELBOW
                                0.8,  # LWRIST
                                0.5,  # RSHOULDER
                                0.5,  # RELBOW
                                1.5,  # RWRIST
                            ],
                            device=gt_loss.device,
                        )  # (17,)
                        weight = weight * kp_weights.unsqueeze(0)  # (B, K) broadcast

                        per_kp_loss = ((student_xy_norm - teacher_global) ** 2).sum(
                            dim=-1
                        )  # (B, K)

                        # ARCH-3: Normalize by visible count per image, not weight_sum
                        visible_count = vis_mask.sum(dim=-1).clamp(min=1.0)  # (B,)
                        coord_loss = (weight * per_kp_loss).sum() / visible_count.sum().clamp(
                            min=1.0
                        )
                        coord_loss = coord_loss * B  # FIX T4: scale to match batch-scaled GT loss

        # Total loss
        total_loss = gt_loss + self.coord_alpha * w_kd * coord_loss

        kd_items = torch.cat(
            [
                loss_items,
                torch.tensor([coord_loss.item(), w_kd], device=gt_loss.device),
            ]
        )
        return total_loss, kd_items

    def _rebuild_optimizer(self, model):
        """Add backbone params to optimizer with 0.1x LR after unfreeze.

        Splits backbone params into weight (with decay) and bias/bn (no decay) groups,
        matching Ultralytics' build_optimizer pattern.
        """
        trainer = self._trainer_ref
        if trainer is None:
            print("WARNING: Cannot rebuild optimizer — no trainer reference")
            return

        base_lr = getattr(self, "_base_lr", trainer.optimizer.param_groups[0]["lr"])
        backbone_lr = base_lr * 0.1
        default_wd = trainer.optimizer.param_groups[0].get("weight_decay", 1e-5)

        # Separate backbone params: weight (with decay) vs bias/bn (no decay)
        backbone_params_wd = []
        backbone_params_no_wd = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Skip params already in optimizer (detect/sigma = head)
            already_optimized = any(
                param is p for pg in trainer.optimizer.param_groups for p in pg["params"]
            )
            if already_optimized:
                continue
            if param.ndim == 1 or "bn" in name.lower() or "norm" in name.lower():
                backbone_params_no_wd.append(param)
            else:
                backbone_params_wd.append(param)

        added = 0
        if backbone_params_wd:
            trainer.optimizer.add_param_group(
                {
                    "params": backbone_params_wd,
                    "lr": backbone_lr,
                    "weight_decay": default_wd,
                }
            )
            added += len(backbone_params_wd)
        if backbone_params_no_wd:
            trainer.optimizer.add_param_group(
                {
                    "params": backbone_params_no_wd,
                    "lr": backbone_lr,
                    "weight_decay": 0.0,
                }
            )
            added += len(backbone_params_no_wd)

        if added > 0:
            print(
                f"  Added backbone: {added} params (wd={len(backbone_params_wd)}, no_wd={len(backbone_params_no_wd)}), lr={backbone_lr:.6f} (0.1x)"
            )

    def restore_model(self, model: nn.Module):
        if self._original_loss is not None:
            model.loss = self._original_loss
        if self._coord_loader is not None:
            self._coord_loader.close()

    def get_loss_names(self, base_names: tuple[str, ...]) -> tuple[str, ...]:
        return (*base_names, "kd_coord_loss", "kd_weight")


# ---------------------------------------------------------------------------
# Ultralytics PoseTrainer integration
# ---------------------------------------------------------------------------


def make_distil_pose_trainer(
    teacher_coords_path: str | Path | None = None,
    coord_alpha: float = 0.05,
    warmup_epochs: int = 3,
    unfreeze_epoch: int = 8,
    freeze_backbone: bool = False,
    stage2: bool = False,
):
    """Create a PoseTrainer subclass with v35b KD support (coordinate-only).

    Returns a class that can be used with Ultralytics CLI or Python API:
        trainer = make_distil_pose_trainer(teacher_coords_path="coords.h5")
        t = trainer(overrides={...})
        t.train()
    """
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import unwrap_model

    kd_config = {
        "teacher_coords_path": teacher_coords_path,
        "coord_alpha": coord_alpha,
        "warmup_epochs": warmup_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "freeze_backbone": freeze_backbone,
        "stage2": stage2,
    }

    class _DistilPoseTrainer(PoseTrainer):
        """PoseTrainer with Coordinate Knowledge Distillation."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg, overrides, _callbacks)
            self._kd = DistilPoseTrainer(**kd_config)

        def setup_model(self):
            """Set up model and patch with KD loss."""
            ckpt = super().setup_model()
            self._kd.setup_model(unwrap_model(self.model))
            self._kd.set_max_epochs(self.epochs)
            self._kd._trainer_ref = self  # Store reference for optimizer rebuild
            return ckpt

        def build_optimizer(self, model, **kwargs):
            """Override to add backbone param group with differential LR (0.1x)."""
            # Call parent to get standard optimizer with weight/bn/bias groups
            optimizer = super().build_optimizer(model, **kwargs)
            # Store base LR for later use in unfreeze
            self._kd._base_lr = optimizer.param_groups[0]["lr"]
            return optimizer

        def preprocess_batch(self, batch):
            """Attach epoch info for KD warmup."""
            batch = super().preprocess_batch(batch)
            # FIX: Safety net — verify self.epoch is not None/0 unexpectedly
            if hasattr(self, "epoch") and self.epoch is not None and self.epoch > 0:
                self._kd.set_epoch(self.epoch)
            else:
                print(
                    "[KD WARNING] self.epoch is None or 0 in preprocess_batch, skipping set_epoch"
                )
            return batch

        def get_validator(self):
            """Return validator with extended loss names."""
            validator = super().get_validator()
            base_names = self.loss_names
            self.loss_names = self._kd.get_loss_names(base_names)
            return validator

        def label_loss_items(self, loss_items=None, prefix="train"):
            """Label loss items including KD components."""
            labels = super().label_loss_items(loss_items, prefix)
            if loss_items is not None and len(loss_items) > len(labels):
                extra = len(loss_items) - len(labels)
                for i in range(extra):
                    labels[f"{prefix}_kd_{i}"] = loss_items[len(labels) + i].item()
            return labels

    _DistilPoseTrainer.__name__ = "DistilPoseTrainer"
    _DistilPoseTrainer.__qualname__ = "DistilPoseTrainer"
    return _DistilPoseTrainer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def train():
    """Launch v35b KD training via Ultralytics CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="DistilPoseTrainer: v35b KD for YOLO26-Pose")
    parser.add_argument("--model", type=str, default="yolo26s-pose.yaml", help="Student model path")
    parser.add_argument("--data", type=str, required=True, help="data.yaml path")
    parser.add_argument(
        "--teacher-coords", type=str, required=True, help="Teacher coords HDF5 path"
    )
    parser.add_argument("--epochs", type=int, default=210)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--coord-alpha", type=float, default=0.05, help="Coordinate KD loss weight")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-epoch", type=int, default=8, help="Epoch to unfreeze backbone")
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--no-freeze", dest="freeze_backbone", action="store_false")
    parser.add_argument("--stage2", action="store_true", help="Enable Stage 2 self-KD")
    parser.add_argument("--name", type=str, default="distil_pose")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    TrainerClass = make_distil_pose_trainer(
        teacher_coords_path=args.teacher_coords,
        coord_alpha=args.coord_alpha,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
        freeze_backbone=args.freeze_backbone,
        stage2=args.stage2,
    )

    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "name": args.name,
        "mosaic": 0.0,
        "cos_lr": True,
        "lr0": 0.001,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": "AdamW",
        "mixup": 0.1,
        "patience": 30,
        "rect": True,
        "device": args.device,
    }

    trainer = TrainerClass(overrides=overrides)
    trainer.train()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def run_tests():
    """Run unit tests without GPU (mock data)."""
    import traceback

    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except (AssertionError, RuntimeError, ValueError) as e:
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("DistilPoseTrainer unit tests (v35c coordinate-only)")
    print("=" * 60)

    # Test 1: TeacherCoordLoader graceful handling
    def test_loader_no_file():
        loader = TeacherCoordLoader(h5_path="/nonexistent/path.h5")
        assert loader.file is None

    test("TeacherCoordLoader init with missing file", test_loader_no_file)

    # Test 2: Inverse affine with known values
    def test_inverse_affine():
        # Crop params: x1=100, y1=50, crop_w=300, crop_h=200, img_w=1000, img_h=750
        # Teacher heatmap size: 384x288
        crop_coords = torch.tensor([[[0.5, 0.5]]])  # center of crop
        crop_params = torch.tensor([[100.0, 50.0, 300.0, 200.0, 1000.0, 750.0]])

        kd = DistilPoseTrainer()
        global_coords = kd._inverse_affine_transform(crop_coords, crop_params)

        # Expected: 100 + 0.5*384 * (300/384) = 100 + 192 * 0.78125 = 100 + 150 = 250
        # Then 250/1000 = 0.25
        expected_x = (100.0 + 0.5 * 384 * (300.0 / 384.0)) / 1000.0
        expected_y = (50.0 + 0.5 * 288 * (200.0 / 288.0)) / 750.0
        assert abs(global_coords[0, 0, 0].item() - expected_x) < 1e-5, (
            f"Expected x={expected_x}, got {global_coords[0, 0, 0].item()}"
        )
        assert abs(global_coords[0, 0, 1].item() - expected_y) < 1e-5, (
            f"Expected y={expected_y}, got {global_coords[0, 0, 1].item()}"
        )

    test("Inverse affine transform with known values", test_inverse_affine)

    # Test 3: Inverse affine with edge clamping
    def test_inverse_affine_edge_clamp():
        # Crop at image edge: x1=0 (clamped from negative), img_w=800
        crop_coords = torch.tensor([[[0.0, 0.0]]])  # top-left of crop
        crop_params = torch.tensor([[0.0, 0.0, 200.0, 150.0, 800.0, 600.0]])

        kd = DistilPoseTrainer()
        global_coords = kd._inverse_affine_transform(crop_coords, crop_params)

        # Top-left of crop maps to (0, 0) in image
        assert global_coords[0, 0, 0].item() < 0.01, (
            f"Expected x~0, got {global_coords[0, 0, 0].item()}"
        )
        assert global_coords[0, 0, 1].item() < 0.01, (
            f"Expected y~0, got {global_coords[0, 0, 1].item()}"
        )

    test("Inverse affine with edge clamping", test_inverse_affine_edge_clamp)

    # Test 4: IoU matching with perfect overlap
    def test_iou_matching():
        B, K = 2, 3
        N_anchors = 10

        student_kpts = torch.zeros(B, N_anchors, K, 3)
        # Image 0: anchor 3 has bbox matching GT at (10, 10, 100, 100)
        student_kpts[0, 3, 0] = torch.tensor([10.0, 10.0, 1.0])
        student_kpts[0, 3, 1] = torch.tensor([100.0, 100.0, 1.0])
        student_kpts[0, 3, 2] = torch.tensor([50.0, 50.0, 1.0])
        # Image 0: anchor 7 has different bbox far away
        student_kpts[0, 7, 0] = torch.tensor([200.0, 200.0, 1.0])
        student_kpts[0, 7, 1] = torch.tensor([300.0, 300.0, 1.0])
        student_kpts[0, 7, 2] = torch.tensor([250.0, 250.0, 1.0])

        # Image 1: anchor 5 has matching bbox
        student_kpts[1, 5, 0] = torch.tensor([50.0, 50.0, 1.0])
        student_kpts[1, 5, 1] = torch.tensor([150.0, 150.0, 1.0])
        student_kpts[1, 5, 2] = torch.tensor([100.0, 100.0, 1.0])

        batch_idx = torch.tensor([0, 1])  # one GT per image
        gt_kpts = torch.zeros(2, K, 3)
        gt_kpts[0, 0] = torch.tensor([10.0, 10.0, 1.0])
        gt_kpts[0, 1] = torch.tensor([100.0, 100.0, 1.0])
        gt_kpts[0, 2] = torch.tensor([50.0, 50.0, 1.0])
        gt_kpts[1, 0] = torch.tensor([50.0, 50.0, 1.0])
        gt_kpts[1, 1] = torch.tensor([150.0, 150.0, 1.0])
        gt_kpts[1, 2] = torch.tensor([100.0, 100.0, 1.0])

        kd = DistilPoseTrainer()
        selected = kd._select_best_anchor(student_kpts, batch_idx, gt_kpts, B, K)

        assert selected.shape == (B, K, 3), f"Expected ({B}, {K}, 3), got {selected.shape}"
        # Image 0 should select anchor 3 (perfect IoU with GT bbox (10,10,100,100))
        assert torch.equal(selected[0], student_kpts[0, 3]), (
            f"Image 0: expected anchor 3, got {selected[0]}"
        )
        # Image 1 should select anchor 5
        assert torch.equal(selected[1], student_kpts[1, 5]), (
            f"Image 1: expected anchor 5, got {selected[1]}"
        )

    test("IoU matching selects correct anchor", test_iou_matching)

    # Test 5: Compute KD weight — progressive growth schedule
    def test_kd_weight():
        kd = DistilPoseTrainer(warmup_epochs=3)
        kd.set_max_epochs(210)

        # During warmup: weight = 0
        kd.set_epoch(1)
        w = kd.compute_kd_weight()
        assert w == 0.0, f"Epoch 1 (warmup): expected w=0.0, got {w}"

        kd.set_epoch(3)
        w = kd.compute_kd_weight()
        assert w == 0.0, f"Epoch 3 (warmup): expected w=0.0, got {w}"

        # First epoch after warmup: 1/15
        kd.set_epoch(4)
        w = kd.compute_kd_weight()
        assert abs(w - 1.0 / 15.0) < 1e-6, f"Epoch 4: expected w~{1 / 15:.4f}, got {w}"

        # Mid growth: epoch 11 -> (11-3)/15 = 8/15
        kd.set_epoch(11)
        w = kd.compute_kd_weight()
        assert abs(w - 8.0 / 15.0) < 1e-6, f"Epoch 11: expected w~{8 / 15:.4f}, got {w}"

        # End of growth: epoch 18 -> (18-3)/15 = 15/15 = 1.0
        kd.set_epoch(18)
        w = kd.compute_kd_weight()
        assert w == 1.0, f"Epoch 18: expected w=1.0, got {w}"

        # Sustain at 1.0
        kd.set_epoch(210)
        w = kd.compute_kd_weight()
        assert w == 1.0, f"Epoch 210: expected w=1.0, got {w}"

    test("KD weight progressive growth", test_kd_weight)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = run_tests()
        sys.exit(0 if ok else 1)
    else:
        train()
