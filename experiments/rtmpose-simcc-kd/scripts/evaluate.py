#!/usr/bin/env python3
"""Evaluate RTMPose-s and compare against YOLO26s v36b.

Usage:
    python evaluate.py \
        --rtmpose-config configs/rtmpose_s_coco17_skating.py \
        --rtmpose-weights work_dirs/rtmpose_s_kd/best_coco_AP_epoch_*.pth \
        --yolo-weights ../yolo26-pose-kd/yolo26s-v36b-skating-kd.pt \
        --data-config ../yolo26-pose-kd/configs/data.yaml \
        --output results/comparison.json
"""

from __future__ import annotations

import argparse
import json

from mmengine.config import Config
from mmengine.runner import Runner


def evaluate_rtmpose(config_path, weights_path, work_dir):
    """Run mmpose evaluation."""
    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.load_from = weights_path

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    return {
        "AP": metrics.get("coco/AP", 0.0),
        "AP50": metrics.get("coco/AP50", 0.0),
        "AP75": metrics.get("coco/AP75", 0.0),
    }


def evaluate_yolo(weights_path, data_config, _work_dir):
    """Run YOLO26s evaluation via Ultralytics val."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(data=data_config, imgsz=384, batch=32)

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_pose": float(metrics.pose.map50),
        "mAP50_95_pose": float(metrics.pose.map),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtmpose-config")
    parser.add_argument("--rtmpose-weights")
    parser.add_argument("--yolo-weights")
    parser.add_argument("--data-config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}

    if args.rtmpose_config and args.rtmpose_weights:
        print("Evaluating RTMPose-s...")
        results["rtmpose_s"] = evaluate_rtmpose(
            args.rtmpose_config, args.rtmpose_weights, "work_dirs/eval_rtmpose"
        )

    if args.yolo_weights:
        print("Evaluating YOLO26s...")
        results["yolo26s_v36b"] = evaluate_yolo(
            args.yolo_weights, args.data_config, "work_dirs/eval_yolo"
        )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
