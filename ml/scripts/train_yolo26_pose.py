#!/usr/bin/env python3
"""YOLO26-Pose training script with HP search support."""

import argparse

import yaml
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config):
    print(f"Training: {config['name']}")
    print(f"Model: {config['model']}")
    print(f"Data: {config['data']}")
    print(f"LR: {config.get('lr0', 0.0005)}")
    print(f"Freeze: {config.get('freeze', 5)}")

    model = YOLO(config["model"])

    kwargs = {
        "data": config["data"],
        "epochs": config.get("epochs", 50),
        "batch": config.get("batch", 16),
        "imgsz": config.get("imgsz", 640),
        "lr0": config.get("lr0", 0.0005),
        "lrf": config.get("lrf", 0.88),
        "freeze": config.get("freeze", 5),
        "mosaic": config.get("mosaic", 0.9),
        "mixup": config.get("mixup", 0.05),
        "fliplr": config.get("fliplr", 0.5),
        "copy_paste": config.get("copy_paste", 0.0),
        "val": config.get("val", True),
        "save_period": config.get("save_period", 10),
        "project": config.get("project", "/root/yolo_runs"),
        "name": config["name"],
        "exist_ok": True,
        "device": config.get("device", 0),
        "workers": config.get("workers", 8),
        "seed": config.get("seed", 42),
    }

    results = model.train(**kwargs)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
