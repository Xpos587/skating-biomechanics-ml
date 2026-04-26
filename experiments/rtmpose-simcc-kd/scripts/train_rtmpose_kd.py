#!/usr/bin/env python3
"""Train RTMPose-s with DWPose-style distillation.

Usage:
    python train_rtmpose_kd.py \
        --config configs/rtmpose_s_kd.py \
        --work-dir work_dirs/rtmpose_s_kd \
        --teacher-simcc data/teacher_simcc.npz
"""

from __future__ import annotations

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Train config file path")
    parser.add_argument("--work-dir", required=True, help="Working directory")
    parser.add_argument("--teacher-simcc", help="Teacher SimCC labels (.npz)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.teacher_simcc_path = args.teacher_simcc

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
