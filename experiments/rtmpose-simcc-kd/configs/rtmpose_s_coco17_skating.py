"""RTMPose-s config for COCO 17kp figure skating."""

from mmengine.config import read_base

with read_base():
    from mmpose.configs._base_.default_runtime import *
    from mmpose.configs._base_.schedules.schedule_420e import *

from mmengine.dataset import DefaultSampler
from mmpose.datasets import CocoDataset
from mmpose.datasets.transforms import (
    GetBBoxCenterScale,
    LoadImage,
    PackPoseInputs,
    RandomFlip,
    RandomHalfBody,
    TopdownAffine,
)
from mmpose.engine.hooks import ExpMomentumEMA
from mmpose.models import TopdownPoseEstimator
from mmpose.models.backbones import CSPNeXt
from mmpose.models.heads import RTMCCHead
from mmpose.models.losses import KLDiscretLoss

# === Model ===
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type=CSPNeXt,
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth",
            prefix="backbone.",
        ),
    ),
    head=dict(
        type=RTMCCHead,
        in_channels=384,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(24, 32),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type=KLDiscretLoss,
            use_target_weight=True,
            beta=1.0,
            label_softmax=True,
        ),
        decoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=6.0,
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
    ),
    test_cfg=dict(flip_test=True),
)

# === Data ===
dataset_type = CocoDataset
data_root = "data"

train_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction="horizontal"),
    dict(type=RandomHalfBody),
    dict(type=TopdownAffine, input_size=(192, 256)),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.5,
            ),
        ],
    ),
    dict(
        type="GenerateTarget",
        encoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=6.0,
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
    ),
    dict(type=PackPoseInputs),
]

val_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=TopdownAffine, input_size=(192, 256)),
    dict(type=PackPoseInputs),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode="topdown",
        ann_file="coco_skating_train.json",
        data_prefix=dict(img=""),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode="topdown",
        ann_file="coco_skating_val.json",
        data_prefix=dict(img=""),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# === Evaluator ===
val_evaluator = dict(
    type="CocoMetric",
    ann_file="data/coco_skating_val.json",
)
test_evaluator = val_evaluator

# === Optimizer ===
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
)

# === Training ===
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# === Hooks ===
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=10, max_keep_ckpts=3, save_best="coco/AP", rule="greater"
    ),
    logger=dict(type="LoggerHook", interval=50),
)

custom_hooks = [dict(type=ExpMomentumEMA, momentum=0.0002, priority=49)]

# === Runtime ===
load_from = "checkpoints/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth"
resume = False
