"""KD overrides for RTMPose-s training."""

from mmengine.config import read_base

with read_base():
    from .rtmpose_s_coco17_skating import *

# KD-specific overrides
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
        use_teacher_simcc=True,
    ),
    dict(type=PackPoseInputs),
]

# KD loss weights
model = dict(
    head=dict(
        loss=dict(
            type="CombinedLoss",
            losses=[
                dict(
                    type=KLDiscretLoss,
                    use_target_weight=True,
                    beta=1.0,
                    label_softmax=True,
                    loss_weight=1.0,
                ),
                dict(type="KLDistillationLoss", use_target_weight=True, loss_weight=0.5),
                dict(type="L1CoordinateLoss", loss_weight=0.1),
            ],
        ),
    ),
)

# Smaller batch for KD
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
)

# Shorter training for KD
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
