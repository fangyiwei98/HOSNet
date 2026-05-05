_base_ = [
    '../../../configs/_base_/datasets/P2V.py',
    '../../../configs/_base_/default_runtime.py'
]

source_included_classes = [
    'impervious_surface', 'building', 'low_vegetation',
    'tree', 'car'
]
target_included_classes = [
    'impervious_surface', 'building', 'low_vegetation',
    'tree', 'car', 'clutter'
]

closed_set = len(source_included_classes) == len(target_included_classes)


FULL_CLASS_WEIGHT = {
    'impervious_surface': 1.0,
    'building': 1.0,
    'low_vegetation': 1.0,
    'tree': 1.25,
    'car': 1.5,
    'clutter': 1.5,
}
source_class_weight = [FULL_CLASS_WEIGHT[c] for c in source_included_classes]
target_class_weight = [FULL_CLASS_WEIGHT[c] for c in target_included_classes]
unknown_class_weight = [
    FULL_CLASS_WEIGHT[c]
    for c in target_included_classes
    if c not in source_included_classes
]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='OSNet',
    closed_set=closed_set,
    source_classes=source_included_classes,
    target_classes=target_included_classes,
    pretrained='open-mmlab://resnet50_v1c',

    backbone_s=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),

    decode_head_s=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=len(source_included_classes),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=source_class_weight)
    ),

    decode_head_t=dict(
        type='DecoupledOSHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_known_classes=len(source_included_classes),
        num_unknown_classes=len(target_included_classes) - len(source_included_classes),

        # 建议先保持 True，更稳定
        detach_unknown_from_trunk=True,

        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=source_class_weight)
    ),

    contrast_cfg=dict(
        proj_dim=256,
        momentum=0.99,
        known_conf_thresh=0.7,
        discrepancy_thresh=0.2,
        tau_unified=0.07,
        loss_contrast_weight=0.1,
        loss_unknown_seg_weight=0.1,
        max_samples=4096,
        min_pixels_per_anchor=10,
        unknown_pseudo_thresh=0.0,

        # new
        infer_known_conf_thresh=0.7,
        infer_unknown_logit_bias=0.0,

        balance_unknown_pseudo=True,
        unknown_balance_ratio=0.5,
        min_unknown_pixels_per_class=16,
        unknown_balance_warmup_iters=4000,
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole', decode_head='decode_head_t')
)

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

# optimizer setting
optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    feat_proj=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01)
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        img_dir='Potsdam_RGB/img_dir/train',
        ann_dir='Potsdam_RGB/ann_dir/train',
        split='Potsdam_RGB/train.txt',
        source_included_classes=source_included_classes),
    val=dict(source_included_classes=target_included_classes),
    test=dict(source_included_classes=target_included_classes)
)
total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
runner = None
find_unused_parameters = True
