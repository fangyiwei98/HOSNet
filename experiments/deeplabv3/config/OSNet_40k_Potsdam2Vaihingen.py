_base_ = [
    '../../../configs/_base_/datasets/P2V.py', '../../../configs/_base_/default_runtime.py'
]


# 固定全量类别 + 对应权重（绝对顺序不变）
FULL_CLASSES = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']
FULL_CLASS_WEIGHT = [1.0, 1.0, 1.0, 1.25, 1.5, 1.5]

# 源域 / 目标域类别
source_included_classes = ['impervious_surface', 'low_vegetation',  'tree', 'car', 'clutter']
target_included_classes = FULL_CLASSES

# 🔥 自适应计算源域权重（自动匹配）
class_weight_s = [
    FULL_CLASS_WEIGHT[FULL_CLASSES.index(cls)]
    for cls in source_included_classes
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='OSNet',
    # --- 新增这两行 ---
    source_classes=source_included_classes,
    target_classes=target_included_classes,
    # -----------------
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
        num_classes=len(source_included_classes),  # 动态计算类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight_s)
    ),
    decode_head_t=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=FULL_CLASS_WEIGHT)
    ),
    cross_EMA = dict(
        type='decoder_only_t',
        training_ratio=1,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.01, 1.51, 1.51, 2.01, 2.01],
        backbone_EMA=dict(
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
        decode_head_EMA=dict(
            type='DepthwiseSeparableASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                class_weight=FULL_CLASS_WEIGHT))
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',decode_head='decode_head_t'))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

# optimizer setting
optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
    )

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        source_included_classes=source_included_classes  # 同步类别列表到数据集
    ),
    val=dict(
        source_included_classes=target_included_classes
    ),
    test=dict(
        source_included_classes=target_included_classes)
)
total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
runner = None
find_unused_parameters = True