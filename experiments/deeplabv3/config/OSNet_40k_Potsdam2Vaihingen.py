_base_ = [
    '../../../configs/_base_/datasets/P2V.py',
    '../../../configs/_base_/default_runtime.py'
]

source_included_classes = [
    'impervious_surface', 'low_vegetation',
    'tree', 'car', 'clutter'
]
target_included_classes = [
    'impervious_surface', 'building', 'low_vegetation',
    'tree', 'car', 'clutter'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='OSNet',
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
        num_classes=len(source_included_classes),  # 动态计算类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0,  1.0, 1.25, 1.5, 1.5])
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
        num_classes=len(target_included_classes),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])
    ),

    cross_EMA=dict(
        type='decoder_only_t',
        training_ratio=0.25,
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
            num_classes=len(target_included_classes),
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5]))
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole', decode_head='decode_head_t')
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        source_included_classes=source_included_classes,
        pipeline=[
            dict(type='LoadImageFromFile_forAdap'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='MapPVLabelTrain', source_included_classes=source_included_classes, ignore_label=255),
            dict(type='Resize', img_scale=(512, 512), B_img_scale=(384, 384), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(384, 384), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'B_img', 'gt_semantic_seg']),
        ]
    ),
    val=dict(
        source_included_classes=target_included_classes
    ),
    test=dict(
        source_included_classes=target_included_classes
    )
)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
runner = None
find_unused_parameters = True