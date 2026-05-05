_base_ = [
    '../../../configs/_base_/datasets/P2V.py',
    '../../../configs/_base_/default_runtime.py',
]

source_included_classes = [
    'impervious_surface', 'building', 'low_vegetation',
    'tree', 'car', 'clutter'
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


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='OSNet',
    closed_set=closed_set,
    source_classes=source_included_classes,
    target_classes=target_included_classes,
    pretrained=None,

    backbone_s=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),

    decode_head_s=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=len(source_included_classes),
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=source_class_weight)),

    decode_head_t=dict(
        type='DecoupledSegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,

        num_known_classes=len(source_included_classes),
        num_unknown_classes=len(target_included_classes) - len(source_included_classes),

        # 建议先保持 True，更稳定
        detach_unknown_from_trunk=True,

        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
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
    test_cfg=dict(
        mode='slide',
        crop_size=(1024, 1024),
        stride=(768, 768),
        decode_head='decode_head_t'
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        img_dir='Potsdam_RGB/img_dir/train',
        ann_dir='Potsdam_RGB/ann_dir/train',
        split='Potsdam_RGB/train.txt',
        source_included_classes=source_included_classes  # 同步类别列表到数据集
    ),
    val=dict(source_included_classes=target_included_classes),
    test=dict(source_included_classes=target_included_classes)
)



# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

# optimizer setting
optimizer = dict(
    backbone_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    feat_proj=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01)
)

runner = None
find_unused_parameters = True
