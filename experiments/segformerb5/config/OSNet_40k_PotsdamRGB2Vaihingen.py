_base_ = [
    '../../../configs/_base_/datasets/P2V.py', '../../../configs/_base_/default_runtime.py',
]

# 同步控制源域类别数和解码器num_classes！！！
source_included_classes = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car']
target_included_classes = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='OSNet',
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
    # ------------------- 源域解码器（动态设置num_classes） -------------------
    decode_head_s=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=len(source_included_classes),  # 动态计算类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # ------------------- 目标域解码器（保留全类别） -------------------
    decode_head_t=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # ------------------- 判别器-------------------
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='IN'),
        in_channels=512),
    # ------------------- EMA教师网络（适配Open-Set） -------------------
    cross_EMA = dict(
        type='single_t',
        training_ratio=0.25,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=None,
        backbone_EMA=dict(
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
        decode_head_EMA=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            ignore_index=255,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768),
                  decode_head='decode_head_t'))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        img_dir='Potsdam_RGB/img_dir/train',
        ann_dir='Potsdam_RGB/ann_dir/train',
        split='Potsdam_RGB/train.txt',
        source_included_classes=source_included_classes  # 同步类别列表到数据集
    ),
    val=dict(
        source_included_classes=target_included_classes
    ),
    test=dict(
        source_included_classes=target_included_classes)
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
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='mIoU', pre_eval=True)

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
    discriminator_s=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)))

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True
