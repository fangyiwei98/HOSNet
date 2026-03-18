# dataset settings
dataset_type = 'PVDataset_forAdap'
data_root = '/data/fywdata/ISPRS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)


train_pipeline = [
    dict(type='LoadImageFromFile_forAdap'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), B_img_scale=crop_size, ratio_range=(0.5, 2.0)), #recommed to set img_scale because img and B_img may have different scals
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'B_img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Vaihingen_IRRG/img_dir/train',
        ann_dir='Vaihingen_IRRG/ann_dir/train',
        split='Vaihingen_IRRG/train.txt',
        ignore_label=255,
        B_img_dir = 'Potsdam_IRRG/img_dir/train',
        B_split = 'Potsdam_IRRG/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Potsdam_IRRG/img_dir/val',
        ann_dir='Potsdam_IRRG/ann_dir/val',
        split='Potsdam_IRRG/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Potsdam_IRRG/img_dir/val',
        ann_dir='Potsdam_IRRG/ann_dir/val',
        split='Potsdam_IRRG/val.txt',
        pipeline=test_pipeline))