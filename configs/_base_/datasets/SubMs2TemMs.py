# dataset settings
dataset_type = 'CASIDDataset_forAdap'
data_root = '/data/fywdata/CASID/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile_forAdap'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1000, 1000), B_img_scale=crop_size, ratio_range=(0.5, 2.0)), #recommed to set img_scale because img and B_img may have different scals
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
        img_scale=(1000, 1000),
        #img_ratios=[0.75, 1.0, 1.25, 1.5],
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
        img_dir='SubMs/img_dir',
        ann_dir='SubMs/ann_dir',
        split='SubMs/SubMs.txt',
        B_img_dir='TemMs/img_dir',
        B_split='TemMs/TemMs.txt',
        pipeline=train_pipeline,
        ignore_index=0,  # 这里设置忽略的索引
    ),
    # target domain for validation
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TemMs/img_dir',
        ann_dir='TemMs/ann_dir',
        split='TemMs/TemMs.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TemMs/img_dir',
        ann_dir='TemMs/ann_dir',
        split='TemMs/TemMs.txt',
        pipeline=test_pipeline))
