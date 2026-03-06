# dataset settings
dataset_type = 'PVDataset_forAdap'
data_root = '/data/fywdata/ISPRS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)

# Open-Set核心配置：仅保留前5个前景类，忽略第6类clutter
# 类别名称与数据集CLASSES前5个完全一致（原始ID 0-4）
classes = [
    'impervious_surface',
    'building',
    'low_vegetation',
    'tree',
    'car'
]
# 调色板对应前5个前景类（原始ID 0-4）
palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]]

train_pipeline = [
    dict(type='LoadImageFromFile_forAdap'),
    # 关键：显式设置reduce_zero_label=False（保留原始0类，Open-Set必须）
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), B_img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
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
            # 测试阶段同样保留原始标签，忽略clutter
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Potsdam_IRRG/img_dir/train',
        ann_dir='Potsdam_IRRG/ann_dir/train',
        split='Potsdam_IRRG/train.txt',
        B_img_dir = 'Vaihingen_IRRG/img_dir/train',
        B_ann_dir = 'Vaihingen_IRRG/ann_dir/train',  # 补充目标域标注路径
        B_split = 'Vaihingen_IRRG/train.txt',
        pipeline=train_pipeline,
        # Open-Set核心配置
        classes=classes,          # 源域仅训练5个前景类
        palette=palette,          # 前景类调色板
        reduce_zero_label=False,  # 保留原始0类（关键！）
        ignore_index=5,           # 忽略原始ID 5（clutter，Open-Set未知类）
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Vaihingen_IRRG/img_dir/val',
        ann_dir='Vaihingen_IRRG/ann_dir/val',
        split='Vaihingen_IRRG/val.txt',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        ignore_index=5,  # 验证集忽略clutter（原始ID 5）
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Vaihingen_IRRG/img_dir/val',
        ann_dir='Vaihingen_IRRG/ann_dir/val',
        split='Vaihingen_IRRG/val.txt',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        ignore_index=5,  # 测试集忽略clutter（原始ID 5）
    ))