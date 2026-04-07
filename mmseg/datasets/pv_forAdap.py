# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PVDataset_forAdap(CustomDataset):
    """Potsdam and Vaihingen dataset for Domain adaptation (Open-set).

    Args:
        split (str): Split txt file for domain A.
        source_included_classes (list): Source domain classes to keep (default: all 6 classes)
        ignore_label (int): Label value for ignored classes (default: 255)
    """

    # 完整类别定义（目标域始终使用）
    FULL_CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
                    'car', 'clutter')
    FULL_PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                    [255, 255, 0], [255, 0, 0]]

    # 对外暴露的类别（兼容原有逻辑）
    CLASSES = FULL_CLASSES
    PALETTE = FULL_PALETTE

    def __init__(self,
                 split,
                 B_split=None,
                 B_img_dir=None,
                 B_img_suffix='.png',
                 B_ann_dir=None,
                 B_seg_map_suffix='.png',
                 source_included_classes=None,  # 新增：源域参与训练的类别
                 ignore_label=255,  # 新增：忽略标签值
                 **kwargs):
        # 初始化源域类别筛选
        self.ignore_label = ignore_label
        self.source_included_classes = source_included_classes if source_included_classes is not None else self.FULL_CLASSES


        # 构建源域标签映射表：保留类→连续索引，排除类→ignore_label
        self.source_label_map = {}
        valid_idx = 0
        # 修改映射：不压缩索引
        for cls_idx, cls_name in enumerate(self.FULL_CLASSES):
            if cls_name in source_included_classes:
                self.source_label_map[cls_idx] = cls_idx
            else:
                self.source_label_map[cls_idx] = self.ignore_label

        # 源域有效类别数
        self.source_num_classes = len(self.source_included_classes)

        super(PVDataset_forAdap, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=True, split=split, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None

        self.B_img_dir = B_img_dir
        self.B_img_suffix = B_img_suffix
        self.B_ann_dir = B_ann_dir
        self.B_seg_map_suffix = B_seg_map_suffix
        self.B_split = B_split

        # join paths if data_root is specified
        if self.B_img_dir is not None:
            if not osp.isabs(self.B_img_dir):
                self.B_img_dir = osp.join(self.data_root, self.B_img_dir)
            if not (self.B_ann_dir is None or osp.isabs(self.B_ann_dir)):
                self.B_ann_dir = osp.join(self.data_root, self.B_ann_dir)
            if not (self.B_split is None or osp.isabs(self.B_split)):
                self.B_split = osp.join(self.data_root, self.B_split)
            # 加载目标域标注（始终保留全部6类）
            self.B_img_infos = self.load_annotations(self.B_img_dir, self.B_img_suffix,
                                                     self.B_ann_dir,
                                                     self.B_seg_map_suffix, self.B_split)
        else:
            self.B_img_infos = None

    def get_gt_seg_map_by_idx(self, idx):
        """重载获取源域标签的方法，过滤不参与训练的类别"""
        # 先获取原始标签
        seg_map = super().get_gt_seg_map_by_idx(idx)
        if self.test_mode:  # 测试/验证阶段保留原始标签（兼容评估）
            return seg_map

        # 训练阶段：应用源域标签映射
        seg_map_np = np.array(seg_map, dtype=np.uint8)
        new_seg_map = np.ones_like(seg_map_np) * self.ignore_label

        # 替换保留类别的标签，排除类设为ignore_label
        for old_label, new_label in self.source_label_map.items():
            new_seg_map[seg_map_np == old_label] = new_label

        return new_seg_map

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)  # 会自动调用get_gt_seg_map_by_idx过滤标签
        assert len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]
        results = dict(img_info=img_info, ann_info=ann_info, B_img_info=B_img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir
            # 传递源域标签映射（供pipeline使用）
            results['source_label_map'] = self.source_label_map
            results['ignore_label'] = self.ignore_label
        if self.custom_classes:
            results['label_map'] = self.label_map