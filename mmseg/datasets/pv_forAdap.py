# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PVDataset_forAdap(CustomDataset):
    """Potsdam and Vaihingen dataset for Domain adaptation (Open-set)."""

    FULL_CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
                    'car', 'clutter')
    FULL_PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                    [255, 255, 0], [255, 0, 0]]

    CLASSES = FULL_CLASSES
    PALETTE = FULL_PALETTE

    def __init__(self,
                 split,
                 B_split=None,
                 B_img_dir=None,
                 B_img_suffix='.png',
                 B_ann_dir=None,
                 B_seg_map_suffix='.png',
                 source_included_classes=None,
                 ignore_label=255,
                 **kwargs):

        self.ignore_label = ignore_label

        if source_included_classes is None:
            self.source_included_classes = list(self.FULL_CLASSES)
        else:
            self.source_included_classes = list(source_included_classes)

        # 检查类别合法性
        for cls_name in self.source_included_classes:
            if cls_name not in self.FULL_CLASSES:
                raise ValueError(f'Class "{cls_name}" not in FULL_CLASSES: {self.FULL_CLASSES}')

        # =========================
        # 核心修复：源域标签映射必须压缩为连续索引
        # 例如:
        # FULL:   [0,1,2,3,4,5]
        # keep:   [0,1,3,4,5]
        # map ->  [0,1,255,2,3,4]
        # =========================
        self.source_label_map = {}
        valid_idx = 0
        for cls_idx, cls_name in enumerate(self.FULL_CLASSES):
            if cls_name in self.source_included_classes:
                self.source_label_map[cls_idx] = valid_idx
                valid_idx += 1
            else:
                self.source_label_map[cls_idx] = self.ignore_label

        self.source_num_classes = len(self.source_included_classes)

        # 记录压缩后的类别顺序与全类别索引的对应关系，供调试/可视化使用
        self.source_class_to_compact_idx = {
            cls_name: i for i, cls_name in enumerate(self.source_included_classes)
        }
        self.compact_to_full_idx = [
            self.FULL_CLASSES.index(cls_name) for cls_name in self.source_included_classes
        ]

        super(PVDataset_forAdap, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            split=split,
            **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None

        self.B_img_dir = B_img_dir
        self.B_img_suffix = B_img_suffix
        self.B_ann_dir = B_ann_dir
        self.B_seg_map_suffix = B_seg_map_suffix
        self.B_split = B_split

        if self.B_img_dir is not None:
            if not osp.isabs(self.B_img_dir):
                self.B_img_dir = osp.join(self.data_root, self.B_img_dir)
            if not (self.B_ann_dir is None or osp.isabs(self.B_ann_dir)):
                self.B_ann_dir = osp.join(self.data_root, self.B_ann_dir)
            if not (self.B_split is None or osp.isabs(self.B_split)):
                self.B_split = osp.join(self.data_root, self.B_split)

            self.B_img_infos = self.load_annotations(
                self.B_img_dir,
                self.B_img_suffix,
                self.B_ann_dir,
                self.B_seg_map_suffix,
                self.B_split)
        else:
            self.B_img_infos = None

    def get_gt_seg_map_by_idx(self, idx):
        """重载获取源域标签的方法，训练阶段将源域标签压缩到连续索引"""
        seg_map = super().get_gt_seg_map_by_idx(idx)

        if self.test_mode:
            # 验证/测试阶段保留原始标签，保证评估仍基于完整目标域类别
            return seg_map

        seg_map_np = np.array(seg_map, dtype=np.uint8)
        new_seg_map = np.ones_like(seg_map_np, dtype=np.uint8) * self.ignore_label

        for old_label, new_label in self.source_label_map.items():
            new_seg_map[seg_map_np == old_label] = new_label

        return new_seg_map

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
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
            results['source_label_map'] = self.source_label_map
            results['ignore_label'] = self.ignore_label
            results['source_included_classes'] = self.source_included_classes
            results['compact_to_full_idx'] = self.compact_to_full_idx
        if self.custom_classes:
            results['label_map'] = self.label_map