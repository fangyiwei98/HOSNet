# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PVDataset_forAdap(CustomDataset):
    """Potsdam and Vaihingen dataset for Domain adaptation."""

    FULL_CLASSES = (
        'impervious_surface', 'building', 'low_vegetation',
        'tree', 'car', 'clutter'
    )

    FULL_PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ]

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
        self.source_included_classes = (
            source_included_classes
            if source_included_classes is not None
            else self.FULL_CLASSES
        )

        for c in self.source_included_classes:
            assert c in self.FULL_CLASSES, f'Unknown class in source_included_classes: {c}'

        # source标签映射：FULL_CLASSES索引 -> source压缩索引
        # 注意：这里只有 train 阶段 dataset 内部 get_gt_seg_map_by_idx 会用
        self.source_label_map = {}
        valid_idx = 0
        for full_idx, cls_name in enumerate(self.FULL_CLASSES):
            if cls_name in self.source_included_classes:
                self.source_label_map[full_idx] = valid_idx
                valid_idx += 1
            else:
                self.source_label_map[full_idx] = self.ignore_label

        self.source_num_classes = len(self.source_included_classes)

        super(PVDataset_forAdap, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,   # 这里改成 False，统一交给 pipeline 做映射
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
                self.B_img_dir, self.B_img_suffix,
                self.B_ann_dir, self.B_seg_map_suffix, self.B_split)
        else:
            self.B_img_infos = None

    def get_gt_seg_map_by_idx(self, idx):
        seg_map = super().get_gt_seg_map_by_idx(idx)
        seg_map = np.array(seg_map, dtype=np.uint8)

        if not self.test_mode:
            return seg_map

        new_seg_map = np.ones_like(seg_map, dtype=np.uint8) * self.ignore_label
        for raw_id in range(1, 7):
            new_seg_map[seg_map == raw_id] = raw_id - 1

        return new_seg_map

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        assert self.B_img_infos is not None and len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]

        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            B_img_info=B_img_info)

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """修复 val/test 使用 LoadAnnotations 时缺 ann_info 的问题。"""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir
            results['source_label_map'] = self.source_label_map
            results['ignore_label'] = self.ignore_label

        if self.custom_classes:
            results['label_map'] = self.label_map