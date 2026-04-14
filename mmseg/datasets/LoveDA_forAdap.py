# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

import mmcv
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LoveDADataset_forAdap(CustomDataset):
    FULL_CLASSES = ('background', 'building', 'road', 'water', 'barren',
                    'forest', 'agricultural')
    FULL_PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                    [159, 129, 183], [0, 255, 0], [255, 195, 128]]

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
            list(source_included_classes)
            if source_included_classes is not None else list(self.FULL_CLASSES)
        )

        # ---------------- 新增：和 ISPRS 一样的 source_label_map 标签映射 ----------------
        # 把 FULL_CLASSES 的索引映射到 source 子集的压缩索引，不在子集里的设为 ignore_label
        self.source_label_map = {}
        valid_idx = 0
        for full_idx, cls_name in enumerate(self.FULL_CLASSES):
            if cls_name in self.source_included_classes:
                self.source_label_map[full_idx] = valid_idx
                valid_idx += 1
            else:
                self.source_label_map[full_idx] = self.ignore_label
        # -----------------------------------------------------------------------------

        self.source_num_classes = len(self.source_included_classes)

        super(LoveDADataset_forAdap, self).__init__(
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
                self.B_img_dir, self.B_img_suffix,
                self.B_ann_dir, self.B_seg_map_suffix, self.B_split)
        else:
            self.B_img_infos = None

    def get_gt_seg_map_by_idx(self, idx):
        """LoveDA在reduce_zero_label=True后标签已为0~6，评估阶段可直接使用。"""
        seg_map = super().get_gt_seg_map_by_idx(idx)
        return np.array(seg_map, dtype=np.uint8)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        assert len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]

        results = dict(img_info=img_info, ann_info=ann_info, B_img_info=B_img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir
            results['source_included_classes'] = self.source_included_classes
            results['ignore_label'] = self.ignore_label

        if self.custom_classes:
            results['label_map'] = self.label_map