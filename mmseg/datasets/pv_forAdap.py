# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PVDataset_forAdap(CustomDataset):
    """Potsdam and Vaihingen dataset for Domain adaptation (Open-set)."""

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
        [255, 0, 0]
    ]

    CLASSES = FULL_CLASSES
    PALETTE = FULL_PALETTE

    RAW_CLASS_IDS = {
        'impervious_surface': 1,
        'building': 2,
        'low_vegetation': 3,
        'tree': 4,
        'car': 5,
        'clutter': 6
    }

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

        for cls_name in self.source_included_classes:
            if cls_name not in self.FULL_CLASSES:
                raise ValueError(
                    f'Class "{cls_name}" not in FULL_CLASSES: {self.FULL_CLASSES}')

        # source raw label -> compact label
        self.source_label_map = {}
        compact_idx = 0
        for cls_name in self.FULL_CLASSES:
            raw_id = self.RAW_CLASS_IDS[cls_name]
            if cls_name in self.source_included_classes:
                self.source_label_map[raw_id] = compact_idx
                compact_idx += 1
            else:
                self.source_label_map[raw_id] = self.ignore_label

        self.source_num_classes = len(self.source_included_classes)

        super(PVDataset_forAdap, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
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

        # 日志打印
        print('=' * 80)
        print(f'【源域训练类别】: {self.source_included_classes}')
        print(f'【源域有效类别数】: {self.source_num_classes}')
        print(f'【源域被忽略的类别】: {[c for c in self.FULL_CLASSES if c not in self.source_included_classes]}')
        print(f'【源域标签映射表】: {self.source_label_map}')
        print(f'【目标域训练类别】: {self.FULL_CLASSES}')
        print(f'【目标域类别数】: {len(self.FULL_CLASSES)}')
        print('=' * 80)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        assert self.B_img_infos is not None and len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]

        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            B_img_info=B_img_info
        )
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        关键修复：
        默认CustomDataset.prepare_test_img不会放ann_info，
        但你的test_pipeline里用了LoadAnnotations，所以这里必须补上。
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        results = dict(
            img_info=img_info,
            ann_info=ann_info
        )
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir

        if self.custom_classes:
            results['label_map'] = self.label_map

    def get_gt_seg_map_by_idx(self, idx):
        seg_map = super().get_gt_seg_map_by_idx(idx)
        seg_map_np = np.array(seg_map, dtype=np.uint8)

        if idx == 0:
            print(f'[PVDataset_forAdap] test_mode={self.test_mode}, raw gt unique = {np.unique(seg_map_np)}')

        return seg_map_np