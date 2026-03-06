# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PVDataset_forAdap(CustomDataset):
    """Potsdam and Vaihingen dataset for Domain adaptation (Open-Set).

    Args:
        split (str): Split txt file for domain A (source) of Potsdam and Vaihingen dataset.
        reduce_zero_label (bool): Whether to reduce zero label, DEFAULT False (critical for Open-Set).
    """

    # 原始6类（保持与数据集标注一致）
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    def __init__(self,
                 split,
                 B_split=None,
                 B_img_dir=None,
                 B_img_suffix='.png',
                 B_ann_dir=None,
                 B_seg_map_suffix='.png',
                 **kwargs):
        # 关键修正：移除强制reduce_zero_label=True，由配置文件传入（Open-Set必须设为False）
        super(PVDataset_forAdap, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=split,
            **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None

        self.B_img_dir = B_img_dir
        self.B_img_suffix = B_img_suffix
        self.B_ann_dir = B_ann_dir
        self.B_seg_map_suffix = B_seg_map_suffix
        self.B_split = B_split

        # 目标域（B域）路径拼接与标注加载（保留原始6类标签）
        if self.B_img_dir is not None:
            if not osp.isabs(self.B_img_dir):
                self.B_img_dir = osp.join(self.data_root, self.B_img_dir)
            if not (self.B_ann_dir is None or osp.isabs(self.B_ann_dir)):
                self.B_ann_dir = osp.join(self.data_root, self.B_ann_dir)
            if not (self.B_split is None or osp.isabs(self.B_split)):
                self.B_split = osp.join(self.data_root, self.B_split)
            # 加载目标域标注（保留原始6类，不修改标签）
            self.B_img_infos = self.load_annotations(
                img_dir=self.B_img_dir,
                img_suffix=self.B_img_suffix,
                ann_dir=self.B_ann_dir,
                seg_map_suffix=self.B_seg_map_suffix,
                split=self.B_split)
        else:
            self.B_img_infos = None

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline (适配Open-Set)."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        assert len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]
        results = dict(img_info=img_info, ann_info=ann_info, B_img_info=B_img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline (补充目标域前缀)."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir
            results['B_seg_prefix'] = self.B_ann_dir  # 补充目标域标注前缀
        if self.custom_classes:
            results['label_map'] = self.label_map