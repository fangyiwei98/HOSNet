import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class MapPVLabelTrain(object):
    """将Potsdam/Vaihingen原始标签映射为source训练用的压缩标签。

    原始标签:
        0: ignore/background
        1: impervious_surface
        2: building
        3: low_vegetation
        4: tree
        5: car
        6: clutter

    输出标签:
        source保留类 -> 0..K-1
        其余 -> ignore_label
    """

    FULL_CLASSES = (
        'impervious_surface', 'building', 'low_vegetation',
        'tree', 'car', 'clutter'
    )

    RAW_CLASS_IDS = {
        'impervious_surface': 1,
        'building': 2,
        'low_vegetation': 3,
        'tree': 4,
        'car': 5,
        'clutter': 6
    }

    def __init__(self, source_included_classes, ignore_label=255):
        self.source_included_classes = list(source_included_classes)
        self.ignore_label = ignore_label

        self.source_label_map = {}
        compact_idx = 0
        for cls_name in self.FULL_CLASSES:
            raw_id = self.RAW_CLASS_IDS[cls_name]
            if cls_name in self.source_included_classes:
                self.source_label_map[raw_id] = compact_idx
                compact_idx += 1
            else:
                self.source_label_map[raw_id] = self.ignore_label

    def __call__(self, results):
        gt = results['gt_semantic_seg']
        gt = np.array(gt, dtype=np.uint8)

        new_gt = np.ones_like(gt, dtype=np.uint8) * self.ignore_label

        # raw 0 保持 ignore
        for raw_label, compact_label in self.source_label_map.items():
            new_gt[gt == raw_label] = compact_label

        results['gt_semantic_seg'] = new_gt
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(source_included_classes={self.source_included_classes}, ignore_label={self.ignore_label})'


@PIPELINES.register_module()
class MapPVLabelEval(object):
    """将Potsdam/Vaihingen原始标签映射为target评估用的0~5标签。

    原始标签:
        0: ignore/background
        1..6: 六类语义

    输出标签:
        0 -> 255
        1 -> 0
        2 -> 1
        3 -> 2
        4 -> 3
        5 -> 4
        6 -> 5
    """

    FULL_CLASSES = (
        'impervious_surface', 'building', 'low_vegetation',
        'tree', 'car', 'clutter'
    )

    RAW_CLASS_IDS = {
        'impervious_surface': 1,
        'building': 2,
        'low_vegetation': 3,
        'tree': 4,
        'car': 5,
        'clutter': 6
    }

    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label

    def __call__(self, results):
        gt = results['gt_semantic_seg']
        gt = np.array(gt, dtype=np.uint8)

        new_gt = np.ones_like(gt, dtype=np.uint8) * self.ignore_label

        for eval_idx, cls_name in enumerate(self.FULL_CLASSES):
            raw_id = self.RAW_CLASS_IDS[cls_name]
            new_gt[gt == raw_id] = eval_idx

        results['gt_semantic_seg'] = new_gt
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(ignore_label={self.ignore_label})'