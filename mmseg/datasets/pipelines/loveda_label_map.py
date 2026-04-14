import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class MapLoveDALabelTrain(object):
    """Map LoveDA full labels (0~6) to compressed source-train labels."""

    FULL_CLASSES = (
        'background', 'building', 'road', 'water',
        'barren', 'forest', 'agricultural'
    )

    CLASS_TO_LABEL = {
        'background': 0,
        'building': 1,
        'road': 2,
        'water': 3,
        'barren': 4,
        'forest': 5,
        'agricultural': 6
    }

    def __init__(self, source_included_classes=None, ignore_label=255):
        self.source_included_classes = source_included_classes
        self.ignore_label = ignore_label

    def _build_label_map(self, source_included_classes):
        source_included_classes = list(source_included_classes)

        label_map = {}
        compact_idx = 0
        for cls_name in self.FULL_CLASSES:
            old_label = self.CLASS_TO_LABEL[cls_name]
            if cls_name in source_included_classes:
                label_map[old_label] = compact_idx
                compact_idx += 1
            else:
                label_map[old_label] = self.ignore_label
        return label_map

    def __call__(self, results):
        gt = np.array(results['gt_semantic_seg'], dtype=np.uint8)

        if self.source_included_classes is not None:
            source_included_classes = self.source_included_classes
        else:
            if 'source_included_classes' not in results:
                raise KeyError(
                    'source_included_classes is not found in results.')
            source_included_classes = results['source_included_classes']

        label_map = self._build_label_map(source_included_classes)

        new_gt = np.ones_like(gt, dtype=np.uint8) * self.ignore_label
        for old_label, new_label in label_map.items():
            new_gt[gt == old_label] = new_label

        results['gt_semantic_seg'] = new_gt
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(source_included_classes={self.source_included_classes}, '
                f'ignore_label={self.ignore_label})')


@PIPELINES.register_module()
class MapLoveDALabelEval(object):
    """Map LoveDA eval labels. Default target eval keeps 0~6 unchanged."""

    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label

    def __call__(self, results):
        gt = np.array(results['gt_semantic_seg'], dtype=np.uint8)

        new_gt = np.ones_like(gt, dtype=np.uint8) * self.ignore_label
        for k in range(7):
            new_gt[gt == k] = k

        results['gt_semantic_seg'] = new_gt
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(ignore_label={self.ignore_label})'