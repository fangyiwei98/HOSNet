# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch


def accuracy(pred, target, topk=1, thresh=None, ignore_index=255):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    # ========== 关键修改1：兜底处理非法标签 ==========
    num_classes = pred.shape[1] if pred.ndim == 4 else pred.shape[-1]
    # 1. 复制target避免修改原张量
    target = target.clone()
    # 2. 将超出[0, num_classes-1]的标签设为ignore_index（255）
    invalid_mask = (target < 0) | (target >= num_classes)
    target[invalid_mask] = ignore_index

    # ========== 关键修改2：统一设备 ==========
    pred = pred.to(target.device)

    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'

    # ========== 关键修改3：展平张量，避免高维索引 ==========
    # 保存原始形状，用于展平/恢复
    original_target_shape = target.shape
    # 展平pred: [N, C, H, W] -> [N*H*W, C]
    pred = pred.permute(0, *range(2, pred.ndim), 1).reshape(-1, pred.size(1))
    # 展平target: [N, H, W] -> [N*H*W]
    target = target.reshape(-1)

    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N*H*W)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))

    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()

    # ========== 关键修改4：安全处理ignore_index ==========
    if ignore_index is not None:
        # 创建有效掩码，避免高维索引
        valid_mask = (target != ignore_index)
        # 处理空掩码情况（全是ignore_index）
        if not valid_mask.any():
            accu = [pred.new_tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu
        # 只保留有效区域的结果
        correct = correct[:, valid_mask]
    else:
        valid_mask = torch.ones_like(target, dtype=torch.bool)

    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        total_num = valid_mask.sum() + eps  # 使用掩码求和，更安全
        res.append(correct_k.mul_(100.0 / total_num))

    return res[0] if return_single else res


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1,), thresh=None, ignore_index=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh,
                        self.ignore_index)