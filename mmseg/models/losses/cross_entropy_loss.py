# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy` with safety enhancement"""
    # ========== 核心修复1：强制张量内存连续 + 设备统一 ==========
    pred = pred.contiguous()  # 确保内存连续，避免CUDA访问错误
    label = label.contiguous().to(pred.device)  # 统一设备
    if weight is not None:
        weight = weight.contiguous().to(pred.device)

    # ========== 核心修复2：维度校验与修正 ==========
    # pred必须是4D [B,C,H,W]（语义分割）或2D [B,C]（分类）
    # label统一为3D [B,H,W]（分割）或1D [B]（分类）
    if pred.dim() == 4:  # 语义分割场景
        # label移除通道维：[B,1,H,W] → [B,H,W]
        label = label.squeeze(1) if label.dim() == 4 else label
        # 确保空间维度匹配
        assert pred.shape[2:] == label.shape[1:], \
            f"Spatial shape mismatch! pred: {pred.shape[2:]}, label: {label.shape[1:]}"
        # ========== 核心修复3：过滤非法标签值 ==========
        num_classes = pred.shape[1]
        # 仅保留合法标签（0~num_classes-1）和ignore_index
        valid_mask = (label >= 0) & (label < num_classes) | (label == ignore_index)
        label = torch.where(valid_mask, label, torch.tensor(ignore_index, device=label.device))

    # ========== 原有逻辑保留 + 安全增强 ==========
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss with safety enhancement."""
    # ========== 安全增强：设备统一 + 内存连续 ==========
    pred = pred.contiguous()
    label = label.contiguous().to(pred.device)
    if weight is not None:
        weight = weight.contiguous().to(pred.device)

    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks with safety enhancement."""
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None

    # ========== 安全增强：设备统一 + 内存连续 ==========
    pred = pred.contiguous()
    target = target.contiguous().to(pred.device)
    label = label.contiguous().to(pred.device)

    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with safety enhancement for CUDA memory access.

    Enhanced features:
    1. Tensor device unification (avoid CPU/GPU mix)
    2. Illegal label value filtering
    3. Memory contiguous guarantee
    4. Dimension consistency check
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 ignore_index=255):  # 新增：默认忽略标签255
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.ignore_index = ignore_index  # 保存忽略标签值

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function with safety enhancement."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # ========== 核心修复4：传递ignore_index到损失函数 ==========
        if 'ignore_index' not in kwargs:
            kwargs['ignore_index'] = self.ignore_index

        # ========== 原有逻辑保留 ==========
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name