# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class OSNet(BaseSegmentor):
    """
    Decoupled OSNet for Open-set Domain Adaptive Segmentation

    Fixes included:
    1) unknown seg loss for Ku=1 uses BCE instead of CE
    2) inference uses known/unknown gating rather than naive concatenation
    3) multi-unknown balanced pseudo labeling to avoid unknown-class collapse
       and prevent some unknown-class IoU from staying at 0
    """

    def __init__(self,
                 backbone_s,
                 decode_head_s,
                 decode_head_t,
                 source_classes=None,
                 target_classes=None,
                 contrast_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 closed_set=False):
        super(OSNet, self).__init__(init_cfg)

        assert source_classes is not None and target_classes is not None

        self.closed_set = closed_set
        self.source_classes = source_classes
        self.target_classes = target_classes

        source_to_target_indices = [target_classes.index(c) for c in source_classes]

        if closed_set:
            unknown_indices = []
            assert len(source_classes) == len(target_classes), \
                'In closed-set mode, source_classes and target_classes should match.'
        else:
            unknown_indices = [i for i, c in enumerate(target_classes) if c not in source_classes]

        self.unknown_indices = unknown_indices
        self.num_known_classes = len(source_classes)
        self.num_unknown_classes = len(unknown_indices)
        self.num_classes = len(source_classes) if closed_set else len(target_classes)

        self.register_buffer(
            'source_to_target_idx',
            torch.tensor(source_to_target_indices, dtype=torch.long)
        )
        self.register_buffer(
            'unknown_idx',
            torch.tensor(unknown_indices, dtype=torch.long)
        )

        if pretrained is not None:
            assert backbone_s.get('pretrained') is None
            backbone_s.pretrained = pretrained

        self.backbone_s = builder.build_backbone(backbone_s)
        self.decode_head_s = self._init_decode_head(decode_head_s)
        self.decode_head_t = self._init_decode_head(decode_head_t)

        self.align_corners = self.decode_head_t.align_corners
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._parse_train_cfg()
        self._init_contrast_module(contrast_cfg)

    def _init_contrast_module(self, contrast_cfg):
        if contrast_cfg is None:
            contrast_cfg = dict()

        self.proj_dim = contrast_cfg.get('proj_dim', 256)
        self.proto_momentum = contrast_cfg.get('momentum', 0.99)

        self.known_conf_thresh = contrast_cfg.get('known_conf_thresh', 0.7)
        self.discrepancy_thresh = contrast_cfg.get('discrepancy_thresh', 0.2)

        self.tau_unified = contrast_cfg.get('tau_unified', 0.07)

        self.loss_contrast_weight = contrast_cfg.get('loss_contrast_weight', 1.0)
        self.loss_unknown_seg_weight = contrast_cfg.get('loss_unknown_seg_weight', 0.5)

        self.max_samples = contrast_cfg.get('max_samples', 4096)
        self.min_pixels_per_anchor = contrast_cfg.get('min_pixels_per_anchor', 10)

        self.unknown_pseudo_thresh = contrast_cfg.get('unknown_pseudo_thresh', 0.0)

        # inference gating
        self.infer_known_conf_thresh = contrast_cfg.get(
            'infer_known_conf_thresh', self.known_conf_thresh)
        self.infer_unknown_logit_bias = contrast_cfg.get(
            'infer_unknown_logit_bias', 0.0)

        # balanced pseudo labeling for multiple unknown classes
        self.balance_unknown_pseudo = contrast_cfg.get('balance_unknown_pseudo', True)
        self.unknown_balance_ratio = contrast_cfg.get('unknown_balance_ratio', 0.5)
        self.min_unknown_pixels_per_class = contrast_cfg.get('min_unknown_pixels_per_class', 16)
        self.unknown_balance_warmup_iters = contrast_cfg.get('unknown_balance_warmup_iters', 4000)

        in_channels = self.decode_head_s.in_channels
        if isinstance(in_channels, (list, tuple)):
            last_channels = in_channels[-1]
        else:
            last_channels = in_channels

        self.feat_proj = nn.Sequential(
            nn.Conv2d(last_channels, self.proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)
        )

        self.register_buffer(
            'known_anchors',
            F.normalize(torch.randn(len(self.source_classes), self.proj_dim), dim=1)
        )

        if len(self.unknown_indices) > 0:
            self.register_buffer(
                'unknown_anchors',
                F.normalize(torch.randn(len(self.unknown_indices), self.proj_dim), dim=1)
            )
        else:
            self.register_buffer(
                'unknown_anchors',
                torch.zeros(0, self.proj_dim)
            )

    def train_step(self, data_batch, optimizer, **kwargs):
        if not hasattr(self, 'iteration'):
            self.iteration = 0

        optimizer['backbone_s'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_t'].zero_grad()
        if 'feat_proj' in optimizer:
            optimizer['feat_proj'].zero_grad()

        log_vars = dict()

        img_s = data_batch['img']
        img_t = data_batch['B_img']
        gt_s = data_batch['gt_semantic_seg']

        F_s = self.forward_backbone(self.backbone_s, img_s)
        F_t = self.forward_backbone(self.backbone_s, img_t)

        P_s_src = self.forward_decode_head(self.decode_head_s, F_s)
        P_s_tgt_known = self.forward_decode_head(self.decode_head_t, F_s)
        P_t_src = self.forward_decode_head(self.decode_head_s, F_t)

        feat_t_head, P_t_tgt_known, P_t_tgt_unknown = self.forward_decode_head_decoupled(
            self.decode_head_t, F_t)

        loss_seg_s, log_seg_s = self._get_segmentor_loss(
            self.decode_head_s, P_s_src, gt_s)
        log_vars.update(self._rename_log(log_seg_s, '_seg_s'))

        gt_s_target_known = self._remap_source_gt_to_target_known(gt_s)
        loss_seg_t, log_seg_t = self._get_known_branch_loss(
            P_s_tgt_known, gt_s_target_known)
        log_vars.update(self._rename_log(log_seg_t, '_seg_t'))

        feat_s = self._project_feature(F_s[-1])
        feat_t_contrast = self._project_feature(F_t[-1].detach())

        self._update_known_anchors(feat_s, gt_s)

        P_t_src_full = self._scatter_source_logits_to_target_known(P_t_src)
        known_mask, known_label, unknown_region_mask = \
            self._mine_target_masks(P_t_src_full, P_t_tgt_known)

        if self.closed_set:
            unknown_mask = torch.zeros_like(known_mask, dtype=torch.bool)
            unknown_label_local = torch.zeros_like(known_label, dtype=torch.long)
            unknown_conf = torch.zeros_like(known_mask, dtype=feat_t_contrast.dtype)
        else:
            unknown_mask, unknown_label_local, unknown_conf = \
                self._generate_unknown_pseudo_labels_from_head(
                    pred_tgt_unknown=P_t_tgt_unknown,
                    unknown_region_mask=unknown_region_mask
                )

            self._update_unknown_anchors_from_pseudo(
                feat_t_contrast, unknown_mask, unknown_label_local)

        loss_contrast = self._unified_anchor_contrast(
            feat_t_contrast,
            known_mask,
            known_label,
            unknown_mask,
            unknown_label_local
        )

        if self.closed_set:
            loss_unknown_seg = P_t_tgt_known.sum() * 0.0
            total_loss = (
                    loss_seg_s +
                    loss_seg_t +
                    self.loss_contrast_weight * loss_contrast
            )
        else:
            loss_unknown_seg = self._unknown_pseudo_seg_loss(
                pred_tgt_unknown=P_t_tgt_unknown,
                unknown_mask=unknown_mask,
                unknown_label_local=unknown_label_local
            )
            total_loss = (
                    loss_seg_s +
                    loss_seg_t +
                    self.loss_contrast_weight * loss_contrast +
                    self.loss_unknown_seg_weight * loss_unknown_seg
            )

        total_loss.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        if 'feat_proj' in optimizer:
            optimizer['feat_proj'].step()

        log_vars['loss_contrast'] = loss_contrast.item()
        log_vars['loss_unknown_seg'] = loss_unknown_seg.item()
        log_vars['num_known_pixels'] = known_mask.sum().item()

        if self.closed_set:
            log_vars['num_unknown_region_pixels'] = 0
            log_vars['num_unknown_pixels'] = 0
            log_vars['unknown_pseudo_conf'] = 0.0
        else:
            log_vars['num_unknown_region_pixels'] = unknown_region_mask.sum().item()
            log_vars['num_unknown_pixels'] = unknown_mask.sum().item()

            if unknown_mask.sum() > 0:
                log_vars['unknown_pseudo_conf'] = unknown_conf[unknown_mask].mean().item()
            else:
                log_vars['unknown_pseudo_conf'] = 0.0

            if self.num_unknown_classes > 1 and unknown_mask.sum() > 0:
                for u in range(self.num_unknown_classes):
                    log_vars[f'num_unknown_cls_{u}_pixels'] = (
                        (unknown_mask & (unknown_label_local == u)).sum().item()
                    )

        self.iteration += 1

        outputs = dict(
            loss=total_loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas'])
        )
        return outputs

    def _rename_log(self, log_vars, suffix=''):
        out = {}
        for k, v in log_vars.items():
            out[k + suffix] = v
        return out

    def _project_feature(self, feat):
        feat = self.feat_proj(feat)
        feat = F.normalize(feat, dim=1)
        return feat

    def _remap_source_gt_to_target_known(self, gt_s):
        gt_new = gt_s.clone()
        valid_mask = (gt_new != 255)
        mapped = torch.full_like(gt_new, 255)
        for s_idx in range(len(self.source_classes)):
            mapped[(gt_new == s_idx) & valid_mask] = s_idx
        return mapped

    def _scatter_source_logits_to_target_known(self, pred_s):
        return pred_s

    @torch.no_grad()
    def _update_known_anchors(self, feat_s, gt_s):
        feat_s = resize(
            feat_s,
            size=gt_s.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        gt = gt_s.squeeze(1)
        for cls_id in range(len(self.source_classes)):
            mask = (gt == cls_id)
            if mask.sum() < self.min_pixels_per_anchor:
                continue
            cls_feat = feat_s.permute(0, 2, 3, 1)[mask]
            cls_anchor = cls_feat.mean(dim=0)
            cls_anchor = F.normalize(cls_anchor, dim=0)
            self.known_anchors[cls_id] = F.normalize(
                self.proto_momentum * self.known_anchors[cls_id] +
                (1.0 - self.proto_momentum) * cls_anchor,
                dim=0
            )

    @torch.no_grad()
    def _balanced_unknown_assignment(self, prob_unknown, unknown_region_mask):
        """
        Balanced pseudo labeling for multiple unknown classes.

        Goal:
        avoid collapse where only a subset of unknown classes receives pseudo labels,
        causing some unknown-class IoU to stay 0 forever.
        """
        B, Ku, H, W = prob_unknown.shape
        device = prob_unknown.device

        # default outputs
        unknown_label_local = torch.zeros((B, H, W), dtype=torch.long, device=device)
        unknown_conf = torch.zeros((B, H, W), dtype=prob_unknown.dtype, device=device)
        unknown_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

        base_conf, base_label = torch.max(prob_unknown, dim=1)

        use_balance = self.balance_unknown_pseudo and (Ku > 1)
        if hasattr(self, 'iteration'):
            if self.iteration > self.unknown_balance_warmup_iters:
                # after warmup still keep balancing, but weaker
                effective_ratio = self.unknown_balance_ratio * 0.5
            else:
                effective_ratio = self.unknown_balance_ratio
        else:
            effective_ratio = self.unknown_balance_ratio

        for b in range(B):
            region_mask_b = unknown_region_mask[b]
            n_region = int(region_mask_b.sum().item())

            if n_region == 0:
                continue

            base_conf_b = base_conf[b]
            base_label_b = base_label[b]

            # start from regular argmax assignment
            assigned_label = base_label_b.clone()
            assigned_conf = base_conf_b.clone()

            if use_balance:
                # flatten unknown region
                region_idx = torch.nonzero(region_mask_b.view(-1), as_tuple=False).squeeze(1)
                if region_idx.numel() > 0:
                    prob_b = prob_unknown[b].permute(1, 2, 0).reshape(-1, Ku)[region_idx]  # [N, Ku]

                    quota = max(
                        self.min_unknown_pixels_per_class,
                        int(effective_ratio * n_region / Ku)
                    )
                    quota = min(quota, n_region)

                    # store best forced assignment per pixel
                    forced_score = torch.full((region_idx.numel(),), -1.0, device=device, dtype=prob_b.dtype)
                    forced_label = torch.full((region_idx.numel(),), -1, device=device, dtype=torch.long)

                    for u in range(Ku):
                        cls_scores = prob_b[:, u]
                        k = min(quota, cls_scores.numel())
                        if k <= 0:
                            continue

                        topk_scores, topk_idx = torch.topk(cls_scores, k=k, largest=True, sorted=False)

                        # if a pixel is selected by multiple classes, keep higher score class
                        prev = forced_score[topk_idx]
                        better = topk_scores > prev
                        if better.any():
                            better_idx = topk_idx[better]
                            forced_score[better_idx] = topk_scores[better]
                            forced_label[better_idx] = u

                    # write back forced assignments
                    assigned_label_flat = assigned_label.view(-1)
                    assigned_conf_flat = assigned_conf.view(-1)

                    valid_forced = forced_label >= 0
                    if valid_forced.any():
                        forced_global_idx = region_idx[valid_forced]
                        assigned_label_flat[forced_global_idx] = forced_label[valid_forced]
                        assigned_conf_flat[forced_global_idx] = forced_score[valid_forced]

                        assigned_label = assigned_label_flat.view(H, W)
                        assigned_conf = assigned_conf_flat.view(H, W)

            # final valid mask
            cur_mask = region_mask_b & (assigned_conf > self.unknown_pseudo_thresh)

            unknown_label_local[b] = assigned_label
            unknown_conf[b] = assigned_conf
            unknown_mask[b] = cur_mask

        return unknown_mask, unknown_label_local, unknown_conf

    @torch.no_grad()
    def _generate_unknown_pseudo_labels_from_head(self, pred_tgt_unknown, unknown_region_mask):
        """
        For Ku=1, directly use unknown_region_mask as pseudo unknown mask.
        For Ku>1, use balanced pseudo labeling to avoid class collapse.
        """
        if pred_tgt_unknown.shape[1] == 0:
            dummy_label = torch.zeros_like(unknown_region_mask, dtype=torch.long)
            dummy_conf = torch.zeros_like(unknown_region_mask, dtype=pred_tgt_unknown.dtype)
            return torch.zeros_like(unknown_region_mask, dtype=torch.bool), dummy_label, dummy_conf

        pred_tgt_unknown = resize(
            pred_tgt_unknown,
            size=unknown_region_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        # single unknown class: softmax confidence is meaningless (always 1)
        if pred_tgt_unknown.shape[1] == 1:
            unknown_label_local = torch.zeros_like(unknown_region_mask, dtype=torch.long)
            unknown_conf = torch.ones_like(unknown_region_mask, dtype=pred_tgt_unknown.dtype)
            unknown_mask = unknown_region_mask
            return unknown_mask, unknown_label_local, unknown_conf

        prob_unknown = F.softmax(pred_tgt_unknown, dim=1)
        unknown_mask, unknown_label_local, unknown_conf = self._balanced_unknown_assignment(
            prob_unknown, unknown_region_mask)

        return unknown_mask, unknown_label_local, unknown_conf

    @torch.no_grad()
    def _update_unknown_anchors_from_pseudo(self, feat_t, unknown_mask, unknown_label_local):
        if self.unknown_anchors.shape[0] == 0:
            return

        feat_t = resize(
            feat_t,
            size=unknown_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        feat_flat = feat_t.permute(0, 2, 3, 1)

        for u in range(self.unknown_anchors.shape[0]):
            mask = unknown_mask & (unknown_label_local == u)
            if mask.sum() < self.min_pixels_per_anchor:
                continue

            cls_feat = feat_flat[mask]
            cls_anchor = cls_feat.mean(dim=0)
            cls_anchor = F.normalize(cls_anchor, dim=0)
            self.unknown_anchors[u] = F.normalize(
                self.proto_momentum * self.unknown_anchors[u] +
                (1.0 - self.proto_momentum) * cls_anchor,
                dim=0
            )

    def _mine_target_masks(self, pred_src_known, pred_tgt_known):
        if self.num_known_classes == 1:
            # 单类用 sigmoid 得到概率，避免 softmax 恒为 1
            prob_src = torch.sigmoid(pred_src_known)  # (B, 1, H, W)
            prob_tgt = torch.sigmoid(pred_tgt_known)
            src_conf = prob_src[:, 0, :, :]  # (B, H, W)
            tgt_conf = prob_tgt[:, 0, :, :]
            src_cls_local = torch.zeros_like(src_conf, dtype=torch.long)
            tgt_cls_local = torch.zeros_like(tgt_conf, dtype=torch.long)
            discrepancy = torch.abs(prob_src - prob_tgt).squeeze(1)  # (B, H, W)
        else:
            prob_src = F.softmax(pred_src_known, dim=1)
            prob_tgt = F.softmax(pred_tgt_known, dim=1)
            src_conf, src_cls_local = torch.max(prob_src, dim=1)
            tgt_conf, tgt_cls_local = torch.max(prob_tgt, dim=1)
            discrepancy = torch.mean(torch.abs(prob_src - prob_tgt), dim=1)

        known_mask = (
            (src_cls_local == tgt_cls_local) &
            (src_conf > self.known_conf_thresh) &
            (discrepancy < self.discrepancy_thresh)
        )
        known_label = src_cls_local

        unknown_region_mask = (
            (src_conf < self.known_conf_thresh) |
            (discrepancy > self.discrepancy_thresh)
        ) & (~known_mask)

        return known_mask, known_label, unknown_region_mask

    def _sample_vectors(self, feat_map, mask, labels=None, max_samples=4096):
        feat_map = resize(
            feat_map,
            size=mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        feat_vec = feat_map.permute(0, 2, 3, 1)[mask]

        if feat_vec.shape[0] == 0:
            return None, None

        if feat_vec.shape[0] > max_samples:
            idx = torch.randperm(feat_vec.shape[0], device=feat_vec.device)[:max_samples]
            feat_vec = feat_vec[idx]
            label_vec = labels[mask][idx] if labels is not None else None
        else:
            label_vec = labels[mask] if labels is not None else None

        return feat_vec, label_vec

    def _unified_anchor_contrast(self,
                                 feat_t,
                                 known_mask,
                                 known_label,
                                 unknown_mask,
                                 unknown_label_local):
        known_feat, known_targets = self._sample_vectors(
            feat_t, known_mask, known_label, self.max_samples // 2)

        unknown_feat, unknown_targets_local = self._sample_vectors(
            feat_t, unknown_mask, unknown_label_local, self.max_samples // 2)

        feat_list = []
        target_list = []

        num_known = self.known_anchors.shape[0]
        num_unknown = self.unknown_anchors.shape[0]

        if known_feat is not None and known_feat.shape[0] > 0:
            feat_list.append(known_feat)
            target_list.append(known_targets)

        if unknown_feat is not None and unknown_feat.shape[0] > 0 and num_unknown > 0:
            feat_list.append(unknown_feat)
            target_list.append(unknown_targets_local + num_known)

        if len(feat_list) == 0:
            return feat_t.sum() * 0.0

        feat_all = torch.cat(feat_list, dim=0)
        target_all = torch.cat(target_list, dim=0)

        # all_anchors = torch.cat([self.known_anchors, self.unknown_anchors], dim=0)
        # all_anchors = F.normalize(all_anchors, dim=1)
        #
        # logits = torch.matmul(feat_all, all_anchors.t()) / self.tau_unified
        # loss = F.cross_entropy(logits, target_all)
        loss = self.supcon_loss_function(feat_all, target_all, self.tau_unified)

        return loss

    def _unknown_pseudo_seg_loss(self, pred_tgt_unknown, unknown_mask, unknown_label_local):
        """
        Fix:
        - Ku == 0: zero loss
        - Ku == 1: BCEWithLogits on unknown mask
        - Ku > 1: CE on local unknown labels
        """
        num_unknown = pred_tgt_unknown.shape[1]
        if num_unknown == 0:
            return pred_tgt_unknown.sum() * 0.0

        pred_tgt_unknown = resize(
            pred_tgt_unknown,
            size=unknown_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        if unknown_mask.sum() == 0:
            return pred_tgt_unknown.sum() * 0.0

        # single unknown class -> BCE
        if num_unknown == 1:
            logits = pred_tgt_unknown[:, 0, :, :]
            target = unknown_mask.float()
            valid = unknown_mask

            if valid.sum() == 0:
                return logits.sum() * 0.0

            loss_map = F.binary_cross_entropy_with_logits(
                logits, target, reduction='none')

            loss = loss_map[valid].mean()
            return loss

        # multi-unknown class -> CE
        pseudo_gt = torch.full_like(unknown_label_local, 255)
        pseudo_gt[unknown_mask] = unknown_label_local[unknown_mask]

        if (pseudo_gt != 255).sum() == 0:
            return pred_tgt_unknown.sum() * 0.0

        return F.cross_entropy(
            pred_tgt_unknown,
            pseudo_gt,
            ignore_index=255,
            reduction='mean'
        )

    def supcon_loss_function(self, features, labels, temperature=0.07):
        """纯函数版 SupCon Loss，和你的 _unknown_pseudo_seg_loss 风格完全一样"""
        device = features.device
        N = features.shape[0]

        # 同类掩码
        mask = torch.eq(labels.view(N, 1), labels.view(1, N)).float().to(device)

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / temperature

        # 去掉自身
        logits_mask = 1.0 - torch.eye(N, device=device)
        mask = mask * logits_mask

        # 计算 SupCon
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob.mean()

        return loss


    def _init_decode_head(self, decode_head):
        return builder.build_head(decode_head)

    def _parse_train_cfg(self):
        if self.train_cfg is None:
            self.train_cfg = dict()
        self.disc_steps = self.train_cfg.get('disc_steps', 1)
        self.disc_init_steps = self.train_cfg.get('disc_init_steps', 0)

    def extract_feat(self, img):
        return self.backbone_s(img)

    def _merge_known_unknown_logits_with_gate(self, P_t_known, P_t_unknown):
        if self.closed_set:
            return P_t_known

        B, _, H, W = P_t_known.shape
        device = P_t_known.device
        dtype = P_t_known.dtype

        full_logits = torch.full(
            (B, self.num_classes, H, W),
            -100.0,
            device=device,
            dtype=dtype
        )

        full_logits[:, self.source_to_target_idx, :, :] = P_t_known
        if self.num_unknown_classes > 0 and P_t_unknown.shape[1] > 0:
            full_logits[:, self.unknown_idx, :, :] = P_t_unknown + self.infer_unknown_logit_bias

        if self.num_known_classes == 1:
            known_conf = torch.sigmoid(P_t_known).squeeze(1)
        else:
            known_prob = F.softmax(P_t_known, dim=1)
            known_conf, _ = torch.max(known_prob, dim=1)

        unknown_gate = (known_conf < self.infer_known_conf_thresh)
        known_gate = ~unknown_gate

        if self.num_unknown_classes > 0:
            known_idx = self.source_to_target_idx
            unknown_idx = self.unknown_idx

            known_mask_expand = known_gate.unsqueeze(1).expand(-1, len(known_idx), -1, -1)
            unknown_mask_expand = unknown_gate.unsqueeze(1).expand(-1, len(unknown_idx), -1, -1)

            full_logits[:, known_idx, :, :] = torch.where(
                known_mask_expand,
                full_logits[:, known_idx, :, :],
                torch.full_like(full_logits[:, known_idx, :, :], -100.0)
            )
            full_logits[:, unknown_idx, :, :] = torch.where(
                unknown_mask_expand,
                full_logits[:, unknown_idx, :, :],
                torch.full_like(full_logits[:, unknown_idx, :, :], -100.0)
            )

        return full_logits

    def encode_decode(self, img, img_metas):
        F_t = self.forward_backbone(self.backbone_s, img)

        if hasattr(self.decode_head_t, 'forward_decoupled'):
            _, P_t_known, P_t_unknown = self.decode_head_t.forward_decoupled(F_t)

            P_t_known = resize(
                input=P_t_known,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

            if self.closed_set:
                P_t = P_t_known
            else:
                if P_t_unknown.shape[1] > 0:
                    P_t_unknown = resize(
                        input=P_t_unknown,
                        size=img.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)

                P_t = self._merge_known_unknown_logits_with_gate(P_t_known, P_t_unknown)
        else:
            P_t = self.forward_decode_head(self.decode_head_t, F_t)
            P_t = resize(
                input=P_t,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        return P_t

    def _decode_head_forward_test(self, x, img_metas):
        seg_logits = self.decode_head_t.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_dummy(self, img):
        return self.encode_decode(img, None)

    def forward_backbone(self, backbone, img):
        return backbone(img)

    def forward_decode_head(self, decode_head, feature):
        return decode_head(feature)

    def forward_decode_head_decoupled(self, decode_head, feature):
        if hasattr(decode_head, 'forward_decoupled'):
            return decode_head.forward_decoupled(feature)
        else:
            pred = decode_head(feature)
            return feature[-1], pred, pred[:, :0]

    def forward_train(self, img, B_img):
        pass

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg

    def _get_known_branch_loss(self, pred, gt_semantic_seg, gt_weight=None):
        pred = resize(
            input=pred,
            size=gt_semantic_seg.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = gt_semantic_seg.squeeze(1)

        losses = dict()
        loss = self.decode_head_t.loss_decode(
            pred,
            seg_label,
            weight=gt_weight,
            ignore_index=255)

        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses['loss_seg'] = loss

        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg

    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)

                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        preds = preds / count_mat

        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            size = img_meta[0]['ori_shape'][:2] if not torch.onnx.is_in_onnx_export() else img.shape[2:]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        output = F.softmax(seg_logit, dim=1)

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            seg_logit += self.inference(imgs[i], img_metas[i], rescale)
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)