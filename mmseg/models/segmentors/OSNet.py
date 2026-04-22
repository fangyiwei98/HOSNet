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
    Unified-Anchor OSNet for Open-set Domain Adaptive Segmentation

    Main changes:
    1) Remove unknown segmentation pseudo-label loss.
    2) Merge known/unknown contrast into one unified anchor contrastive loss.
    3) Build virtual anchors for multiple unknown classes and update them online.
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
                 init_cfg=None):
        super(OSNet, self).__init__(init_cfg)

        assert source_classes is not None and target_classes is not None

        self.source_classes = source_classes
        self.target_classes = target_classes

        source_to_target_indices = [target_classes.index(c) for c in source_classes]
        unknown_indices = [i for i, c in enumerate(target_classes) if c not in source_classes]

        self.unknown_indices = unknown_indices

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

        self.num_classes = self.decode_head_t.num_classes
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

        self.max_samples = contrast_cfg.get('max_samples', 4096)
        self.min_pixels_per_anchor = contrast_cfg.get('min_pixels_per_anchor', 10)

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
        P_s_tgt = self.forward_decode_head(self.decode_head_t, F_s)

        P_t_src = self.forward_decode_head(self.decode_head_s, F_t)
        P_t_tgt = self.forward_decode_head(self.decode_head_t, F_t)

        # 1) source supervised seg
        loss_seg_s, log_seg_s = self._get_segmentor_loss(
            self.decode_head_s, P_s_src, gt_s)
        log_vars.update(self._rename_log(log_seg_s, '_seg_s'))

        gt_s_target = self._remap_source_gt_to_target(gt_s)
        loss_seg_t, log_seg_t = self._get_segmentor_loss(
            self.decode_head_t, P_s_tgt, gt_s_target)
        log_vars.update(self._rename_log(log_seg_t, '_seg_t'))

        # 2) project features
        feat_s = self._project_feature(F_s[-1])
        feat_t = self._project_feature(F_t[-1])

        # 3) update source known anchors
        self._update_known_anchors(feat_s, gt_s)

        # 4) mine target known / unknown masks
        P_t_src_full = self._scatter_source_logits_to_target(P_t_src)
        known_mask, known_label, unknown_mask, unknown_label_local = \
            self._mine_target_masks(P_t_src_full, P_t_tgt)

        # 5) update target unknown anchors
        self._update_unknown_anchors(feat_t, unknown_mask, unknown_label_local)

        # 6) unified anchor contrast
        loss_contrast = self._unified_anchor_contrast(
            feat_t,
            known_mask,
            known_label,
            unknown_mask,
            unknown_label_local
        )

        log_vars['loss_contrast'] = loss_contrast.item()
        log_vars['num_known_pixels'] = known_mask.sum().item()
        log_vars['num_unknown_pixels'] = unknown_mask.sum().item()

        total_loss = (
            loss_seg_s +
            loss_seg_t +
            self.loss_contrast_weight * loss_contrast
        )

        total_loss.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        if 'feat_proj' in optimizer:
            optimizer['feat_proj'].step()

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

    def _remap_source_gt_to_target(self, gt_s):
        gt_new = gt_s.clone()
        valid_mask = (gt_new != 255)
        mapped = torch.full_like(gt_new, 255)
        for s_idx, t_idx in enumerate(self.source_to_target_idx):
            mapped[(gt_new == s_idx) & valid_mask] = t_idx
        return mapped

    def _scatter_source_logits_to_target(self, pred_s):
        B, _, H, W = pred_s.shape
        out = pred_s.new_full((B, len(self.target_classes), H, W), -100.0)
        out[:, self.source_to_target_idx, :, :] = pred_s
        return out

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
    def _update_unknown_anchors(self, feat_t, unknown_mask, unknown_label_local):
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

    def _mine_target_masks(self, pred_src_full, pred_tgt):
        """
        Return:
            known_mask: [B,H,W]
            known_label: local known class ids in [0, Ks-1]
            unknown_mask: [B,H,W]
            unknown_label_local: local unknown class ids in [0, Ku-1]
        """
        prob_src = F.softmax(pred_src_full, dim=1)
        prob_tgt = F.softmax(pred_tgt, dim=1)

        src_share = prob_src[:, self.source_to_target_idx, :, :]
        tgt_share = prob_tgt[:, self.source_to_target_idx, :, :]

        src_conf, src_cls_local = torch.max(src_share, dim=1)
        tgt_conf, tgt_cls_local = torch.max(tgt_share, dim=1)

        discrepancy = torch.mean(torch.abs(src_share - tgt_share), dim=1)

        known_mask = (
            (src_cls_local == tgt_cls_local) &
            (src_conf > self.known_conf_thresh) &
            (discrepancy < self.discrepancy_thresh)
        )
        known_label = src_cls_local

        unknown_mask = (
            (src_conf < self.known_conf_thresh) |
            (discrepancy > self.discrepancy_thresh)
        ) & (~known_mask)

        if len(self.unknown_idx) > 0:
            prob_tgt_unknown = prob_tgt[:, self.unknown_idx, :, :]  # [B, Ku, H, W]
            _, unknown_label_local = torch.max(prob_tgt_unknown, dim=1)
        else:
            unknown_label_local = torch.zeros_like(known_label)

        return known_mask, known_label, unknown_mask, unknown_label_local

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

        all_anchors = torch.cat([self.known_anchors, self.unknown_anchors], dim=0)
        logits = torch.matmul(feat_all, all_anchors.t()) / self.tau_unified
        loss = F.cross_entropy(logits, target_all)

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

    def encode_decode(self, img, img_metas):
        F_t = self.forward_backbone(self.backbone_s, img)
        P_t = self.forward_decode_head(self.decode_head_t, F_t)
        out = resize(
            input=P_t,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x, img_metas):
        seg_logits = self.decode_head_t.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_dummy(self, img):
        return self.encode_decode(img, None)

    def forward_backbone(self, backbone, img):
        return backbone(img)

    def forward_decode_head(self, decode_head, feature):
        return decode_head(feature)

    def forward_train(self, img, B_img):
        pass

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        # 保持兼容，如果你的 head 不支持 gt_weight，可以去掉这个参数
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
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