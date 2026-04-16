# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class OSNetV1(BaseSegmentor):
    """
    KPU-OSNet:
    Known Prototype Alignment + Unknown Prototype Repulsion

    Only two key innovations:
    1) align target-known pixels to source-known prototypes
    2) repel target-unknown pixels from all source-known prototypes
    """

    def __init__(self,
                 backbone_s,
                 decode_head_s,
                 decode_head_t,
                 source_classes=None,
                 target_classes=None,
                 proto_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OSNetV1, self).__init__(init_cfg)

        assert source_classes is not None and target_classes is not None, \
            "必须提供source_classes和target_classes!"

        self.source_classes = source_classes
        self.target_classes = target_classes

        source_to_target_indices = [target_classes.index(c) for c in source_classes]
        unknown_indices = [i for i, c in enumerate(target_classes) if c not in source_classes]

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
        self._init_proto_module(proto_cfg)

    def _init_proto_module(self, proto_cfg):
        if proto_cfg is None:
            proto_cfg = dict()

        self.proj_dim = proto_cfg.get('proj_dim', 256)
        self.proto_momentum = proto_cfg.get('momentum', 0.99)
        self.known_conf_thresh = proto_cfg.get('known_conf_thresh', 0.7)
        self.unknown_conf_thresh = proto_cfg.get('unknown_conf_thresh', 0.6)
        self.discrepancy_thresh = proto_cfg.get('discrepancy_thresh', 0.2)
        self.unknown_margin = proto_cfg.get('unknown_margin', 0.3)

        self.loss_kpa_weight = proto_cfg.get('loss_kpa_weight', 1.0)
        self.loss_ucr_weight = proto_cfg.get('loss_ucr_weight', 1.0)

        # projector from backbone last stage feature
        last_channels = 512
        self.feat_proj = nn.Sequential(
            nn.Conv2d(last_channels, self.proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)
        )

        self.register_buffer(
            'known_prototypes',
            F.normalize(torch.randn(len(self.source_classes), self.proj_dim), dim=1)
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

        # backbone features
        F_s = self.forward_backbone(self.backbone_s, img_s)
        F_t = self.forward_backbone(self.backbone_s, img_t)

        # decoder outputs
        P_s_src = self.forward_decode_head(self.decode_head_s, F_s)  # source head on source
        P_s_tgt = self.forward_decode_head(self.decode_head_t, F_s)  # target head on source

        P_t_src = self.forward_decode_head(self.decode_head_s, F_t)  # source head on target
        P_t_tgt = self.forward_decode_head(self.decode_head_t, F_t)  # target head on target

        # ------------------------------------------------------------------
        # 1) Source supervised losses
        # ------------------------------------------------------------------
        loss_seg_s, log_seg_s = self._get_segmentor_loss(
            self.decode_head_s, P_s_src, gt_s)
        log_vars.update(self._rename_log(log_seg_s, '_seg_s'))

        gt_s_target = self._remap_source_gt_to_target(gt_s)
        loss_seg_t, log_seg_t = self._get_segmentor_loss(
            self.decode_head_t, P_s_tgt, gt_s_target)
        log_vars.update(self._rename_log(log_seg_t, '_seg_t'))

        # ------------------------------------------------------------------
        # 2) Update source-known prototypes from source labeled data
        # ------------------------------------------------------------------
        feat_s = self._project_feature(F_s[-1])
        self._update_known_prototypes(feat_s, gt_s)

        # ------------------------------------------------------------------
        # 3) Mine target-known / target-unknown pixels via dual-head discrepancy
        # ------------------------------------------------------------------
        P_t_src_full = self._scatter_source_logits_to_target(P_t_src)
        known_mask, known_label, unknown_mask = self._mine_target_masks(P_t_src_full, P_t_tgt)

        # ------------------------------------------------------------------
        # 4) Prototype-based target known alignment and unknown repulsion
        # ------------------------------------------------------------------
        feat_t = self._project_feature(F_t[-1])
        loss_kpa = self._known_prototype_alignment_loss(feat_t, known_mask, known_label)
        loss_ucr = self._unknown_centroid_repulsion_loss(feat_t, unknown_mask)

        log_vars['loss_kpa'] = loss_kpa.item()
        log_vars['loss_ucr'] = loss_ucr.item()

        total_loss = (
            loss_seg_s +
            loss_seg_t +
            self.loss_kpa_weight * loss_kpa +
            self.loss_ucr_weight * loss_ucr
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
    def _update_known_prototypes(self, feat_s, gt_s):
        feat_s = resize(
            feat_s,
            size=gt_s.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        gt = gt_s.squeeze(1)
        for cls_id in range(len(self.source_classes)):
            mask = (gt == cls_id)
            if mask.sum() < 10:
                continue
            cls_feat = feat_s.permute(0, 2, 3, 1)[mask]
            cls_proto = cls_feat.mean(dim=0)
            cls_proto = F.normalize(cls_proto, dim=0)
            self.known_prototypes[cls_id] = F.normalize(
                self.proto_momentum * self.known_prototypes[cls_id] +
                (1 - self.proto_momentum) * cls_proto,
                dim=0
            )

    def _mine_target_masks(self, pred_src_full, pred_tgt):
        """
        Mine:
        1) target-known pixels: both heads agree on shared known classes with high confidence
        2) target-unknown pixels: source head uncertain or target unknown stronger or large discrepancy
        """
        prob_src = F.softmax(pred_src_full, dim=1)
        prob_tgt = F.softmax(pred_tgt, dim=1)

        src_share = prob_src[:, self.source_to_target_idx, :, :]
        tgt_share = prob_tgt[:, self.source_to_target_idx, :, :]

        src_conf, src_cls_local = torch.max(src_share, dim=1)
        tgt_conf, tgt_cls_local = torch.max(tgt_share, dim=1)

        discrepancy = torch.mean(torch.abs(src_share - tgt_share), dim=1)

        # known mask: dual-head agreement + high confidence + low discrepancy
        known_mask = (
            (src_cls_local == tgt_cls_local) &
            (src_conf > self.known_conf_thresh) &
            (tgt_conf > self.known_conf_thresh) &
            (discrepancy < self.discrepancy_thresh)
        )

        known_label = src_cls_local  # local source-class index

        # unknown mask
        if len(self.unknown_idx) > 0:
            tgt_unknown = prob_tgt[:, self.unknown_idx, :, :]
            tgt_unknown_conf, _ = torch.max(tgt_unknown, dim=1)
        else:
            tgt_unknown_conf = torch.zeros_like(src_conf)

        unknown_mask = (
            ((src_conf < self.known_conf_thresh) | (discrepancy > self.discrepancy_thresh)) &
            (tgt_unknown_conf > self.unknown_conf_thresh if len(self.unknown_idx) > 0
             else (src_conf < self.known_conf_thresh))
        )

        return known_mask, known_label, unknown_mask

    def _known_prototype_alignment_loss(self, feat_t, known_mask, known_label):
        if known_mask.sum() < 1:
            return feat_t.sum() * 0.0

        feat_t = resize(
            feat_t,
            size=known_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        feat_vec = feat_t.permute(0, 2, 3, 1)[known_mask]     # [N, D]
        cls_vec = known_label[known_mask]                     # [N]

        proto = self.known_prototypes[cls_vec]                # [N, D]
        sim = torch.sum(feat_vec * proto, dim=1)              # cosine similarity
        loss = (1.0 - sim).mean()
        return loss

    def _unknown_centroid_repulsion_loss(self, feat_t, unknown_mask):
        if unknown_mask.sum() < 1:
            return feat_t.sum() * 0.0

        feat_t = resize(
            feat_t,
            size=unknown_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        feat_vec = feat_t.permute(0, 2, 3, 1)[unknown_mask]   # [N, D]
        sim_all = torch.matmul(feat_vec, self.known_prototypes.t())  # [N, K]
        max_sim = sim_all.max(dim=1)[0]

        loss = F.relu(max_sim - self.unknown_margin).mean()
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