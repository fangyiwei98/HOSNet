# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class OSNet(BaseSegmentor):

    def __init__(self,
                 backbone_s,
                 decode_head_s,
                 decode_head_t,
                 source_classes=None,
                 target_classes=None,
                 cross_EMA=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OSNet, self).__init__(init_cfg)

        assert source_classes is not None and target_classes is not None, \
            "必须提供source_classes和target_classes!"

        self.source_classes = source_classes
        self.target_classes = target_classes

        # source压缩通道 -> target全类别索引
        # 例如:
        # source_classes = [imp, low_veg, tree, car, clutter]
        # target_classes = [imp, building, low_veg, tree, car, clutter]
        # source_to_target_idx = [0, 2, 3, 4, 5]
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

        if cross_EMA is not None:
            self.cross_EMA = cross_EMA
            self._init_cross_EMA(self.cross_EMA)

        self._parse_train_cfg()

    def train_step(self, data_batch, optimizer, **kwargs):
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration

        if curr_iter % 500 == 0:
            print('gt unique:', torch.unique(data_batch['gt_semantic_seg']))

        if curr_iter > 0:
            self._update_cross_EMA(curr_iter)

        optimizer['backbone_s'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_t'].zero_grad()

        log_vars = dict()

        # 1. 提取特征
        F_s = self.forward_backbone(self.backbone_s, data_batch['img'])
        F_t = self.forward_backbone(self.backbone_s, data_batch['B_img'])

        # 2. 前向预测
        P_s = self.forward_decode_head(self.decode_head_s, F_s)
        P_t = self.forward_decode_head(self.decode_head_t, F_t)

        # 3. source supervised loss
        loss_seg_s, log_vars_seg_s = self._get_segmentor_loss(
            self.decode_head_s, P_s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s)

        # 4. target pseudo loss
        pseudo_label, pseudo_weight, _ = self.encode_decode_crossEMA(
            input=data_batch['B_img'],
            dev=data_batch['img'].device)

        loss_seg_t, log_vars_seg_t = self._get_segmentor_loss(
            self.decode_head_t,
            P_t,
            pseudo_label,
            gt_weight=pseudo_weight)

        if 'loss_ce' in log_vars_seg_t:
            log_vars_seg_t['loss_ce_seg_t'] = log_vars_seg_t.pop('loss_ce')
        if 'acc_seg' in log_vars_seg_t:
            log_vars_seg_t['acc_seg_t'] = log_vars_seg_t.pop('acc_seg')
        if 'loss' in log_vars_seg_t:
            log_vars_seg_t['loss_seg_t'] = log_vars_seg_t.pop('loss')

        log_vars.update(log_vars_seg_t)

        loss_seg = loss_seg_s + self.cross_EMA_training_ratio * loss_seg_t
        loss_seg.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()

        self.iteration += 1

        outputs = dict(
            loss=loss_seg,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas'])
        )
        return outputs

    def encode_decode_crossEMA(self, input=None, dev=None):
        F_t_student = self.forward_backbone(self.backbone_s, input)
        F_t_teacher = self.forward_backbone(self.cross_EMA_backbone, input)

        # 学生source head输出: 压缩类别空间
        P_student_source = self.forward_decode_head(self.decode_head_s, F_t_student)

        # 教师target head输出: 完整目标类别空间
        P_teacher_target = self.forward_decode_head(self.cross_EMA_decoder, F_t_teacher)

        # 融合到target完整类别空间
        P_ensemble = torch.zeros_like(P_teacher_target)

        # source压缩通道填到target全类别对应位置
        P_ensemble[:, self.source_to_target_idx, :, :] = P_student_source

        # unknown类别由teacher提供
        if len(self.unknown_idx) > 0:
            P_ensemble[:, self.unknown_idx, :, :] = P_teacher_target[:, self.unknown_idx, :, :]

        P_EMA_KD = P_ensemble.detach()

        P_final_logits = resize(
            input=P_ensemble,
            size=input.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        P_EMA_detach = P_final_logits.detach()
        pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(
            P_EMA_detach, dev=dev)

        return pseudo_label, pseudo_weight, P_EMA_KD

    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        pred_softmax = torch.softmax(pred, dim=1)

        # 已知类 logits/prob 来自 source压缩通道映射到target类别后的那些通道
        pred_known = pred_softmax[:, self.source_to_target_idx, :, :]
        max_known_prob, max_known_local_idx = torch.max(pred_known, dim=1)

        # source局部通道索引 -> target全局类别索引
        pseudo_label = self.source_to_target_idx[max_known_local_idx]

        # open-set判定
        open_set_mask = max_known_prob < 0.5

        # unknown类别只从unknown通道中选
        if len(self.unknown_idx) > 0:
            pred_unknown = pred_softmax[:, self.unknown_idx, :, :]
            max_unknown_prob, max_unknown_local_idx = torch.max(pred_unknown, dim=1)
            best_unknown_label = self.unknown_idx[max_unknown_local_idx]
        else:
            max_unknown_prob = torch.zeros_like(max_known_prob)
            best_unknown_label = pseudo_label

        pseudo_label = torch.where(open_set_mask, best_unknown_label, pseudo_label)
        pseudo_prob = torch.where(open_set_mask, max_unknown_prob, max_known_prob)

        # 高置信度区域比例
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre)
        ps_size = pseudo_label.numel()
        pseudo_weight_ratio = ps_large_p.float().sum().item() / (ps_size + 1e-8)

        pseudo_weight = pseudo_weight_ratio * torch.ones(
            pseudo_prob.shape, device=dev, dtype=torch.float32)

        # 日志打印
        if hasattr(self, 'iteration') and self.iteration % 100 == 0:
            valid_num = ps_large_p.float().sum().item()
            total_num = ps_size
            unknown_pixel_num = open_set_mask.float().sum().item()

            print(f"\n========== Iter {self.iteration} 伪标签统计 ==========")
            print(f"已知类: {self.source_classes}")
            print(f"未知类: {[self.target_classes[i] for i in self.unknown_idx.cpu().numpy()]}")
            print(f"总像素: {total_num} | 高置信度: {int(valid_num)} | 有效比例: {pseudo_weight_ratio * 100:.2f}%")
            print(f"🚨 判定为未知类: {int(unknown_pixel_num)} ({unknown_pixel_num / total_num * 100:.2f}%)")

            pseudo_label_np = pseudo_label.detach().cpu().numpy()
            pseudo_prob_np = pseudo_prob.detach().cpu().numpy()
            ps_large_p_np = ps_large_p.detach().cpu().numpy()
            valid_labels = pseudo_label_np[ps_large_p_np]

            if len(valid_labels) > 0:
                print("---------- 高置信度类别分布 ----------")
                valid_probs = pseudo_prob_np[ps_large_p_np]
                for cls in np.unique(valid_labels):
                    cls_mask = valid_labels == cls
                    cnt = np.sum(cls_mask)
                    avg_prob = np.mean(valid_probs[cls_mask])

                    cls_name = self.target_classes[int(cls)]
                    mark = "🚨" if int(cls) in self.unknown_idx.cpu().numpy().tolist() else "✅"
                    print(f"{mark} 类别 {int(cls)} ({cls_name}) | 数量: {cnt} | 平均置信度: {avg_prob:.4f}")

        # 类别权重
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).float()
            pseudo_weight = pseudo_weight * ps_large_p_rare

            # 安全写法，避免原地替换造成潜在错误
            pseudo_class_weight = torch.ones_like(
                pseudo_label, dtype=torch.float32, device=pseudo_label.device)
            for i, w in enumerate(self.cross_EMA_pseu_cls_weight):
                pseudo_class_weight[pseudo_label == i] = float(w)

            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5

        pseudo_label = pseudo_label.unsqueeze(1)  # [B,1,H,W]
        return pseudo_label, pseudo_weight

    def _init_cross_EMA(self, cfg):
        self.cross_EMA_type = cfg['type']
        self.cross_EMA_alpha = cfg['decay']
        self.cross_EMA_training_ratio = cfg['training_ratio']
        self.cross_EMA_pseu_cls_weight = cfg['pseudo_class_weight']
        self.cross_EMA_pseu_thre = cfg['pseudo_threshold']
        self.cross_EMA_rare_pseu_thre = cfg['pseudo_rare_threshold']
        self.cross_EMA_backbone = builder.build_backbone(cfg['backbone_EMA'])
        self.cross_EMA_decoder = self._init_decode_head(cfg['decode_head_EMA'])

    def _init_decode_head(self, decode_head):
        decode_head = builder.build_head(decode_head)
        return decode_head

    def _parse_train_cfg(self):
        if self.train_cfg is None:
            self.train_cfg = dict()
        self.disc_steps = self.train_cfg.get('disc_steps', 1)
        self.disc_init_steps = self.train_cfg.get('disc_init_steps', 0)

    def extract_feat(self, img):
        x = self.backbone_s(img)
        return x

    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)

        for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_s.parameters()):
            if not target_b.data.shape:
                ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
            else:
                ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]

        for ema_d, target_d in zip(self.cross_EMA_decoder.parameters(), self.decode_head_t.parameters()):
            if not target_d.data.shape:
                ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
            else:
                ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]

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
        seg_logit = self.encode_decode(img, None)
        return seg_logit

    def forward_backbone(self, backbone, img):
        return backbone(img)

    def forward_decode_head(self, decode_head, feature):
        return decode_head(feature)

    def forward_discriminator(self, discriminator, seg_pred):
        return discriminator(seg_pred)

    def forward_train(self, img, B_img):
        pass

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg

    def _get_gan_loss(self, discriminator, pred, domain, target_is_real):
        losses = dict()
        losses[f'loss_gan_{domain}'] = discriminator.gan_loss(pred, target_is_real)
        loss_dis, log_vars_dis = self._parse_losses(losses)
        return loss_dis, log_vars_dis

    def _get_KD_loss(self, teacher, student, pred_name, T=3):
        losses = dict()
        losses[f'loss_KD_{pred_name}'] = self.KL_loss(teacher, student, T)
        loss_KD, log_vars_KD = self._parse_losses(losses)
        return loss_KD, log_vars_KD

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

        assert (count_mat == 0).sum() == 0

        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)

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
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]

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
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)

        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred

        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale

        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def MSE_loss(self, teacher, student):
        mse_loss = nn.MSELoss()
        t = self.sw_softmax(teacher)
        s = self.sw_softmax(student)
        return mse_loss(s, t)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H * W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out

    @staticmethod
    def KL_loss(teacher, student, T=5):
        return nn.KLDivLoss(reduction='mean')(
            F.log_softmax(student / T, dim=1),
            F.softmax(teacher / T, dim=1)
        ) * (T * T)