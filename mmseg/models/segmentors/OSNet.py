# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import copy


@SEGMENTORS.register_module()
class OSNet(BaseSegmentor):

    def __init__(self,
                 backbone_s,
                 decode_head_s,
                 decode_head_t,
                 source_classes=None,  # 新增
                 target_classes=None,  # 新增
                 cross_EMA=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OSNet, self).__init__(init_cfg)

        assert source_classes is not None and target_classes is not None, "必须提供source_classes和target_classes!"
        self.source_classes = source_classes
        self.target_classes = target_classes

        # source decoder 的压缩通道 -> target 全类别索引
        source_to_target_indices = [target_classes.index(c) for c in source_classes]
        unknown_indices = [i for i, c in enumerate(target_classes) if c not in source_classes]

        self.register_buffer('source_to_target_idx', torch.tensor(source_to_target_indices, dtype=torch.long))
        self.register_buffer('unknown_idx', torch.tensor(unknown_indices, dtype=torch.long))

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


    # 训练函数
    def train_step(self, data_batch, optimizer, **kwargs):

        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration

        #通过EMA更新教师网络参数
        if curr_iter > 0:
            self._update_cross_EMA(curr_iter)

        ## 1.初始化需要优化的参数和变量
        optimizer['backbone_s'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_t'].zero_grad()

        # 所有要发送给日志记录器的变量
        log_vars = dict()

        #2.提取特征,img是源域图像,B_img是目标域图像
        F_s = self.forward_backbone(self.backbone_s, data_batch['img'])
        F_t = self.forward_backbone(self.backbone_s, data_batch['B_img'])

        # 3.输出预测
        P_s = self.forward_decode_head(self.decode_head_s, F_s)
        P_t = self.forward_decode_head(self.decode_head_t, F_t)

        # 4. 计算源域损失
        loss_seg_s, log_vars_seg_s = self._get_segmentor_loss(self.decode_head_s, P_s,data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s)

        # 5.计算目标域损失
        # 5.1生成伪标签
        pseudo_label, pseudo_weight, P_EMA_detach= self.encode_decode_crossEMA(input=data_batch['B_img'], dev=data_batch['img'].device)
        # 5.3使用目标域的解码头计算目标域损失
        loss_seg_t, log_vars_seg_t = self._get_segmentor_loss(self.decode_head_t, P_t, pseudo_label,gt_weight=pseudo_weight)
        log_vars_seg_t['loss_ce_seg_t'] = log_vars_seg_t.pop('loss_ce')
        log_vars_seg_t['acc_seg_t'] = log_vars_seg_t.pop('acc_seg')
        log_vars_seg_t['loss_ce_seg_t'] = log_vars_seg_t.pop('loss')
        log_vars.update(log_vars_seg_t)
        # 计算总的分割损失loss_seg
        loss_seg = loss_seg_s + self.cross_EMA_training_ratio * loss_seg_t
        loss_seg.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()


        # 将当前计算的分割损失值赋给loss变量
        loss = loss_seg
        # 检查self对象是否具有'iteration'属性
        # 如果有，说明这是一个迭代过程，需要对迭代次数进行更新
        if hasattr(self, 'iteration'):
            # 迭代次数加1，表示完成了一次迭代过程
            self.iteration += 1

        # 创建一个字典，用于存储本次迭代的相关信息
        outputs = dict(
            loss=loss,
            # 日志变量
            log_vars=log_vars,
            # 当前批次数据的样本数量，通常用于计算平均损失等指标
            num_samples=len(data_batch['img_metas'])
        )

        # 返回封装好的信息字典
        return outputs

    def encode_decode_crossEMA(self, input=None, dev=None):
        F_t = self.forward_backbone(self.backbone_s, input)
        F_ttea = self.forward_backbone(self.cross_EMA_backbone, input)

        P_t = self.forward_decode_head(self.decode_head_s, F_t)  # 学生：源域压缩类别
        P_ttea = self.forward_decode_head(self.cross_EMA_decoder, F_ttea)  # 教师：目标域全类别

        P_ensemble = torch.zeros_like(P_ttea)

        # 将source压缩通道按类别语义填回target全类别空间
        P_ensemble[:, self.source_to_target_idx, :, :] = P_t

        if len(self.unknown_idx) > 0:
            P_ensemble[:, self.unknown_idx, :, :] = P_ttea[:, self.unknown_idx, :, :]

        P_EMA_KD = P_ensemble.detach()

        P_final_logits = resize(
            input=P_ensemble,
            size=input.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        P_EMA_detach = P_final_logits.detach()
        pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(P_EMA_detach, dev=dev)

        return pseudo_label, pseudo_weight, P_EMA_KD

    # 生成伪标签
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        # 1. 生成基本的伪标签
        pred_softmax = torch.softmax(pred, dim=1)
        # 找到概率分布中最大值对应的类别索引，即伪标签
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        # 判断哪些伪标签的概率大于或等于设定的阈值，并转换为长整型张量
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1
        # 计算伪标签的总数量
        ps_size = np.size(np.array(pseudo_label.cpu()))
        # 计算大于或等于阈值的伪标签的比例
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        # 根据比例生成权重张量，所有元素初始化为权重比例值
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)
        # 2. 应用类别平衡策略
        # 2.1 如果设置了类别权重和稀有类别阈值
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            # 判断哪些伪标签的概率大于或等于稀有类别阈值
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).long() == 1
            # 更新权重张量，只有大于或等于稀有类别阈值的伪标签才保留原有权重
            pseudo_weight = pseudo_weight * ps_large_p_rare
            # 创建一个与伪标签形状相同的浮点数张量，用于存储类别权重
            pseudo_class_weight = copy.deepcopy(pseudo_label.float())
            # 遍历类别权重列表，将对应类别的伪标签权重设置为类别权重值
            for i in range(len(self.cross_EMA_pseu_cls_weight)):
                pseudo_class_weight[pseudo_class_weight == i] = self.cross_EMA_pseu_cls_weight[i]
            # 更新权重张量，将类别权重与原有权重相乘
            pseudo_weight = pseudo_class_weight * pseudo_weight
            # 如果权重为0，则设置为权重比例的0.5倍，避免权重完全为0
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5
        # 将伪标签张量扩展一个维度，以便与某些模型或操作兼容
        pseudo_label = pseudo_label[:, None, :, :]
        # 返回生成的伪标签和权重
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


    # 初始化decode_head
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = builder.build_head(decode_head)
        return decode_head

    # 解析训练配置
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))


    # 特征提取函数
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone_s(img)
        return x

    #更新cross_EMA
    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)
        ## 1. update teacher_backbone
        for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_s.parameters()):
            if not target_b.data.shape:
                ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
            else:
                ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]

        ## 2. updata teacher_decoder
        for ema_d, target_d in zip(self.cross_EMA_decoder.parameters(), self.decode_head_t.parameters()):
            ## For scalar params
            if not target_d.data.shape:
                ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
            ## For tensor params
            else:
                ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]

    # 编码和解码函数
    def encode_decode(self, img, img_metas):
        ## 1. forward backbone
        F_t = self.forward_backbone(self.backbone_s, img)
        ## 2. forward decode_head
        #P_t = self.forward_decode_head(self.decode_head_s, F_t)
        P_t = self.forward_decode_head(self.decode_head_t, F_t)
        out = P_t
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    # 用于推理的解码头前向传播
    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        #seg_logits = self.decode_head_s.forward_test(x, img_metas, self.test_cfg)
        seg_logits = self.decode_head_t.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    # 虚拟前向传播函数
    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    # 构建backbone的前向传播
    def forward_backbone(self, backbone, img):
        F_b = backbone(img)
        return F_b

    # 构建decode_head的前向传播
    def forward_decode_head(self, decode_head, feature):
        Pred = decode_head(feature)
        return Pred


    # 训练前向传播函数
    def forward_train(self, img, B_img):
        pass
        """Forward function for training."""

    # 获取分割损失
    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg


    # 获取KD损失
    def _get_KD_loss(self, teacher, student, pred_name, T=3):
        losses = dict()
        losses[f'loss_KD_{pred_name}'] = self.KL_loss(teacher, student, T)
        loss_KD, log_vars_KD = self._parse_losses(losses)
        return loss_KD, log_vars_KD



    # 滑动窗口推理
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

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
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    # 完整图像推理
    def whole_inference(self, img, img_meta, rescale):

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
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

    # 推理函数
    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

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

    # 简单测试
    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # 增强测试
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # MSE损失函数
    def MSE_loss(self, teacher, student):
        MSE_loss = nn.MSELoss()
        t = self.sw_softmax(teacher)
        s = self.sw_softmax(student)
        KD_loss = MSE_loss(s, t)
        return KD_loss

    # 设置网络是否需要梯度
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # softmax函数变体
    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H * W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out

    # KL散度损失函数
    @staticmethod
    def KL_loss(teacher, student, T=5):
        KL_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(student / T, dim=1),
                                                 F.softmax(teacher / T, dim=1)) * (T * T)
        return KL_loss