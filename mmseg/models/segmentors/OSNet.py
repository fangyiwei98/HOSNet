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
                 discriminator_s=None,
                 cross_EMA=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OSNet, self).__init__(init_cfg)
        # 如果提供了预训练权重，则将其设置为backbone的预训练权重
        if pretrained is not None:
            assert backbone_s.get('pretrained') is None, \
                'both backbone_s and segmentor set pretrained weight'
            backbone_s.pretrained = pretrained
        # 构建backbone和decode_head
        self.backbone_s = builder.build_backbone(backbone_s)
        self.decode_head_s = self._init_decode_head(decode_head_s)
        self.decode_head_t = self._init_decode_head(decode_head_t)
        # 类别数是decode_head_t的类别数量
        self.num_classes = self.decode_head_t.num_classes
        self.align_corners = self.decode_head_t.align_corners
        # 训练和测试配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 添加的判别器
        self.discriminator_s = builder.build_discriminator(discriminator_s)
        # 添加的cross_EMA(跟配置里一样)
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
        optimizer['discriminator_s'].zero_grad()

        # 设置一下组件参数为不可训练
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.discriminator_s, False)
        # 所有要发送给日志记录器的变量
        log_vars = dict()
        #需要训练的参数(需要训练的模块)
        self.set_requires_grad(self.backbone_s, True)
        self.set_requires_grad(self.decode_head_s, True)
        self.set_requires_grad(self.decode_head_t, True)

        #2.提取特征,img是源域图像,B_img是目标域图像
        F_s = self.forward_backbone(self.backbone_s, data_batch['img'])
        F_t = self.forward_backbone(self.backbone_s, data_batch['B_img'])

        # 3.输出预测
        P_s = self.forward_decode_head(self.decode_head_s, F_s)
        P_t = self.forward_decode_head(self.decode_head_t, F_t)

        # 4. 计算源域损失
        loss_seg_s, log_vars_seg_s = self._get_segmentor_loss(self.decode_head_s, P_s,data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s)
        loss_seg = loss_seg_s

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
        loss_seg = loss_seg + self.cross_EMA_training_ratio * loss_seg_t
        loss_seg.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)

        #=============第二阶段==================
        ## 训练Ds
        self.set_requires_grad(self.discriminator_s, True)
        F_s = F_s[-1]
        F_t = F_t[-1]
        # 对F_s和F_t应用softmax函数，得到其在空间维度上的概率分布
        F_s_dis_sm = self.sw_softmax(F_s)
        F_t_dis_sm = self.sw_softmax(F_t)
        F_s_dis_detach = F_s_dis_sm.detach()
        F_t_dis_detach = F_t_dis_sm.detach()
        F_s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s_dis_detach)
        F_t_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t_dis_detach)
        F_s_dis_detach_oup = resize(
            input=F_s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_t_dis_detach_oup = resize(
            input=F_t_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_s_ds, log_vars_adv_s_ds = self._get_gan_loss(self.discriminator_s, F_s_dis_detach_oup,
                                                                  'F_s_ds', 1)
        log_vars.update(log_vars_adv_s_ds)
        loss_adv_t_ds, log_vars_adv_t_ds = self._get_gan_loss(self.discriminator_s, F_t_dis_detach_oup,
                                                                  'F_t_ds', 0)
        log_vars.update(log_vars_adv_t_ds)
        loss_Ds=loss_adv_s_ds+loss_adv_t_ds

        #判别器损失
        loss_Ds.backward()
        optimizer['discriminator_s'].step()

        self.set_requires_grad(self.discriminator_s, False)

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


    # 生成伪标签
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        # 1. 计算所有类别的 softmax 概率
        pred_softmax = torch.softmax(pred, dim=1)

        # 动态获取源域的类别数，作为共享类的数量
        src_num_classes = self.decode_head_s.num_classes

        # 提取共享类的概率分布
        known_probs = pred_softmax[:, :src_num_classes, :, :]

        # 获取共享类中的最大概率和对应的初始伪标签
        max_known_prob, pseudo_label = torch.max(known_probs, dim=1)

        # ================= 开放集挖掘逻辑 =================
        # 设置开放集阈值 (超参数，建议在 0.4 到 0.6 之间微调)
        open_set_threshold = 0.45

        # 判定条件：如果像素在所有共享类上的最高概率都达不到阈值，则认为是目标域私有类
        is_open_set = max_known_prob < open_set_threshold

        # 将这些像素的伪标签强制修改为目标域特有类的索引 (例如 5)
        pseudo_label[is_open_set] = src_num_classes

        # 重新构建置信度矩阵 (pseudo_prob)
        pseudo_prob = max_known_prob.clone()
        # 对于被判定为未知类的像素，其置信度设为 (1.0 - max_known_prob)，
        # 意味着它在共享类上的概率越低，我们越确信它是私有类
        pseudo_prob[is_open_set] = 1.0 - max_known_prob[is_open_set]
        # ==================================================

        # 2. 阈值过滤与权重生成
        # 此时的 pseudo_prob 包含了共享类的高置信度和私有类的高置信度
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1

        ps_size = np.size(np.array(pseudo_label.cpu()))
        # 加上 1e-6 防止除以 0
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / (ps_size + 1e-6)

        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)

        # 3. 类别平衡策略 (保留你原有的优秀设计)
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).long() == 1
            pseudo_weight = pseudo_weight * ps_large_p_rare

            pseudo_class_weight = copy.deepcopy(pseudo_label.float())

            # 注意：确保配置文件中的 cross_EMA_pseu_cls_weight 列表长度等于目标域的总类别数 (6)
            for i in range(len(self.cross_EMA_pseu_cls_weight)):
                pseudo_class_weight[pseudo_class_weight == i] = self.cross_EMA_pseu_cls_weight[i]

            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5

        # 扩展维度以匹配损失计算的需求
        pseudo_label = pseudo_label[:, None, :, :]

        return pseudo_label, pseudo_weight

    # 使用cross_EMA生成伪标签
    def encode_decode_crossEMA(self, input=None, dev=None):
        # 提取特征
        #F_t = self.forward_backbone(self.backbone_s, input)
        F_ttea = self.forward_backbone(self.cross_EMA_backbone, input)

        # 使用decode_head_t和cross_EMA_decoder对特征进行解码
        #P_t = self.forward_decode_head(self.decode_head_t, F_t)
        P_ttea = self.forward_decode_head(self.cross_EMA_decoder, F_ttea)

        # 计算P_t2s和P_t的平均值，并调整到与输入图像相同的尺寸
        #P_EMA = (P_t + P_ttea) / 2
        P_EMA = P_ttea
        P_EMA_KD=P_EMA.detach()
        P_EMA = resize(
            input=P_EMA,
            size=input.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        #3. pseudo label generation
        P_EMA_detach = P_EMA.detach()
        pseudo_label,pseudo_weight=self.pseudo_label_generation_crossEMA(P_EMA_detach, dev=dev)

        # 返回生成的伪标签和对应的权重
        return pseudo_label, pseudo_weight,P_EMA_KD



    # 添加的cross_EMA相关的函数
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

    # 更新cross_EMA
    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)

        ## 1. 更新EMA Backbone（保持不变）
        for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_s.parameters()):
            if ema_b.dim() == 0:
                ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
            else:
                ema_b.data[:] = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data

        ## 2. 更新EMA Decoder（融合源域与目标域的知识）
        assert isinstance(self.cross_EMA_decoder, nn.Module), \
            "self.cross_EMA_decoder must be a nn.Module"

        src_num_classes = self.decode_head_s.num_classes  # 源域类别数 (例如 5)
        ema_num_classes = self.cross_EMA_decoder.num_classes  # EMA类别数 (例如 6)

        # 同时获取源域和目标域解码器的参数
        src_params = list(self.decode_head_s.parameters())
        tgt_params = list(self.decode_head_t.parameters())
        ema_params = list(self.cross_EMA_decoder.parameters())

        for idx, (ema_d, src_d, tgt_d) in enumerate(zip(ema_params, src_params, tgt_params)):
            if ema_d.dim() == 0:
                ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * src_d.data
            else:
                # 判定是否为输出层的分类权重/偏置 (维度0等于类别数)
                if ema_d.shape[0] == ema_num_classes and src_d.shape[0] == src_num_classes:
                    # 1. 共享类 (0 到 src_num_classes-1)：从源域解码器吸收稳定知识
                    ema_d.data[:src_num_classes] = alpha_t * ema_d.data[:src_num_classes] + \
                                                   (1 - alpha_t) * src_d.data[:src_num_classes]
                    # 2. 目标域私有类 (src_num_classes 到 ema_num_classes-1)：从目标域解码器吸收新知识
                    ema_d.data[src_num_classes:] = alpha_t * ema_d.data[src_num_classes:] + \
                                                   (1 - alpha_t) * tgt_d.data[src_num_classes:]
                else:
                    # 对于非分类层的通用特征提取参数，依然跟从源域以保持基础语义提取能力
                    ema_d.data[:] = alpha_t * ema_d.data + (1 - alpha_t) * src_d.data

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

    # 构建判别器的前向传播
    def forward_discriminator(self, discriminator, seg_pred):
        dis_pred = discriminator(seg_pred)
        return dis_pred

    # 训练前向传播函数
    def forward_train(self, img, B_img):
        pass
        """Forward function for training."""

    # 获取分割损失
    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        '''print("计算损失时特征图形状：")
        print(pred.shape)'''
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg

    # 获取对抗损失
    def _get_gan_loss(self, discriminator, pred, domain, target_is_real):
        losses = dict()
        losses[f'loss_gan_{domain}'] = discriminator.gan_loss(pred, target_is_real)
        loss_dis, log_vars_dis = self._parse_losses(losses)
        return loss_dis, log_vars_dis

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