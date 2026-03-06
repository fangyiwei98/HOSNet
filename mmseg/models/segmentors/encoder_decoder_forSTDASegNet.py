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
class EncoderDecoder_forSTDASegNet(BaseSegmentor):

    def __init__(self,
                 backbone_s,
                 backbone_t,
                 decode_head_s,
                 decode_head_t,
                 discriminator_s=None,
                 discriminator_t=None,
                 dsk_neck=None,
                 cross_EMA= None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder_forSTDASegNet, self).__init__(init_cfg)
        # 如果提供了预训练权重，则将其设置为backbone的预训练权重
        if pretrained is not None:
            assert backbone_s.get('pretrained') is None, \
                'both backbone_s and segmentor set pretrained weight'
            assert backbone_t.get('pretrained') is None, \
                'both backbone_t and segmentor set pretrained weight'
            backbone_s.pretrained = pretrained
            backbone_t.pretrained = pretrained
        # 构建backbone和decode_head
        self.backbone_s = builder.build_backbone(backbone_s)
        self.backbone_t = builder.build_backbone(backbone_t)
        self.decode_head_s = self._init_decode_head(decode_head_s)
        self.decode_head_t = self._init_decode_head(decode_head_t)
        # 确保decode_head_s和decode_head_t的类别数量相同
        self.num_classes = self.decode_head_t.num_classes
        self.align_corners = self.decode_head_t.align_corners
        assert self.decode_head_s.num_classes == self.decode_head_t.num_classes, \
                'both decode_head_s and decode_head_t must have same num_classes'
        # 训练和测试配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 添加的判别器
        self.discriminator_s = builder.build_discriminator(discriminator_s)
        self.discriminator_t = builder.build_discriminator(discriminator_t)
        # 添加的dsk_neck(功能是融合源域和目标域特征)
        self.dsk_neck = builder.build_neck(dsk_neck)

        # 添加的cross_EMA
        if cross_EMA is not None:
            self.cross_EMA = cross_EMA
            self._init_cross_EMA(self.cross_EMA)
        self._parse_train_cfg()
    
    # 添加的cross_EMA相关的函数
    def _init_cross_EMA(self, cfg):
        self.cross_EMA_type = cfg['type']
        self.cross_EMA_alpha = cfg['decay']
        self.cross_EMA_training_ratio = cfg['training_ratio']
        self.cross_EMA_pseu_cls_weight = cfg['pseudo_class_weight']
        self.cross_EMA_pseu_thre = cfg['pseudo_threshold']
        self.cross_EMA_rare_pseu_thre = cfg['pseudo_rare_threshold']
        if self.cross_EMA_type == 'single_t':
            self.cross_EMA_backbone = builder.build_backbone(cfg['backbone_EMA'])
            self.cross_EMA_decoder = self._init_decode_head(cfg['decode_head_EMA'])
        elif self.cross_EMA_type == 'decoder_only_t':
            self.cross_EMA_decoder_s = self._init_decode_head(cfg['decode_head_EMA'])
            self.cross_EMA_decoder_t = self._init_decode_head(cfg['decode_head_EMA'])
        elif self.cross_EMA_type == 'whole':
            ## DEPRECATED, too much memory cost
            pass
        else:
            ## No cross_EMA
            pass
    # 更新cross_EMA
    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)
        if self.cross_EMA_type == 'single_t':
            ## 1. update target_backbone
            for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_t.parameters()):
                ## For scalar params
                if not target_b.data.shape:
                    ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
                ## For tensor params
                else:
                    ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]

            ## 2. updata target_decoder
            for ema_d, target_d in zip(self.cross_EMA_decoder.parameters(), self.decode_head_t.parameters()):
                ## For scalar params
                if not target_d.data.shape:
                    ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
                ## For tensor params
                else:
                    ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]
        if self.cross_EMA_type == 'decoder_only_t':
            ## 1. updata EMA_source_decoder
            for ema_d_s, source_d in zip(self.cross_EMA_decoder_s.parameters(), self.decode_head_s.parameters()):
                ## For scalar params
                if not source_d.data.shape:
                    ema_d_s.data = alpha_t * ema_d_s.data + (1 - alpha_t) * source_d.data
                ## For tensor params
                else:
                    ema_d_s.data[:] = alpha_t * ema_d_s.data[:] + (1 - alpha_t) * source_d.data[:]
            ## 2. updata EMA_source_decoder
            for ema_d_t, target_d in zip(self.cross_EMA_decoder_t.parameters(), self.decode_head_t.parameters()):
                ## For scalar params
                if not target_d.data.shape:
                    ema_d_t.data = alpha_t * ema_d_t.data + (1 - alpha_t) * target_d.data
                ## For tensor params
                else:
                    ema_d_t.data[:] = alpha_t * ema_d_t.data[:] + (1 - alpha_t) * target_d.data[:]

    # 生成伪标签
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        #1. 生成基本的伪标签
        pred_softmax = torch.softmax(pred, dim=1)
        #找到概率分布中最大值对应的类别索引，即伪标签
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        # 判断哪些伪标签的概率大于或等于设定的阈值，并转换为长整型张量
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1
        # 计算伪标签的总数量
        ps_size = np.size(np.array(pseudo_label.cpu()))
        # 计算大于或等于阈值的伪标签的比例
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        # 根据比例生成权重张量，所有元素初始化为权重比例值
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)
        #2. 应用类别平衡策略
        #2.1 如果设置了类别权重和稀有类别阈值
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

    # 使用cross_EMA进行编码和解码
    def encode_decode_crossEMA(self, input=None, F_t2s=None, F_t2t=None, dev=None):
        """
        提出了“Decoder-only”和“Single-target”两种自训练范式，通过生成目标图像的伪标签来平衡源域和目标域之间的表示倾向。
            根据指定的cross_EMA_type，对输入图像进行编码和解码，并生成伪标签。

            :param input: 输入的图像数据
            :param F_t2s: 从学生模型（backbone_s）中提取的特征，如果为None，则会在函数内部计算
            :param F_t2t: 从教师模型（cross_EMA_backbone）中提取的特征，如果为None，则会在函数内部计算
            :param dev: 设备参数，可能用于伪标签生成
            :return: 返回生成的伪标签和对应的权重
            """
        # 如果cross_EMA_type为'single_t'，则同时推断cross_EMA_teacher（包括cross_EMA_backbone和cross_EMA_decoder）
        if self.cross_EMA_type == 'single_t':
            """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
            ## 1. forward backbone
            # 使用backbone_s和cross_EMA_backbone对输入图像进行编码
            if isinstance(self.dsk_neck.in_channels, int):
                F_t2s = self.forward_backbone(self.backbone_s, input)[-1]
                F_t2t = self.forward_backbone(self.cross_EMA_backbone, input)[-1]
            else:
                F_t2s = self.forward_backbone(self.backbone_s, input)
                F_t2t = self.forward_backbone(self.cross_EMA_backbone, input)  

            ## 2. forward neck
            # 使用dsk_neck对特征进行进一步处理
            F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t) 
            ## 3. forward decode_head
            # 使用decode_head_s和cross_EMA_decoder对特征进行解码
            P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
            P_t2t = self.forward_decode_head(self.cross_EMA_decoder, F_t2t_dsk)

            # 计算P_t2s和P_t2t的平均值，并调整到与输入图像相同的尺寸
            P_EMA = (P_t2s + P_t2t) / 2
            P_EMA = resize(
                input=P_EMA,
                size=input.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            ## 4. pseudo label generation
            # 生成伪标签和对应的权重
            P_EMA_detach = P_EMA.detach()
            pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(P_EMA_detach, dev)

        # 如果cross_EMA_type为'decoder_only_t'，则只推断cross_EMA_decoder_s和cross_EMA_decoder_t
        if self.cross_EMA_type == 'decoder_only_t':
            ## 1. forward decode_head
            # 使用cross_EMA_decoder_s和cross_EMA_decoder_t对特征进行解码
            P_t2s = self.forward_decode_head(self.cross_EMA_decoder_s, F_t2s)
            P_t2t = self.forward_decode_head(self.cross_EMA_decoder_t, F_t2t)
            # 计算P_t2s和P_t2t的平均值，并调整到与输入图像相同的尺寸
            P_EMA = (P_t2s + P_t2t) / 2
            P_EMA = resize(
                input=P_EMA,
                size=input.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ## 2. pseudo label generation
            # 生成伪标签和对应的权重
            P_EMA_detach = P_EMA.detach()
            pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(P_EMA_detach, dev)

        # 返回生成的伪标签和对应的权重
        return pseudo_label, pseudo_weight
    
    ## CODE for cross_EMA
    ##############################
    # 解析训练配置
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

    # 初始化decode_head
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = builder.build_head(decode_head)
        return decode_head

    # 特征提取函数
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone_s(img)
        return x

    # 编码和解码函数
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ## 1. forward backbone
        if isinstance(self.dsk_neck.in_channels, int):
            F_t2s = self.forward_backbone(self.backbone_s, img)[-1]
            F_t2t = self.forward_backbone(self.backbone_t, img)[-1]
        else:
            F_t2s = self.forward_backbone(self.backbone_s, img)
            F_t2t = self.forward_backbone(self.backbone_t, img)
        ## 2. forward neck
        F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)
        
        ## 3. forward decode_head
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
        P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)
        out = (P_t2s + P_t2t) / 2
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
        seg_logits = self.decode_head_s.forward_test(x, img_metas, self.test_cfg)
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

    # 训练步骤函数
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        The whole process including back propagation and 
        optimizer updating is also defined in this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # dirty walkround for not providing running status
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration 
        
        ## CODE for cross_EMA
        if curr_iter > 0:
            self._update_cross_EMA(curr_iter)

        ## 1.初始化需要优化的参数和变量
        optimizer['backbone_s'].zero_grad()
        optimizer['backbone_t'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_t'].zero_grad()
        optimizer['discriminator_s'].zero_grad()
        optimizer['discriminator_t'].zero_grad()
        optimizer['dsk_neck'].zero_grad()
        #设置一下组件参数为不可训练
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.discriminator_s, False)
        self.set_requires_grad(self.discriminator_t, False)
        self.set_requires_grad(self.dsk_neck, False)
        #所有要发送给日志记录器的变量
        log_vars = dict()

        ## 2.提取4种风格的特征
        self.set_requires_grad(self.backbone_s, True)
        self.set_requires_grad(self.decode_head_s, True)
        self.set_requires_grad(self.backbone_t, True)
        self.set_requires_grad(self.decode_head_t, True)
        #img是源域图像,B_img是目标域图像
        F_s2s_all = self.forward_backbone(self.backbone_s, data_batch['img'])  #源域特征且源域风格
        F_t2s_all = self.forward_backbone(self.backbone_s, data_batch['B_img'])  #目标域特征但是源域风格
        F_s2t_all = self.forward_backbone(self.backbone_t, data_batch['img'])   #源域特征但目标域风格
        F_t2t_all = self.forward_backbone(self.backbone_t, data_batch['B_img'])   #目标域特征且目标域风格
        #根据self.dsk_neck.in_channels的类型来决定是使用特征列表的最后一层特征还是使用整个特征列表。如果self.dsk_neck.in_channels是整数，则只取最后一层特征；否则，使用所有层的特征。
        if isinstance(self.dsk_neck.in_channels, int):
            F_s2s = F_s2s_all[-1]
            F_t2s = F_t2s_all[-1]
            F_s2t = F_s2t_all[-1]
            F_t2t = F_t2t_all[-1]
        else:
            F_s2s = F_s2s_all
            F_t2s = F_t2s_all
            F_s2t = F_s2t_all
            F_t2t = F_t2t_all

        ## 3.DSM模块(用于融合源域和目标域特征,输入的是两个特征图，输出的是融合后的特征图),这里实际上是把源域特征和目标域特征分开合并(不管是哪种风格的)
        self.set_requires_grad(self.dsk_neck, True)
        F_s2s_dsk, F_s2t_dsk = self.dsk_neck(F_s2s, F_s2t)
        F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)

        #4.对融合后的图像进行预测(source and target decoder)
        P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s_dsk)
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
        P_s2t = self.forward_decode_head(self.decode_head_t, F_s2t_dsk)
        P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)

        #5. 计算源域损失, 这边计算的是s2s和s2t的分割损失!!!!
        loss_seg_s2s, log_vars_seg_s2s = self._get_segmentor_loss(self.decode_head_s, P_s2s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s2s)
        loss_seg_s2t, log_vars_seg_s2t = self._get_segmentor_loss(self.decode_head_t, P_s2t, data_batch['gt_semantic_seg'])
        log_vars_seg_s2t['loss_ce_seg_s2t'] = log_vars_seg_s2t.pop('loss_ce')
        log_vars_seg_s2t['acc_seg_s2t'] = log_vars_seg_s2t.pop('acc_seg')
        log_vars_seg_s2t['loss_ce_seg_s2t'] = log_vars_seg_s2t.pop('loss')
        log_vars.update(log_vars_seg_s2t)
        loss_seg = loss_seg_s2s + loss_seg_s2t

        #6.计算目标域损失, 这边计算的是t2s和t2t的分割损失!!!!
        #6.1使用目标域图像（B_img）和特征图（F_t2s_dsk, F_t2t_dsk）来生成伪标签（pseudo_label）和伪标签权重（pseudo_weight）,这里用到了Decoder-only和Single-target两种自监督学习方式
        pseudo_label, pseudo_weight = self.encode_decode_crossEMA(input=data_batch['B_img'], F_t2s=F_t2s_dsk, F_t2t=F_t2t_dsk, dev=data_batch['img'].device)
        #6.2使用源域的解码头和伪标签来计算针对P_t2s的分割损失
        loss_seg_t2s, log_vars_seg_t2s = self._get_segmentor_loss(self.decode_head_s, P_t2s, pseudo_label, gt_weight=pseudo_weight)
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss_ce')
        log_vars_seg_t2s['acc_seg_t2s'] = log_vars_seg_t2s.pop('acc_seg')
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss')
        log_vars.update(log_vars_seg_t2s)
        #6.3使用目标域的解码头和相同的伪标签来计算针对P_t2t的分割损失
        loss_seg_t2t, log_vars_seg_t2t = self._get_segmentor_loss(self.decode_head_t, P_t2t, pseudo_label, gt_weight=pseudo_weight)
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss_ce')
        log_vars_seg_t2t['acc_seg_t2t'] = log_vars_seg_t2t.pop('acc_seg')
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss')
        log_vars.update(log_vars_seg_t2t)
        # 计算总的分割损失loss_seg，它是之前计算的损失（loss_seg）加上基于伪标签计算的两个损失（loss_seg_t2s和loss_seg_t2t）的加权和
        # 其中，self.cross_EMA_training_ratio是一个超参数，用于控制伪标签损失在总损失中的权重
        loss_seg = loss_seg + self.cross_EMA_training_ratio * (loss_seg_t2s + loss_seg_t2t)


        #7.训练判别器Ds和Dt，用来区分跨域单风格cross-domain single-style(采用对抗学习)
        #7.1通过F_s2s和F_t2s训练Ds
        F_s2s = F_s2s_all[-1]
        F_t2s = F_t2s_all[-1]
        #对F_s2s和F_t2s应用softmax函数，得到其在空间维度上的概率分布
        F_s2s_dis_sm = self.sw_softmax(F_s2s)
        F_t2s_dis_sm = self.sw_softmax(F_t2s)
        #通过判别器Ds对F_s2s和F_t2s的概率分布进行前向传播，得到判别器的输出，并恢复到原始尺寸
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # 计算判别器Ds的损失，并获取相关的日志变量
        loss_dis_s, log_vars_dis_s = self._get_gan_loss(self.discriminator_s, F_t2s_dis_oup, 'F_t2s_ds_seg', 1)
        log_vars.update(log_vars_dis_s)
        #7.2通过F_s2t和F_t2t训练Dt
        F_s2t = F_s2t_all[-1]
        F_t2t = F_t2t_all[-1]
        F_s2t_dis_sm = self.sw_softmax(F_s2t)
        F_t2t_dis_sm = self.sw_softmax(F_t2t)
        # 通过判别器Dt对F_s2t和F_t2t的概率分布进行前向传播，得到判别器的输出
        F_s2t_dis_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_sm)
        F_s2t_dis_oup = resize(
            input=F_s2t_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # 计算判别器Dt的损失，并获取相关的日志变量
        loss_dis_t, log_vars_dis_t = self._get_gan_loss(self.discriminator_t, F_s2t_dis_oup, 'F_s2t_dt_seg', 1)
        log_vars.update(log_vars_dis_t)
        # 计算总的对抗损失（Dt和Ds的损失之和）
        loss_adv = loss_dis_t + loss_dis_s

        # 计算第一阶段总损失（分割损失与对抗损失之和）
        loss_stage1 = loss_seg + loss_adv
        loss_stage1.backward()
        optimizer['backbone_s'].step()
        optimizer['backbone_t'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        optimizer['dsk_neck'].step()
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.dsk_neck, False)

        ## 再次训练Ds,用来区分跨域单风格特征(这里是F_s2s和F_t2s都用到了)
        self.set_requires_grad(self.discriminator_s, True)
        #分离出F_s2s_dis_sm和F_t2s_dis_sm，这样可以防止在计算Ds的损失时影响到生成器的参数
        F_s2s_dis_detach = F_s2s_dis_sm.detach()
        F_t2s_dis_detach = F_t2s_dis_sm.detach()
        F_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_detach)
        F_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_detach)
        F_t2s_dis_detach_oup = resize(
            input=F_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_detach_oup = resize(
            input=F_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_s, F_s2s_dis_detach_oup, 'F_s2s_ds', 1)
        loss_adv_s2s_ds.backward()
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_s, F_t2s_dis_detach_oup, 'F_t2s_ds', 0)
        loss_adv_t2s_ds.backward()
        log_vars.update(log_vars_adv_t2s_ds)
        optimizer['discriminator_s'].step()
        # 设置判别器Ds的参数为不可训练
        self.set_requires_grad(self.discriminator_s, False)

        ## 再次训练Dt(同上)
        self.set_requires_grad(self.discriminator_t, True)
        F_s2t_dis_detach = F_s2t_dis_sm.detach()
        F_t2t_dis_detach = F_t2t_dis_sm.detach()
        F_s2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach)
        F_t2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach)
        F_t2t_dis_detach_oup = resize(
            input=F_t2t_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup = resize(
            input=F_s2t_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt, log_vars_adv_t2t_dt = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup, 'F_t2t_dt', 1)
        loss_adv_t2t_dt.backward()
        log_vars.update(log_vars_adv_t2t_dt)
        loss_adv_s2t_dt, log_vars_adv_s2t_dt = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup, 'F_s2t_dt', 0)
        loss_adv_s2t_dt.backward()
        log_vars.update(log_vars_adv_s2t_dt)
        optimizer['discriminator_t'].step()
        self.set_requires_grad(self.discriminator_t, False)

        # 将当前计算的分割损失值赋给loss变量
        loss = loss_seg
        # 检查self对象是否具有'iteration'属性
        # 如果有，说明这是一个迭代过程，需要对迭代次数进行更新
        if hasattr(self, 'iteration'):
            # 迭代次数加1，表示完成了一次迭代过程
            self.iteration += 1

        # 创建一个字典，用于存储本次迭代的相关信息
        outputs = dict(
            # 损失值，这是训练过程中需要关注的重要指标
            loss=loss,
            # 日志变量，可能包含其他有助于了解训练过程的指标或数据
            log_vars=log_vars,
            # 当前批次数据的样本数量，通常用于计算平均损失等指标
            num_samples=len(data_batch['img_metas'])
        )

        # 返回封装好的信息字典
        return outputs


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
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

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
        pred_sh = torch.reshape(pred, (N, C, H*W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out

    # KL散度损失函数
    @staticmethod
    def KL_loss(teacher, student, T=5):
        KL_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(student/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        return KL_loss