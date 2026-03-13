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
        # 确保decode_head_s和decode_head_t的类别数量相同
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

        # ===================== 新增：OpenMax相关初始化 =====================
        # OpenMax超参数（可在配置文件中配置，这里默认值）
        self.openmax_alpha = cross_EMA.get('openmax_alpha', 0.5) if cross_EMA else 0.5
        self.unknown_threshold = cross_EMA.get('unknown_threshold', 0.7) if cross_EMA else 0.7
        # ===================== 对比学习相关初始化 =====================
        self.contrast_temperature = cross_EMA.get('contrast_temperature', 0.1) if cross_EMA else 0.1
        self.contrast_weight = cross_EMA.get('contrast_weight', 0.05) if cross_EMA else 0.05

    # 训练函数
    def train_step(self, data_batch, optimizer, **kwargs):

        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration

        # 通过EMA更新教师网络参数
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
        # 需要训练的参数(需要训练的模块)
        self.set_requires_grad(self.backbone_s, True)
        self.set_requires_grad(self.decode_head_s, True)
        self.set_requires_grad(self.decode_head_t, True)

        # 2.提取4种风格的特征,img是源域图像,B_img是目标域图像
        F_s2s = self.forward_backbone(self.backbone_s, data_batch['img'])  # 源域特征且源域风格
        F_t2s = self.forward_backbone(self.backbone_s, data_batch['B_img'])  # 目标域特征但是源域风格

        # ===================== 新增：计算跨域对比损失 =====================
        # 用源域真实标签作为对比学习的类别依据
        contrast_loss = self.info_nce_loss(
            query=F_s2s[-1],  # 源域最后一层特征作为query
            key=F_t2s[-1],  # 目标域最后一层特征作为key
            labels=data_batch['gt_semantic_seg'],  # 源域真实标签
            temperature=self.contrast_temperature
        )
        log_vars['loss_contrast'] = contrast_loss.item()  # 记录对比损失到日志
        # ===================== 对比损失结束 =====================

        # 3.对融合后的图像进行预测(source and target decoder)
        P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s)
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s)

        # 4. 计算源域损失
        loss_seg_s2s, log_vars_seg_s2s = self._get_segmentor_loss(self.decode_head_s, P_s2s,
                                                                  data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s2s)
        loss_seg = loss_seg_s2s

        # 5.计算目标域损失, 这边计算的是t2s和t2t的分割损失!!!!
        # 5.1使用目标域图像（B_img）和特征图（F_t2s_dsk, F_t2t_dsk）来生成伪标签（pseudo_label）和伪标签权重（pseudo_weight）
        pseudo_label, pseudo_weight, P_EMA_detach = self.encode_decode_crossEMA(input=data_batch['B_img'],
                                                                                dev=data_batch['img'].device)
        # 5.2使用源域的解码头和伪标签来计算针对P_t2s的分割损失
        loss_seg_t2s, log_vars_seg_t2s = self._get_segmentor_loss(self.decode_head_s, P_t2s, pseudo_label,
                                                                  gt_weight=pseudo_weight)
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss_ce')
        log_vars_seg_t2s['acc_seg_t2s'] = log_vars_seg_t2s.pop('acc_seg')
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss')
        log_vars.update(log_vars_seg_t2s)
        # 5.3使用目标域的解码头和相同的伪标签来计算针对P_t2s的分割损失
        loss_seg_t2t, log_vars_seg_t2t = self._get_segmentor_loss(self.decode_head_t, P_t2s, pseudo_label,
                                                                  gt_weight=pseudo_weight)
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss_ce')
        log_vars_seg_t2t['acc_seg_t2t'] = log_vars_seg_t2t.pop('acc_seg')
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss')
        log_vars.update(log_vars_seg_t2t)
        # 计算总的分割损失loss_seg，它是之前计算的损失（loss_seg）加上基于伪标签计算的两个损失（loss_seg_t2s和loss_seg_t2t）的加权和
        # 其中，self.cross_EMA_training_ratio是一个超参数，用于控制伪标签损失在总损失中的权重
        loss_seg = loss_seg + self.cross_EMA_training_ratio * (loss_seg_t2s + loss_seg_t2t)

        # ===================== 新增：整合对比损失到总分割损失 =====================
        loss_seg = loss_seg + self.contrast_weight * contrast_loss
        # ===================== 对比损失整合结束 =====================

        # 7.训练判别器Ds和Dt，用来区分跨域单风格cross-domain single-style(采用对抗学习)
        # 7.1通过F_s2s和F_t2s训练Ds
        F_s2s = F_s2s[-1]
        F_t2s = F_t2s[-1]
        # 对F_s2s和F_t2s应用softmax函数，得到其在空间维度上的概率分布
        F_s2s_dis_sm = self.sw_softmax(F_s2s)
        F_t2s_dis_sm = self.sw_softmax(F_t2s)
        # 通过判别器Ds对F_s2s和F_t2s的概率分布进行前向传播，得到判别器的输出，并恢复到原始尺寸
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # 计算判别器Ds的损失，并获取相关的日志变量
        loss_dis_s, log_vars_dis_s = self._get_gan_loss(self.discriminator_s, F_t2s_dis_oup, 'F_t2s_ds_seg', 1)
        log_vars.update(log_vars_dis_s)

        # 计算总的对抗损失
        loss_adv = loss_dis_s

        loss_stage1 = loss_seg + loss_adv
        loss_stage1.backward()

        optimizer['backbone_s'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)

        # =============第二阶段==================
        ## 训练Ds
        self.set_requires_grad(self.discriminator_s, True)
        # 分离出F_s2s_dis_sm和F_t2s_dis_sm，这样可以防止在计算Ds的损失时影响到生成器的参数
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
        loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_s, F_s2s_dis_detach_oup,
                                                                  'F_s2s_ds', 1)
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_s, F_t2s_dis_detach_oup,
                                                                  'F_t2s_ds', 0)
        log_vars.update(log_vars_adv_t2s_ds)
        loss_Ds = loss_adv_s2s_ds + loss_adv_t2s_ds

        # 总的损失
        loss_adv_D = loss_Ds
        loss_adv_D.backward()
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


    def update_pseudo_labels(self, initial_pseudo_probs, modulation_weights):
        # 假设initial_pseudo_probs是初始的软伪标签概率分布，形状为[N, C, H, W]
        # 使用调制权重更新伪标签概率分布（这里简化为直接使用权重作为更新后的伪标签）
        # updated_pseudo_probs = modulation_weights * initial_pseudo_probs  # 你可以根据需要调整这个更新逻辑
        # 将更新的伪标签概率分布转换为硬伪标签（如果需要）
        # hard_pseudo_labels = modulation_weights.argmax(dim=1)
        hard_pseudo_labels = torch.max(modulation_weights, dim=1)
        return hard_pseudo_labels, modulation_weights

    # ===================== 新增：OpenMax分类器函数 =====================
    def openmax_classifier(self, pred, alpha=None, unknown_threshold=None):
        """
        OpenMax分类器：计算已知类置信度，低于阈值判定为未知类
        Args:
            pred: 分类器输出logits [N, C, H, W]
            alpha: 未知类概率分配系数（默认使用初始化的self.openmax_alpha）
            unknown_threshold: 已知类置信度阈值（默认使用self.unknown_threshold）
        Returns:
            openmax_prob: 包含未知类的概率分布 [N, C+1, H, W]
            is_known: 已知类掩码 [N, 1, H, W]（True为已知类）
        """
        alpha = alpha if alpha is not None else self.openmax_alpha
        unknown_threshold = unknown_threshold if unknown_threshold is not None else self.unknown_threshold

        N, C, H, W = pred.shape
        # 1. 计算原始Softmax概率
        pred_flat = pred.reshape(N, C, H * W)
        softmax_prob = F.softmax(pred_flat, dim=1).reshape(N, C, H, W)

        # 2. 计算每个像素的已知类最大置信度
        max_known_prob, _ = torch.max(softmax_prob, dim=1, keepdim=True)
        is_known = max_known_prob >= unknown_threshold  # 已知类掩码（bool）

        # 3. 分配未知类概率（OpenMax核心）
        unknown_prob = alpha * (1 - max_known_prob)  # 未知类概率
        openmax_prob = torch.cat([softmax_prob * (1 - alpha), unknown_prob], dim=1)  # C+1维（最后一维是未知类）

        return openmax_prob, is_known

    # ===================== OpenMax函数结束 =====================

    # 生成伪标签（修改版：加入OpenMax过滤未知类）
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        # ===================== 修改：加入OpenMax区分已知/未知类 =====================
        # 用OpenMax计算已知类掩码
        openmax_prob, is_known = self.openmax_classifier(pred)
        pred_softmax = openmax_prob[:, :-1, :, :]  # 去掉未知类维度，保留已知类
        # ===================== OpenMax过滤结束 =====================

        # 找到概率分布中最大值对应的类别索引，即伪标签
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        # 判断哪些伪标签的概率大于或等于设定的阈值，并转换为长整型张量
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1

        # ===================== 修改：叠加未知类掩码，仅保留已知类区域的有效伪标签 =====================
        ps_large_p = ps_large_p & is_known.squeeze(1)  # 只保留已知类+高置信度区域
        # ===================== 掩码叠加结束 =====================

        # 计算伪标签的总数量（避免除0）
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / (ps_size + 1e-8)  # 防止除0
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

        # ===================== 修改：未知类区域权重置0 =====================
        pseudo_weight = pseudo_weight * is_known.squeeze(1).float()
        # ===================== 权重过滤结束 =====================

        # 将伪标签张量扩展一个维度，以便与某些模型或操作兼容
        pseudo_label = pseudo_label[:, None, :, :]
        # 返回生成的伪标签和权重
        return pseudo_label, pseudo_weight

    # ===================== 新增：InfoNCE对比损失函数 =====================
    def info_nce_loss(self, query, key, labels, temperature=0.1):
        """
        跨域对比损失（InfoNCE）：彻底修复标签与特征图尺寸不匹配问题
        Args:
            query: 源域特征 [N, C, H_feat, W_feat]
            key: 目标域特征 [N, C, H_feat, W_feat]
            labels: 真实/伪标签 [N, 1, H_img, W_img]（图像尺寸）
            temperature: 温度系数（越小对比越严格）
        Returns:
            loss: InfoNCE损失值
        """
        # ========== 1. 调试打印（可选，确认尺寸） ==========
        N, C, H_feat, W_feat = query.shape
        #print(f"特征图尺寸: H_feat={H_feat}, W_feat={W_feat}")
        #print(f"原始标签尺寸: {labels.shape}")

        # ========== 2. 核心修复：标签严格下采样到特征图尺寸 ==========
        # 标签是离散类别，必须用nearest插值（避免类别值被修改）
        # 步骤1：将标签从 [N, 1, H_img, W_img] 插值到 [N, 1, H_feat, W_feat]
        labels_resized = F.interpolate(
            input=labels.float(),  # 临时转float以支持插值
            size=(H_feat, W_feat),  # 强制匹配特征图的高宽
            mode='nearest',  # 离散类别必须用nearest
            align_corners=None  # nearest模式下必须设为None（PyTorch要求）
        ).long()  # 转回long类型（类别标签）
        #print(f"下采样后标签尺寸: {labels_resized.shape}")  # 应输出 [N, 1, H_feat, W_feat]

        # ========== 3. 维度校验（提前报错，避免后续索引错误） ==========
        assert labels_resized.shape[2:] == query.shape[2:], \
            f"标签下采样后尺寸{labels_resized.shape[2:]}与特征图尺寸{query.shape[2:]}不匹配！"

        # ========== 4. 特征归一化（对比学习必备） ==========
        query = F.normalize(query, dim=1)  # [N, C, H_feat, W_feat]
        key = F.normalize(key, dim=1)  # [N, C, H_feat, W_feat]

        # ========== 5. 展平特征和标签（现在维度100%匹配） ==========
        # 特征展平：[N, C, H, W] → [N*H*W, C]
        query_flat = query.permute(0, 2, 3, 1).reshape(-1, C)  # 长度=N*H_feat*W_feat
        key_flat = key.permute(0, 2, 3, 1).reshape(-1, C)  # 长度=N*H_feat*W_feat
        # 标签展平：[N, 1, H, W] → [N*H*W]
        labels_flat = labels_resized.reshape(-1)  # 长度=N*H_feat*W_feat
        #print(f"特征展平长度: {len(query_flat)}, 标签展平长度: {len(labels_flat)}")  # 应相等

        # ========== 6. 过滤未知类样本 ==========
        mask = labels_flat < self.num_classes
        query_flat = query_flat[mask]
        key_flat = key_flat[mask]
        labels_flat = labels_flat[mask]

        # 边界情况：无已知类样本时返回0损失
        if len(query_flat) == 0:
            return torch.tensor(0.0, device=query.device)

        # ========== 7. InfoNCE损失计算（数值稳定版） ==========
        # 计算相似度矩阵 [N', N']（N'是过滤后的已知类像素数）
        sim_matrix = torch.matmul(query_flat, key_flat.T) / temperature
        # 减去最大值防止指数爆炸（数值稳定性）
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

        # 构建正负样本掩码
        pos_mask = (labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)).float()
        neg_mask = 1 - pos_mask

        # 计算InfoNCE损失
        exp_sim = torch.exp(sim_matrix) * neg_mask
        sum_exp = exp_sim.sum(dim=1, keepdim=True) + 1e-8  # 防止除0
        pos_sim_sum = (sim_matrix * pos_mask).sum(dim=1) + 1e-8

        loss = -torch.log(pos_sim_sum / sum_exp).mean()
        return loss

    # ===================== InfoNCE函数结束 =====================

    # 使用cross_EMA生成伪标签
    def encode_decode_crossEMA(self, input=None, dev=None):
        # 1. forward backbone
        #F_t2s = self.forward_backbone(self.backbone_s, input)
        F_t2t = self.forward_backbone(self.cross_EMA_backbone, input)

        ## 2. forward decode_head
        # 使用decode_head_s和cross_EMA_decoder对特征进行解码
        #P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s)
        P_t2t = self.forward_decode_head(self.cross_EMA_decoder, F_t2t)

        # 计算P_t2s和P_t2t的平均值，并调整到与输入图像相同的尺寸
        #P_EMA = (P_t2s + P_t2t) / 2
        P_EMA = P_t2t
        P_EMA_KD = P_EMA.detach()
        P_EMA = resize(
            input=P_EMA,
            size=input.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # 3. pseudo label generation
        P_EMA_detach = P_EMA.detach()
        pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(P_EMA_detach, dev=dev)

        # 返回生成的伪标签和对应的权重
        return pseudo_label, pseudo_weight, P_EMA_KD

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
        ## 1. update target_backbone
        for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_s.parameters()):
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

    # 编码和解码函数
    def encode_decode(self, img, img_metas):
        ## 1. forward backbone
        F_t2s = self.forward_backbone(self.backbone_s, img)

        ## 2. forward decode_head
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s)
        out = P_t2s
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