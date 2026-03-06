# Copyright (c) OpenMMLab. All rights reserved.
import warnings
# 导入MMCV的模型注册表
from mmcv.cnn import MODELS as MMCV_MODELS
#导入MMCV的注意力机制注册表
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
# 导入注册表和构建函数
from mmcv.utils import Registry, build_from_cfg
# 创建模型、注意力机制等的注册表
from torch import nn

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
# 将BACKBONES、NECKS、HEADS、LOSSES、SEGMENTORS设置为模型注册表的别名
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
## 判别器注册表
DISCRIMINATORS = Registry('module')
# 1定义构建不同组件的函数
#1.1 根据配置构建特征提取网络
def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

#1.2 # 根据配置构建特征融合网络
def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

#1.3 # 根据配置构建头部网络
def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

#1.4 # 根据配置构建损失函数
def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

# 根据配置构建判别器
def build_discriminator(cfg, default_args=None):
    """Build a module or modules from a list."""
    return build_module(cfg, DISCRIMINATORS, default_args)

def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    # 根据配置构建分割模型，并传入train_cfg和test_cfg
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_module(cfg, registry, default_args=None):

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.ModuleList(modules)

    return build_from_cfg(cfg, registry, default_args)