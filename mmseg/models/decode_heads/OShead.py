# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn import normal_init

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class OSHead(ASPPHead):
    """DeepLabV3+ head."""

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(OSHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward_feature(self, inputs):
        """Return segmentation feature before classifier."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self.forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class DecoupledOSHead(ASPPHead):
    """
    Decoupled DeepLabV3+ head for target branch.

    - shared trunk
    - known classifier
    - unknown classifier
    - optional detach before unknown branch
    """

    def __init__(self,
                 c1_in_channels,
                 c1_channels,
                 num_known_classes,
                 num_unknown_classes,
                 detach_unknown_from_trunk=True,
                 **kwargs):
        super(DecoupledOSHead, self).__init__(
            num_classes=num_known_classes, **kwargs)

        assert c1_in_channels >= 0

        self.num_known_classes = num_known_classes
        self.num_unknown_classes = num_unknown_classes
        self.detach_unknown_from_trunk = detach_unknown_from_trunk

        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        # 删除父类默认的 conv_seg，并替换成 known/unknown 两个分类头
        if hasattr(self, 'conv_seg'):
            del self.conv_seg

        self.conv_seg_known = nn.Conv2d(
            self.channels, self.num_known_classes, kernel_size=1)

        self.conv_seg_unknown = nn.Conv2d(
            self.channels, self.num_unknown_classes, kernel_size=1)

        # 关键修复：
        # BaseDecodeHead / ASPPHead 默认 init_cfg 会试图初始化 conv_seg
        # 这里手动清掉，避免 mmcv 再找不存在的 conv_seg
        self.init_cfg = None

    def init_weights(self):
        """Initialize weights for known / unknown classifiers."""
        super(DecoupledOSHead, self).init_weights()
        normal_init(self.conv_seg_known, mean=0, std=0.01)
        normal_init(self.conv_seg_unknown, mean=0, std=0.01)

    def forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        return output

    def cls_seg_known(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg_known(feat)

    def cls_seg_unknown(self, feat):
        if self.detach_unknown_from_trunk:
            feat = feat.detach()
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg_unknown(feat)

    def forward(self, inputs):
        """Default forward returns known logits only."""
        feat = self.forward_feature(inputs)
        return self.cls_seg_known(feat)

    def forward_decoupled(self, inputs):
        """Return shared feat, known logits and unknown logits."""
        feat = self.forward_feature(inputs)
        known_logits = self.cls_seg_known(feat)
        unknown_logits = self.cls_seg_unknown(feat)
        return feat, known_logits, unknown_logits

    def forward_test(self, inputs, img_metas, test_cfg):
        """
        For compatibility with mmseg test api.
        Return concatenated known+unknown logits so output num_classes matches target classes.
        """
        feat, known_logits, unknown_logits = self.forward_decoupled(inputs)
        return torch.cat([known_logits, unknown_logits], dim=1)