# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder

from .encoder_decoder_forSTDASegNet import EncoderDecoder_forSTDASegNet

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'EncoderDecoder_forSTDASegNet', 'MyNet', 'OSNet', 'OSNetV1']
