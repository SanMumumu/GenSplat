from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.anysplat import EncoderAnySplatCfg
from ..encoder.gensplat import EncoderGenSplat, EncoderGenSplatCfg
from ..encoder.gensplatpi3 import EncoderGenSplatCfg as EncoderGenSplatPi3Cfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .gensplat import GenSplat

MODELS = {
    "anysplat": GenSplat,
    "gensplat": GenSplat,
    "gensplatpi3": GenSplat,
}

EncoderCfg = Union[EncoderAnySplatCfg, EncoderGenSplatCfg, EncoderGenSplatPi3Cfg]
DecoderCfg = DecoderSplattingCUDACfg


def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS[encoder_cfg.name](encoder_cfg, decoder_cfg)
    return model
