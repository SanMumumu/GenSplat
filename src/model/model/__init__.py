from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.gensplat import EncoderGenSplat, EncoderGenSplatCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .gensplat import GenSplat

MODELS = {
    "gensplat": GenSplat,
}

EncoderCfg = Union[EncoderGenSplatCfg]
DecoderCfg = DecoderSplattingCUDACfg


# hard code for now
def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS['gensplat'](encoder_cfg, decoder_cfg)
    return model
