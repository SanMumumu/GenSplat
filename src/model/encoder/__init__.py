from typing import Optional, Union

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .anysplat import EncoderAnySplat, EncoderAnySplatCfg
from .gensplat import EncoderGenSplat, EncoderGenSplatCfg
from .gensplatpi3 import (
    EncoderGenSplat as EncoderGenSplatPi3,
    EncoderGenSplatCfg as EncoderGenSplatPi3Cfg,
)

ENCODERS = {
    "anysplat": (EncoderAnySplat, None),
    "gensplat": (EncoderGenSplat, None),
    "gensplatpi3": (EncoderGenSplatPi3, None),
}

EncoderCfg = Union[EncoderAnySplatCfg, EncoderGenSplatCfg, EncoderGenSplatPi3Cfg]


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
