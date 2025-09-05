from typing import Optional, Union

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .gensplat import EncoderGenSplat, EncoderGenSplatCfg

ENCODERS = {
    "gensplat": (EncoderGenSplat, None),
}

EncoderCfg = Union[EncoderGenSplatCfg]


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
