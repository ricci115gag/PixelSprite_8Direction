from __future__ import annotations

from typing import Type
import torch.nn as nn


def get_model_class(version: int, model_type: str = "generator") -> Type[nn.Module]:
    """Factory to retrieve generator or discriminator classes based on version (1 or 2)."""
    if version == 1:
        if model_type == "generator":
            from pixel_sprite.model.GAN.generator import UNetGenerator
            return UNetGenerator
        elif model_type == "discriminator":
            from pixel_sprite.model.GAN.discriminator import Discriminator
            return Discriminator
    elif version == 2:
        if model_type == "generator":
            from pixel_sprite.model.GANv2.generatorv2 import UNetGenerator
            return UNetGenerator
        elif model_type == "discriminator":
            from pixel_sprite.model.GANv2.discriminatorv2 import Discriminator
            return Discriminator
    
    raise ValueError(f"Unsupported model version {version} or model type '{model_type}'.")
