from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,  # 4 channels of down + 4 channels of target
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        # Stem: Conv3x3 (in_channels=8 -> base_channels=32)
        # We apply spectral_norm to all convolutional layers to stabilize GAN training.
        self.stem = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Down 1: Stride=2 (32 -> 64), output resolution: 16x24
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Down 2: Stride=2 (64 -> 128), output resolution: 8x12
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Down 3: Stride=2 (128 -> 256), output resolution: 6x4
        self.down3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # PatchGAN Head: outputs a 2D map of classification decisions at 6x4 resolution
        self.final_conv = spectral_norm(nn.Conv2d(base_channels * 8, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Input shape: (B, 8, 32, 48)
        features: list[torch.Tensor] = []

        x = self.stem(x)
        features.append(x)

        x = self.down1(x)
        features.append(x)

        x = self.down2(x)
        features.append(x)

        x = self.down3(x)
        features.append(x)

        # Compute PatchGAN outputs of shape (B, 1, 4, 6)
        patch_out = torch.sigmoid(self.final_conv(x))
        return patch_out, features
