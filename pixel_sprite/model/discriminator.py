from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,  # 4 channels of down + 4 channels of left/fake_left
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        # Stem: Conv3x3 (in_channels=8 -> base_channels=32)
        # We apply spectral_norm to all convolutional and linear layers in D to stabilize GAN training.
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

        # Down 3: Stride=2 (128 -> 256), output resolution: 4x6
        self.down3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global average pooling to reduce spatial size (4x6 -> 1x1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Binary classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(base_channels * 8, 1)),
            nn.Sigmoid(),
        )

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

        x_pooled = self.pool(x)
        out = self.head(x_pooled)
        return out, features
