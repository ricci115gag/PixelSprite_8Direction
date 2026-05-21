from __future__ import annotations

import torch
import torch.nn as nn

from pixel_sprite.model.blocks import ResBlock
from pixel_sprite.model.network import (
    HorizontalAxisAttention,
    VerticalAxisAttention,
    GlobalAttention,
)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        hidden_channels = base_channels * 4  # 128

        # Encoder Level 0
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
        )

        # Encoder Level 1 (Downsampling by 2x)
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GELU(),
        )

        # Bottleneck (Operating at 16x24, channels = 128)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.bottleneck_res_pre = nn.ModuleList(
            [ResBlock(hidden_channels) for _ in range(2)]
        )

        self.horizontal_attn = HorizontalAxisAttention(hidden_channels, 4, base_channels)
        self.vertical_attn = VerticalAxisAttention(hidden_channels, 4, base_channels)
        self.global_attn = GlobalAttention(hidden_channels, 4, base_channels)

        self.bottleneck_res_post = nn.ModuleList(
            [ResBlock(hidden_channels) for _ in range(2)]
        )

        # Decoder Level 1 (Concatenate Bottleneck and Enc1 features)
        # Input shape: (B, 128, 16, 24) concat (B, 64, 16, 24) -> (B, 192, 16, 24)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(hidden_channels + base_channels * 2, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        # Upsampling Level 1 to Level 0 (16x24 to 32x48)
        # Input: 128 channels. Output of PixelShuffle(2): 128 / 4 = 32 channels.
        self.upsample = nn.PixelShuffle(2)

        # Decoder Level 0 (Concatenate Upsampled and Enc0 features)
        # Input shape: (B, 32, 32, 48) concat (B, 32, 32, 48) -> (B, 64, 32, 48)
        self.dec0_conv = nn.Sequential(
            nn.Conv2d(base_channels + base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 1),
        )

    def forward(self, source_image: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_enc0 = self.enc0(source_image)           # (B, 32, 32, 48)
        x_enc1 = self.enc1(x_enc0)                 # (B, 64, 16, 24)

        # Bottleneck
        x = self.bottleneck_conv(x_enc1)           # (B, 128, 16, 24)
        for block in self.bottleneck_res_pre:
            x = block(x)

        x = x + self.horizontal_attn(x) + self.vertical_attn(x) + self.global_attn(x)

        for block in self.bottleneck_res_post:
            x = block(x)                           # (B, 128, 16, 24)

        # Decoder Level 1
        x_dec1 = torch.cat([x, x_enc1], dim=1)     # (B, 192, 16, 24)
        x_dec1 = self.dec1_conv(x_dec1)            # (B, 128, 16, 24)

        # Upsample to Level 0
        x_upsampled = self.upsample(x_dec1)        # (B, 32, 32, 48)

        # Decoder Level 0
        x_dec0 = torch.cat([x_upsampled, x_enc0], dim=1)  # (B, 64, 32, 48)
        output = self.dec0_conv(x_dec0)            # (B, 4, 32, 48)

        return torch.sigmoid(output)
