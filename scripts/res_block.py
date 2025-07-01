import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """带角度条件的残差块"""
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.angle_transform = nn.Sequential(
            nn.Linear(embed_dim, in_channels * 2),
            nn.GELU()
        )
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, in_channels//4), in_channels)
        
    def forward(self, x, angle_embed):
        """
        x: [b, in_channels, h, w] 输入特征图
        angle_embed: [b, embed_dim] 角度嵌入向量
        return: [b, in_channels, h, w] 输出特征图
        """
        identity = x  # [b, in_channels, h, w]
        
        params = self.angle_transform(angle_embed)  # [b, in_channels*2]
        scale, shift = params.chunk(2, dim=1)  # 各[b, in_channels]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [b, in_channels, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [b, in_channels, 1, 1]
        
        x = self.conv1(x)  # [b, in_channels, h, w]
        x = self.norm(x)   # [b, in_channels, h, w]
        x = x * (1 + scale) + shift  # [b, in_channels, h, w]
        x = F.gelu(x)  # [b, in_channels, h, w]
        
        x = self.conv2(x)  # [b, in_channels, h, w]
        
        return identity + x  # [b, in_channels, h, w]