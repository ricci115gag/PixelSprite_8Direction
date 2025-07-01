import torch
import torch.nn as nn
import math

class AngleEmbedding(nn.Module):
    """角度嵌入层：将角度转换为高维向量"""
    def __init__(self, embed_dim=128):
        super().__init__()
        freq_bands = torch.exp(torch.linspace(0, 5, embed_dim//2)) * 0.1
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, angles):
        """
        angles: [b] 角度值（-180~180）
        return: [b, embed_dim] 嵌入向量
        """
        radians = angles * (math.pi / 180.0)  # [b]
        rad_expanded = radians.unsqueeze(1)   # [b, 1]
        freq_expanded = self.freq_bands.unsqueeze(0)  # [1, embed_dim//2]
        
        sin_values = torch.sin(rad_expanded * freq_expanded)  # [b, embed_dim//2]
        cos_values = torch.cos(rad_expanded * freq_expanded)  # [b, embed_dim//2]
        
        return torch.cat([sin_values, cos_values], dim=1)  # [b, embed_dim]