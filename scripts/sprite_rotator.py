import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.angle_embed import AngleEmbedding
from scripts.res_block import ResBlock  # 正确引用ResBlock

class HorizontalAxisAttention(nn.Module):
    """水平轴向注意力 - 捕捉左右对称性"""
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h, w), qkv)  # [b, heads, c', h, w]
        
        # 水平轴注意力
        q = q.permute(0, 1, 3, 4, 2)  # [b, heads, h, w, c']
        k = k.permute(0, 1, 3, 2, 4)  # [b, heads, h, c', w]
        v = v.permute(0, 1, 3, 4, 2)  # [b, heads, h, w, c']
        
        # 计算注意力得分并应用
        attn = torch.matmul(q, k) * self.scale  # [b, heads, h, w, w]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [b, heads, h, w, c']
        
        # 重新组合
        out = out.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)  # [b, inner_dim, h, w]
        return self.to_out(out)  # [b, dim, h, w]

class VerticalAxisAttention(nn.Module):
    """垂直轴向注意力 - 捕捉上下对称性"""
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h, w), qkv)  # [b, heads, c', h, w]
        
        # 垂直轴注意力
        q = q.permute(0, 1, 4, 2, 3)  # [b, heads, w, c', h]
        k = k.permute(0, 1, 4, 3, 2)  # [b, heads, w, h, c']
        v = v.permute(0, 1, 4, 2, 3)  # [b, heads, w, c', h]
        
        # 计算注意力得分并应用
        attn = torch.matmul(q, k) * self.scale  # [b, heads, w, h, h]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [b, heads, w, c', h]
        
        # 重新组合
        out = out.permute(0, 1, 3, 4, 2).reshape(b, -1, h, w)  # [b, inner_dim, h, w]
        return self.to_out(out)  # [b, dim, h, w]

class MultiHeadAttention(nn.Module):
    """标准多头注意力 - 捕捉全局细节"""
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)  # [b, heads, c', h*w]
        
        # 计算注意力得分并应用
        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale  # [b, heads, h*w, h*w]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-1, -2))  # [b, heads, h*w, c']
        
        # 重新组合
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)  # [b, inner_dim, h, w]
        return self.to_out(out)  # [b, dim, h, w]

class SpriteRotator(nn.Module):
    """像素精灵转身网络 - 整合轴向注意力和多头注意力"""
    def __init__(self, in_channels=4, base_channels=32, embed_dim=128):
        super().__init__()
        self.angle_embedder = AngleEmbedding(embed_dim)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),  # [b, base_channels, h, w]
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # [b, base_channels*2, h/2, w/2]
            nn.GELU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),  # [b, base_channels*4, h/2, w/2]
            nn.GELU()
        )
        
        # 旋转残差块和注意力模块
        self.rotation_blocks = nn.ModuleList([
            ResBlock(base_channels*4, embed_dim) for _ in range(2)
        ])
        
        # 轴向注意力和多头注意力模块
        self.horizontal_attn = HorizontalAxisAttention(
            dim=base_channels*4, 
            heads=4, 
            dim_head=base_channels
        )
        self.vertical_attn = VerticalAxisAttention(
            dim=base_channels*4, 
            heads=4, 
            dim_head=base_channels
        )
        self.multi_head_attn = MultiHeadAttention(
            dim=base_channels*4, 
            heads=4, 
            dim_head=base_channels
        )
        
        # 额外的残差块
        self.post_attn_blocks = nn.ModuleList([
            ResBlock(base_channels*4, embed_dim) for _ in range(2)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),  # [b, base_channels*8, h/2, w/2]
            nn.PixelShuffle(2),  # [b, base_channels*2, h, w]
            nn.Conv2d(base_channels*2, in_channels, 1)  # [b, in_channels, h, w]
        )
    
    def forward(self, base_img, base_angle, target_angle):
        """
        base_img: [b, in_channels, h, w] 基础图像
        base_angle: [b] 基础角度
        target_angle: [b] 目标角度
        return: [b, in_channels, h, w] 旋转后的图像
        """
        # 计算角度差并嵌入
        angle_diff = target_angle - base_angle  # [b]
        angle_diff = (angle_diff + 180) % 360 - 180  # [b]
        angle_embed = self.angle_embedder(angle_diff)  # [b, embed_dim]
        
        # 编码
        x = self.encoder(base_img)  # [b, base_channels*4, h/2, w/2]
        
        # 初始残差块
        for block in self.rotation_blocks:
            x = block(x, angle_embed)  # [b, base_channels*4, h/2, w/2]
        
        # 应用轴向注意力和多头注意力
        x_horizontal = self.horizontal_attn(x)
        x_vertical = self.vertical_attn(x)
        x_global = self.multi_head_attn(x)
        
        # 融合注意力结果
        x = x + x_horizontal + x_vertical + x_global
        
        # 后续残差块
        for block in self.post_attn_blocks:
            x = block(x, angle_embed)  # [b, base_channels*4, h/2, w/2]
        
        # 解码
        out = self.decoder(x)  # [b, in_channels, h, w]
        
        return torch.sigmoid(out)  # [b, in_channels, h, w]