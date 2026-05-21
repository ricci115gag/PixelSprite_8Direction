from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock


class HorizontalAxisAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h, w),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q.permute(0, 1, 3, 4, 2)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 4, 2)
        attn = torch.matmul(q, k) * self.scale
        out = torch.matmul(F.softmax(attn, dim=-1), v)
        out = out.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)
        return self.to_out(out)


class VerticalAxisAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h, w),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q.permute(0, 1, 4, 2, 3)
        k = k.permute(0, 1, 4, 3, 2)
        v = v.permute(0, 1, 4, 2, 3)
        attn = torch.matmul(q, k) * self.scale
        out = torch.matmul(F.softmax(attn, dim=-1), v)
        out = out.permute(0, 1, 3, 4, 2).reshape(b, -1, h, w)
        return self.to_out(out)


class GlobalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h * w),
            self.to_qkv(x).chunk(3, dim=1),
        )
        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale
        out = torch.matmul(F.softmax(attn, dim=-1), v.transpose(-1, -2))
        out = out.permute(0, 1, 3, 4, 2).reshape(b, -1, h, w)
        return self.to_out(out)
