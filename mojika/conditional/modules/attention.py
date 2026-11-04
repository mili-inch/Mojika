import torch
import torch.nn as nn
import torch.nn.functional as F
from mojika.modules.positional_embedding import PositiveRotary2DMixedPositionalEmbedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.rope = PositiveRotary2DMixedPositionalEmbedding(
            num_heads=num_heads, rope_dim=self.head_dim, rope_theta=rope_theta
        )

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
            self.q_scale = nn.Parameter(torch.ones(1, 1, num_heads, self.head_dim))
        else:
            self.q_norm = None
            self.k_norm = None
            self.q_scale = None

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, positions):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        if self.qk_norm:
            q = self.q_norm(q) * self.q_scale
            k = self.k_norm(k)

        q, k = self.rope(q, k, positions)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
