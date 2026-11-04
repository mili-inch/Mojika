# SPDX-License-Identifier: MIT
# Copyright (c) 2026 mili-inch

import torch
from torch import nn


class Rotary2DMixedPositionalEmbedding(nn.Module):
    def __init__(self, num_heads: int, rope_dim: int, rope_theta: float = 10.0):
        self.rope_dim = rope_dim

        super().__init__()

        freqs = self.init_2d_freqs(
            rope_dim=rope_dim,
            num_heads=num_heads,
            theta=rope_theta,
        )
        self.freqs = nn.Parameter(
            freqs,
            requires_grad=True,
        )  # (2, num_heads, rope_dim // 2)

    def init_2d_freqs(
        self,
        rope_dim: int,
        num_heads: int,
        theta: float,
    ) -> torch.Tensor:
        idx = torch.arange(0, rope_dim, 4, dtype=torch.float32).unsqueeze(0)
        mag = 1.0 / (theta ** (idx / rope_dim))
        angles = torch.rand(num_heads, 1) * 2.0 * torch.pi
        cos_angles = mag * torch.cos(angles)
        sin_angles = mag * torch.sin(angles)

        return torch.stack(
            [
                torch.cat([cos_angles, -sin_angles], dim=-1),
                torch.cat([sin_angles, cos_angles], dim=-1),
            ],
            dim=0,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """q: (batch, seqLen, num_heads, head_dim)
        k: (batch, seqLen, num_heads, head_dim)
        positions: (2, seqLen)
        """
        with torch.autocast(device_type=q.device.type, enabled=False):
            theta = torch.einsum(
                "d s, d h f -> s h f",
                positions.to(torch.float32),
                self.freqs,
            )

            theta = theta.unsqueeze(0).unsqueeze(-1)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            def apply_rotary(x: torch.Tensor):
                x_float = x[..., : self.rope_dim].to(torch.float32)
                x_reshaped = x_float.view(*x.shape[:-1], -1, 2)

                x_rotated = torch.stack(
                    [-x_reshaped[..., 1], x_reshaped[..., 0]],
                    dim=-1,
                )

                out = (
                    (x_reshaped * cos_theta + x_rotated * sin_theta)
                    .flatten(-2)
                    .to(x.dtype)
                )
                out = torch.cat([out, x[..., self.rope_dim :]], dim=-1)
                return out

            q_out = apply_rotary(q)
            k_out = apply_rotary(k)

        return q_out, k_out


class PositiveRotary2DMixedPositionalEmbedding(nn.Module):
    def __init__(self, num_heads: int, rope_dim: int, rope_theta: float = 10.0):
        self.rope_dim = rope_dim

        super().__init__()

        freqs = self.init_2d_freqs(
            rope_dim=rope_dim,
            num_heads=num_heads,
            theta=rope_theta,
        )
        self.raw_freqs = nn.Parameter(
            freqs,
            requires_grad=True,
        )  # (2, num_heads, rope_dim // 2)

    def init_2d_freqs(
        self, rope_dim: int, num_heads: int, theta: float, eps: float = 1e-8
    ) -> torch.Tensor:
        idx = torch.arange(0, rope_dim, 4, dtype=torch.float32).unsqueeze(0)
        mag = 1.0 / (theta ** (idx / rope_dim))

        angles = torch.rand(4, num_heads, 1, device=mag.device) * torch.pi
        s = mag * torch.sin(angles) + eps  # (4, H, L) where L = rope_dim/4
        s = s + torch.log(-torch.expm1(-s))

        # (4,H,L) -> (2,2,H,L) -> (2,H,2,L) -> (2,H,2L)
        return s.view(2, 2, num_heads, -1).permute(0, 2, 1, 3).reshape(2, num_heads, -1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """q: (batch, seqLen, num_heads, head_dim)
        k: (batch, seqLen, num_heads, head_dim)
        positions: (2, seqLen)
        """
        with torch.autocast(device_type=q.device.type, enabled=False):
            theta = torch.einsum(
                "d s, d h f -> s h f",
                positions.to(torch.float32),
                nn.functional.softplus(self.raw_freqs),
            )

            theta = theta.unsqueeze(0).unsqueeze(-1)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            def apply_rotary(x: torch.Tensor):
                x_float = x[..., : self.rope_dim].to(torch.float32)
                x_reshaped = x_float.view(*x.shape[:-1], -1, 2)

                x_rotated = torch.stack(
                    [-x_reshaped[..., 1], x_reshaped[..., 0]],
                    dim=-1,
                )

                out = (
                    (x_reshaped * cos_theta + x_rotated * sin_theta)
                    .flatten(-2)
                    .to(x.dtype)
                )
                out = torch.cat([out, x[..., self.rope_dim :]], dim=-1)
                return out

            q_out = apply_rotary(q)
            k_out = apply_rotary(k)

        return q_out, k_out
