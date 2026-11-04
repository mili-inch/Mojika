# SPDX-License-Identifier: MIT
# Copyright (c) 2026 mili-inch

import math

import torch
from torch import nn


class DiMRTimestepParameter(nn.Module):
    """Mix learned endpoints with a positive time slope and a trainable offset."""

    def __init__(self, hidden_dim: int, init_std: float):
        super().__init__()
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError(
                "Endpoint initialization std must be finite and nonnegative"
            )

        self.param0 = nn.Parameter(torch.empty(hidden_dim).normal_(std=init_std))
        self.param1 = nn.Parameter(torch.empty(hidden_dim).normal_(std=init_std))

        self.raw_slope = nn.Parameter(
            torch.full((hidden_dim,), math.log(math.expm1(4.0)))
        )
        self.bias = nn.Parameter(torch.full((hidden_dim,), -2.0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(
            t.unsqueeze(-1) * nn.functional.softplus(self.raw_slope) + self.bias
        )
        out = s * self.param0 + (1.0 - s) * self.param1
        return out
