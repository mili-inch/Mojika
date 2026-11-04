# SPDX-License-Identifier: MIT
# Copyright (c) 2026 mili-inch

import torch


def init_positions_2d(end_x: int, end_y: int):
    y_range = torch.arange(end_y, dtype=torch.float32)
    x_range = torch.arange(end_x, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(y_range, x_range, indexing="ij"), dim=0)
    return grid.flatten(1)  # (2, end_x * end_y)
