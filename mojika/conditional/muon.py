"""Batched Muon updates adapted from nanochat's optim.py.

Source: https://github.com/karpathy/nanochat/blob/
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/optim.py

MIT License
Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from collections import defaultdict

import torch


POLAR_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def muon_direction(grad, momentum_buffer, variance_buffer, momentum, beta2):
    """Update moments and return a normalized matrix direction without changing grad."""
    momentum_buffer.lerp_(grad, 1 - momentum)
    x = grad.lerp(momentum_buffer, momentum).bfloat16()

    # MuonEq: equalize row norms before the polar iteration.
    target = x.float().norm(dim=(-2, -1), keepdim=True) / math.sqrt(x.size(-2))
    row_norm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = x * (target / row_norm).to(x.dtype)

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_COEFFICIENTS:
        if x.size(-2) > x.size(-1):
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x

    direction = x.to(grad.dtype)

    # NorMuon: adapt rows or columns while preserving the matrix norm.
    reduction_dim = -1 if grad.size(-2) >= grad.size(-1) else -2
    mean_square = direction.float().square().mean(dim=reduction_dim, keepdim=True)
    reduction_size = direction.size(reduction_dim)
    original_norm = (
        mean_square.sum(dim=(-2, -1), keepdim=True) * reduction_size
    ).sqrt()
    variance_buffer.lerp_(mean_square.to(variance_buffer.dtype), 1 - beta2)
    scale = variance_buffer.clamp_min(1e-10).rsqrt()
    scaled_norm = (
        (mean_square * reduction_size * scale.float().square())
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
    )
    scale = scale * (original_norm / scaled_norm.clamp_min(1e-10))
    return direction * scale.to(direction.dtype)


def muon_step(
    parameters, grads, momentum_buffer, variance_buffer, momentum, beta2, lr, wd
):
    direction = muon_direction(grads, momentum_buffer, variance_buffer, momentum, beta2)
    decay_mask = direction * parameters >= 0
    parameters.sub_(lr * direction + lr * wd * parameters * decay_mask)


class Muon(torch.optim.Optimizer):
    """MuonEq + Polar Express + NorMuon + cautious weight decay."""

    def __init__(self, params, *, lr, momentum, beta2, weight_decay, compile):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Muon learning rate must be finite and nonnegative")
        if not 0 <= momentum < 1 or not 0 <= beta2 < 1:
            raise ValueError("Muon momentum and beta2 must be in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Muon weight decay must be finite and nonnegative")
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay),
        )
        self.update = (
            torch.compile(
                muon_step,
                fullgraph=True,
                dynamic=False,
                options={"emulate_precision_casts": True},
            )
            if compile
            else muon_step
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2 or parameter.dtype != torch.float32:
                    raise ValueError("Muon requires two-dimensional float32 parameters")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # Round coefficients before arithmetic, as in nanochat. Tiny changes
            # before the BF16 cast can be amplified by the polar iterations.
            momentum = torch.tensor(
                group["momentum"], dtype=torch.float32, device="cpu"
            )
            beta2 = torch.tensor(group["beta2"], dtype=torch.float32, device="cpu")
            weight_decay = torch.tensor(
                group["weight_decay"], dtype=torch.float32, device="cpu"
            )
            batches = defaultdict(list)
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise ValueError("Muon requires dense gradients")
                state = self.state[parameter]
                if not state:
                    rows, columns = parameter.shape
                    variance_shape = (rows, 1) if rows >= columns else (1, columns)
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["variance_buffer"] = parameter.new_zeros(variance_shape)
                batches[(parameter.device, parameter.shape)].append(parameter)
            for parameters in batches.values():
                moments = [self.state[p]["momentum_buffer"] for p in parameters]
                variances = [self.state[p]["variance_buffer"] for p in parameters]
                stacked = torch.stack(parameters)
                stacked_moments = torch.stack(moments)
                stacked_variances = torch.stack(variances)
                rows, columns = parameters[0].shape
                lr = torch.tensor(
                    group["lr"] * math.sqrt(max(1, rows / columns)),
                    dtype=torch.float32,
                    device="cpu",
                )
                self.update(
                    stacked,
                    torch.stack([p.grad for p in parameters]),
                    stacked_moments,
                    stacked_variances,
                    momentum,
                    beta2,
                    lr,
                    weight_decay,
                )
                torch._foreach_copy_(parameters, stacked.unbind())
                torch._foreach_copy_(moments, stacked_moments.unbind())
                torch._foreach_copy_(variances, stacked_variances.unbind())
