"""Adam with decoupled cautious weight decay (https://arxiv.org/abs/2510.12402)."""

import math
from collections import defaultdict

import torch


def adamw_step(parameters, gradients, averages, squares, coefficients, decay):
    beta1_weight, beta2, beta2_weight, bias1, sqrt_bias2, eps, lr, wd = coefficients
    for parameter, grad, average, square in zip(
        parameters, gradients, averages, squares, strict=True
    ):
        average.lerp_(grad, beta1_weight)
        square.mul_(beta2).addcmul_(grad, grad, value=beta2_weight)
        denominator = square.sqrt() / sqrt_bias2
        denominator.add_(eps)
        direction = (average / bias1) / denominator
        if decay:
            aligned = direction * parameter >= 0
            parameter.add_(parameter * aligned, alpha=-lr * wd)
        parameter.add_(direction, alpha=-lr)


class CautiousAdamW(torch.optim.Optimizer):
    """FP32 Adam moments, bias correction, and decay aligned with the Adam update."""

    def __init__(self, params, *, lr, betas, eps, weight_decay, compile):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("AdamW learning rate must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("AdamW epsilon must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("AdamW weight decay must be finite and nonnegative")
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )
        self.update = (
            torch.compile(adamw_step, fullgraph=True, dynamic=False)
            if compile
            else adamw_step
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("CautiousAdamW requires float32 parameters")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            batches = defaultdict(list)
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise ValueError("CautiousAdamW requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                batches[(parameter.device, state["step"])].append(parameter)
            for (_, step), parameters in batches.items():
                coefficients = tuple(
                    torch.tensor(value, dtype=torch.float32, device="cpu")
                    for value in (
                        1 - beta1,
                        beta2,
                        1 - beta2,
                        1 - beta1**step,
                        math.sqrt(1 - beta2**step),
                        group["eps"],
                        group["lr"],
                        group["weight_decay"],
                    )
                )
                self.update(
                    parameters,
                    [p.grad for p in parameters],
                    [self.state[p]["exp_avg"] for p in parameters],
                    [self.state[p]["exp_avg_sq"] for p in parameters],
                    coefficients,
                    group["weight_decay"] != 0,
                )
