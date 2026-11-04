import math

import torch

from .adamw import CautiousAdamW
from .config import AdamWConfig, MuonAdamWConfig
from .muon import Muon
from .schedule import WSD


def muon_hyperparameters(config: MuonAdamWConfig, schedule: WSD, step: int):
    """Momentum and weight decay follow a branch without a Stable end date."""
    schedule.multiplier(step)
    warmup_step = step if schedule.decay is None else schedule.decay.start_step
    fraction = (
        min(warmup_step / config.momentum_warmup_steps, 1.0)
        if config.momentum_warmup_steps
        else 1.0
    )
    momentum = (
        config.momentum_start + (config.momentum - config.momentum_start) * fraction
    )
    weight_decay = config.muon_weight_decay
    if schedule.decay is not None:
        progress = (step - schedule.decay.start_step) / schedule.decay.steps
        momentum += (config.momentum_end - momentum) * progress
        ratio = config.decay_weight_decay_ratio
        weight_decay *= ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2
    return momentum, weight_decay


class Optimizers:
    """Matrix updates and AdamW with ordinary per-parameter optimizer state."""

    def __init__(
        self, model, conditioner, config: AdamWConfig | MuonAdamWConfig, *, compile
    ):
        self.config = config
        roots = [
            (prefix, module)
            for prefix, module in (("model", model), ("conditioner", conditioner))
            if module is not None
        ]
        if not roots:
            raise ValueError("At least one optimizer module is required")
        named = [
            (f"{prefix}.{name}", parameter)
            for prefix, module in roots
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        if len({id(p) for _, p in named}) != len(named):
            raise ValueError("Optimizer parameters must not be shared between modules")
        self.items = {}
        self.layout = {}
        if isinstance(config, MuonAdamWConfig) and model is not None:
            hidden = [
                parameter
                for block in model.blocks
                for parameter in (
                    block.attn.qkv.weight,
                    block.attn.proj.weight,
                    block.ffn.in_proj.weight,
                    block.ffn.gate_proj.weight,
                    block.ffn.out_proj.weight,
                )
            ]
            ids = {id(p) for p in hidden}
            if len(ids) != len(hidden) or any(
                p.ndim != 2 or not p.requires_grad for p in hidden
            ):
                raise ValueError("Muon requires unique, trainable hidden matrices")
            hidden_named = [(name, p) for name, p in named if id(p) in ids]
            if len(hidden_named) != len(hidden):
                raise ValueError("Muon matrix assignment is incomplete")
            self.items["muon"] = Muon(
                [{"params": [p for _, p in hidden_named], "name": "muon"}],
                lr=config.muon_lr,
                weight_decay=config.muon_weight_decay,
                momentum=config.momentum,
                beta2=config.muon_beta2,
                compile=compile,
            )
            self.layout["muon"] = [name for name, _ in hidden_named]
            adam_named = [(name, p) for name, p in named if id(p) not in ids]
        else:
            adam_named = named

        # Only Linear/Conv weights decay. Frequency, time, gate and bias parameters
        # keep their own meaning and receive no shrinkage toward zero.
        decay_ids = {
            id(module.weight)
            for _, root in roots
            for module in root.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        }
        decay = [(name, p) for name, p in adam_named if id(p) in decay_ids]
        no_decay = [(name, p) for name, p in adam_named if id(p) not in decay_ids]
        self.items["adamw"] = CautiousAdamW(
            [
                {
                    "params": [p for _, p in decay],
                    "name": "adamw_decay",
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for _, p in no_decay],
                    "name": "adamw_no_decay",
                    "weight_decay": 0.0,
                },
            ],
            lr=config.adamw_lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            compile=compile,
        )
        self.layout["adamw"] = [name for name, _ in decay + no_decay]
        assigned = [name for group in self.layout.values() for name in group]
        if len(assigned) != len(named) or set(assigned) != {name for name, _ in named}:
            raise ValueError(
                "Every trainable parameter must belong to exactly one optimizer"
            )
        self.base_lrs = self.learning_rates()

    def groups(self):
        return [group for opt in self.items.values() for group in opt.param_groups]

    def learning_rates(self) -> tuple[float, ...]:
        return tuple(group["lr"] for group in self.groups())

    def learning_rate_metrics(self):
        return {group["name"]: group["lr"] for group in self.groups()}

    def set_learning_rates(self, rates: tuple[float, ...]):
        groups = self.groups()
        if len(rates) != len(groups) or any(
            not math.isfinite(lr) or lr < 0 for lr in rates
        ):
            raise ValueError(
                "Learning rates must match optimizer groups and be finite and nonnegative"
            )
        for group, lr in zip(groups, rates, strict=True):
            group["lr"] = lr

    def apply_schedule(self, schedule: WSD, step: int):
        self.set_learning_rates(schedule.learning_rates(step))
        if "muon" in self.items:
            momentum, weight_decay = muon_hyperparameters(self.config, schedule, step)
            group = self.items["muon"].param_groups[0]
            group["momentum"] = momentum
            group["weight_decay"] = weight_decay

    def validate_schedule(self, schedule: WSD, step: int):
        if self.learning_rates() != schedule.learning_rates(step):
            raise ValueError(
                "Saved optimizer learning rates disagree with scheduler progress"
            )
        if "muon" in self.items:
            momentum, weight_decay = muon_hyperparameters(self.config, schedule, step)
            group = self.items["muon"].param_groups[0]
            if group["momentum"] != momentum or group["weight_decay"] != weight_decay:
                raise ValueError(
                    "Saved Muon momentum or weight decay disagrees with scheduler progress"
                )

    def hyperparameter_metrics(self):
        return {
            group["name"]: {
                key: value
                for key, value in group.items()
                if key in {"momentum", "weight_decay"}
            }
            for group in self.groups()
        }

    def zero_grad(self):
        for optimizer in self.items.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        for name, optimizer in self.items.items():
            if name == "muon" and any(
                p.grad is None for p in optimizer.param_groups[0]["params"]
            ):
                raise RuntimeError(
                    "Every Muon hidden matrix must participate in the training step"
                )
            optimizer.step()

    def state_dict(self):
        return {
            "layout": self.layout,
            "states": {name: opt.state_dict() for name, opt in self.items.items()},
        }

    def load_state_dict(self, state: dict):
        if set(state) != {"layout", "states"}:
            raise ValueError("Optimizer state has missing or unknown fields")
        if state["layout"] != self.layout or list(state["states"]) != list(self.items):
            raise ValueError(
                "Optimizer parameter assignment differs from the checkpoint"
            )
        for name, optimizer in self.items.items():
            saved = state["states"][name]
            current_groups = optimizer.state_dict()["param_groups"]
            if len(saved["param_groups"]) != len(current_groups):
                raise ValueError(
                    "Optimizer parameter group count differs from the checkpoint"
                )
            dynamic_keys = (
                {"lr", "momentum", "weight_decay"} if name == "muon" else {"lr"}
            )
            for current_group, saved_group in zip(
                current_groups, saved["param_groups"], strict=True
            ):
                if set(saved_group) != set(current_group):
                    raise ValueError(
                        f"Optimizer group has missing or unknown settings: {name}"
                    )
                for key, value in current_group.items():
                    if key not in dynamic_keys and saved_group[key] != value:
                        raise ValueError(
                            f"Optimizer setting differs from the checkpoint: {name}.{key}"
                        )
            optimizer.load_state_dict(saved)
