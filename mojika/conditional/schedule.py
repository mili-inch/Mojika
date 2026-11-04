import math
from dataclasses import asdict, dataclass

from .config import positive, unit_interval


@dataclass(frozen=True)
class Decay:
    start_step: int
    steps: int
    curve: str
    min_lr_ratio: float

    def __post_init__(self):
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("Decay start must be a nonnegative optimizer step")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("Decay steps must be a positive integer")
        if self.curve not in {"linear", "cosine", "sqrt"}:
            raise ValueError(f"Unknown decay curve: {self.curve}")
        unit_interval(self.min_lr_ratio, "min_lr_ratio")


@dataclass(frozen=True)
class WSD:
    """Learning rates for one-based optimizer updates, independent of run length."""

    warmup_steps: int
    base_lrs: tuple[float, ...]
    decay: Decay | None

    def __post_init__(self):
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if not self.base_lrs:
            raise ValueError("At least one base learning rate is required")
        for lr in self.base_lrs:
            positive(lr, "base_lr")
        if self.decay is not None and self.decay.start_step < self.warmup_steps:
            raise ValueError("Decay must start from a Stable checkpoint after warmup")

    @property
    def end_step(self) -> int | None:
        if self.decay is None:
            return None
        return self.decay.start_step + self.decay.steps

    def multiplier(self, step: int) -> float:
        if type(step) is not int or step < 0:
            raise ValueError("Optimizer step must be a nonnegative integer")
        if self.decay is None:
            if self.warmup_steps and step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        elapsed = step - self.decay.start_step
        if not 0 <= elapsed <= self.decay.steps:
            raise ValueError("Optimizer step is outside this decay branch")
        fraction = elapsed / self.decay.steps
        if self.decay.curve == "linear":
            value = 1 - fraction
        elif self.decay.curve == "sqrt":
            value = 1 - math.sqrt(fraction)
        else:
            value = (1 + math.cos(math.pi * fraction)) / 2
        return self.decay.min_lr_ratio + (1 - self.decay.min_lr_ratio) * value

    def learning_rates(self, step: int) -> tuple[float, ...]:
        multiplier = self.multiplier(step)
        return tuple(lr * multiplier for lr in self.base_lrs)

    def phase(self, step: int) -> str:
        self.multiplier(step)
        if self.decay is not None:
            return "complete" if step == self.end_step else "decay"
        return "warmup" if step < self.warmup_steps else "stable"

    def fork_decay(self, step: int, steps: int, curve: str, min_lr_ratio: float):
        if self.decay is not None or self.phase(step) != "stable":
            raise ValueError("Only a Stable checkpoint can start a new decay branch")
        return WSD(
            self.warmup_steps, self.base_lrs, Decay(step, steps, curve, min_lr_ratio)
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        if set(data) != {"warmup_steps", "base_lrs", "decay"}:
            raise ValueError("Scheduler state has missing or unknown fields")
        decay = None if data["decay"] is None else Decay(**data["decay"])
        return cls(data["warmup_steps"], tuple(data["base_lrs"]), decay)
