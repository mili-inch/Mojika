import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from mojika.utils.timestep_distribution import TimeSamplingParameters


def positive(value: float, name: str):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def unit_interval(value: float, name: str):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ModelConfig:
    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    rope_theta: float
    qk_norm: bool

    def __post_init__(self):
        if (self.img_size, self.patch_size, self.in_chans) != (256, 16, 1):
            raise ValueError(
                "The conditional dataset requires 256x256 grayscale images and 16x16 patches"
            )
        if self.embed_dim <= 0 or self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.embed_dim % self.num_heads or (self.embed_dim // self.num_heads) % 4:
            raise ValueError(
                "Each attention head must have a dimension divisible by four"
            )
        positive(self.mlp_ratio, "mlp_ratio")
        positive(self.rope_theta, "rope_theta")


@dataclass(frozen=True)
class AdamWConfig:
    name: str
    adamw_lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

    def __post_init__(self):
        positive(self.adamw_lr, "adamw_lr")
        positive(self.eps, "eps")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class MuonAdamWConfig(AdamWConfig):
    muon_lr: float
    muon_weight_decay: float
    momentum: float
    muon_beta2: float
    momentum_start: float
    momentum_warmup_steps: int
    momentum_end: float
    decay_weight_decay_ratio: float

    def __post_init__(self):
        super().__post_init__()
        positive(self.muon_lr, "muon_lr")
        if not math.isfinite(self.muon_weight_decay) or self.muon_weight_decay < 0:
            raise ValueError("muon_weight_decay must be finite and nonnegative")
        if not 0 <= self.momentum < 1 or not 0 <= self.muon_beta2 < 1:
            raise ValueError("Muon momentum and beta2 must be in [0, 1)")
        if not 0 <= self.momentum_start < 1 or not 0 <= self.momentum_end < 1:
            raise ValueError("Momentum endpoints must be in [0, 1)")
        if (
            type(self.momentum_warmup_steps) is not int
            or self.momentum_warmup_steps < 0
        ):
            raise ValueError("momentum_warmup_steps must be a nonnegative integer")
        unit_interval(self.decay_weight_decay_ratio, "decay_weight_decay_ratio")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    time_patch_sizes: tuple[int, ...]
    time_sampling: TimeSamplingParameters
    grad_clip_norm: float
    seed: int
    num_workers: int
    compile: bool
    save_every_steps: int
    keep_every_steps: int
    sample_every_steps: int
    sample_steps: int
    log_every_steps: int
    check_loss_every_steps: int

    def __post_init__(self):
        if not self.time_patch_sizes or any(
            type(size) is not int or size <= 0 for size in self.time_patch_sizes
        ):
            raise ValueError(
                "time_patch_sizes must contain positive integer pixel sizes"
            )
        if len(set(self.time_patch_sizes)) != len(self.time_patch_sizes):
            raise ValueError("time_patch_sizes must be unique")
        for name in (
            "batch_size",
            "save_every_steps",
            "keep_every_steps",
            "sample_every_steps",
            "sample_steps",
            "log_every_steps",
            "check_loss_every_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("Worker count and seed must be nonnegative")
        if self.keep_every_steps % self.save_every_steps:
            raise ValueError("keep_every_steps must be a multiple of save_every_steps")
        if self.sample_every_steps % self.save_every_steps:
            raise ValueError(
                "sample_every_steps must be a multiple of save_every_steps"
            )
        positive(self.grad_clip_norm, "grad_clip_norm")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: AdamWConfig | MuonAdamWConfig
    training: TrainingConfig
    warmup_steps: int

    def __post_init__(self):
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        for size in self.training.time_patch_sizes:
            if size % self.model.patch_size or self.model.img_size % size:
                raise ValueError(
                    "Time patch sizes must be multiples of the model patch size and divide the image size"
                )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        if set(data) != {
            "model",
            "optimizer",
            "training",
            "warmup_steps",
        }:
            raise ValueError("Experiment configuration has missing or unknown sections")
        name = data["optimizer"]["name"]
        if name == "adamw":
            optimizer = AdamWConfig(**data["optimizer"])
        elif name == "muon_adamw":
            optimizer = MuonAdamWConfig(**data["optimizer"])
        else:
            raise ValueError(f"Unknown optimizer: {name}")
        training = dict(data["training"])
        training["time_patch_sizes"] = tuple(training["time_patch_sizes"])
        training["time_sampling"] = TimeSamplingParameters(**training["time_sampling"])
        return cls(
            model=ModelConfig(**data["model"]),
            optimizer=optimizer,
            training=TrainingConfig(**training),
            warmup_steps=data["warmup_steps"],
        )

    @classmethod
    def read(cls, path: Path):
        with path.open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))
