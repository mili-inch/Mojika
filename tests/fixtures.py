from dataclasses import asdict, replace
from pathlib import Path

import torch

from mojika.conditional.conditioner import Conditioner
from mojika.conditional.config import AdamWConfig, ExperimentConfig
from mojika.conditional.flow import interpolate_patches, sample_patch_times
from mojika.conditional.model import DiffusionTransformer
from mojika.conditional.optimizers import Optimizers
from mojika.conditional.schedule import WSD
from train import TrainingState


def small_config(optimizer: str):
    config = ExperimentConfig.read(
        Path(__file__).resolve().parents[1] / "configs/train.toml"
    )
    if optimizer == "adamw":
        optimizer_config = AdamWConfig("adamw", 0.001, 0.9, 0.95, 1e-8, 0.01)
    elif optimizer == "muon_adamw":
        optimizer_config = replace(
            config.optimizer,
            adamw_lr=0.001,
            muon_lr=0.01,
            momentum_warmup_steps=4,
            decay_weight_decay_ratio=0.25,
        )
    else:
        raise ValueError(optimizer)
    return replace(
        config,
        model=replace(config.model, embed_dim=24, depth=1, num_heads=3, mlp_ratio=2),
        optimizer=optimizer_config,
        warmup_steps=2,
        training=replace(
            config.training,
            batch_size=2,
            num_workers=0,
            save_every_steps=2,
            keep_every_steps=4,
            sample_every_steps=4,
            sample_steps=2,
            log_every_steps=1,
        ),
    )


def new_state(optimizer: str):
    config = small_config(optimizer)
    model = DiffusionTransformer(**asdict(config.model))
    conditioner = Conditioner(
        torch.randn(1280),
        torch.randn(512),
        config.model.embed_dim,
        16,
        config.model.depth,
    )
    optimizers = Optimizers(
        model, conditioner, config.optimizer, compile=config.training.compile
    )
    schedule = WSD(config.warmup_steps, optimizers.base_lrs, None)
    optimizers.apply_schedule(schedule, 0)
    return TrainingState(
        config,
        model,
        conditioner,
        optimizers,
        schedule,
        {"root": "/unit-test", "counts": {"images": 7}},
        0,
        0,
        0,
        torch.zeros((), dtype=torch.float64),
        None,
    )


def step(state):
    state.optimizers.zero_grad()
    x = torch.randn(2, 1, 256, 256)
    noise = torch.randn_like(x)
    times, _, _ = sample_patch_times(
        x.size(0),
        256,
        16,
        state.config.training.time_patch_sizes,
        x.device,
        time_sampling=state.config.training.time_sampling,
    )
    noisy = interpolate_patches(x, noise, times, state.config.model.patch_size)
    patch, extra = state.conditioner(
        noisy,
        gpt=torch.randn(2, 1280),
        clip=torch.randn(2, 512),
        mask=torch.randn(2, 16, 16),
    )
    loss = (state.model(noisy, times, patch, extra) - (x - noise)).square().mean()
    loss.backward()
    state.optimizers.apply_schedule(state.schedule, state.global_step + 1)
    state.optimizers.step()
    state.record_batch(loss, x.size(0))
    return loss.item()
