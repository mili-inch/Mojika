import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .conditioner import Conditioner
from .config import ExperimentConfig
from .model import DiffusionTransformer


def save_checkpoint(path: Path, checkpoint: dict):
    """Commit a complete checkpoint without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_models(checkpoint: dict):
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = DiffusionTransformer(**asdict(config.model))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state = checkpoint["conditioner_state_dict"]
    conditioner = Conditioner(
        state["gpt_embedder.mean"],
        state["clip_embedder.mean"],
        embed_dim=config.model.embed_dim,
        patch_grid=config.model.img_size // config.model.patch_size,
        depth=config.model.depth,
    )
    conditioner.load_state_dict(state, strict=True)
    return model, conditioner
