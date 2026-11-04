"""Export inference weights without training state or machine-specific paths."""

import argparse
from pathlib import Path

import torch

from .checkpoint import load_models, save_checkpoint
from .config import ExperimentConfig


def export_weights(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    payload = {
        "config": config.to_dict(),
        "model_state_dict": checkpoint["model_state_dict"],
        "conditioner_state_dict": checkpoint["conditioner_state_dict"],
    }
    # Validate all model tensors against the same contract used by the server.
    load_models(payload)
    save_checkpoint(destination, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_weights(args.checkpoint, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
