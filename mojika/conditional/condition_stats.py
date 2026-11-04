"""Compute fixed means of RMS-normalized GPT/CLIP training embeddings."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from .dataset import embedding_paths, image_paths, read_embedding
from .modules.embedders import normalize_embedding


@torch.inference_mode()
def compute_condition_means(root: Path) -> dict[str, torch.Tensor]:
    """Weight characters equally, and variants uniformly within each character."""
    root = root.resolve(strict=True)
    codes = [path.stem for path in image_paths(root)]
    means = {}
    for name, directory, shape in (
        ("gpt", "sarashina_embeddings", (1280,)),
        ("clip", "clip_embeddings", (1, 512)),
    ):
        paths = embedding_paths(root, codes, directory)
        total = torch.zeros(shape[-1], dtype=torch.float64)
        for variants in paths.values():
            character = torch.zeros_like(total)
            for path in variants:
                normalized = normalize_embedding(read_embedding(path, shape))
                if not torch.isfinite(normalized).all():
                    raise ValueError(f"Non-finite normalized embedding: {path}")
                character.add_(normalized.double())
            total.add_(character / len(variants))
        means[name] = (total / len(codes)).float()
    return means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.dataset_root / "condition_means.safetensors"
    if destination.exists():
        raise FileExistsError(f"Condition means already exist: {destination}")
    torch.set_num_threads(2)
    means = compute_condition_means(args.dataset_root)
    save_file(means, str(destination))
    print(destination.resolve())


if __name__ == "__main__":
    main()
