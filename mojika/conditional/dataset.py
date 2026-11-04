from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from .perlin_noise import generate_perlin_noise_2d_float
from .condition_sampling import ConditionSample, embedding_variants


def read_embedding(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    tensors = load_file(str(path))
    if set(tensors) != {"embedding"}:
        raise ValueError(f"Expected only an embedding tensor: {path}")
    embedding = tensors["embedding"]
    if (
        embedding.shape != shape
        or not embedding.is_floating_point()
        or not torch.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid embedding, expected finite {shape}: {path}")
    return embedding.reshape(-1).float()


def image_paths(root: Path) -> list[Path]:
    images = sorted((root / "images").glob("*.png"))
    if not images:
        raise ValueError(f"No training images in {root}")
    return images


def embedding_paths(
    root: Path, codes: list[str], directory: str
) -> dict[str, list[Path]]:
    paths = {code: [] for code in codes}
    for path in sorted((root / directory).glob("*.safetensors")):
        code = path.stem.split("_")[0]
        if code not in paths:
            raise ValueError(f"Embedding has no matching image: {path}")
        paths[code].append(path)
    missing = [code for code, variants in paths.items() if not variants]
    if missing:
        raise ValueError(
            f"Missing {directory} for {len(missing)} images: {missing[:10]}"
        )
    return paths


def partial_mask(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    pooled = F.avg_pool2d(image, 16, 16).squeeze(0)
    strokes = pooled < 0.98
    threshold = torch.rand((), generator=generator).item() * 0.34
    frequency = torch.rand((), generator=generator).item() * 7.7 + 4
    noise = generate_perlin_noise_2d_float((16, 16), (frequency, frequency), generator)
    selected = (noise > threshold) & strokes
    # A complete glyph would reveal the answer; use the unconditional mask instead.
    if selected.sum() == strokes.sum():
        selected = torch.zeros_like(selected)
    return torch.where(selected, pooled, torch.ones_like(pooled))


class ConditionalKanjiDataset(Dataset):
    def __init__(self, root: Path, seed: int):
        self.root = root.resolve(strict=True)
        self.seed = seed
        self.epoch = 0
        self.images = image_paths(self.root)
        self.codes = [path.stem for path in self.images]
        self.gpt_paths = embedding_paths(self.root, self.codes, "sarashina_embeddings")
        self.clip_paths = embedding_paths(self.root, self.codes, "clip_embeddings")
        means = load_file(str(self.root / "condition_means.safetensors"))
        if set(means) != {"gpt", "clip"}:
            raise ValueError(
                "Condition means must contain exactly gpt and clip tensors"
            )
        for name, dim in (("gpt", 1280), ("clip", 512)):
            value = means[name]
            if (
                value.shape != (dim,)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Invalid {name} mean: expected finite float32 ({dim},)"
                )
        self.gpt_mean, self.clip_mean = means["gpt"], means["clip"]

    def __len__(self):
        return len(self.images)

    def read_image(self, index: int) -> torch.Tensor:
        path = self.images[index]
        image = read_image(str(path))
        if image.shape != (1, 256, 256) or image.dtype != torch.uint8:
            raise ValueError(f"Expected a grayscale 256x256 uint8 image: {path}")
        return image.float().div(127.5).sub(1)

    def __getitem__(self, sample: ConditionSample):
        code = self.codes[sample.index]
        gpt_index, clip_index = embedding_variants(
            sample, self.seed, len(self.gpt_paths[code]), len(self.clip_paths[code])
        )
        return self.read_sample(sample.index, gpt_index, clip_index, sample.mode)

    def read_sample(self, index: int, gpt_index: int, clip_index: int, mode: int):
        # Stable augmentation across workers and when resuming within an epoch.
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * len(self) + index
        )
        code = self.codes[index]
        image = self.read_image(index)
        gpt_paths = self.gpt_paths[code]
        clip_paths = self.clip_paths[code]
        return {
            "unicode": code,
            "condition_mode": mode,
            "image": image,
            "mask": partial_mask(image, generator) if mode & 1 else torch.ones(16, 16),
            "gpt": read_embedding(gpt_paths[gpt_index], (1280,))
            if mode & 4
            else torch.zeros(1280),
            "clip": read_embedding(clip_paths[clip_index], (1, 512))
            if mode & 2
            else torch.zeros(512),
        }

    def validate_all(self) -> dict[str, int]:
        for index in range(len(self)):
            self.read_image(index)
        for variants in self.gpt_paths.values():
            for path in variants:
                read_embedding(path, (1280,))
        for variants in self.clip_paths.values():
            for path in variants:
                read_embedding(path, (1, 512))
        return self.counts()

    def counts(self) -> dict[str, int]:
        return {
            "images": len(self),
            "gpt_embeddings": sum(map(len, self.gpt_paths.values())),
            "clip_embeddings": sum(map(len, self.clip_paths.values())),
        }
