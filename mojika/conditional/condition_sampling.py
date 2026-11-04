"""Balanced condition modes with deterministic per-character coverage."""

from dataclasses import dataclass
from hashlib import blake2b

import torch


@dataclass(frozen=True)
class ConditionSample:
    index: int
    mode: int
    visit: int


def sampling_seed(seed: int, *coordinates: int) -> int:
    encoded = ":".join(map(str, (seed, *coordinates))).encode("ascii")
    return int.from_bytes(blake2b(encoded, digest_size=8).digest(), "little") % (2**63)


def epoch_samples(size: int, batch_size: int, seed: int, epoch: int):
    """Every batch covers eight modes; every included glyph covers them in 8 epochs.

    Groups of eight assign shuffled Latin-square modes over eight epochs.
    Images are shuffled independently within each mode every epoch, then combined
    into balanced batches with shuffled positions. The incomplete batch is omitted
    for that whole cycle. Its membership rotates through a seeded permutation, so
    omissions and prior visits are reproducible from the epoch alone, including
    after a mid-epoch resume.
    """
    if batch_size <= 0 or batch_size % 8:
        raise ValueError("Balanced conditions require a batch size divisible by 8")
    if size < batch_size or epoch < 0 or seed < 0:
        raise ValueError("Expected a full training batch and nonnegative epoch/seed")
    cycle, phase = divmod(epoch, 8)
    remainder = size % batch_size
    generator = torch.Generator()
    identities = torch.randperm(
        size, generator=generator.manual_seed(sampling_seed(seed, 0))
    )
    ranks = torch.arange(size)
    included = (ranks - cycle * remainder) % size >= remainder
    ranks = ranks[included]
    # Count omissions before this cycle, rather than advancing unseen examples.
    visits = cycle - (cycle * remainder + size - 1 - ranks) // size
    generator.manual_seed(sampling_seed(seed, 1, cycle))
    order = torch.randperm(len(ranks), generator=generator)
    indices = identities[ranks[order]].reshape(-1, 8)
    visits = visits[order].reshape(-1, 8)
    groups = len(indices)
    labels = torch.rand(groups, 8, generator=generator).argsort(dim=1)
    phases = torch.rand(groups, 8, generator=generator).argsort(dim=1)
    positions = (torch.arange(8) + phases[:, phase, None]) % 8
    modes = labels.gather(1, positions).flatten()
    indices = indices.flatten()
    visits = visits.flatten()

    # Mix images across assignment groups while retaining each mode and visit.
    mode_orders = []
    for mode in range(8):
        positions = (modes == mode).nonzero(as_tuple=True)[0]
        generator.manual_seed(sampling_seed(seed, 2, epoch, mode))
        mode_orders.append(positions[torch.randperm(groups, generator=generator)])
    order = torch.stack(mode_orders, dim=1).reshape(-1, batch_size)
    generator.manual_seed(sampling_seed(seed, 4, epoch))
    batch_positions = torch.stack(
        [torch.randperm(batch_size, generator=generator) for _ in range(len(order))]
    )
    order = order.gather(1, batch_positions).flatten()
    return [
        ConditionSample(index, mode, visit)
        for index, mode, visit in zip(
            indices[order].tolist(),
            modes[order].tolist(),
            visits[order].tolist(),
            strict=True,
        )
    ]


def embedding_variants(
    sample: ConditionSample, seed: int, gpt_count: int, clip_count: int
):
    """Cycle through the Cartesian product of the enabled embedding variants."""
    if not 0 <= sample.mode < 8 or sample.visit < 0 or min(gpt_count, clip_count) < 1:
        raise ValueError("Invalid condition sample or embedding counts")
    gpt_count = gpt_count if sample.mode & 4 else 1
    clip_count = clip_count if sample.mode & 2 else 1
    count = gpt_count * clip_count
    cycle, position = divmod(sample.visit, count)
    generator = torch.Generator().manual_seed(
        sampling_seed(seed, 3, sample.index, sample.mode, cycle)
    )
    variant = torch.randperm(count, generator=generator)[position].item()
    return divmod(variant, clip_count)
