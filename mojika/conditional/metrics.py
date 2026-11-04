"""Collect detached training metrics on device and transfer them in batches."""

import torch


def _map_tensors(value, convert):
    if isinstance(value, torch.Tensor):
        return convert(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, convert) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, convert) for item in value]
    return value


class MetricBuffer:
    """Snapshot JSON records with tensor leaves; drain with one device-to-host copy."""

    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(_map_tensors(record, lambda value: value.detach().clone()))

    def drain(self):
        if not self.records:
            return []
        tensors = []

        def collect(value):
            tensors.append(value)

        _map_tensors(self.records, collect)
        # FP64 preserves epoch accumulation and the small integer time-region fields.
        packed = torch.cat([value.reshape(-1).to(torch.float64) for value in tensors])
        values = iter(packed.cpu().split([value.numel() for value in tensors]))

        def unpack(value):
            return next(values).reshape(value.shape).to(value.dtype).tolist()

        records = _map_tensors(self.records, unpack)
        self.records.clear()
        return records


@torch.no_grad()
def image_loss_metrics(prediction, target, modes):
    losses = (prediction.detach().float() - target).square().flatten(1).mean(1)
    counts = losses.new_zeros(8).scatter_add_(0, modes, torch.ones_like(losses))
    sums = losses.new_zeros(8).scatter_add_(0, modes, losses)
    means = sums / counts
    return {
        "image_loss_max": losses.max(),
        "image_loss_p99": torch.quantile(losses, 0.99),
        "condition_mean_losses": {f"{mode:03b}": means[mode] for mode in range(8)},
    }


@torch.no_grad()
def patch_time_metrics(times, sizes, offsets):
    return {
        "min": times.min(),
        "max": times.max(),
        "mean": times.mean(),
        "mean_within_image_std": times.flatten(1).std(dim=1, correction=0).mean(),
        "region_size_pixels": sizes,
        "region_offset_pixels_yx": offsets,
    }
