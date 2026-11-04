"""Flow matching with one denoising time per spatial image patch."""

import torch

from mojika.utils.timestep_distribution import TimeSamplingParameters, sample_times


def sample_patch_times(
    batch_size,
    image_size,
    patch_size,
    time_patch_sizes,
    device,
    *,
    time_sampling: TimeSamplingParameters,
):
    """Draw region sizes, independent Y/X offsets, and configured region times.

    Offsets are crop coordinates within a region, aligned to model patches.
    Border regions are clipped to the image and receive independent times.
    A full-image region has zero offset and one shared time for all patches.
    Returns times [batch, 1, 1] for image-wide sampling, otherwise a patch grid,
    region sizes in pixels, and pixel offsets [batch, YX].
    """
    grid = image_size // patch_size
    if time_patch_sizes == (image_size,):
        # Only B independent times are needed when every image is one region.
        draws = sample_times(
            time_sampling, (batch_size,), device=device, generator=None
        )
        times = draws[:, None, None]
        selected = torch.full(
            (batch_size,), image_size, device=device, dtype=torch.int64
        )
        offsets = torch.zeros((batch_size, 2), device=device, dtype=torch.int64)
        return times, selected, offsets

    sizes = torch.tensor(time_patch_sizes, device=device)
    selected = sizes[torch.randint(len(sizes), (batch_size,), device=device)]
    span = selected // patch_size
    # Every allowed span divides grid, so integer remainders are exactly uniform.
    offsets = torch.randint(grid, (batch_size, 2), device=device) % span[:, None]
    offsets = torch.where(span[:, None] == grid, 0, offsets)
    coordinates = torch.arange(grid, device=device)[None, :]
    rows = (coordinates + offsets[:, 0, None]) // span[:, None]
    columns = (coordinates + offsets[:, 1, None]) // span[:, None]
    indices = rows[:, :, None] * grid + columns[:, None, :]
    draws = sample_times(
        time_sampling, (batch_size, grid * grid), device=device, generator=None
    )
    times = draws.gather(1, indices.flatten(1)).reshape(batch_size, grid, grid)
    return times, selected, offsets * patch_size


def interpolate_patches(image, noise, times, patch_size: int):
    """Interpolate with one time per image or one time per spatial patch."""
    if image.ndim != 4 or noise.shape != image.shape:
        raise ValueError(
            "Image and noise must have the same [batch, channels, height, width] shape"
        )
    if patch_size <= 0 or any(size % patch_size for size in image.shape[-2:]):
        raise ValueError(
            "Image dimensions must be divisible by the positive patch size"
        )
    expected = (image.size(0), image.size(2) // patch_size, image.size(3) // patch_size)
    if times.shape == (image.size(0), 1, 1):
        pixels = times
    elif times.shape == expected:
        pixels = times.repeat_interleave(patch_size, dim=1).repeat_interleave(
            patch_size, dim=2
        )
    else:
        raise ValueError("Timesteps must be image-wide or match the spatial patch grid")
    pixels = pixels.unsqueeze(1)
    return pixels * image + (1 - pixels) * noise


@torch.inference_mode()
def euler_sample(model, noise, patch_emb, ext_emb, steps: int):
    """Denoise every patch together along the usual shared-time Euler trajectory."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("Sampling steps must be a positive integer")
    x = noise.clone()
    for step in range(steps):
        times = torch.full((x.size(0), 1, 1), step / steps, device=x.device)
        velocity = model(x, times, patch_emb, ext_emb)
        x = x + velocity.float() / steps
    return x
