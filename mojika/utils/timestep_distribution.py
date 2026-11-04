"""A logit-normal density with independently set endpoint densities.

p(t) = (1 - (a + b)/2) * q(t) + a*(1-t) + b*t,
where a=noise_density, b=image_density, and q is logit-normal.
Its density tends to zero at both endpoints, so p(0)=a and p(1)=b.
The constraints a,b >= 0 and a+b <= 2 ensure normalization.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TimeSamplingParameters:
    location: float
    scale: float
    noise_density: float
    image_density: float

    def __post_init__(self):
        for name in (
            "location",
            "scale",
            "noise_density",
            "image_density",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.noise_density < 0 or self.image_density < 0:
            raise ValueError("Endpoint densities must be nonnegative")
        if self.noise_density + self.image_density > 2:
            raise ValueError("Endpoint densities must sum to at most 2")


def logit_normal_cdf(times: Tensor, location, scale) -> Tensor:
    """CDF on [0, 1]; location is finite and scale is positive."""
    normal = (torch.logit(times) - location) / scale
    return 0.5 * torch.erfc(-normal / math.sqrt(2))


def time_cdf(times: Tensor, location, scale, noise_density, image_density) -> Tensor:
    """CDF on [0, 1]; callers supply validated scalars or constrained fit tensors."""
    core = logit_normal_cdf(times, location, scale)
    core_weight = 1 - (noise_density + image_density) / 2
    baseline = (
        noise_density * (times - times.square() / 2)
        + image_density * times.square() / 2
    )
    return core_weight * core + baseline


def time_density(
    times: Tensor, location, scale, noise_density, image_density
) -> Tensor:
    """Density on [0, 1], with the analytic endpoint limits evaluated explicitly."""
    endpoint = (times == 0) | (times == 1)
    interior = torch.where(endpoint, 0.5, times)
    normal = (torch.logit(interior) - location) / scale
    log_density = (
        -0.5 * normal.square()
        - 0.5 * math.log(2 * math.pi)
        - math.log(scale)
        - torch.log(interior)
        - torch.log1p(-interior)
    )
    core = torch.where(endpoint, 0.0, log_density.exp())
    return (
        (1 - (noise_density + image_density) / 2) * core
        + noise_density * (1 - times)
        + image_density * times
    )


def draws_to_time(
    normal: Tensor,
    uniform: Tensor,
    component: Tensor,
    location,
    scale,
    noise_density,
    image_density,
) -> Tensor:
    """Map independent normal/uniform draws to the normalized mixture.

    The continuous triangular components have densities 2*(1-t) and 2*t.
    Their respective probabilities are noise_density/2 and image_density/2.
    """
    core = torch.sigmoid(location + scale * normal)
    toward_noise = uniform / (1 + torch.sqrt(1 - uniform))
    toward_image = torch.sqrt(uniform)
    return torch.where(
        component < noise_density / 2,
        toward_noise,
        torch.where(
            component < (noise_density + image_density) / 2, toward_image, core
        ),
    )


def sample_times(
    parameters: TimeSamplingParameters,
    sample_shape: tuple[int, ...],
    *,
    device: torch.device | str,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw float32 times on the requested device without CPU synchronization.

    The floor is a continuous density, not probability atoms at 0 or 1.
    Float32 arithmetic can round extreme samples to the endpoints.
    """
    normal = torch.randn(
        sample_shape, dtype=torch.float32, device=device, generator=generator
    )
    uniform, component = torch.rand(
        (2, *sample_shape), dtype=torch.float32, device=device, generator=generator
    ).unbind(0)
    return draws_to_time(
        normal,
        uniform,
        component,
        parameters.location,
        parameters.scale,
        parameters.noise_density,
        parameters.image_density,
    )
