"""Quintic Hermite polynomials and finite C2 condition transitions."""

from dataclasses import dataclass
import math

import torch


def quintic_coefficients(
    start, velocity, acceleration, end, end_velocity, end_acceleration, duration
):
    delta = end - start
    v0, v1 = velocity * duration, end_velocity * duration
    a0, a1 = acceleration * duration**2, end_acceleration * duration**2
    return (
        start,
        v0,
        a0 / 2,
        10 * delta - 6 * v0 - 4 * v1 - 1.5 * a0 + 0.5 * a1,
        -15 * delta + 8 * v0 + 7 * v1 + 1.5 * a0 - a1,
        6 * delta - 3 * v0 - 3 * v1 - 0.5 * a0 + 0.5 * a1,
    )


def differentiate(coefficients, duration):
    return tuple(i * value / duration for i, value in enumerate(coefficients) if i)


def evaluate(coefficients, u):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * u + coefficient
    return value


def validate_segment(start, velocity, acceleration, target, now, duration):
    if not math.isfinite(now) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("A spline requires a finite start time and positive duration.")
    for value in (start, velocity, acceleration, target):
        if (
            value.shape != start.shape
            or value.device != start.device
            or value.dtype != start.dtype
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "Spline tensors must be finite and share shape/device/dtype."
            )


def progress(now, start_time, duration):
    if not math.isfinite(now) or now < start_time:
        raise ValueError("Spline time must be finite and cannot precede its start.")
    return min((now - start_time) / duration, 1.0)


@dataclass(frozen=True)
class HermitePath:
    start_time: float
    duration: float
    start: torch.Tensor
    start_velocity: torch.Tensor
    start_acceleration: torch.Tensor
    target: torch.Tensor
    coefficients: tuple
    velocity_coefficients: tuple
    acceleration_coefficients: tuple

    @classmethod
    def create(cls, start, velocity, acceleration, target, now, duration):
        validate_segment(start, velocity, acceleration, target, now, duration)
        zero = torch.zeros_like(target)
        # Evaluate from the nearer endpoint to limit float32 cancellation when
        # a fast-moving condition is retargeted over a much longer duration.
        coefficients = (
            quintic_coefficients(
                start, velocity, acceleration, target, zero, zero, duration
            ),
            quintic_coefficients(
                target, zero, zero, start, velocity, acceleration, -duration
            ),
        )
        velocities = tuple(
            differentiate(c, d) for c, d in zip(coefficients, (duration, -duration))
        )
        return cls(
            now,
            duration,
            start,
            velocity,
            acceleration,
            target,
            coefficients,
            velocities,
            tuple(
                differentiate(c, d) for c, d in zip(velocities, (duration, -duration))
            ),
        )

    def sample(self, now):
        u = progress(now, self.start_time, self.duration)
        if u == 0:
            return self.start, self.start_velocity, self.start_acceleration
        if u == 1:
            return (
                self.target,
                torch.zeros_like(self.target),
                torch.zeros_like(self.target),
            )
        index, local_u = (0, u) if u <= 0.5 else (1, 1 - u)
        return tuple(
            evaluate(coefficients[index], local_u)
            for coefficients in (
                self.coefficients,
                self.velocity_coefficients,
                self.acceleration_coefficients,
            )
        )
