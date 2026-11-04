"""Quintic C2 noise paths normalized by theoretical covariance.

Jointly Gaussian inputs and sample-independent controls give N(0, I) at every
time slice. At most five scalar coordinates describe a segment; their covariance
stays on the CPU. Indexed Gaussian bridges allow O(log n) target lookup.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from .spline import differentiate, evaluate, quintic_coefficients


def mix_noise(source, fresh, mix):
    """Keep unit marginal variance when source and fresh are independent N(0, I)."""
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("Noise mix must be finite and between 0 and 1.")
    return math.sqrt(1 - mix * mix) * source + mix * fresh


class NoiseTargets:
    """A seeded Gaussian AR(1) sequence, with random access to target indices.

    Consecutive targets have correlation sqrt(1 - mix**2). Power-of-two anchors
    and conditional Gaussian midpoints give the same sequence for any access
    order. Only logarithmically many anchors and four recent points are stored.
    """

    def __init__(self, initial, seed, mix):
        if not math.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError("Noise mix must be finite and between 0 and 1.")
        self.initial, self.seed, self.mix = initial, seed, mix
        self.log_correlation = 0.5 * math.log1p(-mix * mix) if 0 < mix < 1 else 0
        self.anchors = {0: initial}
        self.recent = OrderedDict()

    def normal(self, tag):
        digest = hashlib.blake2b(f"{self.seed}:{tag}".encode(), digest_size=8).digest()
        generator = torch.Generator(device=self.initial.device)
        generator.manual_seed(int.from_bytes(digest, "little"))
        return torch.randn(
            self.initial.shape,
            dtype=self.initial.dtype,
            device=self.initial.device,
            generator=generator,
        )

    def decay(self, distance):
        log_decay = self.log_correlation * distance
        return math.exp(log_decay), -math.expm1(2 * log_decay)

    def point(self, index):
        if index < 0:
            raise ValueError("Target index cannot be negative.")
        if index == 0 or self.mix == 0:
            return self.initial
        if index in self.recent:
            self.recent.move_to_end(index)
            return self.recent[index]
        point = self.normal(index + 1) if self.mix == 1 else self.bridge(index)
        self.recent[index] = point
        if len(self.recent) > 4:
            self.recent.popitem(last=False)
        return point

    def bridge(self, index):
        right = 1
        while True:
            left = right // 2
            if right not in self.anchors:
                decay, variance = self.decay(right - left)
                self.anchors[right] = decay * self.anchors[left] + math.sqrt(
                    variance
                ) * self.normal(f"anchor:{right}")
            if right >= index:
                break
            right *= 2
        if right == index:
            return self.anchors[right]
        start, end = self.anchors[left], self.anchors[right]
        while True:
            middle = (left + right) // 2
            decay, variance = self.decay(middle - left)
            denominator = 1 + decay * decay
            point = decay / denominator * (start + end) + math.sqrt(
                variance / denominator
            ) * self.normal(f"bridge:{left}:{right}")
            if middle == index:
                return point
            if index < middle:
                right, end = middle, point
            else:
                left, start = middle, point


def combine(weights, values):
    return sum(
        (float(weight) * value for weight, value in zip(weights, values) if weight),
        torch.zeros_like(values[0]),
    )


@dataclass(frozen=True)
class NoiseSegment:
    duration: float
    start: torch.Tensor
    start_velocity: torch.Tensor
    start_acceleration: torch.Tensor
    end: torch.Tensor
    end_velocity: torch.Tensor
    end_acceleration: torch.Tensor
    coefficients: tuple
    velocity_coefficients: tuple
    acceleration_coefficients: tuple
    weights: tuple
    covariance: np.ndarray
    end_weights: np.ndarray

    @classmethod
    def create(cls, duration, basis, covariance, endpoints):
        start, velocity, acceleration, end, end_velocity, end_acceleration = endpoints
        weights = quintic_coefficients(*endpoints, duration)
        reverse = quintic_coefficients(
            end,
            end_velocity,
            end_acceleration,
            start,
            velocity,
            acceleration,
            -duration,
        )
        # Evaluate from the nearer endpoint. This avoids cancellation near u=1
        # when a short transition is stretched to a long duration in float32.
        coefficients = tuple(
            tuple(combine(row, basis) for row in w) for w in (weights, reverse)
        )
        velocities = tuple(
            differentiate(c, d) for c, d in zip(coefficients, (duration, -duration))
        )
        return cls(
            duration,
            *(
                combine(row, basis)
                for row in (
                    start,
                    velocity,
                    acceleration,
                    end,
                    end_velocity,
                    end_acceleration,
                )
            ),
            coefficients,
            velocities,
            tuple(
                differentiate(c, d) for c, d in zip(velocities, (duration, -duration))
            ),
            (weights, reverse),
            covariance,
            end,
        )

    def normalization(self, u):
        if not math.isfinite(u) or not 0 <= u <= 1:
            raise ValueError("Segment progress must be finite and between 0 and 1.")
        index, progress = (0, u) if u <= 0.5 else (1, 1 - u)
        duration = self.duration if index == 0 else -self.duration
        first = differentiate(self.weights[index], duration)
        weights, derivative, second = (
            evaluate(coefficients, progress)
            for coefficients in (
                self.weights[index],
                first,
                differentiate(first, duration),
            )
        )
        variance = float(weights @ self.covariance @ weights)
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("Theoretical noise variance must be positive and finite.")
        scale = 1 / math.sqrt(variance)
        log_speed = float(derivative @ self.covariance @ weights) / variance
        correction = (
            3 * log_speed**2
            - float(
                second @ self.covariance @ weights
                + derivative @ self.covariance @ derivative
            )
            / variance
        )
        return (
            scale,
            log_speed,
            correction,
            np.stack(
                (
                    weights,
                    derivative - weights * log_speed,
                    second - 2 * log_speed * derivative + correction * weights,
                )
            )
            * scale,
        )

    def sample(self, u):
        scale, log_speed, _, _ = self.normalization(u)
        if u == 0:
            return self.start, self.start_velocity
        if u == 1:
            return self.end, self.end_velocity
        index, progress = (0, u) if u <= 0.5 else (1, 1 - u)
        position = evaluate(self.coefficients[index], progress)
        velocity = evaluate(self.velocity_coefficients[index], progress)
        return position * scale, (velocity - position * log_speed) * scale

    def acceleration(self, u):
        scale, log_speed, correction, _ = self.normalization(u)
        if u == 0:
            return self.start_acceleration
        if u == 1:
            return self.end_acceleration
        index, progress = (0, u) if u <= 0.5 else (1, 1 - u)
        position = evaluate(self.coefficients[index], progress)
        velocity = evaluate(self.velocity_coefficients[index], progress)
        acceleration = evaluate(self.acceleration_coefficients[index], progress)
        return (acceleration - 2 * log_speed * velocity + correction * position) * scale

    def moments(self, u):
        """Covariance of [position, velocity, acceleration, destination]."""
        _, _, _, state = self.normalization(u)
        weights = np.vstack((state, self.end_weights))
        covariance = weights @ self.covariance @ weights.T
        # Matrix products can round the two triangles differently when velocity
        # and acceleration have very different scales. Covariance is symmetric.
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(covariance).all() or np.min(np.diag(covariance)) < -1e-10:
            raise ValueError("Invalid theoretical state covariance.")
        # Enforce the analytic unit-variance identities after float64 arithmetic.
        covariance[0, 0] = covariance[3, 3] = 1
        covariance[0, 1] = covariance[1, 0] = 0
        covariance[1, 1] = max(0.0, covariance[1, 1])
        covariance[0, 2] = covariance[2, 0] = -covariance[1, 1]
        return covariance


class NoiseSpline:
    """Covariance-normalized Hermite through correlated Gaussian targets.

    covariance describes [start, velocity, acceleration, target]. Unit position
    variance requires Cov(z, v)=0 and Cov(z, a)=-Var(v). Quintic replanning
    preserves position, velocity and acceleration.
    """

    def __init__(
        self,
        start,
        velocity,
        acceleration,
        target,
        now,
        interval,
        automatic,
        seed,
        mix,
        covariance,
    ):
        if not math.isfinite(now) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("A path requires a finite start and positive interval.")
        covariance = np.array(covariance, dtype=np.float64, copy=True)
        if (
            covariance.shape != (4, 4)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0)
            or not np.allclose(covariance[[0, 3], [0, 3]], 1, atol=1e-10, rtol=0)
            or abs(covariance[0, 1]) > 1e-10
            or not np.isclose(
                covariance[0, 2], -covariance[1, 1], atol=1e-10, rtol=1e-10
            )
            or np.linalg.eigvalsh(covariance).min()
            < -1e-10 * max(1, np.max(np.abs(covariance)))
        ):
            raise ValueError(
                "Expected a valid unit-position Gaussian state covariance."
            )
        self.covariance = covariance
        self.start, self.velocity, self.target = start, velocity, target
        self.start_acceleration = acceleration
        self.epoch, self.interval = now, interval
        self.automatic = automatic
        self.targets = NoiseTargets(target, seed, mix)
        self.index = 0
        self.segment = self._segment(0)

    def point(self, index):
        if index == 0:
            return self.start
        return self.targets.point(index - 1)

    def point_covariance(self, left, right):
        if left == right:
            return 1.0
        rho = math.sqrt(1 - self.targets.mix**2)
        if left == 0 or right == 0:
            return self.covariance[0, 3] * rho ** (max(left, right) - 1)
        return rho ** abs(left - right)

    def _segment(self, index):
        duration = self.interval
        if index == 0:
            # [current position, velocity, acceleration, target, following target]
            basis = (
                self.start,
                self.velocity,
                self.start_acceleration,
                self.point(1),
                self.point(2),
            )
            covariance = np.empty((5, 5))
            covariance[:4, :4] = self.covariance
            rho = math.sqrt(1 - self.targets.mix**2)
            covariance[4, :4] = covariance[:4, 4] = rho * self.covariance[:, 3]
            covariance[4, 4] = 1
            start, velocity, acceleration, end, following = np.eye(5)
        else:
            indices = range(index - 1, index + 3)
            basis = tuple(self.point(i) for i in indices)
            covariance = np.array(
                [[self.point_covariance(i, j) for j in indices] for i in indices]
            )
            start, end = np.eye(4)[1:3]
            previous, following = np.eye(4)[[0, 3]]
            velocity, acceleration = self.knot_state(previous, start, end, covariance)
        if self.automatic:
            end_velocity, end_acceleration = self.knot_state(
                start, end, following, covariance
            )
        else:
            end_velocity = end_acceleration = np.zeros(len(basis))
        return NoiseSegment.create(
            duration,
            basis,
            covariance,
            (start, velocity, acceleration, end, end_velocity, end_acceleration),
        )

    def knot_state(self, previous, position, following, covariance):
        velocity = (following - previous) / (2 * self.interval)
        velocity -= float(position @ covariance @ velocity) * position
        acceleration = (previous - 2 * position + following) / self.interval**2
        # Unit theoretical variance must have zero first AND second derivative.
        acceleration -= (
            float(
                position @ covariance @ acceleration + velocity @ covariance @ velocity
            )
            * position
        )
        return velocity, acceleration

    def _select_segment(self, now):
        if not math.isfinite(now) or now < self.epoch:
            raise ValueError("Path time must be finite and cannot precede its start.")
        progress = (now - self.epoch) / self.interval
        if not math.isfinite(progress):
            raise ValueError("Path progress must be finite.")
        # Split one time coordinate; subtracting a rounded absolute knot time
        # independently can put the sample outside the selected segment.
        index = int(progress) if self.automatic else 0
        u = progress - index if self.automatic else min(progress, 1.0)
        if index != self.index:
            self.segment, self.index = self._segment(index), index
        return self.segment, u

    def sample(self, now):
        segment, u = self._select_segment(now)
        return segment.sample(u)

    def moments(self, now):
        """Covariance of [position, velocity, acceleration, destination]."""
        segment, u = self._select_segment(now)
        covariance = segment.moments(u)
        if not self.automatic and u == 1:
            covariance[1:3, :] = covariance[:, 1:3] = 0
        return covariance

    def acceleration(self, now):
        segment, u = self._select_segment(now)
        if not self.automatic and u == 1:
            return torch.zeros_like(segment.end)
        return segment.acceleration(u)
