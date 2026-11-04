"""Spline transitions, independent of HTTP and browser sessions.

GPT/CLIP values are RMS-normalized and centered. Hints are unnormalized pixels.
"""

import random

import torch

from .noise import NoiseSpline, mix_noise
from .schema import Controls
from .spline import HermitePath


class Interpolation:
    def __init__(self, unconditional, device, seed, now):
        self.unconditional = unconditional
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.seeds = random.Random(seed)
        self.targets = {key: value.clone() for key, value in self.unconditional.items()}
        self.targets["noise"] = self.random_noise()
        self.values = {key: value.clone() for key, value in self.targets.items()}
        self.velocities = {
            key: torch.zeros_like(value) for key, value in self.values.items()
        }
        self.accelerations = {
            key: torch.zeros_like(value) for key, value in self.unconditional.items()
        }
        self.controls = Controls.initial()
        self.condition_paths = {}
        rho = (1 - self.controls.noise_mix**2) ** 0.5
        self.path = NoiseSpline(
            self.values["noise"],
            self.velocities["noise"],
            torch.zeros_like(self.values["noise"]),
            self.next_noise(self.controls),
            now,
            self.controls.transition_seconds,
            self.controls.noise_auto,
            self.seeds.getrandbits(64),
            self.controls.noise_mix,
            [[1, 0, 0, rho], [0, 0, 0, 0], [0, 0, 0, 0], [rho, 0, 0, 1]],
        )
        self.targets["noise"] = self.path.segment.end

    def random_noise(self):
        return torch.randn(1, 1, 256, 256, device=self.device, generator=self.generator)

    def next_noise(self, controls):
        return mix_noise(self.values["noise"], self.random_noise(), controls.noise_mix)

    def noise_path(self, now, controls, refresh):
        start, velocity = self.values["noise"], self.velocities["noise"]
        interval, automatic, mix = (
            controls.transition_seconds,
            controls.noise_auto,
            controls.noise_mix,
        )
        seed = self.seeds.getrandbits(64)
        covariance = self.path.moments(now)
        acceleration = self.path.acceleration(now)
        target = self.targets["noise"]
        if refresh:
            target = self.next_noise(controls)
            rho = (1 - mix * mix) ** 0.5
            covariance[:3, 3] = covariance[3, :3] = rho * covariance[:3, 0]
            covariance[3, 3] = 1
        return NoiseSpline(
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
        )

    def condition_path(self, key, target, now, controls):
        return HermitePath.create(
            self.values[key],
            self.velocities[key],
            self.accelerations[key],
            target,
            now,
            controls.transition_seconds,
        )

    def update(self, kind, value, now):
        self.advance(now)
        if kind in ("clear_condition", "clear_conditions"):
            keys = self.unconditional.keys() if kind == "clear_conditions" else (value,)
            targets = {key: self.unconditional[key].clone() for key in keys}
            paths = {
                key: self.condition_path(key, target, now, self.controls)
                for key, target in targets.items()
            }
            self.targets.update(targets)
            self.condition_paths.update(paths)
            return
        if kind == "noise":
            self.path = self.noise_path(now, self.controls, refresh=True)
            self.targets["noise"] = self.path.segment.end
            return
        if kind in self.unconditional:
            # Construct before committing: invalid inputs leave targets intact.
            path = self.condition_path(kind, value, now, self.controls)
            self.condition_paths[kind] = path
            self.targets[kind] = value
            return

        controls = Controls.model_validate({**self.controls.model_dump(), kind: value})
        if controls == self.controls:
            return
        if kind == "noise_mix" and not controls.noise_auto:
            self.controls = controls
            return
        paths = self.condition_paths
        if kind == "transition_seconds":
            paths = {
                key: self.condition_path(key, self.targets[key], now, controls)
                for key in self.condition_paths
            }
        path = self.noise_path(
            now,
            controls,
            refresh=kind == "noise_mix" or (kind == "noise_auto" and value),
        )
        self.condition_paths, self.path = paths, path
        self.targets["noise"] = path.segment.end
        self.controls = controls

    def advance(self, now):
        for key, path in list(self.condition_paths.items()):
            self.values[key], self.velocities[key], self.accelerations[key] = (
                path.sample(now)
            )
            if now >= path.start_time + path.duration:
                del self.condition_paths[key]
        self.values["noise"], self.velocities["noise"] = self.path.sample(now)
        self.targets["noise"] = self.path.segment.end
