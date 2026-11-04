"""Staggered AB2 trajectories with one batched model evaluation per step."""

import torch


class AB2Stream:
    def __init__(self, model, noise, patch_emb, ext_emb, steps: int):
        if type(steps) is not int or steps < 1:
            raise ValueError("Sampling steps must be a positive integer")
        if noise.shape != (1, 1, model.img_size, model.img_size):
            raise ValueError("Stream noise must contain one grayscale image")
        self.model = model
        self.steps = steps
        self.images = noise.expand(steps, -1, -1, -1).clone()
        self.patch_emb = patch_emb.expand(steps, -1, -1).clone()
        self.ext_emb = ext_emb.expand(steps, -1).clone()
        self.previous_velocity = torch.zeros_like(self.images)
        self.ages = torch.arange(steps, device=noise.device)
        self.iteration = 0

    @torch.inference_mode()
    def step(self, noise, patch_emb, ext_emb):
        # Each image has one time shared by all its spatial patches.
        times = (self.ages.float() / self.steps)[:, None, None]
        velocity = self.model(self.images, times, self.patch_emb, self.ext_emb).float()
        self.images.add_(velocity, alpha=1 / self.steps)
        if self.iteration > 0:
            # Euler initializes every trajectory; AB2 then uses its own history.
            correct = (self.ages > 0)[:, None, None, None]
            self.images.add_(
                (velocity - self.previous_velocity) * correct,
                alpha=0.5 / self.steps,
            )
        self.previous_velocity.copy_(velocity)
        head = (self.steps - 1 - self.iteration) % self.steps
        self.iteration += 1
        # Initial staggered lanes have incomplete trajectories; never display them.
        completed = self.images[head].clone() if self.iteration >= self.steps else None
        self.images[head].copy_(noise[0])
        self.patch_emb[head].copy_(patch_emb[0])
        self.ext_emb[head].copy_(ext_emb[0])
        self.previous_velocity[head].zero_()
        self.ages.add_(1).remainder_(self.steps)
        return completed
