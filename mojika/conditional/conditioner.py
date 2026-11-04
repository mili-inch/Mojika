import torch
from torch import nn

from .modules.embedders import MaskEmbedder, VectorEmbedder


class Conditioner(nn.Module):
    """Use independent rank-256 text projections for each block and the output."""

    def __init__(
        self,
        gpt_mean: torch.Tensor,
        clip_mean: torch.Tensor,
        embed_dim: int,
        patch_grid: int,
        depth: int,
    ):
        super().__init__()
        if embed_dim <= 0 or patch_grid <= 0 or depth <= 0:
            raise ValueError("Conditioner dimensions must be positive")
        self.register_buffer("mask_uncond", torch.ones(1, patch_grid, patch_grid))
        rank = 256
        latent_dim = (depth + 1) * rank
        # Each rank-sized row slice is an independent layer's input projection A.
        self.gpt_embedder = VectorEmbedder(1280, latent_dim, gpt_mean)
        self.clip_embedder = VectorEmbedder(512, latent_dim, clip_mean)
        self.modulation_projections = nn.ModuleList(
            nn.Linear(rank, count * embed_dim, bias=False)
            for count in [6] * depth + [2]
        )
        self.mask_embedder = MaskEmbedder(embed_dim=embed_dim)

        for embedder in (self.gpt_embedder, self.clip_embedder):
            for weight in embedder.linear.weight.split(rank):
                nn.init.xavier_uniform_(weight)
        for projection in self.modulation_projections:
            nn.init.zeros_(projection.weight)

    def forward(self, x, mask=None, gpt=None, clip=None, condition_mode=None):
        """Prepare raw inputs; mode bits 4/2 select GPT/CLIP rows in a mixed batch."""
        batch_size = x.size(0)
        if condition_mode is not None and (
            condition_mode.shape != (batch_size,) or condition_mode.dtype != torch.int64
        ):
            raise ValueError("condition_mode must be an int64 tensor of shape (batch,)")
        gpt = (
            self.gpt_embedder.mean.new_zeros(batch_size, 1280)
            if gpt is None
            else self.gpt_embedder.prepare(gpt)
        )
        clip = (
            self.clip_embedder.mean.new_zeros(batch_size, 512)
            if clip is None
            else self.clip_embedder.prepare(clip)
        )
        if condition_mode is not None:
            gpt = torch.where((condition_mode & 4).bool().unsqueeze(-1), gpt, 0)
            clip = torch.where((condition_mode & 2).bool().unsqueeze(-1), clip, 0)
        if mask is None:
            mask = self.mask_uncond.expand(batch_size, -1, -1)
        return self.project_centered(mask, gpt, clip)

    def project_centered(self, mask, gpt, clip):
        """Project centered features, preserving zero conditions and interpolated magnitudes."""
        patch_emb = self.mask_embedder(mask.unsqueeze(1))
        latent = self.gpt_embedder.linear(gpt) + self.clip_embedder.linear(clip)
        ext_emb = torch.cat(
            [
                projection(layer_latent)
                for projection, layer_latent in zip(
                    self.modulation_projections,
                    latent.chunk(len(self.modulation_projections), dim=-1),
                    strict=True,
                )
            ],
            dim=-1,
        )
        return patch_emb, ext_emb
