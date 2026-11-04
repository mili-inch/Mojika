import torch.nn as nn
import torch
from torch.nn import functional as F


def normalize_embedding(x: torch.Tensor) -> torch.Tensor:
    """Normalize encoder features in FP32, identically for statistics and inference."""
    return F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)


class MaskEmbedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mask_embed = nn.Conv2d(1, embed_dim, kernel_size=1, stride=1, bias=False)
        self.mask_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # [B, 1, H, W]
        # [B, embed_dim, H/P, W/P]
        m = self.mask_embed(x)
        # [B, embed_dim, H/P, W/P]
        # [B, (H/P)*(W/P), embed_dim]
        m = m.flatten(2).transpose(1, 2)
        # [B, (H/P)*(W/P), embed_dim]
        mask_emb = m * self.mask_gate
        # [B, (H/P)*(W/P), embed_dim]
        return mask_emb


class VectorEmbedder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mean: torch.Tensor):
        super().__init__()
        if (
            mean.shape != (input_dim,)
            or not mean.is_floating_point()
            or not torch.isfinite(mean).all()
        ):
            raise ValueError(
                f"Expected a finite floating-point mean of shape ({input_dim},)"
            )
        self.register_buffer("mean", mean.detach().float().clone())
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def prepare(self, x):
        return normalize_embedding(x) - self.mean

    def forward(self, x):
        return self.linear(self.prepare(x))
