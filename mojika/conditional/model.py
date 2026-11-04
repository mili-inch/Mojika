from mojika.utils.init_positions_2d import init_positions_2d
from torch import nn

from .modules.modulation import DiMRModulation
from .modules.transformer_block import TransformerBlock


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        rope_theta: float,
        qk_norm: bool,
    ):
        super().__init__()
        if img_size <= 0 or patch_size <= 0 or img_size % patch_size:
            raise ValueError("Image size must be a positive multiple of patch size")
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by the head count")
        if (embed_dim // num_heads) % 4:
            raise ValueError("RoPE requires the head dimension to be divisible by four")
        if depth <= 0 or in_chans <= 0 or mlp_ratio <= 0 or rope_theta <= 0:
            raise ValueError("Model dimensions and RoPE theta must be positive")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, patch_size, patch_size, bias=False
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, qk_norm, rope_theta)
                for _ in range(depth)
            ]
        )
        self.norm_final = nn.RMSNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.final_modulation = DiMRModulation(
            embed_dim, {"scale": 0.01, "shift": 0.01}
        )
        self.proj_out = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=False)
        grid_size = img_size // patch_size
        self.register_buffer(
            "positions", init_positions_2d(grid_size, grid_size), persistent=False
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.proj_out.weight)

    def unpatchify(self, x):
        batch = x.shape[0]
        grid = self.img_size // self.patch_size
        x = x.reshape(
            batch, grid, grid, self.patch_size, self.patch_size, self.in_chans
        )
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self.in_chans, self.img_size, self.img_size
        )

    def forward(self, x, t, patch_emb=None, ext_emb=None):
        """Denoise with image-wide or local times and packed text coefficients."""
        grid = self.img_size // self.patch_size
        if t.shape not in ((x.size(0), 1, 1), (x.size(0), grid, grid)):
            raise ValueError(
                "Timesteps must have shape [batch, 1, 1] for image-wide times "
                "or [batch, patch_rows, patch_columns] for local times"
            )
        # Packed coefficients: six per block, output affine.
        count = 6 * len(self.blocks) + 2
        if ext_emb is None:
            conditions = (None,) * (len(self.blocks) + 1)
        else:
            if ext_emb.shape != (x.size(0), count * self.embed_dim):
                raise ValueError(
                    "Text coefficients must have shape [batch, (6 * depth + 2) * embed_dim]"
                )
            conditions = ext_emb.unflatten(-1, (count, self.embed_dim)).split(
                (*(6 for _ in self.blocks), 2), dim=1
            )
        t = t.flatten(1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if patch_emb is not None:
            x = x + patch_emb.to(dtype=x.dtype)
        for index, block in enumerate(self.blocks):
            x = block(x, t, conditions[index], self.positions)
        scale, shift = self.final_modulation(t, conditions[-1])
        x = self.norm_final(x) * (1 + scale) + shift
        return self.unpatchify(self.proj_out(x))
