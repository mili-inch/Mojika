import torch.nn as nn
from .attention import Attention
from .mlp import MLP
from .modulation import DiMRModulation


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qk_norm,
        rope_theta,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
        )
        ffn_hidden_dim = int(dim * mlp_ratio)
        self.ffn = MLP(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
        )
        self.modulation = DiMRModulation(
            dim,
            {
                "attn_scale": 0.01,
                "attn_shift": 0.01,
                "attn_gate": 0.0,
                "ffn_scale": 0.01,
                "ffn_shift": 0.01,
                "ffn_gate": 0.0,
            },
        )

    def forward(self, x, t, condition, positions):
        scale_norm1, shift_norm1, gate_norm1, scale_norm2, shift_norm2, gate_norm2 = (
            self.modulation(t, condition)
        )

        # Apply modulation
        x_modulated_attn = self.norm1(x) * (1 + scale_norm1) + shift_norm1
        attn_out = self.attn(x_modulated_attn, positions)
        x = x + gate_norm1 * attn_out

        x_modulated_ffn = self.norm2(x) * (1 + scale_norm2) + shift_norm2
        ffn_out = self.ffn(x_modulated_ffn)
        x = x + gate_norm2 * ffn_out

        return x
