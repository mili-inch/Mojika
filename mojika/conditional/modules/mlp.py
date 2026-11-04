import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        hidden = self.in_proj(x)
        gate = self.act(self.gate_proj(x))
        return self.out_proj(hidden * gate)
