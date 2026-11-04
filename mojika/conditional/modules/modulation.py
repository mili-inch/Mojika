import torch
from torch import nn

from mojika.modules.timestep_parameter import DiMRTimestepParameter


class DiMRModulation(nn.Module):
    """Add text coefficients to independently parameterized time coefficients."""

    def __init__(self, dim: int, init_stds: dict[str, float]):
        super().__init__()
        self.time = nn.ModuleDict(
            {
                name: DiMRTimestepParameter(dim, init_std)
                for name, init_std in init_stds.items()
            }
        )

    def forward(self, t, condition):
        coefficients = torch.stack([module(t) for module in self.time.values()], dim=2)
        if condition is not None:
            coefficients = coefficients + condition.to(coefficients.dtype).unsqueeze(1)
        return coefficients.unbind(2)
