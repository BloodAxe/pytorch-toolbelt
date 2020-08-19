import torch
from torch import nn

__all__ = ["Normalize"]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float().reshape(1, len(mean), 1, 1).contiguous())
        self.register_buffer("std", torch.tensor(std).float().reshape(1, len(std), 1, 1).reciprocal().contiguous())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input.type_as(self.mean) - self.mean) * self.std
