from torch import nn, Tensor
import torch

__all__ = ["GlobalResponseNormalization2d"]


class GlobalResponseNormalization2d(nn.Module):
    """
    Global Response Normalization layer from ConvNextV2
    https://arxiv.org/abs/2301.00808
    """
    
    def __init__(self, in_channels: int, epsilon: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.epsilon = epsilon
        
        torch.nn.init.constant_(self.gamma, 0.5)
        torch.nn.init.constant_(self.beta, 0.0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, C, H, W]
        """
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / gx.mean(dim=-1, keepdim=True).add_(self.epsilon)
        return self.gamma * (x * nx) + self.beta + x
