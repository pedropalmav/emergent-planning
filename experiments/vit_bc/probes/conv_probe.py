import torch
import torch.nn as nn
from typing import Optional


class ConvProbe(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        kernel_size: int,
        padding: int = 0,
        nl: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()

    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        out = self.conv(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
